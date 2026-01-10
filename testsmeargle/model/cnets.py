# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""
import copy
import os
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download


try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor


from fla.layers.mamba2 import Mamba2
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules.l2warp import l2_warp
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


class Mamba2Cache:
    """
    Cache holder for FLA Mamba2 selective scan and convolution states.
    """

    def __init__(
        self,
        config: "Mamba2Config",
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: str | None = None,
    ):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.intermediate_size + 2 * self.n_groups * self.state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=device,
            dtype=dtype,
        )

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_init: bool = False,
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class Mamba2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mixer = Mamba2(
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
            state_size=config.state_size,
            expand=config.expand,
            n_groups=config.n_groups,
            conv_kernel=config.conv_kernel,
            use_conv_bias=config.use_conv_bias,
            hidden_act=config.hidden_act,
            rms_norm=config.rms_norm,
            chunk_size=config.chunk_size,
            time_step_rank=config.time_step_rank,
            time_step_limit=config.time_step_limit,
            time_step_min=config.time_step_min,
            time_step_max=config.time_step_max,
            use_bias=config.use_bias,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states,
        cache_params: Mamba2Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        if self.residual_in_fp32:
            hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
        return hidden_states

class Mamba2Config(PretrainedConfig):
    model_type = "mamba2"

    def __init__(
        self,
        head_dim: int = 64,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        state_size: int = 128,
        num_hidden_layers: int = 48,
        norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 1,
        conv_kernel: int = 2,
        n_groups: int = 1,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        residual_in_fp32: bool = False,
        time_step_rank: str = "auto",
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        time_step_limit=(0.0, float("inf")),
        rescale_prenorm_residual: bool = True,
        use_cache: bool = True,
        rms_norm: bool = True,
        chunk_size: int = 64,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.num_heads = int(self.expand * self.hidden_size / self.head_dim)
        self.rms_norm = rms_norm
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.tie_word_embeddings = tie_word_embeddings

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True simultaneously.")
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled; if you see divergence consider disabling this setting.",
            )

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register with HF auto classes
AutoConfig.register(Mamba2Config.model_type, Mamba2Config, exist_ok=True)

class MambaBlock(nn.Module):
    """MAMBA block wrapping the inlined FLA Mamba2Block with cache support."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        d_model = config.hidden_size * 2
        d_state = getattr(config, "ssm_state_size", 16)
        d_conv = getattr(config, "ssm_conv_kernel", 4)
        expand = getattr(config, "ssm_expand", 2)

        # Fix: Ensure total_channels (intermediate_size + 2*n_groups*state_size) is divisible by 8
        # This is required by causal_conv1d CUDA kernel
        # intermediate_size = expand * d_model (calculated inside Mamba2Cache)
        # total_channels = intermediate_size + 2 * n_groups * state_size
        n_groups = 1  # Default n_groups in Mamba2Config
        extra_channels = 2 * n_groups * d_state  # 2 * 1 * 16 = 32
        
        # Calculate what intermediate_size should be to make total_channels divisible by 8
        # Since extra_channels=32 is divisible by 8, we need intermediate_size to also be divisible by 8
        # to ensure total_channels is divisible by 8
        intermediate_size_base = expand * d_model
        
        # Round up intermediate_size to next multiple of 8
        if intermediate_size_base % 8 != 0:
            intermediate_size_target = ((intermediate_size_base + 7) // 8) * 8
            # Adjust expand to achieve the target intermediate_size
            expand_adjusted = intermediate_size_target / d_model
            expand = expand_adjusted
            print(f"WARNING: Adjusted expand from {getattr(config, 'ssm_expand', 2)} to {expand:.6f} to ensure intermediate_size (and thus total_channels) is divisible by 8")
        
        d_model_expanded = int(d_model * expand)
        head_dim = math.gcd(d_model_expanded, 64) or 64
        num_heads = max(1, d_model_expanded // head_dim)
        head_dim = d_model_expanded // num_heads
        
        intermediate_size_final = int(expand * d_model)
        total_channels_final = intermediate_size_final + extra_channels
        
        print("d_model_expanded", d_model_expanded)
        print("head_dim", head_dim)
        print("num_heads", num_heads)
        print("expand", expand)
        print("intermediate_size (expand * hidden_size)", intermediate_size_final)
        print("total_channels (intermediate + 2*n_groups*state_size)", total_channels_final)
        print("total_channels divisible by 8:", total_channels_final % 8 == 0)
        mamba2_config = Mamba2Config(
            hidden_size=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=d_state,
            conv_kernel=d_conv,
            expand=expand,
            num_hidden_layers=1,
            layer_norm_epsilon=config.rms_norm_eps if hasattr(config, "rms_norm_eps") else 1e-5,
            use_bias=False,
            use_conv_bias=True,
            use_cache=True,
        )

        self.mamba2_block = Mamba2Block(mamba2_config, layer_idx=0)
        self.mamba2_config = mamba2_config
        self.out_proj = nn.Linear(d_model, config.hidden_size, bias=False)
        self.d_state = d_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Mamba2Cache]]:
        output = self.mamba2_block(
            hidden_states=hidden_states,
            cache_params=cache_params if use_cache else None,
            cache_position=cache_position if use_cache else None,
            attention_mask=None,
        )

        output = output.to(self.out_proj.weight.dtype)
        output_proj = self.out_proj(output)

        if use_cache:
            return output_proj, cache_params
        return output_proj, None

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Replace attention with MAMBA block
        self.mamba_block = MambaBlock(config=config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Mamba2Cache]]:
        """
        Args:
            input_emb: Input embeddings (batch, seq_len, hidden_size)
            hidden_states: Hidden states from target model (batch, seq_len, hidden_size)
            cache_params: Mamba2Cache for incremental processing
            cache_position: Tensor indicating position in sequence for cache
            use_cache: Whether to return updated cache
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        # Concatenate input_emb and hidden_states for MAMBA block
        hidden_states = torch.cat(
            (input_emb, hidden_states), dim=-1
        )  # (batch, seq_len, hidden_size * 2)

        return_hidden = hidden_states

        # MAMBA block (replaces attention) with HF cache support
        hidden_states, updated_cache_params = self.mamba_block(
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)

        return outputs, updated_cache_params


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def len_list(x, n):
    return [i for i in x if len(i) <= n]


class Model(nn.Module):
    def __init__(
        self,
        config,
        load_emb=False,
        path=None,
        bias=True,
        total_tokens=63,
        depth=5,
        top_k=8,
        threshold=1.0,
    ):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.lm_head = nn.Linear(
            config.hidden_size, self.draft_vocab_size, bias=False
        )
        if load_emb and not hasattr(config, "target_hidden_size"):
            from safetensors import safe_open
            import json

            try:
                index_json_path = os.path.join(path, "model.safetensors.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(
                        path, "model.safetensors.index.json"
                    )
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                with safe_open(local_emb_path, framework="pt", device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                index_json_path = os.path.join(path, "pytorch_model.bin.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(
                        path, "pytorch_model.bin.index.json"
                    )
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                weights = torch.load(local_emb_path)
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        self.hidden_size = config.hidden_size
        self.midlayer = LlamaDecoderLayeremb(config)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(
                config.target_hidden_size * 3, self.hidden_size, bias=False
            )
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        if "d2t" not in self._buffers:
            self.register_buffer(
                "d2t", torch.zeros((self.draft_vocab_size), dtype=torch.long)
            )
        if "t2d" not in self._buffers:
            self.register_buffer(
                "t2d", torch.zeros((self.vocab_size), dtype=torch.bool)
            )

        # Check if d2t is not all zeros (warn if so)
        if torch.all(self.d2t == 0):
            print("Warning: d2t is all zeros. Draft-to-target mapping might not be properly initialized.")
            cache = torch.load("/workspace/cache.pt")
            if cache["d2t"].shape == self.d2t.shape:
                self.d2t.copy_(cache["d2t"])
            else:
                self.register_buffer("d2t", cache["d2t"])
            if cache["t2d"].shape == self.t2d.shape:
                self.t2d.copy_(cache["t2d"])
            else:
                self.register_buffer("t2d", cache["t2d"])

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(
            self.top_k, device=self.embed_tokens.weight.device
        )[None, None]
        self.position_ids = torch.zeros(
            self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long
        )
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None,
    ):
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        # Project hidden_states to model hidden_size if mismatched (e.g., concat of multiple layers)
        if hidden_states.shape[-1] != self.hidden_size:
            hidden_states = self.fc(hidden_states)
        
        # Align sequence lengths before concat (hidden_states vs inputs_embeds)
        if hidden_states.shape[1] != inputs_embeds.shape[1]:
            seq_len = min(hidden_states.shape[1], inputs_embeds.shape[1])
            hidden_states = hidden_states[:, -seq_len:, :]
            inputs_embeds = inputs_embeds[:, -seq_len:, :]

        # Normalize cache input shape: tuple(cache_params, cache_position) or None
        if past_key_values is not None:
            cache_params, cache_position = past_key_values
        else:
            cache_params, cache_position = None, None

        (layer_outputs, updated_cache_params) = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            use_cache=use_cache if use_cache is not None else True,
            input_ids=input_ids,
        )

        # Unpack layer outputs: (hidden_states, return_hidden)
        hidden_states = layer_outputs[0]

        if use_cache:
            # cache_position tracks last processed index; update based on processed tokens
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            if cache_position is None:
                cache_position = torch.full(
                    (batch_size,),
                    seq_len - 1,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            else:
                cache_position = cache_position + seq_len
            return hidden_states, (updated_cache_params, cache_position)

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        # Calculate position BEFORE any alignment/truncation (matches EAGLE3)
        len_posi = input_ids.shape[1]
        self.reset()

        # Align input_ids with hidden_states sequence length for Mamba
        # hidden_states only cover accepted tokens; align tokens to match
        seq_len = hidden_states.shape[1]
        input_ids_aligned = input_ids[:, -seq_len:]

        # Prefill once (or reuse stable cache) with correct cache_position tracking
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            cache_params, cache_position = self.stable_kv
            past_key_values = (cache_params, cache_position)
            start_idx = cache_position[0].item() + 1 if cache_position is not None else 0
            incremental_ids = input_ids_aligned[:, start_idx:]
            if incremental_ids.numel() == 0:
                out_hidden, past_key_values = hidden_states, (cache_params, cache_position)
            else:
                out_hidden, past_key_values = self(
                    hidden_states,
                    input_ids=incremental_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
        else:
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids_aligned,
                past_key_values=None,
                use_cache=True,
            )
            if past_key_values is not None:
                cache_params = past_key_values[0]
                batch_size = hidden_states.shape[0]
                cache_position = torch.full(
                    (batch_size,),
                    input_ids_aligned.shape[1] - 1,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                past_key_values = (cache_params, cache_position)

        self.stable_kv = past_key_values
        cache_params, cache_position = past_key_values

        last_hidden = out_hidden[:, -1]
        if last_hidden.shape[-1] != self.hidden_size:
            # fc expects hidden_size*3 input; fall back to a simple linear projection if shapes differ
            if last_hidden.shape[-1] == self.hidden_size * 3:
                last_hidden = self.fc(last_hidden)
            else:
                proj = torch.nn.Linear(
                    last_hidden.shape[-1], self.hidden_size, bias=False
                ).to(last_hidden.device, dtype=last_hidden.dtype)
                last_hidden = proj(last_hidden)

        last_headout = self.lm_head(self.norm(last_hidden))

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.vocab_size == self.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index])
            input_ids = topk_index + self.d2t[topk_index]
        # Initialize: we have top_k branches, each needs its own cache
        # For incremental processing, we need hidden states for the new token position
        # Since we're processing one new token per branch, we use the last hidden state
        # Ensure last_hidden is 1D: (hidden_size,)
        if last_hidden.dim() > 1:
            last_hidden = last_hidden.squeeze(0)  # Remove batch dimension if present
        # Reshape to batch dimension: (hidden_size) -> (top_k, 1, hidden_size)
        input_hidden = last_hidden.unsqueeze(0).unsqueeze(0).repeat(top_k, 1, 1)  # (top_k, 1, hidden_size)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        
        # Expand cache to batch_size=top_k for parallel branch processing
        # If cache exists, we currently avoid expanding it per-branch; fall back to no-cache reuse.
        batch_cache_params = None
        batch_cache_position = None

        for i in range(depth):
            self.tree_mask = tree_mask
            # position_ids for batched branches: each branch is at position len_posi
            # For Mamba, position_ids might not be critical, but we pass it for compatibility
            # Use same position for all branches since they're at the same depth
            position_ids = (len_posi + self.position_ids).unsqueeze(0)  # (1, top_k) - keep original shape for compatibility
            
            # Reshape input_ids to batch dimension: (1, top_k) -> (top_k, 1)
            input_ids_batch = input_ids.squeeze(0).unsqueeze(1)  # (top_k, 1)
            
            # Process all branches incrementally with cache + cache_position
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids_batch,
                past_key_values=(batch_cache_params, batch_cache_position)
                if batch_cache_params is not None
                else None,
                position_ids=position_ids,
                use_cache=True,
            )

            if past_key_values is not None:
                batch_cache_params, batch_cache_position = past_key_values
                # Note: cache_position is already updated by forward() method (adds seq_len=1)
                # No need to increment again here
            
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            # out_hidden is (top_k, 1, hidden_size), get last token for each branch
            last_headout = self.lm_head(self.norm(out_hidden[:, -1]))  # (top_k, vocab_size)
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)  # (top_k, top_k)
            topk_index, topk_p = top.indices, top.values

            # Flatten to (top_k * top_k,) for selection
            cu_scores = topk_p + scores[:, None]  # (1, top_k) + (top_k, top_k) -> (top_k, top_k)

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            # Select which branches to keep
            out_ids = topk_cs_index // top_k  # Which branch
            input_hidden = out_hidden[out_ids]  # (top_k, 1, hidden_size) - select branches

            # Select which tokens from those branches
            token_ids = topk_cs_index % top_k  # Which token within branch
            input_ids = topk_index[out_ids, token_ids].unsqueeze(0)  # (1, top_k)

            if self.vocab_size == self.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                # Keep tokens in base vocab throughout: base_id = draft + d2t[draft]
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])

            scores_list.append(cu_scores)
            tree_mask = torch.cat(
                (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3
            )

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(
            top_scores_index, draft_parents - 1, right=False
        )

        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents


        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del (
            mask_index,
            mask_index_list,
            noleaf_index,
            noleaf_num,
            leaf_num,
            max_depth,
            rid,
        )
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def sequential_generate(self, hidden_states, input_ids, head, logits_processor, num_draft_tokens=None):
        """Generate draft tokens sequentially (non-tree) for speculative decoding.
        
        Args:
            hidden_states: Hidden states from base model
            input_ids: Current input token IDs
            head: Base model LM head (unused, kept for compatibility)
            logits_processor: Logits processor for sampling
            num_draft_tokens: Number of draft tokens to generate (defaults to total_tokens)
        
        Returns:
            draft_tokens: Tensor of shape (num_draft_tokens + 1,) containing sample_token + draft tokens
            hidden_state: Last hidden state for next iteration
        """
        if num_draft_tokens is None:
            num_draft_tokens = self.total_tokens
        
        input_ids = input_ids.to(hidden_states.device)
        sample_token = input_ids[:, -1]
        
        # Align input_ids with hidden_states sequence length for Mamba
        seq_len = hidden_states.shape[1]
        input_ids_aligned = input_ids[:, -seq_len:]
        
        # Prefill once (or reuse stable cache)
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            cache_params, cache_position = self.stable_kv
            past_key_values = (cache_params, cache_position)
            start_idx = cache_position[0].item() + 1 if cache_position is not None else 0
            incremental_ids = input_ids_aligned[:, start_idx:]
            if incremental_ids.numel() == 0:
                out_hidden, past_key_values = hidden_states, (cache_params, cache_position)
            else:
                out_hidden, past_key_values = self(
                    hidden_states,
                    input_ids=incremental_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
        else:
            # Create cache if it doesn't exist - use the same config as the mixer
            if not hasattr(self, "stable_kv") or self.stable_kv is None:
                batch_size = hidden_states.shape[0]
                device = hidden_states.device
                dtype = hidden_states.dtype
                # Get config from midlayer's mamba_block to ensure it matches the mixer weights
                mamba_config = self.midlayer.mamba_block.mamba2_config
                cache_params = Mamba2Cache(
                    mamba_config,
                    batch_size,
                    dtype=dtype,
                    device=device,
                )
                cache_position = torch.full(
                    (batch_size,),
                    input_ids_aligned.shape[1] - 1,
                    device=device,
                    dtype=torch.long,
                )
                past_key_values = (cache_params, cache_position)
            else:
                cache_params, cache_position = self.stable_kv
                past_key_values = (cache_params, cache_position)
            
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids_aligned,
                past_key_values=past_key_values,
                use_cache=True,
            )
            if past_key_values is not None:
                cache_params = past_key_values[0]
                batch_size = hidden_states.shape[0]
                cache_position = torch.full(
                    (batch_size,),
                    input_ids_aligned.shape[1] - 1,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                past_key_values = (cache_params, cache_position)
        
        self.stable_kv = past_key_values
        cache_params, cache_position = past_key_values
        
        # Trim conv_state to width-1 length for causal_conv1d_update compatibility
        # The kernel expects conv_state.shape[-1] = width - 1, but after prefill
        # it may have width elements. Trim to the last element to satisfy the requirement.
        if cache_params is not None:
            if cache_params.conv_states.shape[-1] > 1:
                cache_params.conv_states = cache_params.conv_states[..., -1:].contiguous()
        
        # Generate draft tokens sequentially
        draft_tokens_list = []
        # Take only the last token from out_hidden for sequential generation
        current_hidden = out_hidden[:, -1:]  # (1, 1, hidden_size)
        
        for _ in range(num_draft_tokens):
            # Project hidden state if needed
            if current_hidden.shape[-1] != self.hidden_size:
                if current_hidden.shape[-1] == self.hidden_size * 3:
                    current_hidden = self.fc(current_hidden)
                else:
                    proj = torch.nn.Linear(
                        current_hidden.shape[-1], self.hidden_size, bias=False
                    ).to(current_hidden.device, dtype=current_hidden.dtype)
                    current_hidden = proj(current_hidden)
            
            # Get logits and sample next token
            logits = self.lm_head(self.norm(current_hidden[:, -1]))
            logits = logits.float()
            
            if logits_processor is not None:
                processed_logits = logits_processor(None, logits)
                probs = torch.nn.functional.softmax(processed_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Map draft token to base vocab if needed
            if self.vocab_size != self.draft_vocab_size:
                next_token_base = next_token + self.d2t[next_token]
            else:
                next_token_base = next_token
            
            draft_tokens_list.append(next_token_base.squeeze(-1).item())
            
            # Forward pass to get next hidden state
            # next_token is (1, 1), so embed_tokens returns (1, 1, embed_dim)
            next_token_emb = self.embed_tokens(next_token).to(current_hidden.dtype)  # (1, 1, embed_dim)
            
            # Process with Mamba layer
            layer_outputs, cache_params = self.midlayer(
                input_emb=next_token_emb,
                hidden_states=current_hidden,
                cache_params=cache_params,
                cache_position=cache_position,
                use_cache=True,
                input_ids=next_token,
            )
            
            current_hidden = layer_outputs[0]  # (1, 1, hidden_size)
            
            # Update cache position
            if cache_position is not None:
                cache_position = cache_position + 1
            past_key_values = (cache_params, cache_position)
        
        # Combine sample token with draft tokens
        draft_tokens = torch.tensor([sample_token.item()] + draft_tokens_list, 
                                   device=hidden_states.device, dtype=torch.long)
        
        return draft_tokens, current_hidden

import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())