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
import math
import time
from typing import List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import warnings
from dataclasses import dataclass
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.configuration_utils import PretrainedConfig
from modeling_llama_kv import LlamaForCausalLM
from configs import EConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing
from fla.layers.mamba2 import Mamba2
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules.l2warp import l2_warp
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


logger = logging.get_logger(__name__)


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


class Mamba2PreTrainedModel(PreTrainedModel):
    """
    Base Mamba2 pretrain wrapper for weight init and registration.
    """

    config_class = None  # defined after Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(
        self,
        module: nn.Module,
        num_residuals_per_layer: int = 1,
    ):
        if isinstance(module, Mamba2):
            A = torch.arange(1, module.num_heads + 1)
            with torch.no_grad():
                if not isinstance(module.A_log, DTensor):
                    module.A_log.copy_(torch.log(A))
                else:
                    logger.warning_once("`A_log` is a DTensor, skipping initialization")
            module.A_log._no_weight_decay = True

            nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min),
            ).clamp(min=self.config.time_step_floor)

            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                if not isinstance(module.dt_bias, DTensor):
                    module.dt_bias.copy_(inv_dt)
                else:
                    logger.warning_once("`dt_bias` is a DTensor, skipping initialization")
            module.dt_bias._no_reinit = True

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                if hasattr(module.bias, "_no_reinit"):
                    raise ValueError("Unexpected _no_reinit flag on bias")
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if self.config.rescale_prenorm_residual:
            p = None
            if hasattr(module, "o_proj"):
                raise ValueError("Unexpected o_proj during init")
            elif hasattr(module, "out_proj"):
                p = module.out_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


@dataclass
class Mamba2Output(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    cache_params: Mamba2Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


@dataclass
class Mamba2CausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    cache_params: Mamba2Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


class Mamba2Model(Mamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in list(state_dict.keys()):
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.LongTensor | None = None,
        cache_params: Mamba2Cache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | Mamba2Output:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype,
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                raise ValueError(
                    "cache_position must be provided when use_cache=True and cache_params is passed; "
                    "omit cache_params during prefill to auto-init."
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class Mamba2ForCausalLM(Mamba2PreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_params: Mamba2Cache | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | None = 0,
        **kwargs,
    ) -> tuple | Mamba2CausalLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]

        loss, logits = None, None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )


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


# Bind config class after definition
Mamba2PreTrainedModel.config_class = Mamba2Config

# Register with HF auto classes
AutoConfig.register(Mamba2Config.model_type, Mamba2Config, exist_ok=True)
AutoModel.register(Mamba2Config, Mamba2Model, exist_ok=True)
AutoModelForCausalLM.register(Mamba2Config, Mamba2ForCausalLM, exist_ok=True)

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

        d_model_expanded = d_model * expand
        head_dim = math.gcd(d_model_expanded, 64) or 64
        num_heads = max(1, d_model_expanded // head_dim)
        head_dim = d_model_expanded // num_heads

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
            cache_state: Previous MAMBA state (batch, d_state) or None
            use_cache: Whether to return updated state
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat(
            (input_emb, hidden_states), dim=-1
        )

        return_hidden = hidden_states

        # MAMBA block
        hidden_states, updated_state = self.mamba_block(
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

        return outputs, updated_state


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def process_data(data_chunk):

    token_dict = Counter()
    input_ids = data_chunk["input_ids"]
    loss_mask = data_chunk["loss_mask"]
    for i in range(len(input_ids)):
        ids = input_ids[i][0]
        mask = loss_mask[i][0]
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1

    return token_dict


def merge_dicts(dicts):
    """合并多个 Counter 字典"""
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


class Model(nn.Module):
    def __init__(
        self,
        config,
        ds_config,
        training_config,
        load_head=False,
        load_emb=True,
        path=None,
    ):
        super().__init__()
        # self.layers = nn.ModuleList(
        #     [LlamaDecoderLayer(config, index=index) for index in range(config.num_hidden_layers)])
        self.train_config = training_config
        # Settng dschf to allow efficient ZeRO-3 usage between hf and ds.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None

        self.midlayer = LlamaDecoderLayeremb(config)
        self.gradient_checkpointing = self.train_config["gradient_checkpoint"]
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.length = 7
        # Lazy load target_model to avoid OOM before DeepSpeed initialization
        self._target_model = None
        self._target_model_path = path
        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)

        if not load_emb:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )

        else:

            from safetensors import safe_open
            import json
            import os

            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(
                    os.path.join(path, emb_path), framework="pt", device="cpu"
                ) as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx, _weight=tensor
            )

        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    @property
    def target_model(self):
        """Lazy load target model on first access to avoid OOM before DeepSpeed init"""
        if self._target_model is None:
            import os

            # Try to get local rank from environment to load only on one process if possible
            # But still load on all ranks since target_model is used during forward pass
            # Use low_cpu_mem_usage to minimize memory footprint
            self._target_model = LlamaForCausalLM.from_pretrained(
                self._target_model_path, dtype=torch.float16, low_cpu_mem_usage=True
            )
            self._target_model.eval()
            for param in self._target_model.parameters():
                param.requires_grad = False
        return self._target_model

    def scandata(self, datapath, tokenizerpath, local_rank):
        N = self.draft_vocab_size

        if local_rank != 0 and not os.path.exists("cache.pt"):
            while not os.path.exists("cache.pt"):
                time.sleep(1)
            cache = torch.load("cache.pt")
            d2t = cache["d2t"]
            t2d = cache["t2d"]
        elif local_rank == 0 and not os.path.exists("cache.pt"):
            tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
            dataset = load_dataset("json", data_files=datapath)
            dataset = dataset["train"]
            # dataset = dataset.select(range(96))
            original_columns1 = dataset.column_names
            num_proc = 48

            def preprocess_function(examples):
                new_examples = {
                    # "conversation": [],
                    "input_ids": [],
                    "loss_mask": [],
                }
                for i in range(len(examples["id"])):
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                        },
                    ]
                    convroles = ["user", "assistant"]
                    roles = {"human": "user", "gpt": "assistant"}
                    source = examples["conversations"][i]
                    if not source:
                        continue
                    if roles[source[0]["from"]] != "user":
                        # Skip the first one if it is not from human
                        source = source[1:]
                    for j, sentence in enumerate(source):
                        role = roles[sentence["from"]]
                        assert role == convroles[j % 2], f"{i}"
                        # if sentence["from"]=="gpt":
                        #     sentence["value"]=" "+sentence["value"]
                        messages.append({"role": role, "content": sentence["value"]})
                    conversation = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    if not tokenizer.pad_token_id:
                        tokenizer.pad_token_id = tokenizer.unk_token_id

                    input_ids = tokenizer(
                        conversation,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids[0]
                    # When construct draft model vocab,
                    # filter out samples which is longer than max_len,
                    # instead of truncating them.
                    if len(input_ids) > self.train_config["max_len"]:
                        continue
                    loss_mask = torch.ones_like(input_ids)
                    # print(i)

                    sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                    total_len = len(input_ids)

                    sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
                    turns = conversation.split(sep2)

                    turns[1] = turns[0] + sep2 + turns[1]
                    turns = turns[1:]

                    cur_len = 1
                    loss_mask[:cur_len] = 0
                    for i, turn in enumerate(turns):
                        if turn == "":
                            break
                        turn_len = len(tokenizer(turn).input_ids)

                        parts = turn.split(sep)
                        if len(parts) != 2:
                            break
                        parts[0] += sep
                        # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                        instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                        # Ignore the user instructions
                        if i == 0:
                            loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                        else:
                            loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
                        cur_len += turn_len
                        if i != 0:
                            cur_len += 3
                        # cur_len+=2

                        # if i != 0 and not tokenizer.legacy:
                        #     # The legacy and non-legacy modes handle special tokens differently
                        #     cur_len -= 1

                    loss_mask[cur_len:] = 0

                    # new_examples["conversation"].append(conversation)
                    new_examples["input_ids"].append(input_ids[None, :])
                    new_examples["loss_mask"].append(loss_mask[None, :])

                return new_examples

            dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns1,
                load_from_cache_file=False,
            )
            # dataset.set_format(type="torch")
            num_processes = num_proc
            chunk_size = len(dataset) // num_processes + (
                len(dataset) % num_processes > 0
            )
            chunks = [
                dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)
            ]

            # 创建进程池
            with multiprocessing.Pool(num_processes) as pool:
                # 并行处理数据块
                results = pool.map(process_data, chunks)

            # 合并结果
            token_dict = merge_dicts(results)

            total_frequency = sum(token_dict.values())
            top_N = token_dict.most_common(N)
            top_N_frequency_sum = sum(freq for key, freq in top_N)
            top_N_ratio = top_N_frequency_sum / total_frequency
            print(f"top {N} token frequency ratio: {top_N_ratio:.2%}")
            used_tokens = [key for key, freq in top_N]
            used_tokens.sort()
            d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
            t2d = [i in used_tokens for i in range(self.vocab_size)]
            d2t = torch.tensor(d2t)
            t2d = torch.tensor(t2d)
            cache = {"d2t": d2t, "t2d": t2d}
            torch.save(cache, "cache.pt")
        else:
            cache = torch.load("cache.pt")
            d2t = cache["d2t"]
            t2d = cache["t2d"]
 
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        # Update draft_vocab_size to match actual computed size and fix lm_head if needed
        actual_draft_vocab_size = int(t2d.sum().item())
        if actual_draft_vocab_size != self.draft_vocab_size:
            print(
                f"Warning: config draft_vocab_size ({self.draft_vocab_size}) != actual ({actual_draft_vocab_size}). Updating lm_head..."
            )
            self.draft_vocab_size = actual_draft_vocab_size
            # Recreate lm_head with correct size
            old_lm_head = self.lm_head
            self.lm_head = nn.Linear(
                old_lm_head.in_features, actual_draft_vocab_size, bias=False
            )
            # Initialize with existing weights if possible (first actual_draft_vocab_size outputs)
            if old_lm_head.weight.shape[0] >= actual_draft_vocab_size:
                self.lm_head.weight.data = old_lm_head.weight.data[
                    :actual_draft_vocab_size
                ].clone()
        self.l1smooth = nn.SmoothL1Loss(reduction="none")

    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        # Ensure target_model is on the same device as inputs
        target_model = self.target_model
        model_device = next(target_model.parameters()).device
        if model_device != device:
            target_model = target_model.to(device)
            self._target_model = target_model  # Update cached model on correct device
        outs = target_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states0 = outs.hidden_states[0]
        hidden_states1 = outs.hidden_states[1]
        hidden_states2 = outs.hidden_states[2]
        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )
        # hidden_states=torch.cat((hidden_states0,hidden_states1),dim=-1)
        target = outs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states, target, loss_mask, input_ids = self.dataprepare(
            input_ids, attention_mask, loss_mask
        )

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if (
            self.training
            and self.gradient_checkpointing
            and not hidden_states.requires_grad
        ):
            hidden_states.requires_grad = True

        hidden_states = self.fc(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        self.t2d = self.t2d.to(hidden_states.device)

        cache_params: Optional[Mamba2Cache] = None
        cache_position: Optional[torch.LongTensor] = None
        plosses = []  # Initialize plosses list
        acces = []  # Initialize acces list
        last_target_max = None
        last_logits = None
        for idx in range(self.length):

            inputs_embeds = self.embed_tokens(input_ids)

            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            layer_outputs, cache_params = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                use_cache=True,
            )

            hidden_states_out = layer_outputs[0]
            hidden_states = hidden_states_out
            hidden_states_out = self.norm(hidden_states_out)
            logits = self.lm_head(hidden_states_out)
            logits = logits.float()

            with torch.no_grad():
                target_head = target
                target_max_token = target_head.argmax(-1)
                # Move d2t to the same device as target_max_token
                self.t2d = self.t2d.to(target_max_token.device)
                target_mask = self.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head = target_head[..., self.t2d]
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)

            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            sum_logit = torch.sum(position_mask * plogp, 2)
            loss = -sum_logit.mean()
            plosses.append(loss)
            last_target_max = target_max_token
            last_logits = logits

            # Optional debug logging (tokenizer init only)
            if os.getenv("TRAIN_DEBUG", ""):
                try:
                    if not hasattr(self, "_debug_tokenizer"):
                        tok_path = (
                            self._target_model_path
                            if hasattr(self, "_target_model_path")
                            else None
                        )
                        if tok_path is None:
                            tok_path = getattr(self, "path", None)
                        self._debug_tokenizer = AutoTokenizer.from_pretrained(
                            tok_path if tok_path is not None else "",
                            use_fast=False,
                        )
                except Exception:
                    if not hasattr(self, "_debug_log_failed"):
                        self._debug_log_failed = True
                        print("Debug tokenizer init failed")

            if len(acces) == 0 or acces[-1] > 0:
                acces.append(
                    ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1))
                    .sum()
                    .item()
                    / (loss_mask.sum().item() + 1e-6)
                )
            else:
                acces.append(0)

            if cache_params is not None:
                if cache_position is None:
                    cache_position = torch.full(
                        (hidden_states.shape[0],),
                        hidden_states.shape[1] - 1,
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
                else:
                    cache_position = cache_position + hidden_states.shape[1]

            if idx < self.length - 1:
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)

        # After loop: log all target/pred tokens for valid positions
        if os.getenv("TRAIN_DEBUG", "") and last_target_max is not None and last_logits is not None:
            try:
                tok = getattr(self, "_debug_tokenizer", None)
                if tok is not None:
                    import json
                    valid_positions = (loss_mask[0] > 0).nonzero(as_tuple=True)[0]
                    target_ids = last_target_max[0, valid_positions].detach().cpu().tolist()
                    pred_ids = last_logits.argmax(-1)[0, valid_positions].detach().cpu()
                    if self.vocab_size != self.draft_vocab_size:
                        d2t = self.d2t.to(pred_ids.device)
                        pred_ids = (pred_ids + d2t[pred_ids]).cpu()
                    pred_ids_list = pred_ids.tolist()
                    record = {
                        "pos": valid_positions.tolist(),
                        "target_ids": target_ids,
                        "pred_ids": pred_ids_list,
                        "target_text": tok.decode(target_ids, skip_special_tokens=True),
                        "pred_text": tok.decode(pred_ids_list, skip_special_tokens=True),
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    with open("train_debug.jsonl", "a+", encoding="utf-8") as f:
                        f.write(line)
            except Exception as e:
                if not hasattr(self, "_debug_log_failed"):
                    self._debug_log_failed = True
                    print(f"Debug log failed: {e}")

        return plosses, acces