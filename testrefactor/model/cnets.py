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
import time
import math
import json
from typing import List, Optional, Tuple
from collections import Counter
import torch
import torch.nn.functional as F
from torch import nn
import os
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoTokenizer
from .modeling_llama import LlamaForCausalLM
from .configs import SmeargleConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache, Mamba2Block
from transformers.integrations import use_kernel_forward_from_hub


class Mamba2(nn.Module):
    """MAMBA2 with cache support."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.residual_size

        self.mamba2 = Mamba2Block(config, layer_idx=0)
        self.out_proj = nn.Linear(config.hidden_size, config.residual_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Mamba2Cache], Optional[torch.LongTensor]]:

        output = self.mamba2(
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=None,
        )

        output = output.to(self.out_proj.weight.dtype)
        output_proj = self.out_proj(output)

        return output_proj, cache_params, cache_position


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.residual_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@use_kernel_forward_from_hub("RMSNorm")
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

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SmeargleDecoderLayeremb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.residual_size
        self.mamba2 = Mamba2(config=config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.residual_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.residual_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = LlamaRMSNorm(
            config.residual_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Mamba2Cache], Optional[torch.LongTensor]]:
        """
        Args:
            input_emb: Input embeddings (batch, seq_len, hidden_size)
            hidden_states: Hidden states from target model (batch, seq_len, hidden_size)
            cache_params: Mamba2Cache for incremental processing
            cache_position: Tensor indicating position in sequence for cache
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat(
            (input_emb, hidden_states), dim=-1
        )

        # MAMBA block
        hidden_states, cache_params, cache_position = self.mamba2(
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache_params, cache_position


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


class Model(nn.Module):
    def __init__(
        self,
        config,
        path=None,
        total_token: int = 60,
        depth: int = 5,
        top_k: int = 8,
        threshold: float = 1.0,
    ):
        super().__init__()

        self.total_tokens = total_token - 1
        self.depth = depth
        self.top_k = top_k
        self.threshold = math.log(threshold)
        self.hidden_size = config.residual_size

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size

        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.hidden_size, self.padding_idx
        )
        self.lm_head = nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )

        self.midlayer = SmeargleDecoderLayeremb(config)

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

        assert tensor is not None, "Embedding tensor is None"
        self.embed_tokens.weight.data = tensor
        
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        self.norm = LlamaRMSNorm(self.hidden_size, eps=self.config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        d2t = torch.zeros((self.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((self.vocab_size), dtype=torch.bool)
        
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Mamba2Cache], Optional[torch.LongTensor]]:
        inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        hidden_states, cache_params, cache_position = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
        )

        return hidden_states, cache_params, cache_position

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        pass

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def print_model_summary(model: nn.Module):
    print(model)

if __name__ == "__main__":
    config = SmeargleConfig.from_pretrained('config.json')
    ds_config = json.load(open('ds_config.json'))
    training_config = {
        "bs": ds_config["train_micro_batch_size_per_gpu"],
        "num_epochs": 1,
        "num_workers": 2,
        "max_len": 2048,
        "config_path": "config.json",
        "gradient_checkpoint": True,
    }
    model = Model(config, path="models/llama_3_1_8b_instruct")
    print(f"Number of parameters: {count_parameters(model):,}")
    print_model_summary(model)