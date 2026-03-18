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
"""SMEARGLE draft model with MAMBA2 + tree speculative decoding."""
import copy
import os
from safetensors import safe_open
import json
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache, Mamba2Block

try:
    from .configs import SmeargleConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import SmeargleConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor


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


def clone_mamba_cache(cache):
    """Deep clone a Mamba2Cache (conv_states + ssm_states)."""
    new_cache = copy.copy(cache)
    new_cache.conv_states = cache.conv_states.clone()
    new_cache.ssm_states = cache.ssm_states.clone()
    return new_cache


def expand_mamba_cache(cache, n):
    """Expand batch dimension from 1 to n by repeating.

    Cache tensors have shape [num_layers, batch_size, ...].
    Batch dimension is at index 1.
    """
    ndim_conv = cache.conv_states.dim()
    repeat_conv = [1] * ndim_conv
    repeat_conv[1] = n
    cache.conv_states = cache.conv_states.repeat(*repeat_conv)

    ndim_ssm = cache.ssm_states.dim()
    repeat_ssm = [1] * ndim_ssm
    repeat_ssm[1] = n
    cache.ssm_states = cache.ssm_states.repeat(*repeat_ssm)

    return cache


def reindex_mamba_cache(cache, indices):
    """Select batch elements by indices along the batch dimension (dim=1)."""
    cache.conv_states = cache.conv_states[:, indices].contiguous()
    cache.ssm_states = cache.ssm_states[:, indices].contiguous()
    return cache


class Model(nn.Module):
    def __init__(
        self,
        config,
        path=None,
        total_tokens=63,
        depth=5,
        top_k=8,
        threshold=1.0,
    ):
        super().__init__()

        self.total_tokens = total_tokens - 1
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

        self.stable_cache_params = None
        self.stable_cache_len = 0

    def init_tree(self):
        # No tree masks needed for MAMBA2 (no attention mechanism),
        # but method kept for compatibility with smeargle_model.py
        pass

    def reset_mamba_state(self):
        self.stable_cache_params = None
        self.stable_cache_len = 0

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

        if hidden_states.shape[-1] != self.hidden_size:
            hidden_states = self.fc(hidden_states)

        hidden_states, cache_params, cache_position = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
        )

        return hidden_states, cache_params, cache_position

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

        len_posi = input_ids.shape[1]

        # === Prefill ===
        if self.stable_cache_params is not None:
            kv_len = self.stable_cache_len
            cache_params = clone_mamba_cache(self.stable_cache_params)
            # Process new tokens one at a time (MAMBA2 incremental path only handles seq_len=1)
            for t in range(kv_len, input_ids.shape[1]):
                cache_position = torch.tensor([t], device=hidden_states.device, dtype=torch.long)
                out_hidden, cache_params, cache_position = self.forward(
                    hidden_states[:, t - kv_len:t - kv_len + 1],
                    input_ids=input_ids[:, t:t + 1],
                    cache_params=cache_params,
                    cache_position=cache_position,
                )
        else:
            cache_params = Mamba2Cache(
                self.config, batch_size=1,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            cache_position = torch.arange(
                0, input_ids.shape[1],
                device=hidden_states.device, dtype=torch.long,
            )
            out_hidden, cache_params, cache_position = self.forward(
                hidden_states,
                input_ids=input_ids,
                cache_params=cache_params,
                cache_position=cache_position,
            )

        # Save stable state (batch_size=1, before expansion)
        self.stable_cache_params = clone_mamba_cache(cache_params)
        self.stable_cache_len = input_ids.shape[1]

        # === Initial top_k from last hidden state ===
        last_hidden = out_hidden[:, -1]

        last_headout = self.lm_head(self.norm(last_hidden))

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index])
            input_ids = topk_index + self.d2t[topk_index]

        # === Prepare for batched tree generation ===
        # Replicate last_hidden for each candidate: [1, hidden_size] -> [top_k, 1, hidden_size]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)  # [1, top_k, hidden_size]
        input_hidden = input_hidden.transpose(0, 1)  # [top_k, 1, hidden_size]
        input_ids = input_ids.transpose(0, 1)  # [top_k, 1]

        # Expand MAMBA cache: batch_size=1 -> batch_size=top_k
        cache_params = expand_mamba_cache(cache_params, top_k)

        topk_cs_index = torch.arange(top_k, device=hidden_states.device)

        # === Tree generation loop (depth iterations) ===
        for i in range(depth):
            cache_position = torch.tensor(
                [len_posi], device=hidden_states.device, dtype=torch.long
            )

            # Forward through MAMBA2 (batched — each candidate has own state)
            out_hidden, cache_params, cache_position = self.forward(
                input_hidden,
                input_ids=input_ids,
                cache_params=cache_params,
                cache_position=cache_position,
            )
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            # Get top_k logits per candidate: out_hidden is [top_k, 1, hidden_size]
            last_headout = self.lm_head(self.norm(out_hidden[:, -1]))  # [top_k, draft_vocab]
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values  # [top_k, top_k]

            cu_scores = topk_p + scores[:, None]  # [top_k, top_k]

            # Select best top_k by cumulative score
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k

            # Reindex MAMBA states for winning parents
            cache_params = reindex_mamba_cache(cache_params, out_ids)

            input_hidden = out_hidden[out_ids, -1:]  # [top_k, 1, hidden_size]

            input_ids = topk_index.view(-1)[topk_cs_index][:, None]  # [top_k, 1]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])
            scores_list.append(cu_scores)

        # === Build final tree structure (identical to EAGLE) ===
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


import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = SmeargleConfig.from_pretrained("config.json")
    model = Model(config)
    print(model)
