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
import json
from typing import Callable, List, Optional, Tuple
from collections import Counter
import torch
from torch import nn, Tensor
import os
from transformers.activations import ACT2FN
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM
from configs import EagleConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing
from transformers.cache_utils import Cache
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_layers import GradientCheckpointingLayer


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


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


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: EagleConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: EagleConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size * 2, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config.attn_implementation is not None:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config.attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class EagleDecoderLayeremb(GradientCheckpointingLayer):
    def __init__(self, config: EagleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=0)

        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
 
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@torch.no_grad()
def padding(tensor: torch.Tensor, left: bool = True):
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
    """Merge multiple Counter dictionaries"""
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


class Model(nn.Module):
    def __init__(
        self,
        config: EagleConfig,
        training_config: dict,
        path: str = None,
    ):
        super().__init__()
        self.train_config = training_config
        self.config = config
        self.midlayer = EagleDecoderLayeremb(config)
        self.rotary_emb = LlamaRotaryEmbedding(self.config)
        self.gradient_checkpointing = self.train_config["gradient_checkpoint"]
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.length = 7

        self._target_model = LlamaForCausalLM.from_pretrained(
            path, dtype=torch.float16, low_cpu_mem_usage=True
        )
        self._target_model.eval()
        for param in self._target_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        
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

        assert tensor is not None, "Embedding tensor is None"
        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.hidden_size, self.padding_idx, _weight=tensor
        )

        self.lm_head = nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def scandata(self, datapath: str, tokenizerpath: str, local_rank: int):
        if os.path.exists("cache.pt"):
            cache = torch.load("cache.pt")
            d2t = cache["d2t"]
            t2d = cache["t2d"]
        elif local_rank != 0:         
            while not os.path.exists("cache.pt"):
                time.sleep(1)
            cache = torch.load("cache.pt")
            d2t = cache["d2t"]
            t2d = cache["t2d"]
        elif local_rank == 0:
            tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
            dataset = load_dataset("json", data_files=datapath)
            dataset = dataset["train"]

            original_columns1 = dataset.column_names
            num_proc = 48

            def preprocess_function(examples):
                new_examples = {
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
                        source = source[1:]
                    for j, sentence in enumerate(source):
                        role = roles[sentence["from"]]
                        assert role == convroles[j % 2], f"{i}"
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

                    sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

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

                        if i == 0:
                            loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                        else:
                            loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
                        cur_len += turn_len
                        if i != 0:
                            cur_len += 3

                    loss_mask[cur_len:] = 0

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
            num_processes = num_proc
            chunk_size = len(dataset) // num_processes + (
                len(dataset) % num_processes > 0
            )
            chunks = [
                dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)
            ]

            with multiprocessing.Pool(num_processes) as pool:
                results = pool.map(process_data, chunks)

            token_dict = merge_dicts(results)

            total_frequency = sum(token_dict.values())
            top_draft_tokens = token_dict.most_common(self.draft_vocab_size)
            top_draft_tokens_frequency_sum = sum(freq for key, freq in top_draft_tokens)
            top_draft_tokens_ratio = top_draft_tokens_frequency_sum / total_frequency
            print(f"top {self.draft_vocab_size} token frequency ratio: {top_draft_tokens_ratio:.2%}")
            used_tokens = [key for key, freq in top_draft_tokens]
            used_tokens.sort()

            d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
            t2d = [i in used_tokens for i in range(self.vocab_size)]

            d2t = torch.tensor(d2t)
            t2d = torch.tensor(t2d)
            
            cache = {"d2t": d2t, "t2d": t2d}
            torch.save(cache, "cache.pt")

        assert d2t is not None and d2t.sum() > 0, "d2t is all zeros"
        assert t2d is not None and (~t2d).any(), "t2d does not contain any False values"

        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        actual_draft_vocab_size = int(t2d.sum().item())
        assert actual_draft_vocab_size == self.draft_vocab_size, f"actual draft_vocab_size ({actual_draft_vocab_size}) != draft_vocab_size ({self.draft_vocab_size})"

    def _prepare_decoder_attention_mask(
        self, attention_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], inputs_embeds: torch.Tensor, past_key_values_length: int = 0
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @torch.no_grad()
    def dataprepare(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor):
        device = input_ids.device
        target_model = self._target_model
        model_device = next(target_model.parameters()).device
        if model_device != device:
            target_model = target_model.to(device)
            self._target_model = target_model
        outs = target_model(input_ids=input_ids, attention_mask=attention_mask)
        assert len(outs.hidden_states) == 3, "target model hidden states length is not 3"
        hidden_states = torch.cat(outs.hidden_states, dim=-1)
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        hidden_states, target, loss_mask, input_ids = self.dataprepare(
            input_ids, attention_mask, loss_mask
        )

        batch_size, seq_length, _ = hidden_states.shape

        if (
            self.training
            and self.gradient_checkpointing
            and not hidden_states.requires_grad
        ):
            hidden_states.requires_grad = True

        hidden_states = hidden_states.to(self.fc.weight.dtype)
        hidden_states = self.fc(hidden_states)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            0,
        )

        if position_ids is None:
            position_ids = torch.arange(
                seq_length,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        self.t2d = self.t2d.to(hidden_states.device)

        plosses = []
        acces = []
        for idx in range(self.length):

            inputs_embeds = self.embed_tokens(input_ids)

            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            hidden_states = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
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

            student_p = nn.Softmax(dim=2)(logits)
            smooth_l1_per_elem = self.smooth_l1(student_p, target_p)
            smooth_l1_per_pos = smooth_l1_per_elem.mean(dim=2)
            smooth_l1_masked = (position_mask.squeeze(-1) * smooth_l1_per_pos).sum() / (position_mask.sum() + 1e-6)
            
            loss = loss + 0.1 * smooth_l1_masked

            plosses.append(loss)
        
            if len(acces) == 0 or acces[-1] > 0:
                acces.append(
                    ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1))
                    .sum()
                    .item()
                    / (loss_mask.sum().item() + 1e-6)
                )
            else:
                acces.append(0)

            if idx < self.length - 1:
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)

        return plosses, acces


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def print_model_summary(model: nn.Module):
    print(model)

if __name__ == "__main__":
    config = EagleConfig.from_pretrained('config.json')
    ds_config = json.load(open('ds_config.json'))
    training_config = {
        "bs": ds_config["train_micro_batch_size_per_gpu"],
        "num_epochs": 1,
        "num_workers": 2,
        "max_len": 2048,
        "config_path": "config.json",
        "gradient_checkpoint": True,
    }
    model = Model(config, training_config, path="models/llama_3_1_8b_instruct")
    print(f"Number of parameters: {count_parameters(model):,}")
    print_model_summary(model)