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
from typing import List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoTokenizer
from modeling_llama_kv import LlamaForCausalLM
from configs import EConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing
from mamba_ssm import Mamba2


class MambaBlock(nn.Module):
    """MAMBA block replacing attention mechanism using official Mamba2 implementation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Input dimension is hidden_size * 2 (concatenated input_emb and hidden_states)
        d_model = config.hidden_size * 2

        # Get Mamba2 parameters from config with defaults
        d_state = config.ssm_state_size
        d_conv = config.ssm_conv_kernel
        expand = config.ssm_expand

        # Initialize official Mamba2 block
        # Note: Mamba2 combines token mixing, SSM, and channel mixing internally
        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Output projection: (hidden_size * 2) -> hidden_size
        self.out_proj = nn.Linear(d_model, config.hidden_size, bias=False)

        # Store d_state for state caching compatibility
        self.d_state = d_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_state: Optional[torch.Tensor] = None,
        use_cache: bool = True,  # Controls state caching
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of MAMBA block.

        Args:
            hidden_states: (batch, seq_len, hidden_size * 2) - concatenated input_emb and hidden_states
            cache_state: (batch, d_state) or None - previous SSM state (currently not used by Mamba2)
            use_cache: Whether to return updated state

        Returns:
            output: (batch, seq_len, hidden_size)
            cache_state: (batch, d_state) or None
        """
        # Forward pass through Mamba2
        # Mamba2 internally handles token mixing, SSM, and channel mixing
        # Note: cache_state parameter is ignored as Mamba2 manages state internally
        x = self.mamba2(hidden_states)

        # Output projection: (hidden_size * 2) -> hidden_size
        output = self.out_proj(x)

        # Note: The official Mamba2 manages state internally and doesn't expose
        # it in the same way as the custom implementation. For compatibility with
        # the existing interface, we return None for state. If state caching is
        # required, Mamba2's internal state management should be sufficient for
        # sequential processing within a single forward pass.
        if use_cache:
            # Return None as Mamba2 handles state internally
            # The caller should not rely on this state for cross-forward-pass caching
            return output, None
        else:
            return output, None


class LlamaMLP(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.last = last
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # if last:
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # else:
        #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size * 2, bias=False)
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
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Replace attention with MAMBA block
        self.mamba_block = MambaBlock(config=config)
        self.mlp = LlamaMLP(config, last=last)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_state: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor]]:
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

        # Concatenate input_emb and hidden_states for MAMBA block
        hidden_states = torch.cat(
            (input_emb, hidden_states), dim=-1
        )  # (batch, seq_len, hidden_size * 2)

        return_hidden = hidden_states

        # MAMBA block (replaces attention)
        hidden_states, updated_state = self.mamba_block(
            hidden_states=hidden_states,
            cache_state=cache_state,
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


def build_shifts(tensor, num_shifts):
    """Pre-compute shifted versions of a tensor to avoid repeated padding calls."""
    shifts = []
    cur = tensor
    for _ in range(num_shifts):
        shifts.append(cur)
        zeropad = torch.zeros_like(cur[:, -1:])
        cur = torch.cat([cur[:, 1:], zeropad], dim=1)
    return shifts


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

        # Pre-compute shifted tensors to avoid repeated padding calls
        input_ids_shifts = build_shifts(input_ids, self.length)
        target_shifts = build_shifts(target, self.length)
        loss_mask_shifts = build_shifts(loss_mask, self.length)

        cache_hidden = [
            [],
            [],
        ]  # Initialize cache_hidden as list of two empty lists (cache_k, cache_v)
        plosses = []  # Initialize plosses list
        acces = []  # Initialize acces list
        for idx in range(self.length):
            last = idx == self.length - 1

            # Use pre-computed shifts instead of calling padding()
            input_ids = input_ids_shifts[idx]
            target = target_shifts[idx]
            loss_mask = loss_mask_shifts[idx]

            inputs_embeds = self.embed_tokens(input_ids)

            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            layer_outputs, cache_hidden = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
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

            if len(acces) == 0 or acces[-1] > 0:
                acces.append(
                    ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1))
                    .sum()
                    .item()
                    / (loss_mask.sum().item() + 1e-6)
                )
            else:
                acces.append(0)

        return plosses, acces