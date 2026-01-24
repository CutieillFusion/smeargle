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
from typing import List, Optional, Tuple
from collections import Counter
import torch
from torch import nn
import os
from transformers.activations import ACT2FN
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM
from configs import SmeargleConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache, Mamba2Block
from transformers.integrations import use_kernel_forward_from_hub


class Mamba2(nn.Module):
    """MAMBA2 with cache support."""

    def __init__(self, config: SmeargleConfig):
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
    def __init__(self, config: SmeargleConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.residual_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SmeargleDecoderLayeremb(nn.Module):
    def __init__(self, config: SmeargleConfig):
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

        outputs = (hidden_states)

        return outputs, cache_params, cache_position


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
        config: SmeargleConfig,
        training_config: dict,
        path: str = None,
    ):
        super().__init__()
        self.train_config = training_config
        self.config = config
        self.midlayer = SmeargleDecoderLayeremb(config)
        self.gradient_checkpointing = self.train_config["gradient_checkpoint"]
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.residual_size
        self.draft_vocab_size = config.draft_vocab_size
        self.norm = LlamaRMSNorm(config.residual_size, eps=config.rms_norm_eps)
        self.length = 7

        # Lazy load target model to avoid OOM before DeepSpeed initialization
        self._target_model = None
        self._target_model_path = path

        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)

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

    @property
    def target_model(self):
        """Lazy load target model on first access to avoid OOM before DeepSpeed init"""
        if self._target_model is None:
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

    @torch.no_grad()
    def dataprepare(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor):
        device = input_ids.device
        target_model = self.target_model
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

        hidden_states = self.fc(hidden_states)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        cache_params: Optional[Mamba2Cache] = None
        cache_position: Optional[torch.LongTensor] = None
        if use_cache:
            cache_params = Mamba2Cache(self.config, batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
            cache_position = torch.arange(0, seq_length, device=hidden_states.device, dtype=torch.long)

        self.t2d = self.t2d.to(hidden_states.device)

        plosses = []
        acces = []
        for idx in range(self.length):

            inputs_embeds = self.embed_tokens(input_ids)

            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            hidden_states, cache_params, cache_position = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
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
    model = Model(config, training_config, path="models/llama_3_1_8b_instruct")
    print(f"Number of parameters: {count_parameters(model):,}")
    print_model_summary(model)