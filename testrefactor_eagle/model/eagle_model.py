import json
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache, DynamicCache

from .modeling_llama import LlamaForCausalLM
from .utils import prepare_logits_processor
from .cnets import Model
from .configs import EConfig


def clone_kv(past_key_values):
    """Clone draft model KV cache. Format: ((key, value),) for single layer."""
    return tuple((k.clone(), v.clone()) for k, v in past_key_values)


class EagleModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        base_model_name_or_path: str,
        eagle_model_path: str,
        total_token: int,
        depth: int,
        top_k: int,
        threshold: float,
        eagle_layer_state_dict: torch.Tensor,
    ):
        super().__init__()
        self.total_token = total_token
        self.depth = depth
        self.top_k = top_k
        self.threshold = threshold

        self.base_model = base_model
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )

        self.eagle_config = EConfig.from_pretrained(eagle_model_path)
        with open(eagle_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True

        self.eagle_layer = Model(
            self.eagle_config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            path=base_model_name_or_path,
        )

        # If vocab sizes match then masking is not needed
        if self.eagle_config.vocab_size == self.eagle_config.draft_vocab_size:
            del self.eagle_layer.d2t, self.eagle_layer.t2d

        device = self.base_model.device
        load_ = self.eagle_layer.load_state_dict(eagle_layer_state_dict, strict=False)
        self.eagle_layer.to(self.base_model.dtype).to(device)

        self.stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        eagle_model_path: str,
        total_token: int = 60,
        depth: int = 7,
        top_k: int = 10,
        threshold: float = 1.0,
        **kwargs,
    ):
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == "LlamaForCausalLM":
            base_model = LlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            assert False, f"Unsupported model type: {Type}"

        configpath = os.path.join(eagle_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(eagle_model_path, "config.json")

        try:
            load_model_path = os.path.join(eagle_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(eagle_model_path, "pytorch_model.bin")
            eagle_layer_state_dict = torch.load(
                load_model_path, map_location=base_model.device
            )
        except:
            load_model_path = os.path.join(eagle_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(eagle_model_path, "model.safetensors")
            eagle_layer_state_dict = load_file(load_model_path)

        return cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            eagle_layer_state_dict,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
    ):
        with torch.inference_mode():
            return self.base_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
            )

    @torch.no_grad()
    def eaglegenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=2048,
        max_length=16384,
        use_cache: bool = True,
        log: bool = False,
    ):
        accept_lengths = []
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k) if temperature > 1e-5 else None

        input_len = input_ids.shape[1]
        model_input = input_ids

        # Prefill: run full input through target model
        target_outputs = self.forward(
            model_input,
            use_cache=use_cache,
        )

        if logits_processor is not None:
            target_logits = target_outputs.logits[:, -1]
            target_logits = logits_processor(model_input, target_logits)
            target_probabilities = torch.nn.functional.softmax(target_logits, dim=-1)
            target_token_id = torch.multinomial(target_probabilities, 1)
        else:
            target_token_id = target_outputs.logits[:, -1:].argmax(dim=-1)

        if target_token_id.item() in [self.tokenizer.eos_token_id, self.stop_token_id]:
            return torch.cat([input_ids, target_token_id], dim=-1)
        if input_ids.shape[1] - input_len >= max_new_tokens or input_ids.shape[1] >= max_length:
            return torch.cat([input_ids, target_token_id], dim=-1)

        input_ids = torch.cat([input_ids, target_token_id], dim=-1)

        # Prefill draft model: pass concatenated hidden states through fc + eagle layer
        # hidden_states from target model has shape [batch, seq, hidden*3] (3 layers concatenated)
        hidden_states = torch.cat(target_outputs.hidden_states, dim=-1)

        # Reset draft model KV cache
        self.eagle_layer.reset_kv()

        # Prefill draft model with full sequence
        out_hidden, draft_kv = self.eagle_layer(
            hidden_states=hidden_states,
            input_ids=input_ids[:, 1:],
            use_cache=True,
        )
        self.eagle_layer.stable_kv = draft_kv

        # After prefill, only keep last hidden state for drafting
        hidden_states = out_hidden[:, -1:]

        draft_input = target_token_id
        model_input = target_token_id
        for _ in range(max_length):
            # Save draft KV states for rollback
            draft_kv_saves = [clone_kv(self.eagle_layer.stable_kv)]

            for i in range(self.depth):
                out_hidden, draft_kv = self.eagle_layer(
                    hidden_states=hidden_states,
                    input_ids=draft_input,
                    past_key_values=self.eagle_layer.stable_kv,
                    use_cache=True,
                )
                self.eagle_layer.stable_kv = draft_kv

                draft_kv_saves.append(clone_kv(draft_kv))

                draft_outputs = self.eagle_layer.lm_head(self.eagle_layer.norm(out_hidden))

                if logits_processor is not None:
                    draft_logits = draft_outputs[:, -1]
                    draft_logits = logits_processor(draft_input, draft_logits)
                    draft_probabilities = torch.nn.functional.softmax(draft_logits, dim=-1)
                    draft_token_id = torch.multinomial(draft_probabilities, 1)
                else:
                    draft_token_id = draft_outputs[:, -1:].argmax(dim=-1)

                if self.eagle_config.vocab_size != self.eagle_config.draft_vocab_size:
                    draft_token_id = draft_token_id + self.eagle_layer.d2t[draft_token_id]

                hidden_states = out_hidden[:, -1:]
                model_input = torch.cat([model_input, draft_token_id], dim=-1)
                draft_input = draft_token_id

            # Verify draft tokens with target model
            cache_start = target_outputs.past_key_values.get_seq_length() if target_outputs.past_key_values is not None else 0

            target_outputs = self.forward(
                model_input,
                past_key_values=target_outputs.past_key_values,
                use_cache=use_cache,
            )

            if logits_processor is not None:
                target_logits = target_outputs.logits[:, -(self.depth + 1):]
                target_logits = logits_processor(model_input, target_logits)
                target_probabilities = torch.nn.functional.softmax(target_logits, dim=-1)
                target_token_ids = torch.multinomial(target_probabilities, 1)
            else:
                target_token_ids = target_outputs.logits[:, -(self.depth + 1):].argmax(dim=-1)

            posterior_mask = (
                model_input[:, 1:] == target_token_ids[:, :-1]
            ).int()
            accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1).max().item()
            accept_lengths.append(accept_length)
            accepted_tokens = model_input[:, 1:1+accept_length]

            # Get hidden states for the accepted position to seed next draft round
            hidden_states = torch.cat(target_outputs.hidden_states, dim=-1)[:, accept_length:accept_length+1]

            if accepted_tokens.shape[1] > 0:
                input_ids = torch.cat([input_ids, accepted_tokens], dim=-1)

            # Sample the next token from target model at the accepted position
            sample_logits = target_outputs.logits[:, accept_length:accept_length+1]
            if logits_processor is not None:
                sample_logits = logits_processor(input_ids, sample_logits)
                sample_p = torch.nn.functional.softmax(sample_logits, dim=-1)
                target_token_id = torch.multinomial(sample_p, 1).squeeze(-1)
            else:
                target_token_id = sample_logits.argmax(dim=-1)

            model_input = target_token_id
            draft_input = target_token_id

            input_ids = torch.cat([input_ids, target_token_id], dim=-1)

            if target_token_id.item() in [self.tokenizer.eos_token_id, self.stop_token_id]:
                break

            if input_ids.shape[1] - input_len >= max_new_tokens or input_ids.shape[1] >= max_length:
                break

            # Trim target model KV cache to remove rejected draft tokens
            if use_cache:
                if target_outputs.past_key_values is not None:
                    num_layers = len(target_outputs.past_key_values)
                    for layer_idx in range(num_layers):
                        key_cache, value_cache = target_outputs.past_key_values[layer_idx]
                        cache_seq_len = key_cache.shape[2]
                        end_idx = min(cache_start + accept_length + 1, cache_seq_len)

                        select_indices = torch.arange(
                            0, end_idx,
                            device=key_cache.device,
                            dtype=torch.long,
                        )

                        selected_keys = key_cache.index_select(2, select_indices)
                        selected_values = value_cache.index_select(2, select_indices)

                        target_outputs.past_key_values.layers[layer_idx].keys = selected_keys
                        target_outputs.past_key_values.layers[layer_idx].values = selected_values

                # Rollback draft model KV cache to the accepted position
                self.eagle_layer.stable_kv = draft_kv_saves[accept_length]

        if not log:
            return input_ids
        else:
            return input_ids, accept_lengths

    @torch.no_grad()
    def naivegenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=2048,
        max_length=16384,
        use_cache: bool = True,
        log: bool = False,
    ):
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k) if temperature > 1e-5 else None

        input_len = input_ids.shape[1]

        new_token = 0
        outputs = None
        model_input = input_ids
        for _ in range(max_length):
            outputs = self.forward(
                model_input,
                past_key_values=outputs.past_key_values if outputs is not None else None,
                use_cache=use_cache,
            )

            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(model_input, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                token_id = torch.multinomial(probabilities, 1)
            else:
                token_id = outputs.logits[:, -1:].argmax(dim=-1)

            input_ids = torch.cat([input_ids, token_id], dim=-1)
            model_input = token_id
            new_token += 1

            if token_id.item() in [self.tokenizer.eos_token_id, self.stop_token_id]:
                break
            if new_token >= max_new_tokens or input_ids.shape[1] >= max_length:
                break

        if not log:
            return input_ids
        else:
            return input_ids, [0]
