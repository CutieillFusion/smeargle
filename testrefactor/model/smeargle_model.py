import copy
import json
import time
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

from .modeling_llama import LlamaForCausalLM
from .utils import *

from .cnets import Model
from .configs import SmeargleConfig

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def start(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def stop(self):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')

class SmeargleModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        base_model_name_or_path: str,
        smeargle_model_path: str,
        total_token: int,
        depth: int,
        top_k: int,
        threshold: float,
        smeargle_layer_state_dict: torch.Tensor,
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

        self.smeargle_config = SmeargleConfig.from_pretrained(smeargle_model_path)
        self.smeargle_layer = Model(
            config=self.smeargle_config,
            path=base_model_name_or_path,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
        )

        # If vocab sizes match then masking is not needed
        if self.smeargle_config.vocab_size == self.smeargle_config.draft_vocab_size:
            del self.smeargle_layer.d2t, self.smeargle_layer.t2d

        device = self.base_model.device
        load_ = self.smeargle_layer.load_state_dict(smeargle_layer_state_dict, strict=False)
        self.smeargle_layer.to(self.base_model.dtype).to(device)

        assert self.smeargle_layer.d2t is not None and self.smeargle_layer.d2t.sum() > 0, "d2t is all zeros"
        assert self.smeargle_layer.t2d is not None and (~self.smeargle_layer.t2d).any(), "t2d does not contain any False values"

        self.stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        smeargle_model_path: str,
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

        config_path = os.path.join(smeargle_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(smeargle_model_path, "config.json")

        try:
            load_model_path = os.path.join(smeargle_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(smeargle_model_path, "pytorch_model.bin")
            smeargle_layer_state_dict = torch.load(
                load_model_path, map_location=base_model.device
            )
        except:
            load_model_path = os.path.join(smeargle_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(smeargle_model_path, "model.safetensors")
            smeargle_layer_state_dict = load_file(load_model_path)

        assert smeargle_layer_state_dict is not None, "Smeargle layer state dict is None"

        return cls(
            base_model,
            base_model_path,
            smeargle_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            smeargle_layer_state_dict,
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
            # Pass input through the base model
            return self.base_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
            )

    @torch.no_grad()
    def smearglegenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        use_cache: bool = True,
        log: bool = False,
    ):
        accept_lengths = []
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k) if temperature > 1e-5 else None

        input_len = input_ids.shape[1]
        model_input = input_ids

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
        
        if use_cache:
            model_dtype = self.base_model.dtype
            cache_params = Mamba2Cache(self.smeargle_config, input_ids.shape[0], device=input_ids.device, dtype=model_dtype)
            cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        
        hidden_states = self.smeargle_layer.fc(torch.cat(target_outputs.hidden_states, dim=-1))

        print("hidden_states shape: ", hidden_states.shape)

        draft_input = input_ids[:, 1:]
        model_input = target_token_id
        for _ in range(max_length):
            ssm_states_list = [cache_params.ssm_states.clone()]
            conv_states_list = [cache_params.conv_states.clone()]
            cache_position_list = [cache_position.clone()] # DEBUG

            for i in range(self.depth):
                print(f"depth {i} cache_position: {cache_position}")
                print("draft_input shape: ", draft_input.shape)
                hidden_states, cache_params, cache_position = self.smeargle_layer(
                    hidden_states=hidden_states,
                    input_ids=draft_input,
                    cache_params=cache_params,
                    cache_position=cache_position,
                )

                ssm_states_list.append(cache_params.ssm_states.clone())
                conv_states_list.append(cache_params.conv_states.clone())
                cache_position_list.append(cache_position.clone()) # DEBUG
                
                draft_outputs = self.smeargle_layer.lm_head(self.smeargle_layer.norm(hidden_states))

                if logits_processor is not None:
                    draft_logits = draft_outputs[:, -1]
                    draft_logits = logits_processor(draft_input, draft_logits)
                    draft_probabilities = torch.nn.functional.softmax(draft_logits, dim=-1)
                    draft_token_id = torch.multinomial(draft_probabilities, 1)
                else:
                    draft_token_id = draft_outputs[:, -1:].argmax(dim=-1)

                if self.smeargle_config.vocab_size != self.smeargle_config.draft_vocab_size:
                    draft_token_id = draft_token_id + self.smeargle_layer.d2t[draft_token_id]

                hidden_states = hidden_states[:, -1:]
                model_input = torch.cat([model_input, draft_token_id], dim=-1)
                draft_input = draft_token_id

            cache_start = target_outputs.past_key_values.get_seq_length() if target_outputs.past_key_values is not None else 0
            
            target_outputs = self.forward(
                model_input,
                past_key_values=target_outputs.past_key_values,
                use_cache=use_cache
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
            print("accept_length: ", accept_length)
            hidden_states = self.smeargle_layer.fc(torch.cat(target_outputs.hidden_states, dim=-1)[:, accept_length:accept_length+1])

            if accepted_tokens.shape[0] > 0:
                input_ids = torch.cat([input_ids, accepted_tokens], dim=-1)
                        
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

            if use_cache:
                if target_outputs.past_key_values is not None:
                    num_layers = len(target_outputs.past_key_values)
                    for layer_idx in range(num_layers):
                        key_cache, value_cache = target_outputs.past_key_values[layer_idx]
                        cache_seq_len = key_cache.shape[2]
                        start_idx = 0
                        end_idx = min(cache_start + accept_length + 1, cache_seq_len)
                        
                        select_indices = torch.arange(
                            start_idx,
                            end_idx,
                            device=key_cache.device,
                            dtype=torch.long
                        )
                        
                        selected_keys = key_cache.index_select(2, select_indices)
                        selected_values = value_cache.index_select(2, select_indices)
                        
                        target_outputs.past_key_values.layers[layer_idx].keys = selected_keys
                        target_outputs.past_key_values.layers[layer_idx].values = selected_values
                
                if cache_params is not None:
                    cache_params.ssm_states = ssm_states_list[accept_length]
                    cache_params.conv_states = conv_states_list[accept_length]
                    cache_position = torch.arange(cache_start + accept_length + 1, cache_start + accept_length + 2, device=input_ids.device, dtype=torch.long)

            if input_ids.shape[1] > 150 and input_ids.shape[1] < 200:
                print("Accept length:", accept_length)
                print("Input ids shape:", input_ids.shape)
                print("Number of past keys:", target_outputs.past_key_values[0][0].shape[2])
                print("Cache position list:", cache_position_list[accept_length])
                print("Cache position:", cache_position.item())

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
        max_new_tokens=512,
        max_length=2048,
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
                use_cache=use_cache
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

    @torch.no_grad()
    def smeargle_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
    ):
        pass

    @torch.no_grad()
    def naive_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
    ):
        pass
