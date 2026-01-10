import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .configs import EConfig


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
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )

        config = EConfig.from_pretrained(smeargle_model_path)
        with open(smeargle_model_path, "r") as f:
            con = json.loads(f.read())

        try:
            bias = con["bias"]
        except:
            bias = True

        self.smeargle_layer = Model(
            config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            path=base_model_name_or_path,
            load_emb=True,
        )

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.smeargle_layer.diff_device = True
            if not low_memory:
                self.smeargle_layer.headweight = base_model.lm_head.weight.clone().to(
                    device
                )
            else:
                self.smeargle_layer.layer_device = device
        else:
            self.smeargle_layer.diff_device = False
        if config.vocab_size == config.draft_vocab_size:
            del self.smeargle_layer.d2t, self.smeargle_layer.t2d

        load_ = self.smeargle_layer.load_state_dict(smeargle_layer_state_dict, strict=False)
        self.smeargle_layer.to(self.base_model.dtype).to(device)
        self.smeargle_layer.init_tree()

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
            base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
        else:
            assert False, f"Unsupported model type: {Type}"
        # elif Type == 'Qwen2ForCausalLM':
        #     base_model = KVQwen2ForCausalLM.from_pretrained(base_model_path, **kwargs)
        # elif Type == 'Qwen3ForCausalLM':
        #     base_model = KVQwen3ForCausalLM.from_pretrained(base_model_path, **kwargs)
        # else:
        #     base_model = KVMixtralForCausalLM.from_pretrained(base_model_path, **kwargs)

        # Check local filesystem, download from Hugging Face if not found
        configpath = os.path.join(smeargle_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(smeargle_model_path, "config.json")

        try:
            load_model_path = os.path.join(smeargle_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(smeargle_model_path, "pytorch_model.bin")
            smeargle_layer_state_dict = torch.load(
                load_model_path, map_location=base_model.device
            )
        except:
            from safetensors.torch import load_file

            load_model_path = os.path.join(smeargle_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(smeargle_model_path, "model.safetensors")
            smeargle_layer_state_dict = load_file(load_model_path)

        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            smeargle_layer_state_dict,
        )

        # If total_token is -1, find the optimal total_token by measuring the inference time
        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(
                    0, model.config.vocab_size - 200, (1, length)
                ).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.smeargle_layer.total_tokens = total_token - 1

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def smearglegenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        # Manage Mamba cache internally per prompt
        self.smeargle_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        # prefill
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            logits,
            hidden_state,
            sample_token,
        ) = initialize_tree(input_ids, self, past_key_values, logits_processor)

        new_token = 0
        max_length = max_length - self.smeargle_layer.total_tokens - 10
        accept_lengths = []
        
        # Optional debug logging
        debug_enabled = os.getenv("TEST_DEBUG", "")
        if debug_enabled:
            debug_log_file = os.getenv("TEST_DEBUG_FILE", "inference_debug.jsonl")
        
        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)

            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accept_lengths.append(accept_length)
            
            # Debug logging
            if debug_enabled:
                try:
                    # Candidates are already in base vocab space (mapped in topK_genrate if needed)
                    all_candidates = candidates.detach().cpu()
                    
                    # Get accepted tokens from best candidate
                    best_candidate_idx = best_candidate.item() if isinstance(best_candidate, torch.Tensor) else best_candidate
                    accept_len = accept_length.item() if isinstance(accept_length, torch.Tensor) else accept_length
                    accepted_tokens = all_candidates[best_candidate_idx, :accept_len + 1].detach().cpu()  # +1 for the sample token
                    
                    # Filter out padding tokens (-1)
                    valid_accepted = accepted_tokens[accepted_tokens >= 0].tolist()
                    all_candidates_list = []
                    for i in range(all_candidates.shape[0]):
                        candidate_tokens = all_candidates[i].detach().cpu()
                        valid_candidate = candidate_tokens[candidate_tokens >= 0].tolist()
                        all_candidates_list.append(valid_candidate)
                    
                    record = {
                        "step": idx,
                        "best_candidate": best_candidate_idx,
                        "accept_length": accept_len,
                        "accepted_ids": valid_accepted,
                        "accepted_text": self.tokenizer.decode(valid_accepted, skip_special_tokens=True),
                        "all_candidates_ids": all_candidates_list,
                        "all_candidates_text": [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in all_candidates_list],
                    }
                    
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    with open(debug_log_file, "a+", encoding="utf-8") as f:
                        f.write(line)
                except Exception as e:
                    if not hasattr(self, "_debug_log_failed"):
                        self._debug_log_failed = True
                        print(f"Inference debug log failed: {e}")

            # Adjusting the input sequence, draft model forward
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                hidden_state,
                sample_token,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, accept_lengths

    @torch.no_grad()
    def smearglegenerate_sequential(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
    ):
        """Sequential speculative decoding without tree structure.
        Generates draft tokens one at a time and verifies them sequentially."""
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        input_ids = input_ids.clone()
        self.smeargle_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # Initial forward pass to get hidden states
        outputs, orig, hidden_states = self(
            input_ids, past_key_values=past_key_values, output_orig=True
        )
        
        # Sample initial token
        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            sample_token = torch.multinomial(probabilities, 1)
        else:
            sample_token = torch.argmax(orig[:, -1])
            sample_token = sample_token[None, None]
        
        input_ids = torch.cat((input_ids, sample_token.to(input_ids.device)), dim=1)
        
        # Get hidden states for draft model
        ea_device = self.smeargle_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states = torch.cat(outputs["hidden_states"], dim=-1)

        new_token = 0
        max_length = max_length - self.smeargle_layer.total_tokens - 10
        accept_lengths = []
        
        # Optional debug logging
        debug_enabled = os.getenv("TEST_DEBUG", "")
        if debug_enabled:
            debug_log_file = os.getenv("TEST_DEBUG_FILE", "inference_debug.jsonl")
        
        for idx in range(max_length):
            # Generate draft tokens sequentially
            draft_tokens, draft_hidden_state = self.smeargle_layer.sequential_generate(
                hidden_states, input_ids, self.base_model.lm_head, logits_processor
            )
            
            # Verify draft tokens sequentially with base model
            accepted_tokens = []
            accept_length = 0
            current_context = input_ids
            current_past_kv = past_key_values
            
            # Verify each draft token one by one
            # sample_token (draft_tokens[0]) is already in input_ids, so start from draft_tokens[1]
            for i in range(1, len(draft_tokens)):
                draft_token = draft_tokens[i].item()
                
                # Get base model's prediction for next token given current context
                base_outputs = self.base_model(
                    current_context[:, -1:], use_cache=True, past_key_values=current_past_kv
                )
                base_logits = base_outputs.logits[:, -1]  # (1, vocab_size)
                
                # Get base model's predicted token
                if logits_processor is not None:
                    processed_logits = logits_processor(None, base_logits)
                    base_probs = torch.nn.functional.softmax(processed_logits, dim=-1)
                    base_token = torch.multinomial(base_probs, 1).item()
                else:
                    base_token = base_logits.argmax(dim=-1).item()
                
                # Check if draft token matches base model's prediction
                if draft_token == base_token:
                    accepted_tokens.append(draft_token)
                    accept_length += 1
                    # Update context with accepted token
                    draft_token_tensor = torch.tensor([[draft_token]], device=input_ids.device)
                    current_context = torch.cat([current_context, draft_token_tensor], dim=-1)
                    # Forward accepted token to update cache
                    base_outputs = self.base_model(
                        draft_token_tensor, use_cache=True, past_key_values=current_past_kv
                    )
                    current_past_kv = base_outputs.past_key_values
                else:
                    # Reject: use base model's token and stop
                    base_token_tensor = torch.tensor([[base_token]], device=input_ids.device)
                    current_context = torch.cat([current_context, base_token_tensor], dim=-1)
                    current_past_kv = base_outputs.past_key_values
                    break
            
            # Update input_ids and past_key_values
            input_ids = current_context
            past_key_values = current_past_kv
            
            # Ensure past_key_values is valid (not None and not containing None elements)
            # The base model expects a tuple of KV caches where each element is a tuple of (key, value)
            if past_key_values is None:
                # Use the stored past_key_values if current_past_kv is None
                past_key_values = self.past_key_values
            elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                # Check if the first layer's KV cache is None
                if past_key_values[0] is None or (isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0 and past_key_values[0][0] is None):
                    # Use the stored past_key_values if current_past_kv has None elements
                    past_key_values = self.past_key_values
            
            new_token += len(accepted_tokens) + 1  # +1 for the base model token
            
            # Update hidden states for next iteration
            outputs, orig, hidden_states = self(
                input_ids, past_key_values=past_key_values, output_orig=True
            )
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
            
            accept_lengths.append(accept_length)
            
            # Debug logging
            if debug_enabled:
                try:
                    all_draft = draft_tokens[1:].detach().cpu().tolist()  # Skip sample token
                    record = {
                        "step": idx,
                        "best_candidate": 0,  # Sequential has only one candidate
                        "accept_length": accept_length,
                        "accepted_ids": accepted_tokens,
                        "accepted_text": self.tokenizer.decode(accepted_tokens, skip_special_tokens=True),
                        "all_candidates_ids": [all_draft],
                        "all_candidates_text": [self.tokenizer.decode(all_draft, skip_special_tokens=True)],
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    with open(debug_log_file, "a+", encoding="utf-8") as f:
                        f.write(line)
                except Exception as e:
                    if not hasattr(self, "_debug_log_failed"):
                        self._debug_log_failed = True
                        print(f"Inference debug log failed: {e}")

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, accept_lengths

    @torch.no_grad()
    def naivegenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.smeargle_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(
            input_ids, past_key_values=past_key_values, use_cache=True
        )
        new_token = 0
        max_length = max_length - self.smeargle_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(
                input_id, use_cache=True, past_key_values=past_key_values
            )
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, [0]

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
        is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.smeargle_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            logits,
            hidden_state,
            sample_token,
        ) = initialize_tree(input_ids, self, past_key_values, logits_processor)
        new_token = 0
        max_length = max_length - self.smeargle_layer.total_tokens - 10
        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                hidden_state,
                sample_token,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

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
        is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.smeargle_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(
            input_ids, past_key_values=past_key_values, use_cache=True
        )
        new_token = 0
        max_length = max_length - self.smeargle_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(
                input_id, use_cache=True, past_key_values=past_key_values
            )
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
