import functools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "benchang1110/mamba2-130m-hf"
PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 500

# Partial-sequence test:
PREFIX_REPEAT = 800          # bigger => longer prefix (prefill cost grows)
PARTIAL_NEW_TOKENS = 2000    # tokens generated after the prefix

DO_SAMPLE = False
TEMPERATURE = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# ----------------------------
# Helpers
# ----------------------------
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def now():
    return time.time()

@torch.inference_mode()
def pick_next_token(logits: torch.Tensor, do_sample: bool, temperature: float):
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probs, num_samples=1)

# ----------------------------
# Forward-call tracing for generate()
# ----------------------------
@dataclass
class ForwardTrace:
    call_shapes: List[Tuple[int, int]] = field(default_factory=list)  # (batch, seqlen)
    passed_cache: List[bool] = field(default_factory=list)
    returned_cache: List[bool] = field(default_factory=list)

def attach_forward_logger(model, verbose=True, print_head=6, print_tail=4):
    """
    Monkeypatch model.forward to log input_ids shape and cache usage.
    Returns (trace, detach_fn, print_tail_fn).
    """
    trace = ForwardTrace()
    orig_forward = model.forward  # this is already a bound method

    @functools.wraps(orig_forward)
    def logged_forward(*args, **kwargs):
        # HF may pass input_ids positionally (args[0]) OR by kw OR both.
        # Normalize so we log correctly AND ensure orig_forward sees only one.
        input_ids = None

        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_ids = args[0]

        if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
            if input_ids is None:
                input_ids = kwargs["input_ids"]
            else:
                # If both are provided, drop the kwarg version to avoid "multiple values"
                kwargs = dict(kwargs)
                kwargs.pop("input_ids", None)

        bsz, seqlen = (-1, -1)
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 2:
                bsz, seqlen = int(input_ids.shape[0]), int(input_ids.shape[1])
            elif input_ids.ndim == 1:
                bsz, seqlen = int(input_ids.shape[0]), 1

        # For Mamba2, cache is passed as cache_params
        passed = ("cache_params" in kwargs) and (kwargs["cache_params"] is not None)
        trace.call_shapes.append((bsz, seqlen))
        trace.passed_cache.append(bool(passed))

        out = orig_forward(*args, **kwargs)

        returned = hasattr(out, "cache_params") and (out.cache_params is not None)
        trace.returned_cache.append(bool(returned))

        i = len(trace.call_shapes) - 1
        if verbose and i < print_head:
            print(f"[forward #{i:03d}] input_ids shape={(bsz, seqlen)} "
                  f"passed_cache={passed} returned_cache={returned}")

        return out

    model.forward = logged_forward

    def detach():
        model.forward = orig_forward

    def print_tail_calls():
        if not verbose:
            return
        n = len(trace.call_shapes)
        start = max(0, n - print_tail)
        print("... (snip) ...")
        for i in range(start, n):
            bsz, seqlen = trace.call_shapes[i]
            print(f"[forward #{i:03d}] input_ids shape={(bsz, seqlen)} "
                  f"passed_cache={trace.passed_cache[i]} returned_cache={trace.returned_cache[i]}")

    return trace, detach, print_tail_calls

# ----------------------------
# generate() experiment
# ----------------------------
@torch.inference_mode()
def run_generate_expected(model, tokenizer, prompt: str, max_new_tokens: int, use_cache: bool):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=DO_SAMPLE,
        use_cache=use_cache,
        return_dict_in_generate=False,
    )
    if DO_SAMPLE:
        gen_kwargs["temperature"] = max(TEMPERATURE, 1e-6)
    # ensure pad token exists
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    out_ids = model.generate(**inputs, **gen_kwargs)
    return out_ids

# ----------------------------
# Manual partial-sequence test: cached vs no-cache
# ----------------------------
@torch.inference_mode()
def prefill(model, input_ids: torch.LongTensor):
    # For single sequence, attention_mask is optional; keep it None.
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = out.cache_params
    cache_pos = torch.tensor([input_ids.shape[1]], device=input_ids.device, dtype=torch.long)
    logits = out.logits[:, -1, :]
    return logits, cache, cache_pos

@torch.inference_mode()
def step_cached(model, next_id: torch.LongTensor, cache, cache_pos: torch.LongTensor):
    out = model(
        input_ids=next_id,
        use_cache=True,
        cache_params=cache,
        cache_position=cache_pos,
        return_dict=True,
    )
    return out.logits[:, -1, :], out.cache_params, cache_pos + 1

@torch.inference_mode()
def partial_generate_manual_cached(model, tokenizer, prefix_text: str, new_tokens: int):
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)

    # Prefill once
    t0 = now()
    logits, cache, cache_pos = prefill(model, prefix_ids)
    cuda_sync()
    t_prefill = now() - t0

    # Decode K tokens using only (B,1) inputs
    next_id = pick_next_token(logits, DO_SAMPLE, TEMPERATURE)
    times = []
    for _ in range(new_tokens):
        t1 = now()
        logits, cache, cache_pos = step_cached(model, next_id, cache, cache_pos)
        cuda_sync()
        times.append(now() - t1)
        next_id = pick_next_token(logits, DO_SAMPLE, TEMPERATURE)

    return t_prefill, times

@torch.inference_mode()
def partial_generate_manual_nocache(model, tokenizer, prefix_text: str, new_tokens: int):
    # This is "correct uncached": always re-run full forward on growing prefix
    ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)

    t0 = now()
    cuda_sync()
    t_prefill = 0.0  # no explicit prefill; all cost is in steps below

    times = []
    for _ in range(new_tokens):
        t1 = now()
        out = model(input_ids=ids, use_cache=False, return_dict=True)
        logits = out.logits[:, -1, :]
        next_id = pick_next_token(logits, DO_SAMPLE, TEMPERATURE)
        ids = torch.cat([ids, next_id], dim=1)
        cuda_sync()
        times.append(now() - t1)

    return t_prefill, times

# ----------------------------
# Main
# ----------------------------
def main():
    # perf flags
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Mamba2ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

    print("param dtype:", next(model.parameters()).dtype)
    print("device:", next(model.parameters()).device)
    print()

    # Warmup (important)
    print("Warming up...")
    _ = run_generate_expected(model, tokenizer, PROMPT, 10, use_cache=True)
    _ = run_generate_expected(model, tokenizer, PROMPT, 10, use_cache=False)
    cuda_sync()
    print("Warmup done.\n")

    # ----------------------------
    # 1) Show what generate() does with/without cache
    # ----------------------------
    print("=== TRACE: model.generate(use_cache=True) ===")
    trace, detach, print_tail = attach_forward_logger(model, verbose=True)
    t0 = now()
    _ = run_generate_expected(model, tokenizer, PROMPT, MAX_NEW_TOKENS, use_cache=True)
    cuda_sync()
    dt = now() - t0
    print_tail()
    detach()

    shapes = trace.call_shapes
    # Count how many calls used seqlen==1 (decode steps)
    ones = sum(1 for (_, s) in shapes if s == 1)
    print(f"\nTotal forward calls: {len(shapes)}")
    print(f"Calls with seqlen==1: {ones} (these are your cached decode steps)")
    print(f"Time: {dt:.4f}s\n")

    print("=== TRACE: model.generate(use_cache=False) ===")
    trace2, detach2, print_tail2 = attach_forward_logger(model, verbose=True)
    t0 = now()
    _ = run_generate_expected(model, tokenizer, PROMPT, MAX_NEW_TOKENS, use_cache=False)
    cuda_sync()
    dt2 = now() - t0
    print_tail2()
    detach2()

    shapes2 = trace2.call_shapes
    ones2 = sum(1 for (_, s) in shapes2 if s == 1)
    print(f"\nTotal forward calls: {len(shapes2)}")
    print(f"Calls with seqlen==1: {ones2}")
    # show first few seqlens to illustrate growth
    first_seqlens = [s for (_, s) in shapes2[:8]]
    print(f"First 8 seqlens seen (use_cache=False): {first_seqlens}")
    print(f"Time: {dt2:.4f}s")
    print(f"Speedup (cache): {dt2/dt:.2f}x\n")

    # ----------------------------
    # 2) Manual partial-sequence test (what you asked)
    # ----------------------------
    prefix_text = (PROMPT + " ") * PREFIX_REPEAT
    print("=== PARTIAL SEQUENCE TEST (manual) ===")
    print(f"Prefix repeats: {PREFIX_REPEAT}  |  New tokens: {PARTIAL_NEW_TOKENS}")
    print("This tests: prefill prefix once, then generate from that partial cached state.\n")

    # Cached partial
    t_prefill, step_times = partial_generate_manual_cached(model, tokenizer, prefix_text, PARTIAL_NEW_TOKENS)
    avg_step = sum(step_times) / len(step_times)
    print("[CACHED]")
    print(f"prefill time: {t_prefill:.4f}s")
    print(f"avg decode step: {avg_step*1000:.3f} ms/token")
    print(f"first 5 steps (ms): {[round(t*1000,3) for t in step_times[:5]]}")
    print(f"last 5 steps (ms):  {[round(t*1000,3) for t in step_times[-5:]]}")
    print()

    # Uncached partial
    t_prefill2, step_times2 = partial_generate_manual_nocache(model, tokenizer, prefix_text, PARTIAL_NEW_TOKENS)
    avg_step2 = sum(step_times2) / len(step_times2)
    print("[NO CACHE]")
    print(f"prefill time: {t_prefill2:.4f}s (n/a)")
    print(f"avg decode step: {avg_step2*1000:.3f} ms/token")
    print(f"first 5 steps (ms): {[round(t*1000,3) for t in step_times2[:5]]}")
    print(f"last 5 steps (ms):  {[round(t*1000,3) for t in step_times2[-5:]]}")
    print()

    print(f"Partial-sequence speedup (decode step avg): {avg_step2/avg_step:.2f}x")
    print("If caching is working, cached avg step should stay ~flat even for long prefixes.\n")

if __name__ == "__main__":
    main()
