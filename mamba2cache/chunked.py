import time
import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM
from time import sleep
# ----------------------------
# Config
# ----------------------------
MODEL_ID = "benchang1110/mamba2-130m-hf"
PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 1000

# "Generate n tokens every step"
CHUNK_SIZE = 16

DO_SAMPLE = False
TEMPERATURE = 1.0  # ignored for greedy

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def pick_next_token(logits: torch.Tensor, do_sample: bool, temperature: float) -> torch.LongTensor:
    # logits: (B, V)
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probs, num_samples=1)  # (B, 1)


@torch.inference_mode()
def prefill(model, prompt_ids: torch.LongTensor):
    # No padding here, so attention_mask is optional.
    out = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    cache = out.cache_params
    cache_pos = torch.tensor([prompt_ids.shape[1]], device=prompt_ids.device, dtype=torch.long)
    logits = out.logits[:, -1, :]  # (B, V)
    return logits, cache, cache_pos


@torch.inference_mode()
def step_cached(model, next_id: torch.LongTensor, cache, cache_pos: torch.LongTensor):
    # next_id: (B, 1)
    out = model(
        input_ids=next_id,
        use_cache=True,
        cache_params=cache,
        cache_position=cache_pos,
        return_dict=True,
    )
    logits = out.logits[:, -1, :]
    cache = out.cache_params
    cache_pos = cache_pos + 1
    return logits, cache, cache_pos


@torch.inference_mode()
def generate_in_chunks(model, tokenizer, prompt_text: str, max_new_tokens: int, chunk_size: int):
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    B, P = prompt_ids.shape
    T = max_new_tokens

    # Preallocate output token buffer
    out_ids = torch.empty((B, P + T), dtype=prompt_ids.dtype, device=device)
    out_ids[:, :P] = prompt_ids
    cur = P

    # Prefill once
    cuda_sync()
    t0 = time.time()
    logits, cache, cache_pos = prefill(model, prompt_ids)
    cuda_sync()
    prefill_time = time.time() - t0

    # Chunked decode
    chunk_times = []
    total_steps = 0
    while total_steps < max_new_tokens:
        this_chunk = min(chunk_size, max_new_tokens - total_steps)

        cuda_sync()
        t_chunk = time.time()

        for _ in range(this_chunk):
            next_id = pick_next_token(logits, DO_SAMPLE, TEMPERATURE)  # (B,1)
            out_ids[:, cur : cur + 1] = next_id
            cur += 1

            logits, cache, cache_pos = step_cached(model, next_id, cache, cache_pos)
            if _ == 0:
                print(cache.ssm_states.shape)
                print(cache.conv_states.shape)
            total_steps += 1

        cuda_sync()
        chunk_times.append(time.time() - t_chunk)

        # Print progress
        avg_ms_per_tok = (chunk_times[-1] / this_chunk) * 1000.0
        print(
            f"chunk_end={total_steps:4d}/{max_new_tokens} "
            f"chunk_time={chunk_times[-1]:.4f}s "
            f"avg={avg_ms_per_tok:.3f} ms/token"
        )

    # Decode final text
    text = tokenizer.decode(out_ids[0, :cur].tolist(), skip_special_tokens=True)
    return text, prefill_time, chunk_times


def main():
    # perf flags
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Your env warns: "`torch_dtype` is deprecated! Use `dtype` instead!"
    model = Mamba2ForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()

    print("param dtype:", next(model.parameters()).dtype)
    print("device:", next(model.parameters()).device)
    print()

    # Warmup (important for stable timing)
    _ = generate_in_chunks(model, tokenizer, PROMPT, max_new_tokens=32, chunk_size=8)
    print("\nWarmup done.\n")

    print(f"=== Chunked cached generation (CHUNK_SIZE={CHUNK_SIZE}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}) ===")
    cuda_sync()
    t0 = time.time()
    text, prefill_time, chunk_times = generate_in_chunks(
        model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, chunk_size=CHUNK_SIZE
    )
    cuda_sync()
    total_time = time.time() - t0

    decode_time = total_time - prefill_time - sum(chunk_times)  # rough (includes Python/printing overhead)
    avg_ms_per_tok = (sum(chunk_times) / MAX_NEW_TOKENS) * 1000.0

    print("\n=== Summary ===")
    print(f"prefill_time: {prefill_time:.4f}s")
    print(f"decode_time (kernel-only-ish): {sum(chunk_times):.4f}s")
    print(f"avg decode: {avg_ms_per_tok:.3f} ms/token")
    print(f"total_time: {total_time:.4f}s")
    print(f"(decode+prefill) sanity: {(prefill_time + sum(chunk_times)):.4f}s")
    print("\nOutput (first 200 chars):")
    print(text[:200])


if __name__ == "__main__":
    main()
