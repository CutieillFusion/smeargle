import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM
import time
# ----------------------------
# Config
# ----------------------------
MODEL_ID = "benchang1110/mamba2-130m-hf"  # small Mamba2 HF-compatible checkpoint
PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 1000
DO_SAMPLE = False
TEMPERATURE = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is usually best on modern GPUs; fall back to fp32 on CPU
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = Mamba2ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()

# Set TF32 flags for better performance
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

@torch.inference_mode()
def generate_expected(prompt_text: str, max_new_tokens: int, use_cache: bool):
    """Use HF's model.generate() with cache on/off for expected baseline."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Make generation deterministic like your greedy loop if DO_SAMPLE=False
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=DO_SAMPLE,
        use_cache=use_cache,
    )

    # If sampling
    if DO_SAMPLE:
        gen_kwargs.update(dict(temperature=max(TEMPERATURE, 1e-6)))
    else:
        # Greedy
        gen_kwargs.update(dict(temperature=1.0))

    # Some models don't define pad token; safe default:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    out_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def time_generate(prompt_text: str, max_new_tokens: int, use_cache: bool, repeat: int = 3):
    """Time generation with warmup and multiple runs, returning best time."""
    # warmup
    _ = generate_expected(prompt_text, 10, use_cache=use_cache)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    best = 1e9
    last_text = None
    for _ in range(repeat):
        t0 = time.time()
        last_text = generate_expected(prompt_text, max_new_tokens, use_cache=use_cache)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best, last_text


if __name__ == "__main__":
    print("param dtype:", next(model.parameters()).dtype)
    print("device:", next(model.parameters()).device)
    print()
    
    # Expected baseline using model.generate()
    print("=== Expected baseline: model.generate() ===")
    t_cache, _ = time_generate(PROMPT, MAX_NEW_TOKENS, use_cache=True, repeat=1)
    t_nocache, _ = time_generate(PROMPT, MAX_NEW_TOKENS, use_cache=False, repeat=1)

    print("\n=== generate() with cache ===")
    print(f"time: {t_cache:.4f}s")

    print("\n=== generate() without cache ===")
    print(f"time: {t_nocache:.4f}s")

    print(f"\nSpeedup (cache): {t_nocache / t_cache:.2f}x")
