import time
import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "benchang1110/mamba2-130m-hf"
PROMPT = "What is 15 + 15?"
MAX_NEW_TOKENS = 13  # total new tokens AFTER prompt
DECODE_CHUNK = 1     # how many decode steps between prints

DO_SAMPLE = False
TEMPERATURE = 1.0  # ignored for greedy

# EAGLE3/Medusa-style "choices" (paths from root)
MC_CHOICES = [
    [0],
    [1],
    [2],
    [3],
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 1],
    [2, 0],
    [2, 1],
    [3, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 2],
    [0, 1, 0],
    [0, 1, 1],
    [0, 2, 0],
    [0, 2, 1],
    [1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
]

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
    out = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    cache = out.cache_params
    cache_pos = torch.full(
        (prompt_ids.shape[0],),
        prompt_ids.shape[1],
        device=prompt_ids.device,
        dtype=torch.long,
    )
    logits = out.logits[:, -1, :]  # (B, V)
    return logits, cache, cache_pos


def _alloc_cache_like(model, batch_size: int, device, dtype):
    # For HF Mamba2: cache class is transformers.models.mamba2.modeling_mamba2.Mamba2Cache
    return model.backbone.__class__.forward.__globals__["Mamba2Cache"](
        model.backbone.config, batch_size, device=device, dtype=dtype
    )


@torch.inference_mode()
def step_cached_batch(model, cache, cache_pos, logits, row_indices: torch.LongTensor, next_ids: torch.LongTensor):
    """
    Update ONLY the rows in row_indices with the provided next_ids (Bsub,1).
    Writes updated cache + logits back into the big cache/logits tensors.
    """
    assert next_ids.ndim == 2 and next_ids.shape[1] == 1
    bsub = row_indices.numel()

    subcache = _alloc_cache_like(
        model,
        bsub,
        device=cache.conv_states.device,
        dtype=cache.conv_states.dtype,
    )
    # copy selected rows into subcache
    subcache.conv_states.copy_(cache.conv_states[:, row_indices])
    subcache.ssm_states.copy_(cache.ssm_states[:, row_indices])
    subpos = cache_pos[row_indices].clone()

    out = model(
        input_ids=next_ids,
        use_cache=True,
        cache_params=subcache,
        cache_position=subpos,
        return_dict=True,
    )

    # write back
    cache.conv_states[:, row_indices].copy_(out.cache_params.conv_states)
    cache.ssm_states[:, row_indices].copy_(out.cache_params.ssm_states)
    cache_pos[row_indices] = subpos + 1
    logits[row_indices] = out.logits[:, -1, :]

    return cache, cache_pos, logits


@torch.inference_mode()
def generate_tree_then_decode(model, tokenizer, prompt_text: str):
    # ---- setup prompt
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    B0, P = prompt_ids.shape
    assert B0 == 1, "This demo assumes batch=1 prompt."

    # ---- derive TOPK from MC_CHOICES
    max_choice = max(max(p) for p in MC_CHOICES) if MC_CHOICES else 0
    TOPK = max_choice + 1

    # ---- build full set of nodes = all prefixes + leaves
    leaf_paths = [tuple(p) for p in MC_CHOICES]
    all_paths = {()}  # root
    for lp in leaf_paths:
        for d in range(1, len(lp) + 1):
            all_paths.add(lp[:d])

    # sort by depth (root first)
    paths_by_depth = sorted(all_paths, key=lambda t: (len(t), t))
    max_depth = max(len(p) for p in all_paths)

    # mapping path -> row index in big batch
    path2row = {}

    # pre-allocate big batch for ALL nodes (root + internal + leaves)
    N = len(all_paths)
    # token buffer per node: prompt + max_depth(branch tokens) + decode tokens
    # Each leaf will end up with: prompt + len(leaf_path) + (MAX_NEW_TOKENS - len(leaf_path))
    out_ids = torch.full((N, P + MAX_NEW_TOKENS), fill_value=0, dtype=prompt_ids.dtype, device=device)
    out_ids[0, :P] = prompt_ids[0]
    cur_len = torch.full((N,), P, dtype=torch.long, device=device)

    # ---- prefill root into row 0
    cuda_sync()
    t0 = time.time()
    root_logits, root_cache, root_pos = prefill(model, prompt_ids)
    cuda_sync()
    prefill_time = time.time() - t0

    # allocate big cache/logits/pos for all nodes
    cache = _alloc_cache_like(model, N, device=root_cache.conv_states.device, dtype=root_cache.conv_states.dtype)
    cache.conv_states.zero_()
    cache.ssm_states.zero_()
    cache.conv_states[:, 0].copy_(root_cache.conv_states[:, 0])
    cache.ssm_states[:, 0].copy_(root_cache.ssm_states[:, 0])

    cache_pos = torch.zeros((N,), dtype=root_pos.dtype, device=device)
    cache_pos[0] = root_pos[0]

    logits = torch.empty((N, root_logits.shape[-1]), dtype=root_logits.dtype, device=device)
    logits[0] = root_logits[0]

    path2row[()] = 0
    next_free_row = 1

    # ---- build the tree level-by-level
    for depth in range(1, max_depth + 1):
        nodes = [p for p in paths_by_depth if len(p) == depth]
        if not nodes:
            continue

        parent_rows = []
        child_rows = []
        child_next_ids = []

        for p in nodes:
            parent = p[:-1]
            choice_idx = p[-1]
            parent_row = path2row[parent]

            # assign this node a new row
            row = next_free_row
            next_free_row += 1
            path2row[p] = row

            # clone parent cache state into child row
            cache.conv_states[:, row].copy_(cache.conv_states[:, parent_row])
            cache.ssm_states[:, row].copy_(cache.ssm_states[:, parent_row])
            cache_pos[row] = cache_pos[parent_row]

            # copy tokens so far
            out_ids[row, :cur_len[parent_row]].copy_(out_ids[parent_row, :cur_len[parent_row]])
            cur_len[row] = cur_len[parent_row]

            # choose the token as "choice_idx-th best" from parent's logits
            topk_ids = torch.topk(logits[parent_row], k=TOPK, dim=-1).indices  # (TOPK,)
            tok = topk_ids[choice_idx].view(1, 1)  # (1,1)

            # write that token into this node's sequence
            out_ids[row, cur_len[row]] = tok[0, 0]
            cur_len[row] += 1

            parent_rows.append(parent_row)
            child_rows.append(row)
            child_next_ids.append(tok)

        # advance ALL new nodes one step (batched)
        row_indices = torch.tensor(child_rows, device=device, dtype=torch.long)
        next_ids = torch.cat(child_next_ids, dim=0).to(device)  # (Bsub,1)

        cache, cache_pos, logits = step_cached_batch(model, cache, cache_pos, logits, row_indices, next_ids)

    # ---- pick leaves and decode until MAX_NEW_TOKENS total
    leaf_rows = torch.tensor([path2row[lp] for lp in leaf_paths], device=device, dtype=torch.long)
    leaf_depths = torch.tensor([len(lp) for lp in leaf_paths], device=device, dtype=torch.long)

    # Each leaf already consumed `len(path)` branch tokens.
    remaining = MAX_NEW_TOKENS - leaf_depths.min().item()  # print-only; we actually stop by per-row cur_len.
    print(f"Built tree: total_nodes={len(all_paths)} leaves={leaf_rows.numel()} TOPK={TOPK} max_depth={max_depth}")
    print(f"prefill_time={prefill_time:.4f}s")
    print("Decoding leaves...")

    # decode step-by-step for all leaves; stop when EVERY leaf has produced MAX_NEW_TOKENS after prompt
    # i.e. sequence length reaches P + MAX_NEW_TOKENS
    target_len = P + MAX_NEW_TOKENS
    steps = 0
    decode_time_accum = 0.0

    while True:
        # which leaves still need tokens?
        need = (cur_len[leaf_rows] < target_len)
        if not torch.any(need):
            break

        active_leaf_rows = leaf_rows[need]
        # pick next token for active leaves
        next_id = pick_next_token(logits[active_leaf_rows], DO_SAMPLE, TEMPERATURE)  # (Bactive,1)

        # write tokens
        for i, r in enumerate(active_leaf_rows.tolist()):
            out_ids[r, cur_len[r]] = next_id[i, 0]
            cur_len[r] += 1

        # step cached for active leaves
        cuda_sync()
        t1 = time.time()
        cache, cache_pos, logits = step_cached_batch(model, cache, cache_pos, logits, active_leaf_rows, next_id)
        cuda_sync()
        decode_time_accum += (time.time() - t1)

        steps += 1
        if steps % max(1, DECODE_CHUNK) == 0:
            print(f"  step={steps:4d} active_leaves={active_leaf_rows.numel():3d}")

    # ---- decode text for each leaf
    texts = []
    for lp in leaf_paths:
        r = path2row[lp]
        seq = out_ids[r, :cur_len[r]].tolist()
        texts.append((lp, tokenizer.decode(seq, skip_special_tokens=True)))

    return texts


def main():
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Mamba2ForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()

    print("param dtype:", next(model.parameters()).dtype)
    print("device:", next(model.parameters()).device)
    print()

    # warmup
    _ = generate_tree_then_decode(model, tokenizer, "Hello",)
    print("\nWarmup done.\n")

    texts = generate_tree_then_decode(model, tokenizer, PROMPT)
    print("\n=== Leaf outputs (first 160 chars) ===")
    for lp, txt in texts:
        preview = txt.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"path={list(lp)!s:18s} | {preview}")


if __name__ == "__main__":
    main()
