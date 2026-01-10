import time
import math
import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM
from time import sleep

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "benchang1110/mamba2-130m-hf"
PROMPT = "What is 15 + 15?"
MAX_NEW_TOKENS = 13

CHUNK_SIZE = 1
BRANCHING_CONFIG = [2, 1, 2, 1, 1, 2]  # multipliers: [depth0→depth1, depth1→depth2, ...]
# Each element is a multiplier: target_branches = current_branches × multiplier
# Calculate final total branches: 1 × 3 × 2 × 2 × 1 = 12
TOTAL_BRANCHES = 1
for n in BRANCHING_CONFIG:
    TOTAL_BRANCHES *= n
TREE_DEPTH = len(BRANCHING_CONFIG)

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
    out = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    cache = out.cache_params

    # For Mamba2 in HF, cache_position is basically a "cache initialized?" switch.
    # Any positive value works for decode steps. Keep it scalar per batch element.
    cache_pos = torch.full((prompt_ids.shape[0],), prompt_ids.shape[1], device=prompt_ids.device, dtype=torch.long)

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
    logits = out.logits[:, -1, :]  # (B, V)
    cache = out.cache_params
    cache_pos = cache_pos + 1
    return logits, cache, cache_pos


def _expand_cache_for_children(model, cache, cache_pos, logits, parent_indices: list, children_per_parent):
    """
    Returns (new_cache, new_cache_pos, new_logits) with batch size expanded by
    sum(children_per_parent), where each parent creates the specified number of children.
    
    Args:
        parent_indices: list of parent batch indices to branch from
        children_per_parent: list of how many children each parent creates (or int if all same)
    """
    old_bsz = cache_pos.shape[0]
    
    # Handle both list and int for children_per_parent
    if isinstance(children_per_parent, int):
        children_per_parent = [children_per_parent] * len(parent_indices)
    
    num_new = sum(children_per_parent)
    new_bsz = old_bsz + num_new

    # Create a fresh cache of the larger batch size and copy old state in.
    new_cache = type(cache)(model.backbone.config, new_bsz, device=cache.conv_states.device, dtype=cache.conv_states.dtype)

    # Copy existing batches [0:old_bsz]
    new_cache.conv_states[:, :old_bsz].copy_(cache.conv_states)
    new_cache.ssm_states[:, :old_bsz].copy_(cache.ssm_states)

    # For each parent, clone it the specified number of times
    new_idx = old_bsz
    for parent_idx, num_children in zip(parent_indices, children_per_parent):
        for _ in range(num_children):
            new_cache.conv_states[:, new_idx].copy_(cache.conv_states[:, parent_idx])
            new_cache.ssm_states[:, new_idx].copy_(cache.ssm_states[:, parent_idx])
            new_idx += 1

    # Expand cache_pos + logits
    new_cache_pos = torch.empty((new_bsz,), device=cache_pos.device, dtype=cache_pos.dtype)
    new_cache_pos[:old_bsz] = cache_pos
    
    new_idx = old_bsz
    for parent_idx, num_children in zip(parent_indices, children_per_parent):
        for _ in range(num_children):
            new_cache_pos[new_idx] = cache_pos[parent_idx]
            new_idx += 1

    new_logits = torch.empty((new_bsz, logits.shape[-1]), device=logits.device, dtype=logits.dtype)
    new_logits[:old_bsz] = logits
    
    new_idx = old_bsz
    for parent_idx, num_children in zip(parent_indices, children_per_parent):
        for _ in range(num_children):
            new_logits[new_idx] = logits[parent_idx]
            new_idx += 1

    return new_cache, new_cache_pos, new_logits


@torch.inference_mode()
def generate_in_chunks(model, tokenizer, prompt_text: str, max_new_tokens: int, chunk_size: int):
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    base_B, P = prompt_ids.shape
    T = max_new_tokens

    # Calculate maximum batch size: final target is TOTAL_BRANCHES (12)
    # But we need space for all intermediate branches too
    # Calculate cumulative: 1 → 3 → 6 → 12 → 12
    current = base_B
    max_branches_needed = current
    for multiplier in BRANCHING_CONFIG:
        current = current * multiplier
        max_branches_needed = max(max_branches_needed, current)
    max_B = max_branches_needed

    # Preallocate token buffer for the maximum possible batch size
    out_ids = torch.empty((max_B, P + T), dtype=prompt_ids.dtype, device=device)
    out_ids[:base_B, :P] = prompt_ids
    cur = P

    # Prefill once (batch starts at base_B)
    cuda_sync()
    t0 = time.time()
    logits, cache, cache_pos = prefill(model, prompt_ids)
    cuda_sync()
    prefill_time = time.time() - t0

    active_B = base_B
    
    # Track tree structure: branch_depth[i] = depth level of branch i
    # Depth 0 = root, depth 1 = first level children, etc.
    branch_depth = [0] * active_B  # root starts at depth 0
    
    # Track parent-child relationships: branch_parent[i] = parent branch index (None for root)
    branch_parent = [None] * active_B  # root has no parent
    
    # Track which branches belong to each depth level
    depth_to_branches = {0: list(range(base_B))}  # root branches

    # Create the full tree structure by interleaving token generation with branching
    # At each depth, we first generate one token for all current branches (to get diverged logits),
    # then use those diverged logits to create children for the next depth
    forced_tokens_all = {}  # {branch_idx: token_id} for forced divergence, accumulated across all depths
    
    print(f"\n*** Creating tree structure with interleaved token generation and branching ***")
    for depth in range(TREE_DEPTH):
        if depth not in depth_to_branches:
            print(f"  WARNING: Depth {depth} not in depth_to_branches, breaking")
            break
            
        parent_indices = depth_to_branches[depth]
        current_branches = len(parent_indices)
        multiplier = BRANCHING_CONFIG[depth]
        
        # Calculate target branches at next depth
        target_branches = current_branches * multiplier
        num_new_branches = target_branches - current_branches
        
        print(f"  Depth {depth} -> {depth+1}: {current_branches} branches × {multiplier} = {target_branches} target branches")
        print(f"    Need to add {num_new_branches} new branches")
        print(f"    Parent indices at depth {depth}: {parent_indices}")
        
        # FIRST: Generate one token for all current branches at this depth to get diverged logits
        # This ensures each parent has unique logits when creating children
        # At depth 0, we use prefill logits (no need to generate first)
        # At depth > 0, we need to generate one token so parents have diverged logits
        if depth > 0 and num_new_branches > 0:
            print(f"    Generating one token for {len(parent_indices)} branches at depth {depth} to get diverged logits...")
            
            # Pick next token for all current branches
            next_id = pick_next_token(logits[:active_B], DO_SAMPLE, TEMPERATURE)  # (active_B,1)
            
            # Apply forced tokens for branches that need divergence (from previous depth)
            for branch_idx, forced_token_id in forced_tokens_all.items():
                if branch_idx < active_B and branch_idx in parent_indices:
                    next_id[branch_idx] = forced_token_id
            
            # Write tokens
            out_ids[:active_B, cur : cur + 1] = next_id
            cur += 1
            
            # Advance cache for active rows - this gives us diverged logits
            logits, cache, cache_pos = step_cached(model, next_id, cache, cache_pos)
            print(f"    Generated token at position {cur-1}, logits now diverged for all parents")
        
        if num_new_branches > 0 and active_B + num_new_branches <= max_B:
            old_active_B = active_B
            
            # Distribute new branches among parents
            # Calculate how many children each parent should create
            num_parents = len(parent_indices)
            children_per_parent_base = num_new_branches // num_parents
            extra_children = num_new_branches % num_parents  # Distribute remainder
            
            # Create list of children per parent
            children_per_parent = [children_per_parent_base] * num_parents
            for i in range(extra_children):
                children_per_parent[i] += 1
            
            print(f"    Distribution: {children_per_parent} children per parent")
            print(f"    Parents and their children: {list(zip(parent_indices, children_per_parent))}")
            print(f"    Batch size: {current_branches} -> {target_branches}")
            
            # Expand cache for all children
            cache, cache_pos, logits = _expand_cache_for_children(
                model, cache, cache_pos, logits, parent_indices, children_per_parent
            )
            
            # Copy tokens from parents to children and set up forced divergence
            new_branch_indices = []
            for parent_num, parent_idx in enumerate(parent_indices):
                num_children_for_this_parent = children_per_parent[parent_num]
                
                if num_children_for_this_parent > 0:
                    # Get top-k tokens for this parent (k = num_children + 1 to have enough options)
                    parent_logits = logits[parent_idx]  # (V,)
                    topk = torch.topk(parent_logits, k=num_children_for_this_parent + 1, dim=-1).indices  # (num_children+1,)
                    
                    # Create children
                    children_for_this_parent = []
                    for child_num in range(num_children_for_this_parent):
                        child_idx = active_B + len(new_branch_indices)
                        new_branch_indices.append(child_idx)
                        children_for_this_parent.append(child_idx)
                        
                        # Copy parent's token sequence (all at the same position cur)
                        out_ids[child_idx, :cur].copy_(out_ids[parent_idx, :cur])
                        
                        # Force divergence: child 0 gets 2nd-best, child 1 gets 3rd-best, etc.
                        forced_token = topk[child_num + 1].item()  # +1 to skip the best (parent will take it)
                        forced_tokens_all[child_idx] = forced_token
                        
                        # Track depth and parent
                        branch_depth.append(depth + 1)
                        branch_parent.append(parent_idx)  # Track parent-child relationship
                    
                    print(f"      Parent {parent_idx} created children: {children_for_this_parent}")
            
            active_B += len(new_branch_indices)
            
            # Update depth_to_branches for next level
            # Next level includes all current branches (parents) plus new children
            if depth + 1 not in depth_to_branches:
                depth_to_branches[depth + 1] = []
            # Keep parents and add children
            depth_to_branches[depth + 1].extend(parent_indices)  # Keep parents
            depth_to_branches[depth + 1].extend(new_branch_indices)  # Add children
            
            print(f"    Created {len(new_branch_indices)} new branches (total active_B={active_B}, branches at depth {depth+1}: {len(depth_to_branches[depth + 1])})")
        elif num_new_branches == 0:
            print(f"    No new branches needed (target={target_branches} == current={current_branches})")
            # Still update depth_to_branches to include current branches at next level
            if depth + 1 not in depth_to_branches:
                depth_to_branches[depth + 1] = []
            depth_to_branches[depth + 1].extend(parent_indices)
        else:
            print(f"    SKIPPED: active_B + num_new={active_B + num_new_branches} > max_B={max_B}")
    
    # Calculate final branch count at each depth for reporting
    final_branches_by_depth = {}
    for i in range(active_B):
        d = branch_depth[i] if i < len(branch_depth) else -1
        if d not in final_branches_by_depth:
            final_branches_by_depth[d] = 0
        final_branches_by_depth[d] += 1
    
    print(f"*** Tree structure complete at position {cur}: {active_B} total branches ***")
    print(f"  Branches by depth: {final_branches_by_depth}")
    print(f"  All branches are at position {cur} and will generate in parallel\n")

    chunk_times = []
    total_steps = 0
    chunk_idx = 0

    while total_steps < max_new_tokens:
        this_chunk = min(chunk_size, max_new_tokens - total_steps)

        cuda_sync()
        t_chunk = time.time()

        for j in range(this_chunk):
            # Pick next token for all active rows
            next_id = pick_next_token(logits[:active_B], DO_SAMPLE, TEMPERATURE)  # (active_B,1)

            # Apply forced tokens for branches that need divergence (only on first token)
            if forced_tokens_all and total_steps == 0 and j == 0:
                for branch_idx, forced_token_id in forced_tokens_all.items():
                    if branch_idx < active_B:
                        next_id[branch_idx] = forced_token_id

            # Write tokens
            out_ids[:active_B, cur : cur + 1] = next_id
            cur += 1

            # Advance cache for active rows
            logits, cache, cache_pos = step_cached(model, next_id, cache, cache_pos)
            total_steps += 1

        cuda_sync()
        chunk_times.append(time.time() - t_chunk)

        avg_ms_per_tok = (chunk_times[-1] / this_chunk) * 1000.0
        print(
            f"chunk_idx={chunk_idx:4d}  active_B={active_B:2d}  "
            f"chunk_end={total_steps:4d}/{max_new_tokens}  "
            f"chunk_time={chunk_times[-1]:.4f}s  avg={avg_ms_per_tok:.3f} ms/token"
        )

        chunk_idx += 1

    # Decode each batch row
    texts = []
    token_ids_list = []
    for b in range(active_B):
        token_ids = out_ids[b, :cur].tolist()
        token_ids_list.append(token_ids)
        texts.append(tokenizer.decode(token_ids, skip_special_tokens=True))

    return texts, token_ids_list, prefill_time, chunk_times, active_B, branch_depth, branch_parent


def main():
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Mamba2ForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()

    print("param dtype:", next(model.parameters()).dtype)
    print("device:", next(model.parameters()).device)
    print()

    # Warmup
    _ = generate_in_chunks(model, tokenizer, PROMPT, max_new_tokens=32, chunk_size=8)
    print("\nWarmup done.\n")

    print(f"=== Tree-structured cached generation with branching ===")
    print(f"BRANCHING_CONFIG={BRANCHING_CONFIG}")
    print(f"TOTAL_BRANCHES={TOTAL_BRANCHES}, TREE_DEPTH={TREE_DEPTH}")
    print(f"CHUNK_SIZE={CHUNK_SIZE}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}\n")

    cuda_sync()
    t0 = time.time()
    texts, token_ids_list, prefill_time, chunk_times, active_B, branch_depth, branch_parent = generate_in_chunks(
        model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, chunk_size=CHUNK_SIZE
    )
    cuda_sync()
    total_time = time.time() - t0

    avg_ms_per_tok = (sum(chunk_times) / MAX_NEW_TOKENS) * 1000.0

    print("\n=== Summary ===")
    print(f"prefill_time: {prefill_time:.4f}s")
    print(f"decode_time (sum chunk times): {sum(chunk_times):.4f}s")
    print(f"avg decode: {avg_ms_per_tok:.3f} ms/token")
    print(f"total_time: {total_time:.4f}s")
    print(f"final active_B: {active_B}")

    # Build lineage for each branch
    # Each branch's lineage shows which branch index appears at each depth level
    def build_lineages():
        """Build the lineage (path from root) for each branch.
        Returns: {branch_idx: [branch_idx_at_depth_0, branch_idx_at_depth_1, ...]}"""
        lineages = {}
        
        # Find max depth
        max_depth = 0
        for i in range(active_B):
            depth = branch_depth[i] if i < len(branch_depth) else 0
            max_depth = max(max_depth, depth)
        
        # For each branch, trace its lineage
        for branch_idx in range(active_B):
            lineage = []
            creation_depth = branch_depth[branch_idx] if branch_idx < len(branch_depth) else 0
            
            # Build path from root to this branch
            path_from_root = []
            current = branch_idx
            while current is not None:
                path_from_root.insert(0, current)  # Insert at beginning to build root->branch path
                if current < len(branch_parent) and branch_parent[current] is not None:
                    current = branch_parent[current]
                else:
                    break
            
            # Now build lineage: at each depth, which branch index appears
            # A branch appears at its creation depth and all subsequent depths
            # Before its creation depth, it "is" its ancestor at that depth
            for depth in range(max_depth + 1):
                if depth < creation_depth:
                    # Before this branch was created, find which ancestor it "was" at this depth
                    # The ancestor should be the one in path_from_root that was created at or before this depth
                    branch_at_depth = path_from_root[0]  # Default to root
                    for ancestor in path_from_root:
                        ancestor_depth = branch_depth[ancestor] if ancestor < len(branch_depth) else 0
                        if ancestor_depth <= depth:
                            branch_at_depth = ancestor
                        else:
                            break  # Ancestors are in order, so we can stop
                    lineage.append(branch_at_depth)
                else:
                    # At or after creation depth, this branch appears as itself
                    lineage.append(branch_idx)
            
            lineages[branch_idx] = lineage
        
        return lineages, max_depth
    
    lineages, max_depth = build_lineages()
    
    print("\n" + "="*80)
    print("=== Branch Lineages ===")
    print("="*80)
    print()
    
    # Print lineage for each branch with generated text and token IDs
    for branch_idx in range(active_B):
        lineage = lineages[branch_idx]
        lineage_str = "[" + ", ".join(f"{b}" for b in lineage) + "]"
        print(f"Branch {branch_idx:2d}: {lineage_str}")
        # Print the generated text for this branch (first 200 chars)
        text_preview = texts[branch_idx][:200].replace('\n', ' ')
        if len(texts[branch_idx]) > 200:
            text_preview += "..."
        print(f"           {text_preview}")
        # Print token IDs
        token_ids = token_ids_list[branch_idx]
        token_ids_str = "[" + ", ".join(f"{tid}" for tid in token_ids) + "]"
        print(f"           Token IDs: {token_ids_str}")
        print()


if __name__ == "__main__":
    main()
