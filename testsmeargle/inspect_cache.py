import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect cache.pt mappings (d2t/t2d).")
    parser.add_argument(
        "cache_path",
        nargs="?",
        default="cache.pt",
        help="Path to cache.pt (default: cache.pt in CWD)",
    )
    parser.add_argument(
        "--show-mapping",
        type=int,
        default=0,
        help="Print first N draft->base mappings (default: 0, no listing).",
    )
    args = parser.parse_args()

    cache_file = Path(args.cache_path).expanduser().resolve()
    if not cache_file.exists():
        raise FileNotFoundError(f"cache.pt not found at {cache_file}")

    cache = torch.load(cache_file, map_location="cpu")
    d2t = cache["d2t"]
    t2d = cache["t2d"]

    draft_vocab = d2t.numel()
    base_vocab = t2d.numel()
    used_base = int(t2d.sum().item())

    mapped = torch.arange(draft_vocab) + d2t
    valid_mask = (mapped >= 0) & (mapped < base_vocab)
    valid_mapped = mapped[valid_mask]
    unique_mapped = torch.unique(valid_mapped)
    overlap_used = t2d[valid_mapped].sum().item()

    print(f"cache path: {cache_file}")
    print(f"draft_vocab_size: {draft_vocab}")
    print(f"base_vocab_size:  {base_vocab}")
    print(f"used_base_tokens (t2d==True): {used_base}")
    print(f"mapped_base_ids valid: {int(valid_mask.sum())} / {draft_vocab}")
    print(f"unique mapped_base_ids: {unique_mapped.numel()}")
    print(f"overlap with used_base (t2d): {int(overlap_used)}")
    print(f"d2t min/max: {int(d2t.min())}, {int(d2t.max())}")

    if args.show_mapping > 0:
        limit = min(args.show_mapping, draft_vocab)
        print("\nFirst draft->base mappings:")
        for i in range(limit):
            base_id = int(mapped[i]) if valid_mask[i] else None
            print(f"  draft {i:5d} -> base {base_id}")


if __name__ == "__main__":
    main()

