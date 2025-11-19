#!/usr/bin/env python3
import argparse
import os
import sys
from huggingface_hub import snapshot_download, login

def main():
    p = argparse.ArgumentParser(description="Download an HF model snapshot to a local directory.")
    p.add_argument("--repo_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                   help="HF Hub repo id (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    p.add_argument("--local_dir", type=str, required=True,
                   help="Local target directory to place the model snapshot")
    p.add_argument("--revision", type=str, default=None, help="Optional git revision/tag/commit")
    p.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"),
                   help="HF token (or set HF_TOKEN in env).")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Allow loading repos with custom code (safest to leave off for snapshots).")
    p.add_argument("--symlinks", action="store_true",
                   help="Use symlinks in cache (default False copies real files).")
    args = p.parse_args()

    if not args.token:
        print("ERROR: No HF token provided. Set HF_TOKEN env or pass --token.", file=sys.stderr)
        sys.exit(1)

    try:
        login(token=args.token)
    except Exception as e:
        print(f"ERROR: HF login failed: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.local_dir, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=args.local_dir,
            local_dir_use_symlinks=args.symlinks,
            token=args.token,
        )
        print(f"Downloaded to: {path}")
        print("Now use --basepath set to this directory.")
    except Exception as e:
        print("ERROR: Snapshot download failed.", file=sys.stderr)
        print("- Ensure you accepted the model license and have access.", file=sys.stderr)
        print("- Verify HF_TOKEN is valid and has necessary permissions.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()