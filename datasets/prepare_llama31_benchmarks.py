"""Orchestrator for Llama-3.1-8B model-card benchmark prep.

Runs each dataset's prepare.py as a subprocess so one failure doesn't kill the rest,
then prints a summary table.
"""

import argparse
import os
import subprocess
import sys


BASE_BENCHES = [
    "mmlu", "mmlu_pro", "agieval_en", "commonsense_qa", "winogrande",
    "bbh", "arc_challenge", "trivia_qa", "squad", "quac", "boolq", "drop",
]

INSTRUCT_BENCHES = [
    "mmlu_cot", "gpqa", "ifeval", "humaneval", "mbpp_plus", "gsm8k", "math",
]

TOOLUSE_BENCHES = ["apibank", "bfcl", "gorilla", "nexus"]

LARGE_BENCHES = {"mmlu", "mmlu_cot", "mmlu_pro", "trivia_qa", "drop", "quac"}

ALL_BENCHES = BASE_BENCHES + INSTRUCT_BENCHES + TOOLUSE_BENCHES


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def run_one(bench: str, root: str) -> tuple[str, int, str]:
    script = os.path.join(root, bench, "prepare.py")
    if not os.path.exists(script):
        return bench, 0, "MISSING_SCRIPT"
    try:
        subprocess.run(
            [sys.executable, script],
            check=True,
            cwd=root,
        )
    except subprocess.CalledProcessError as e:
        return bench, 0, f"FAILED(rc={e.returncode})"

    q_path = os.path.join(root, bench, "question.jsonl")
    n = count_lines(q_path)
    return bench, n, ("OK" if n > 0 else "EMPTY")


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run every benchmark.")
    group.add_argument("--benchmarks", type=str, help="Comma-separated list.")
    p.add_argument("--only-small", action="store_true",
                   help="Skip the multi-thousand-sample sets (MMLU, TriviaQA, DROP, QuAC).")
    args = p.parse_args()

    if args.all:
        benches = list(ALL_BENCHES)
    else:
        benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        for b in benches:
            if b not in ALL_BENCHES:
                sys.exit(f"Unknown benchmark: {b}. Known: {ALL_BENCHES}")

    if args.only_small:
        benches = [b for b in benches if b not in LARGE_BENCHES]

    root = os.path.dirname(os.path.abspath(__file__))

    results = []
    for b in benches:
        print(f"\n=== Preparing {b} ===", flush=True)
        results.append(run_one(b, root))

    print("\n\n===== SUMMARY =====")
    print(f"{'benchmark':<20} {'n_examples':>12}  status")
    print("-" * 50)
    for name, n, status in results:
        print(f"{name:<20} {n:>12}  {status}")


if __name__ == "__main__":
    main()
