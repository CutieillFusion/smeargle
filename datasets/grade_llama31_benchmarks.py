"""Run every per-benchmark grader and print a summary table.

Expects each benchmark's answer JSONL under <answers-dir>/ with filenames
matching gen_answer_llama_3_1_8b_bench.py's model_id convention, which always
ends in `_<benchmark_dirname>[.jsonl]` (optionally with `_window_N`).
"""

import argparse
import glob
import os
import subprocess
import sys


BASE_BENCHES = [
    "mmlu", "mmlu_pro", "agieval_en", "commonsense_qa", "winogrande",
    "bbh", "arc_challenge", "trivia_qa", "squad", "quac", "boolq", "drop",
]
INSTRUCT_BENCHES = ["mmlu_cot", "gpqa", "ifeval", "humaneval", "mbpp_plus", "gsm8k", "math"]
TOOLUSE_BENCHES = ["apibank", "bfcl", "gorilla", "nexus"]
ALL_BENCHES = BASE_BENCHES + INSTRUCT_BENCHES + TOOLUSE_BENCHES


def find_answer_file(answers_dir: str, bench: str) -> str | None:
    """Pick the newest answer file whose name ends in _<bench>(_window_N)?.jsonl."""
    patterns = [
        os.path.join(answers_dir, f"*_{bench}.jsonl"),
        os.path.join(answers_dir, f"*_{bench}_window_*.jsonl"),
    ]
    hits: list[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def run_one(bench: str, root: str, answers: str) -> tuple[str, str]:
    script = os.path.join(root, bench, "grade.py")
    if not os.path.exists(script):
        return bench, "MISSING_GRADER"
    try:
        result = subprocess.run(
            [sys.executable, script, "--answers", answers],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return bench, f"FAILED({e.returncode}): {e.stderr.strip()[:80]}"
    # Grader prints a single line like: "MMLU      macro_acc 66.73%  n=14042 ..."
    last = [l for l in result.stdout.splitlines() if l.strip()]
    return bench, last[-1] if last else "NO_OUTPUT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers-dir", required=True,
                   help="Directory containing answer JSONLs (e.g. testeagle3/).")
    p.add_argument("--benchmarks", type=str,
                   help="Comma-separated list; default = all.")
    args = p.parse_args()

    benches = (args.benchmarks.split(",") if args.benchmarks else ALL_BENCHES)
    root = os.path.dirname(os.path.abspath(__file__))

    rows: list[tuple[str, str, str]] = []
    for b in benches:
        ans = find_answer_file(args.answers_dir, b)
        if ans is None:
            rows.append((b, "NO_ANSWER_FILE", ""))
            continue
        _, line = run_one(b, root, ans)
        rows.append((b, line, os.path.basename(ans)))

    print("\n===== SUMMARY =====")
    for bench, line, f in rows:
        print(f"{bench:<16} | {line}")
        if f:
            print(f"{'':<16}   src: {f}")


if __name__ == "__main__":
    main()
