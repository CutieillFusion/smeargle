"""Gorilla grader — function-name match (loose AST check)."""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 8.2  # Instruct Gorilla Benchmark API Bench (approx)


_CALL = re.compile(r"([A-Za-z_][\w\.]*)\s*\(")


def first_fn_name(text: str) -> str | None:
    if not text:
        return None
    m = _CALL.search(text)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    correct = 0
    n = 0
    for qid, out in answers.items():
        if qid not in gold:
            continue
        ref_fn = first_fn_name(gold[qid].get("reference") or "")
        pred_fn = first_fn_name(out)
        if ref_fn and pred_fn == ref_fn:
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("Gorilla", "fn_acc", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
