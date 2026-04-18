"""Winogrande grader — option-text accuracy (model-card acc_char)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, normalize_text, print_score

MODEL_CARD_TARGET = 60.5  # base 5-shot acc_char (Llama-3.1-8B)


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
        ref = gold[qid].get("reference") or ""
        first_line = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
        if normalize_text(ref) and normalize_text(first_line).startswith(normalize_text(ref)):
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("Winogrande", "acc", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
