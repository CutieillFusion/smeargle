"""ARC-Challenge grader — letter-match accuracy."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, extract_letter, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 79.7  # base 25-shot acc_char


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
        ref = (gold[qid].get("reference") or "").strip().upper()
        # ARC labels can be letters A-E or digits 1-5; accept either.
        pred = extract_letter(out, valid="ABCDE")
        if not pred:
            pred_digit = (out or "").strip()[:1]
            pred = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}.get(pred_digit)
        if pred and pred == ref:
            correct += 1
        n += 1

    acc = correct / n if n else 0.0
    print_score("ARC-C", "acc", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
