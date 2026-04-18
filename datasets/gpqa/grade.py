"""GPQA grader — micro-average accuracy."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, extract_letter, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 30.4  # Instruct 0-shot micro_avg/acc (gpqa_main)


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
        pred = extract_letter(out, valid="ABCD")
        if pred == ref:
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("GPQA", "acc", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
