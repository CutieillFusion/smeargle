"""DROP grader — token-level F1 (model-card metric)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import best_f1, default_paths, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 59.5  # base 3-shot f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    f1_total = 0.0
    n = 0
    for qid, out in answers.items():
        if qid not in gold:
            continue
        ref = gold[qid].get("reference") or ""
        pred = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
        f1_total += best_f1(pred, [ref])
        n += 1
    score = f1_total / n if n else 0.0
    print_score("DROP", "f1", score, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
