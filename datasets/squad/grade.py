"""SQuAD grader — normalized exact-match against any of the gold spans."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, exact_match, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 77.0  # base 1-shot em


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
        g = gold[qid]
        refs = list(g.get("aliases") or [g.get("reference", "")])
        refs = [r for r in refs if r]
        pred = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
        if refs and exact_match(pred, refs):
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("SQuAD", "em", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
