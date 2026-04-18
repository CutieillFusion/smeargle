"""MMLU-Pro grader — macro-average accuracy over categories."""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, extract_letter, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 37.1  # base 5-shot CoT macro_avg/acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    correct_by_cat: dict[str, int] = defaultdict(int)
    total_by_cat: dict[str, int] = defaultdict(int)
    n = 0

    for qid, out in answers.items():
        if qid not in gold:
            continue
        g = gold[qid]
        cat = g["category"]
        pred = extract_letter(out, valid="ABCDEFGHIJ")
        ref = (g.get("reference") or "").strip().upper()
        total_by_cat[cat] += 1
        if pred == ref:
            correct_by_cat[cat] += 1
        n += 1

    per_cat = [correct_by_cat[c] / total_by_cat[c] for c in total_by_cat if total_by_cat[c]]
    macro = sum(per_cat) / len(per_cat) if per_cat else 0.0
    print_score("MMLU-Pro", "macro_acc", macro, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
