"""AGIEval-English grader — average accuracy across English subsets."""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, extract_letter, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 47.8  # base 3-5 shot average/acc_char


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    n = 0
    for qid, out in answers.items():
        if qid not in gold:
            continue
        g = gold[qid]
        subset = g["category"]
        pred = extract_letter(out, valid="ABCDE")
        ref = (g.get("reference") or "").strip().upper()
        total[subset] += 1
        if pred and pred == ref:
            correct[subset] += 1
        n += 1

    per_subset = [correct[s] / total[s] for s in total if total[s]]
    avg = sum(per_subset) / len(per_subset) if per_subset else 0.0
    print_score("AGIEval-EN", "avg_acc", avg, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
