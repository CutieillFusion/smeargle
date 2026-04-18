"""MMLU-CoT grader (Instruct) — macro-average accuracy."""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, extract_letter, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 68.5  # Instruct 0-shot CoT macro_avg/acc


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
        subj = g["category"]
        pred = extract_letter(out, valid="ABCD")
        ref = (g.get("reference") or "").strip().upper()
        total[subj] += 1
        if pred == ref:
            correct[subj] += 1
        n += 1

    per = [correct[s] / total[s] for s in total if total[s]]
    macro = sum(per) / len(per) if per else 0.0
    print_score("MMLU-CoT", "macro_acc", macro, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
