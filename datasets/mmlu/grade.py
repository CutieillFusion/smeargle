"""MMLU grader — macro-average accuracy over 57 subjects (model-card metric)."""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, extract_letter, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 66.7  # base Llama-3.1-8B, 5-shot, macro_avg/acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True, help="Path to answer JSONL.")
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    correct_by_subj: dict[str, int] = defaultdict(int)
    total_by_subj: dict[str, int] = defaultdict(int)
    n_graded = 0

    for qid, model_out in answers.items():
        if qid not in gold:
            continue
        g = gold[qid]
        subj = g["category"]
        pred = extract_letter(model_out, valid="ABCD")
        ref = (g.get("reference") or "").strip().upper()
        total_by_subj[subj] += 1
        if pred and pred == ref:
            correct_by_subj[subj] += 1
        n_graded += 1

    per_subj_acc = [
        correct_by_subj[s] / total_by_subj[s]
        for s in total_by_subj if total_by_subj[s] > 0
    ]
    macro = sum(per_subj_acc) / len(per_subj_acc) if per_subj_acc else 0.0
    print_score("MMLU", "macro_acc", macro, n_graded, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
