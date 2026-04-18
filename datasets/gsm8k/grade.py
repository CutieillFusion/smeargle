"""GSM8K grader — final-number exact-match (model-card em_maj1@1 with k=1)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import (
    default_paths, extract_gsm8k_gold, extract_last_number,
    load_answers, load_gold, print_score,
)

MODEL_CARD_TARGET = 84.5  # Instruct 8-shot CoT em_maj1@1


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().rstrip(".").replace(",", "")
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


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
        ref_num = normalize_number(extract_gsm8k_gold(gold[qid].get("reference") or ""))
        # Stop at next "Question:" if the model continued past its answer.
        truncated = (out or "").split("\nQuestion:")[0]
        pred_num = normalize_number(extract_last_number(truncated))
        if ref_num is not None and pred_num == ref_num:
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("GSM8K", "em", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
