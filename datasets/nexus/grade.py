"""Nexus grader — function-call structural match."""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, normalize_text, print_score

MODEL_CARD_TARGET = 38.5  # Instruct 0-shot accuracy


_CALL = re.compile(r"([A-Za-z_][\w\.]*)\s*\((.*)\)", re.DOTALL)


def parse_call(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    m = _CALL.search(text)
    if not m:
        return None
    return m.group(1), m.group(2)


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
        ref = parse_call(gold[qid].get("reference") or "")
        pred = parse_call(out)
        if ref and pred and ref[0] == pred[0] and normalize_text(ref[1]) == normalize_text(pred[1]):
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("Nexus", "acc", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
