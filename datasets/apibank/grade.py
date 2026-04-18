"""API-Bank grader — function-name + args exact match against `reference`."""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, normalize_text, print_score

MODEL_CARD_TARGET = 82.6  # Instruct 0-shot accuracy


_CALL_PAT = re.compile(r"([A-Za-z_][\w\.]*)\s*\((.*)\)", re.DOTALL)


def parse_call(text: str) -> tuple[str, str] | None:
    """Extract (fn_name, arg_string) from a model output or reference."""
    if not text:
        return None
    m = _CALL_PAT.search(text)
    if not m:
        return None
    return m.group(1), m.group(2)


def same_call(a: str, b: str) -> bool:
    pa, pb = parse_call(a), parse_call(b)
    if pa is None or pb is None:
        return False
    if pa[0] != pb[0]:
        return False
    return normalize_text(pa[1]) == normalize_text(pb[1])


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
        ref = gold[qid].get("reference") or ""
        if ref and same_call(out, ref):
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("API-Bank", "acc", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
