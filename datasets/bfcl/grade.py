"""BFCL grader — JSON/AST match against the ground-truth call.

Official BFCL scoring (AST / exec / relevance checks) is complex and tied to
the Berkeley evaluator. This grader uses a simpler structural check: parse the
model's JSON output, compare function name + argument values.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 76.1  # Instruct BFCL summary accuracy (approx)


_JSON_BLOCK = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_json_call(text: str):
    if not text:
        return None
    for m in _JSON_BLOCK.finditer(text):
        blob = m.group(0)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None


def extract_call(obj):
    """Normalise call JSON to {name, args} — tolerate several schemas."""
    if obj is None:
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function") or obj.get("tool_name")
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return {"name": name, "args": args}


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
        ref_raw = gold[qid].get("reference") or ""
        try:
            ref = json.loads(ref_raw) if ref_raw else None
        except json.JSONDecodeError:
            ref = None
        ref_call = extract_call(ref)
        pred_call = extract_call(parse_json_call(out))
        if ref_call and pred_call and ref_call == pred_call:
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("BFCL", "acc", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
