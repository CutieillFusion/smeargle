"""MATH grader — boxed-answer exact-match (model-card final_em).

Uses sympy for symbolic equivalence when available; falls back to string
normalisation otherwise.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, extract_boxed, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 51.9  # Instruct 0-shot final_em


def _strip_latex(s: str) -> str:
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    return s


def _string_equiv(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return _strip_latex(a) == _strip_latex(b)


def _sympy_equiv(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
    except Exception:
        return False
    try:
        ea = parse_latex(a)
        eb = parse_latex(b)
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


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
        ref = extract_boxed(gold[qid].get("reference") or "") or ""
        pred = extract_boxed(out or "") or ""
        if _string_equiv(pred, ref) or _sympy_equiv(pred, ref):
            correct += 1
        n += 1
    acc = correct / n if n else 0.0
    print_score("MATH", "final_em", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
