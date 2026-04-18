"""MBPP-EvalPlus grader — pass@1 against the `test_list` assertions.

Same sandboxing caveat as humaneval/grade.py. For full EvalPlus coverage
(plus-tests), install `evalplus` and run its official sanitizer; this grader
uses only the base asserts baked into each row.
"""

import argparse
import multiprocessing as mp
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 72.8  # Instruct 0-shot pass@1 (MBPP-EvalPlus base)
TIMEOUT_S = 10.0

_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(output: str) -> str:
    m = _CODE_BLOCK.search(output)
    if m:
        return m.group(1)
    return output


def _run_one(code: str, asserts: str, conn):
    try:
        ns: dict = {}
        exec(code, ns)
        exec(asserts, ns)
        conn.send((True, ""))
    except BaseException as e:
        conn.send((False, f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_with_timeout(code: str, asserts: str) -> tuple[bool, str]:
    parent, child = mp.Pipe()
    proc = mp.Process(target=_run_one, args=(code, asserts, child))
    proc.start()
    proc.join(TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "timeout"
    if parent.poll():
        return parent.recv()
    return False, "no result"


def _pull_asserts_from_question(prompt: str) -> str:
    """Recover the assertion block we rendered into the prompt."""
    lines = prompt.splitlines()
    try:
        start = lines.index("Your code should satisfy these assertions:") + 1
    except ValueError:
        return ""
    out = []
    for line in lines[start:]:
        if not line.strip():
            break
        out.append(line)
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    passed = 0
    n = 0
    for qid, out in answers.items():
        if qid not in gold:
            continue
        g = gold[qid]
        asserts = _pull_asserts_from_question(g["turns"][0])
        if not asserts:
            continue
        code = extract_code(out)
        ok, _ = run_with_timeout(code, asserts)
        if ok:
            passed += 1
        n += 1

    acc = passed / n if n else 0.0
    print_score("MBPP+", "pass@1", acc, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
