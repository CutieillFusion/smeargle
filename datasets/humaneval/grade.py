"""HumanEval grader — pass@1 by executing generated code against `test`.

WARNING: this executes untrusted model output in a subprocess. Each subprocess
is isolated with a wall-clock timeout but NOT sandboxed from the filesystem.
Run in a container or VM if you don't trust the model.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 72.6  # Instruct 0-shot pass@1
TIMEOUT_S = 10.0


_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(output: str, prompt: str) -> str:
    """Pull python code out of a model response, falling back to raw text."""
    m = _CODE_BLOCK.search(output)
    if m:
        return m.group(1)
    # If the model just continued the function body, stitch prompt + body.
    if output.lstrip().startswith("def "):
        return output
    return prompt + "\n" + output


def _run_one(code: str, test: str, entry_point: str, conn):
    try:
        ns: dict = {}
        exec(code, ns)
        exec(test, ns)
        ns["check"](ns[entry_point])
        conn.send((True, ""))
    except BaseException as e:
        conn.send((False, f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_with_timeout(code: str, test: str, entry_point: str) -> tuple[bool, str]:
    parent, child = mp.Pipe()
    proc = mp.Process(target=_run_one, args=(code, test, entry_point, child))
    proc.start()
    proc.join(TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "timeout"
    if parent.poll():
        return parent.recv()
    return False, "no result"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    passed = 0
    n = 0
    failures: list[str] = []
    for qid, out in answers.items():
        if qid not in gold:
            continue
        g = gold[qid]
        prompt = g["turns"][0]
        code = extract_code(out, prompt)
        test = g.get("test") or ""
        entry = g.get("entry_point") or ""
        if not test or not entry:
            continue
        ok, err = run_with_timeout(code, test, entry)
        if ok:
            passed += 1
        else:
            failures.append(f"{g.get('task_id', qid)}: {err}")
        n += 1

    acc = passed / n if n else 0.0
    print_score("HumanEval", "pass@1", acc, n, target=MODEL_CARD_TARGET)
    if failures:
        print(f"  first 5 failures: {failures[:5]}")


if __name__ == "__main__":
    main()
