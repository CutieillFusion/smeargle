"""IFEval grader — average of prompt-level + instruction-level strict accuracy.

Uses the official google-research instruction_following_eval verifiers when
available. Install with: `pip install instruction_following_eval` (the
google-research repo also publishes it under several names).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, print_score

MODEL_CARD_TARGET = 80.4  # Instruct 0-shot avg(prompt_lvl_strict, inst_lvl_strict)


def _resolve_verifier():
    """Try several known package names for the IFEval verifier."""
    for mod in (
        "instruction_following_eval.evaluation",
        "instruction_following_eval",
        "ifeval.evaluation",
        "lm_eval.tasks.ifeval.instructions_registry",
    ):
        try:
            return __import__(mod, fromlist=["*"])
        except ImportError:
            continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    verifier = _resolve_verifier()
    if verifier is None:
        print(
            "SKIPPED: IFEval verifier package not installed. "
            "Install with `pip install instruction_following_eval` or vendor the "
            "google-research checker. Raw outputs live in the answer file."
        )
        return

    # Best-effort: both google-research and lm-eval expose an
    # `test_instruction_following_strict`-style function that takes an
    # InputExample + response. We reconstruct InputExample from gold.
    try:
        from instruction_following_eval.evaluation import (
            InputExample, test_instruction_following_strict,
        )
    except ImportError:
        print("SKIPPED: installed IFEval package does not match the expected API.")
        return

    prompt_correct = 0
    inst_total = 0
    inst_correct = 0
    n = 0
    for qid, out in answers.items():
        if qid not in gold:
            continue
        g = gold[qid]
        ex = InputExample(
            key=g.get("key", qid),
            instruction_id_list=g.get("instruction_id_list") or [],
            prompt=g["turns"][0],
            kwargs=g.get("kwargs") or [],
        )
        res = test_instruction_following_strict(ex, {ex.prompt: out})
        prompt_correct += int(res.follow_all_instructions)
        for flag in res.follow_instruction_list:
            inst_total += 1
            inst_correct += int(flag)
        n += 1

    if n == 0:
        print("No IFEval items to grade.")
        return

    prompt_acc = prompt_correct / n
    inst_acc = inst_correct / inst_total if inst_total else 0.0
    avg = (prompt_acc + inst_acc) / 2.0
    print_score(
        "IFEval", "strict_avg", avg, n, target=MODEL_CARD_TARGET,
        extra={"prompt_lvl_strict": f"{prompt_acc * 100:.2f}%",
               "inst_lvl_strict": f"{inst_acc * 100:.2f}%"},
    )


if __name__ == "__main__":
    main()
