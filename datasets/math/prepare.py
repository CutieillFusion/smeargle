"""MATH (0-shot CoT) prep. Uses MATH-500 with fallback to hendrycks/competition_math."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    source = "HuggingFaceH4/MATH-500"
    try:
        ds = safe_load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as e:
        print(f"MATH-500 unavailable ({e}); trying hendrycks/competition_math...")
        try:
            ds = safe_load_dataset("hendrycks/competition_math", split="test")
            source = "hendrycks/competition_math"
        except Exception as e2:
            print(f"SKIPPED: both MATH sources unavailable ({e2})")
            write_questions(here, [], meta={"name": "math", "status": "skipped", "reason": str(e2)})
            return

    records = []
    for qid, row in enumerate(ds):
        problem = row.get("problem") or row.get("question") or ""
        sol = row.get("solution") or row.get("answer") or ""
        category = row.get("subject") or row.get("type") or "math"
        prompt = (
            "Solve the following math problem. Show your work step by step and put your "
            "final answer inside \\boxed{...}.\n\n"
            f"Problem: {problem.strip()}"
        )
        records.append({
            "question_id": qid,
            "category": str(category),
            "turns": [prompt],
            "reference": sol,
        })

    n = write_questions(here, records, meta={
        "name": "math",
        "hf_path": source,
        "split": "test",
        "n_shots": 0,
        "prompting_style": "cot_math",
        "has_chat_variant": True,
    })
    print(f"wrote {n} MATH examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
