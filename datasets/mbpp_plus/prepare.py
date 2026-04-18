"""MBPP-EvalPlus (0-shot code) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ds = safe_load_dataset("evalplus/mbppplus", split="test")

    records = []
    for qid, row in enumerate(ds):
        asserts = row.get("test_list") or []
        assert_str = "\n".join(asserts) if asserts else ""
        prompt = (
            f"{row['prompt'].strip()}\n\n"
            "Your code should satisfy these assertions:\n"
            f"{assert_str}\n\n"
            "Write a Python function that solves the task."
        )
        records.append({
            "question_id": qid,
            "category": "mbpp_plus",
            "turns": [prompt],
            "reference": row.get("code", ""),
            "task_id": row.get("task_id"),
        })

    n = write_questions(here, records, meta={
        "name": "mbpp_plus",
        "hf_path": "evalplus/mbppplus",
        "split": "test",
        "n_shots": 0,
        "prompting_style": "code",
        "has_chat_variant": True,
    })
    print(f"wrote {n} MBPP-EvalPlus examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
