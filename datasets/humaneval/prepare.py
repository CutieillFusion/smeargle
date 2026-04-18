"""HumanEval (0-shot code) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ds = safe_load_dataset("openai/openai_humaneval", split="test")

    records = []
    for qid, row in enumerate(ds):
        prompt = (
            "Complete the following Python function. Return only the function body.\n\n"
            f"{row['prompt']}"
        )
        records.append({
            "question_id": qid,
            "category": "humaneval",
            "turns": [prompt],
            "reference": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "task_id": row["task_id"],
        })

    n = write_questions(here, records, meta={
        "name": "humaneval",
        "hf_path": "openai/openai_humaneval",
        "split": "test",
        "n_shots": 0,
        "prompting_style": "code",
        "has_chat_variant": True,
    })
    print(f"wrote {n} HumanEval examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
