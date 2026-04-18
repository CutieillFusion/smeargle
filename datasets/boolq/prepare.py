"""BoolQ (0-shot) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ds = safe_load_dataset("google/boolq", split="validation")

    records = []
    for qid, row in enumerate(ds):
        prompt = (
            f"Passage: {row['passage'].strip()}\n"
            f"Question: {row['question'].strip()}\n"
            "Answer (yes or no):"
        )
        records.append({
            "question_id": qid,
            "category": "boolq",
            "turns": [prompt],
            "reference": "yes" if row["answer"] else "no",
        })

    n = write_questions(here, records, meta={
        "name": "boolq",
        "hf_path": "google/boolq",
        "split": "validation",
        "n_shots": 0,
        "prompting_style": "yes_no",
        "has_chat_variant": True,
    })
    print(f"wrote {n} BoolQ examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
