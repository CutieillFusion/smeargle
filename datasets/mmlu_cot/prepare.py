"""MMLU 0-shot CoT prep (Instruct variant)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import LETTERS, safe_load_dataset, write_questions


INSTRUCTION = (
    "The following is a multiple choice question about {subject}. Think step by step "
    "and then finish with 'The answer is X' where X is one of A, B, C, D."
)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ds = safe_load_dataset("cais/mmlu", "all", split="test")

    records = []
    for qid, row in enumerate(ds):
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        prompt = (
            INSTRUCTION.format(subject=row["subject"].replace("_", " ")) + "\n\n"
            f"Question: {row['question'].strip()}\n{opts}"
        )
        records.append({
            "question_id": qid,
            "category": row["subject"],
            "turns": [prompt],
            "reference": LETTERS[int(row["answer"])],
        })

    n = write_questions(here, records, meta={
        "name": "mmlu_cot",
        "hf_path": "cais/mmlu",
        "split": "test",
        "n_shots": 0,
        "prompting_style": "cot_mc",
        "has_chat_variant": True,
    })
    print(f"wrote {n} MMLU-CoT examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
