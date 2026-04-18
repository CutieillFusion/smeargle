"""MMLU (5-shot letter-answer) prep for Llama-3.1-8B base-model evaluation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import LETTERS, format_mc_prompt, safe_load_dataset, write_questions


INSTRUCTION = (
    "The following are multiple choice questions (with answers) about the given topic."
)


def _shot_from_row(row: dict) -> dict:
    return {
        "question": row["question"],
        "choices": row["choices"],
        "answer_letter": LETTERS[int(row["answer"])],
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    test_ds = safe_load_dataset("cais/mmlu", "all", split="test")
    dev_ds = safe_load_dataset("cais/mmlu", "all", split="dev")

    shots_by_subject: dict[str, list[dict]] = {}
    for row in dev_ds:
        shots_by_subject.setdefault(row["subject"], []).append(_shot_from_row(row))

    records = []
    for qid, row in enumerate(test_ds):
        subject = row["subject"]
        shots = shots_by_subject.get(subject, [])[:5]
        prompt = format_mc_prompt(
            row["question"],
            row["choices"],
            shots=shots,
            instruction=f"{INSTRUCTION} Topic: {subject.replace('_', ' ')}.",
        )
        records.append({
            "question_id": qid,
            "category": subject,
            "turns": [prompt],
            "reference": LETTERS[int(row["answer"])],
            "choices_list": row["choices"],
        })

    n = write_questions(here, records, meta={
        "name": "mmlu",
        "hf_path": "cais/mmlu",
        "split": "test",
        "n_shots": 5,
        "prompting_style": "mc_letter",
        "has_chat_variant": True,
    })
    print(f"wrote {n} MMLU examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
