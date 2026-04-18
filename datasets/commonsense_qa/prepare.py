"""CommonSenseQA (7-shot) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import format_mc_prompt, sample_shots, safe_load_dataset, write_questions


def _row_to_shot(row: dict) -> dict:
    return {
        "question": row["question"],
        "choices": row["choices"]["text"],
        "answer_letter": row["answerKey"],
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    test_ds = safe_load_dataset("tau/commonsense_qa", split="validation")
    train_ds = safe_load_dataset("tau/commonsense_qa", split="train")

    shot_pool = [_row_to_shot(r) for r in train_ds.select(range(200))]
    shots = sample_shots(shot_pool, k=7, seed=0)

    records = []
    for qid, row in enumerate(test_ds):
        prompt = format_mc_prompt(
            row["question"],
            row["choices"]["text"],
            shots=shots,
            instruction="Answer the following multiple choice questions.",
        )
        records.append({
            "question_id": qid,
            "category": "commonsense_qa",
            "turns": [prompt],
            "reference": row["answerKey"],
        })

    n = write_questions(here, records, meta={
        "name": "commonsense_qa",
        "hf_path": "tau/commonsense_qa",
        "split": "validation",
        "n_shots": 7,
        "prompting_style": "mc_letter",
        "has_chat_variant": True,
    })
    print(f"wrote {n} CommonSenseQA examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
