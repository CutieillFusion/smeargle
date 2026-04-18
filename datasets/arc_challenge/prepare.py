"""ARC-Challenge (25-shot) prep."""

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

    test_ds = safe_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    train_ds = safe_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

    shot_pool = [_row_to_shot(r) for r in train_ds]
    shots = sample_shots(shot_pool, k=25, seed=0)

    records = []
    for qid, row in enumerate(test_ds):
        prompt = format_mc_prompt(
            row["question"],
            row["choices"]["text"],
            shots=shots,
            instruction="Answer the following science multiple choice questions.",
        )
        records.append({
            "question_id": qid,
            "category": "arc_challenge",
            "turns": [prompt],
            "reference": row["answerKey"],
        })

    n = write_questions(here, records, meta={
        "name": "arc_challenge",
        "hf_path": "allenai/ai2_arc",
        "split": "test",
        "n_shots": 25,
        "prompting_style": "mc_letter",
        "has_chat_variant": True,
    })
    print(f"wrote {n} ARC-Challenge examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
