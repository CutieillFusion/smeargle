"""TriviaQA-Wiki (5-shot, nocontext) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import format_open_prompt, sample_shots, safe_load_dataset, write_questions


def _row_to_shot(row: dict) -> dict:
    return {
        "question": row["question"],
        "answer": row["answer"]["value"],
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    test_ds = safe_load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="validation")
    train_ds = safe_load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="train")

    shot_pool = [_row_to_shot(r) for r in train_ds.select(range(200))]
    shots = sample_shots(shot_pool, k=5, seed=0)

    records = []
    for qid, row in enumerate(test_ds):
        prompt = format_open_prompt(
            row["question"],
            shots=shots,
            instruction="Answer the following trivia questions.",
        )
        records.append({
            "question_id": qid,
            "category": "trivia_qa",
            "turns": [prompt],
            "reference": row["answer"]["value"],
            "aliases": list(row["answer"].get("aliases") or []),
        })

    n = write_questions(here, records, meta={
        "name": "trivia_qa",
        "hf_path": "mandarjoshi/trivia_qa",
        "split": "validation",
        "n_shots": 5,
        "prompting_style": "open_qa",
        "has_chat_variant": True,
    })
    print(f"wrote {n} TriviaQA examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
