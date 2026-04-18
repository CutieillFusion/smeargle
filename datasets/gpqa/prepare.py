"""GPQA (0-shot, main split) prep."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import LETTERS, safe_load_dataset, write_questions


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # GPQA is gated on HF; fall through to a friendly skip.
    try:
        ds = safe_load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    except Exception as e:
        print(f"SKIPPED GPQA (gated / unavailable): {e}")
        write_questions(here, [], meta={"name": "gpqa", "status": "skipped", "reason": str(e)})
        return

    rng = random.Random(0)
    records = []
    for qid, row in enumerate(ds):
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        correct_idx = order.index(0)
        opts_text = "\n".join(f"{LETTERS[i]}. {shuffled[i]}" for i in range(4))
        prompt = (
            "Answer the following multiple choice question. Think step by step and "
            "finish with 'The answer is X' where X is one of A, B, C, D.\n\n"
            f"Question: {row['Question'].strip()}\n{opts_text}"
        )
        records.append({
            "question_id": qid,
            "category": row.get("High-level domain", "gpqa"),
            "turns": [prompt],
            "reference": LETTERS[correct_idx],
        })

    n = write_questions(here, records, meta={
        "name": "gpqa",
        "hf_path": "Idavidrein/gpqa",
        "split": "train",
        "n_shots": 0,
        "prompting_style": "cot_mc",
        "has_chat_variant": True,
    })
    print(f"wrote {n} GPQA examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
