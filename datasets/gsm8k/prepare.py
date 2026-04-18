"""GSM8K (8-shot CoT) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import sample_shots, safe_load_dataset, write_questions


def render_shot(q: str, a: str) -> str:
    return f"Question: {q.strip()}\nAnswer: {a.strip()}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    test_ds = safe_load_dataset("openai/gsm8k", "main", split="test")
    train_ds = safe_load_dataset("openai/gsm8k", "main", split="train")

    shots = sample_shots(
        [{"question": r["question"], "answer": r["answer"]} for r in train_ds.select(range(200))],
        k=8, seed=0,
    )
    shot_text = "\n\n".join(render_shot(s["question"], s["answer"]) for s in shots)

    records = []
    for qid, row in enumerate(test_ds):
        prompt = (
            "Solve the following grade-school math problem. Show your work step by step.\n\n"
            f"{shot_text}\n\nQuestion: {row['question'].strip()}\nAnswer:"
        )
        records.append({
            "question_id": qid,
            "category": "gsm8k",
            "turns": [prompt],
            "reference": row["answer"],
        })

    n = write_questions(here, records, meta={
        "name": "gsm8k",
        "hf_path": "openai/gsm8k",
        "split": "test",
        "n_shots": 8,
        "prompting_style": "cot_math",
        "has_chat_variant": True,
    })
    print(f"wrote {n} GSM8K examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
