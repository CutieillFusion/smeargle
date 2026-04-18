"""SQuAD (1-shot) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import sample_shots, safe_load_dataset, write_questions


def render(row: dict, with_answer: bool) -> str:
    ans = (row["answers"]["text"] or [""])[0]
    lines = [
        f"Context: {row['context'].strip()}",
        f"Question: {row['question'].strip()}",
        "Answer:" + (f" {ans}" if with_answer else ""),
    ]
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    test_ds = safe_load_dataset("rajpurkar/squad", split="validation")
    train_ds = safe_load_dataset("rajpurkar/squad", split="train")

    shots = sample_shots(list(train_ds.select(range(200))), k=1, seed=0)

    records = []
    for qid, row in enumerate(test_ds):
        parts = [render(s, with_answer=True) for s in shots]
        parts.append(render(row, with_answer=False))
        prompt = "\n\n".join(parts)
        ans_list = row["answers"]["text"] or [""]
        records.append({
            "question_id": qid,
            "category": "squad",
            "turns": [prompt],
            "reference": ans_list[0],
            "aliases": ans_list,
        })

    n = write_questions(here, records, meta={
        "name": "squad",
        "hf_path": "rajpurkar/squad",
        "split": "validation",
        "n_shots": 1,
        "prompting_style": "span_qa",
        "has_chat_variant": True,
    })
    print(f"wrote {n} SQuAD examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
