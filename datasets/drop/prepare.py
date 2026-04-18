"""DROP (3-shot) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import sample_shots, safe_load_dataset, write_questions


def extract_answer(ans: dict) -> str:
    if ans.get("number"):
        return str(ans["number"])
    spans = ans.get("spans") or []
    if spans:
        return spans[0]
    date = ans.get("date") or {}
    if any(date.values()):
        return f"{date.get('day','')} {date.get('month','')} {date.get('year','')}".strip()
    return ""


def render(row: dict, with_answer: bool) -> str:
    ans = extract_answer(row.get("answers_spans") or row.get("answer") or {})
    lines = [
        f"Passage: {row['passage'].strip()}",
        f"Question: {row['question'].strip()}",
        "Answer:" + (f" {ans}" if with_answer else ""),
    ]
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    test_ds = safe_load_dataset("ucinlp/drop", split="validation")
    train_ds = safe_load_dataset("ucinlp/drop", split="train")

    shots = sample_shots(list(train_ds.select(range(200))), k=3, seed=0)

    records = []
    for qid, row in enumerate(test_ds):
        parts = [render(s, with_answer=True) for s in shots]
        parts.append(render(row, with_answer=False))
        prompt = "\n\n".join(parts)
        records.append({
            "question_id": qid,
            "category": "drop",
            "turns": [prompt],
            "reference": extract_answer(row.get("answers_spans") or {}),
        })

    n = write_questions(here, records, meta={
        "name": "drop",
        "hf_path": "ucinlp/drop",
        "split": "validation",
        "n_shots": 3,
        "prompting_style": "numeric_qa",
        "has_chat_variant": True,
    })
    print(f"wrote {n} DROP examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
