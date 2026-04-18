"""MMLU-Pro (5-shot CoT) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import LETTERS, safe_load_dataset, write_questions


INSTRUCTION = (
    "The following are multiple choice questions (with answers) about "
    "{category}. Think step by step, then finish with 'The answer is X' "
    "where X is the letter."
)


def render_shot(row: dict, with_answer: bool = True) -> str:
    opts = row["options"]
    lines = [f"Question: {row['question'].strip()}"]
    for i, opt in enumerate(opts):
        lines.append(f"{LETTERS[i]}. {opt}")
    if with_answer:
        cot = row.get("cot_content", "") or ""
        # cot_content sometimes starts with "A: " — strip it.
        if cot.startswith("A:"):
            cot = cot[2:].strip()
        lines.append(f"Answer: {cot}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ds = safe_load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    val = safe_load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

    shots_by_cat: dict[str, list[dict]] = {}
    for row in val:
        shots_by_cat.setdefault(row["category"], []).append(row)

    records = []
    for qid, row in enumerate(ds):
        shots = shots_by_cat.get(row["category"], [])[:5]
        parts = [INSTRUCTION.format(category=row["category"])]
        parts.extend(render_shot(s, with_answer=True) for s in shots)
        parts.append(render_shot(row, with_answer=False))
        prompt = "\n\n".join(parts)
        records.append({
            "question_id": qid,
            "category": row["category"],
            "turns": [prompt],
            "reference": row["answer"],
        })

    n = write_questions(here, records, meta={
        "name": "mmlu_pro",
        "hf_path": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "n_shots": 5,
        "prompting_style": "cot_mc",
        "has_chat_variant": True,
    })
    print(f"wrote {n} MMLU-Pro examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
