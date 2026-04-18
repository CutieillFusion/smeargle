"""Winogrande (5-shot) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import sample_shots, safe_load_dataset, write_questions


def render_shot(row: dict, with_answer: bool) -> str:
    text = row["sentence"].replace("_", "____")
    lines = [
        f"Sentence: {text}",
        f"Option 1: {row['option1']}",
        f"Option 2: {row['option2']}",
    ]
    if with_answer:
        ans = row.get("answer", "")
        if ans in ("1", "2") and ans.isdigit():
            lines.append(f"Answer: {row['option' + ans]}")
        else:
            lines.append("Answer:")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    test_ds = safe_load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    train_ds = safe_load_dataset("allenai/winogrande", "winogrande_xl", split="train")

    shot_pool = list(train_ds.select(range(200)))
    shots = sample_shots(shot_pool, k=5, seed=0)

    instruction = (
        "Fill in the blank (____) in the sentence with the option that makes more sense."
    )

    records = []
    for qid, row in enumerate(test_ds):
        parts = [instruction]
        parts.extend(render_shot(s, with_answer=True) for s in shots)
        parts.append(render_shot(row, with_answer=False))
        prompt = "\n\n".join(parts)
        ans = row.get("answer", "")
        ref = row["option" + ans] if ans in ("1", "2") else ""
        records.append({
            "question_id": qid,
            "category": "winogrande",
            "turns": [prompt],
            "reference": ref,
        })

    n = write_questions(here, records, meta={
        "name": "winogrande",
        "hf_path": "allenai/winogrande",
        "split": "validation",
        "n_shots": 5,
        "prompting_style": "mc_text",
        "has_chat_variant": True,
    })
    print(f"wrote {n} Winogrande examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
