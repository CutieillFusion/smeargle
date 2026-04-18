"""IFEval (0-shot instruction following) prep."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # google/IFEval has a single "train" split containing the eval prompts.
    try:
        ds = safe_load_dataset("google/IFEval", split="train")
    except Exception:
        ds = safe_load_dataset("HuggingFaceH4/ifeval", split="train")

    records = []
    for qid, row in enumerate(ds):
        prompt = row["prompt"]
        records.append({
            "question_id": qid,
            "category": "ifeval",
            "turns": [prompt],
            "reference": "",
            "key": row.get("key"),
            "instruction_id_list": row.get("instruction_id_list"),
            "kwargs": row.get("kwargs"),
        })

    n = write_questions(here, records, meta={
        "name": "ifeval",
        "hf_path": "google/IFEval",
        "split": "train",
        "n_shots": 0,
        "prompting_style": "instruction_follow",
        "has_chat_variant": True,
    })
    print(f"wrote {n} IFEval examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
