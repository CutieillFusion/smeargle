"""AGIEval English subsets (3-shot) prep.

`hails/agieval-*` schema: {"query": <full prompt with (A)/(B)/... inline>,
                           "choices": [...], "gold": [int index]}.
"""

import os
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions

LETTERS = list(string.ascii_uppercase)

ENGLISH_SUBSETS = [
    "lsat-ar", "lsat-lr", "lsat-rc",
    "sat-en", "sat-math", "sat-en-without-passage",
    "aqua-rat", "logiqa-en",
]


def render_example(row: dict, with_answer: bool) -> str:
    # AGIEval `query` already contains the full question + option list inline.
    query = row["query"].strip()
    if with_answer:
        gold = row.get("gold") or []
        letter = LETTERS[gold[0]] if gold else ""
        return f"{query}\nAnswer: {letter}"
    return f"{query}\nAnswer:"


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    records = []
    qid = 0
    loaded = 0
    for subset in ENGLISH_SUBSETS:
        try:
            ds = safe_load_dataset(f"hails/agieval-{subset}", split="test")
        except Exception as e:
            print(f"SKIPPED agieval-{subset}: {e}")
            continue
        loaded += 1

        shots = list(ds.select(range(min(3, len(ds)))))
        test_indices = range(len(shots), len(ds))

        for idx in test_indices:
            row = ds[idx]
            parts = [render_example(s, with_answer=True) for s in shots]
            parts.append(render_example(row, with_answer=False))
            prompt = "\n\n".join(parts)
            gold = row.get("gold") or []
            ref_letter = LETTERS[gold[0]] if gold else ""
            records.append({
                "question_id": qid,
                "category": subset,
                "turns": [prompt],
                "reference": ref_letter,
            })
            qid += 1

    if loaded == 0:
        print("SKIPPED: no AGIEval subsets could be loaded")

    n = write_questions(here, records, meta={
        "name": "agieval_en",
        "hf_path": "hails/agieval-*",
        "split": "test",
        "n_shots": 3,
        "prompting_style": "mc_letter",
        "has_chat_variant": True,
    })
    print(f"wrote {n} AGIEval-English examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
