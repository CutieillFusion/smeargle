"""BIG-Bench Hard (3-shot CoT) prep — all 27 subtasks."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions


SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    records = []
    qid = 0
    loaded = 0
    for subtask in SUBTASKS:
        try:
            ds = safe_load_dataset("lukaemon/bbh", subtask, split="test")
        except Exception as e:
            print(f"SKIPPED bbh/{subtask}: {e}")
            continue
        loaded += 1

        # Use first 3 as shots.
        shots = list(ds.select(range(min(3, len(ds)))))
        shot_text = "\n\n".join(
            f"Q: {s['input'].strip()}\nA: {s['target'].strip()}" for s in shots
        )
        test_indices = range(len(shots), len(ds))

        for idx in test_indices:
            row = ds[idx]
            prompt = (
                f"Answer the following questions. Think step by step.\n\n"
                f"{shot_text}\n\nQ: {row['input'].strip()}\nA:"
            )
            records.append({
                "question_id": qid,
                "category": f"bbh_{subtask}",
                "turns": [prompt],
                "reference": row["target"],
            })
            qid += 1

    if loaded == 0:
        print("SKIPPED: no BBH subsets could be loaded")

    n = write_questions(here, records, meta={
        "name": "bbh",
        "hf_path": "lukaemon/bbh",
        "split": "test",
        "n_shots": 3,
        "prompting_style": "cot",
        "has_chat_variant": True,
    })
    print(f"wrote {n} BBH examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
