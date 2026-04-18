"""Nexus / NexusRaven API evaluation (0-shot) prep.

Uses the `standardized_queries` config of Nexusflow/NexusRaven_API_evaluation
(318 examples). Each row has a user prompt, a gold function name, an args dict,
and a list of available tool names for that dataset.
"""

import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import safe_load_dataset, write_questions


def _parse_literal(s):
    """standardized_queries stores lists/dicts as Python literal strings."""
    if s is None:
        return None
    if isinstance(s, (list, dict)):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    try:
        ds = safe_load_dataset(
            "Nexusflow/NexusRaven_API_evaluation",
            "standardized_queries",
            split="train",
        )
    except Exception as e:
        print(f"SKIPPED Nexus: {e}")
        write_questions(here, [], meta={"name": "nexus", "status": "skipped", "reason": str(e)})
        return

    records = []
    for qid, row in enumerate(ds):
        fns = _parse_literal(row.get("context_functions")) or []
        fns_str = ", ".join(fns) if isinstance(fns, list) else str(fns)
        args = _parse_literal(row.get("python_args_dict")) or {}
        gold = f"{row.get('python_function_name')}({json.dumps(args) if isinstance(args, dict) else args})"
        prompt = (
            "You are a function-calling assistant. Given the available tools and the user "
            "request, output a single Python-style function call.\n\n"
            f"Available tools: {fns_str}\n\n"
            f"Request: {row.get('prompt', '').strip()}\n\nCall:"
        )
        records.append({
            "question_id": qid,
            "category": row.get("dataset", "nexus"),
            "turns": [prompt],
            "reference": gold,
        })

    n = write_questions(here, records, meta={
        "name": "nexus",
        "hf_path": "Nexusflow/NexusRaven_API_evaluation:standardized_queries",
        "split": "train",
        "n_shots": 0,
        "prompting_style": "tool_use",
        "has_chat_variant": True,
    })
    print(f"wrote {n} Nexus examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
