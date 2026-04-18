"""BFCL (Berkeley Function-Calling Leaderboard v3) prep.

`load_dataset(...)` breaks on BFCL because its 49 Parquet files have
heterogeneous schemas across categories. We download the raw JSON files from
the HF repo with `hf_hub_download` and parse them per-category.

Multi-turn and REST categories are intentionally skipped — they don't fit the
single-turn MT-Bench prompt format the harness consumes.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import write_questions


REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Single-turn AST categories (ordered as in BFCL leaderboard).
AST_CATEGORIES = [
    "simple", "multiple", "parallel", "parallel_multiple",
    "java", "javascript", "sql",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
    "exec_simple", "exec_multiple", "exec_parallel", "exec_parallel_multiple",
    "irrelevance",
]


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"SKIPPED BFCL: huggingface_hub not installed ({e})")
        write_questions(here, [], meta={"name": "bfcl", "status": "skipped"})
        return

    records = []
    qid = 0
    loaded_categories = 0
    for cat in AST_CATEGORIES:
        try:
            q_path = hf_hub_download(REPO, f"BFCL_v3_{cat}.json", repo_type="dataset")
        except Exception as e:
            print(f"  skipping category {cat}: {e}")
            continue

        gt_by_id: dict[str, list] = {}
        try:
            gt_path = hf_hub_download(
                REPO, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset",
            )
            for row in load_jsonl(gt_path):
                gt_by_id[row["id"]] = row.get("ground_truth")
        except Exception:
            pass

        rows = load_jsonl(q_path)
        loaded_categories += 1
        for row in rows:
            q_list = row.get("question") or []
            # BFCL question is a nested list: [[{role, content}, ...]]
            if q_list and isinstance(q_list[0], list):
                turns = q_list[0]
            else:
                turns = q_list
            user_msg = ""
            for turn in turns:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    user_msg = turn.get("content", "")
                    break
            if not user_msg and turns:
                user_msg = turns[-1].get("content", "") if isinstance(turns[-1], dict) else str(turns[-1])

            tools = row.get("function") or []
            tools_str = json.dumps(tools, indent=2) if tools else "(none)"

            prompt = (
                "You are a function-calling assistant. Given the available tools and the "
                "user request, output the function call(s) as JSON.\n\n"
                f"Tools:\n{tools_str}\n\nRequest:\n{user_msg.strip()}\n\nFunction call:"
            )
            records.append({
                "question_id": qid,
                "category": cat,
                "turns": [prompt],
                "reference": json.dumps(gt_by_id.get(row["id"], "")),
                "bfcl_id": row["id"],
            })
            qid += 1

    if loaded_categories == 0:
        print("SKIPPED BFCL: no categories loaded")

    n = write_questions(here, records, meta={
        "name": "bfcl",
        "hf_path": REPO,
        "split": "train",
        "n_shots": 0,
        "prompting_style": "tool_use",
        "has_chat_variant": True,
    })
    print(f"wrote {n} BFCL examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
