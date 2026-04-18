"""Gorilla / APIBench prep.

`load_dataset(...)` breaks on gorilla-llm/APIBench because `api_arguments` has
inconsistent types across rows (string vs. object). We read the raw JSONL
files directly with `hf_hub_download` and parse per-provider.

Each row: {code: "###Instruction: ... ###Output: ...", api_call, provider, api_data}
We extract the instruction from `code` and use `api_call` as the reference.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import write_questions


REPO = "gorilla-llm/APIBench"
EVAL_FILES = [
    "huggingface_eval.json",
    "torchhub_eval.json",
    "tensorflow_eval.json",
]


_INSTRUCTION_RE = re.compile(
    r"###\s*Instruction\s*:\s*(.*?)(?=###\s*Output\s*:|\Z)", re.DOTALL | re.IGNORECASE,
)


def extract_instruction(code: str) -> str:
    m = _INSTRUCTION_RE.search(code or "")
    if m:
        return m.group(1).strip()
    return (code or "").strip()


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"SKIPPED Gorilla: huggingface_hub not installed ({e})")
        write_questions(here, [], meta={"name": "gorilla", "status": "skipped"})
        return

    records = []
    qid = 0
    loaded = 0
    for fname in EVAL_FILES:
        try:
            path = hf_hub_download(REPO, fname, repo_type="dataset")
        except Exception as e:
            print(f"  skipping {fname}: {e}")
            continue
        provider_tag = fname.replace("_eval.json", "")
        rows = load_jsonl(path)
        loaded += 1
        for row in rows:
            instruction = extract_instruction(row.get("code", ""))
            if not instruction:
                continue
            prompt = (
                "You are the Gorilla API-calling assistant. Given the user request, output "
                "the most appropriate API call (as executable Python).\n\n"
                f"Request: {instruction}\n\nAPI call:"
            )
            records.append({
                "question_id": qid,
                "category": provider_tag,
                "turns": [prompt],
                "reference": row.get("api_call", "") or "",
            })
            qid += 1

    if loaded == 0:
        print("SKIPPED Gorilla: no eval files could be downloaded")

    n = write_questions(here, records, meta={
        "name": "gorilla",
        "hf_path": REPO,
        "split": "eval",
        "n_shots": 0,
        "prompting_style": "tool_use",
        "has_chat_variant": True,
    })
    print(f"wrote {n} Gorilla examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
