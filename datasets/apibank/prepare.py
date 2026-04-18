"""API-Bank prep — scrapes dialog JSONLs from the AlibabaResearch/DAMO-ConvAI
GitHub repo (no HF mirror exists).

API-Bank stores each evaluation dialog as a JSONL file where User/AI/API turns
alternate. For single-turn MT-Bench-style evaluation we emit one example per
API turn: the prompt is the dialog history up to that point plus an instruction
to output the next API call; the reference is `api_name(param_dict_as_json)`.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import write_questions


GH_API = "https://api.github.com/repos/AlibabaResearch/DAMO-ConvAI/contents"
RAW = "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main"

SAMPLE_DIRS = [
    "api-bank/lv1-lv2-samples/level-1-given-desc",
    "api-bank/lv1-lv2-samples/level-2-toolsearcher",
]


def github_list(dir_path: str) -> list[dict]:
    url = f"{GH_API}/{dir_path}?ref=main"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def fetch_raw(path: str) -> str:
    url = f"{RAW}/{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_jsonl(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def render_history_prompt(history_turns: list[dict], instruction: str) -> str:
    lines = [instruction, ""]
    for turn in history_turns:
        role = turn.get("role", "")
        if role == "User":
            lines.append(f"User: {turn.get('text', '').strip()}")
        elif role == "AI":
            lines.append(f"Assistant: {turn.get('text', '').strip()}")
        elif role == "API":
            lines.append(
                f"API call: {turn.get('api_name')}"
                f"({json.dumps(turn.get('param_dict') or {})})"
            )
            result = turn.get("result", {}).get("output")
            lines.append(f"API result: {json.dumps(result)}")
    lines.append("")
    lines.append("Next API call:")
    return "\n".join(lines)


def extract_examples(dialog_turns: list[dict], category: str) -> list[dict]:
    """Emit one example per API turn; each uses prior context as the prompt."""
    out = []
    history: list[dict] = []
    instruction = (
        "You are an assistant with access to a set of APIs. Given the dialog so far, "
        "output the next API call as `api_name(<json-dict-of-params>)`."
    )
    for turn in dialog_turns:
        if turn.get("role") == "API":
            prompt = render_history_prompt(history, instruction)
            gold = f"{turn.get('api_name')}({json.dumps(turn.get('param_dict') or {})})"
            out.append({
                "category": category,
                "prompt": prompt,
                "reference": gold,
            })
        history.append(turn)
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # 1) enumerate dialog files
    file_paths: list[tuple[str, str]] = []  # (category, path)
    for d in SAMPLE_DIRS:
        try:
            listing = github_list(d)
        except Exception as e:
            print(f"  could not list {d}: {e}")
            continue
        cat = d.split("/")[-1]
        for f in listing:
            if f["type"] == "file" and f["name"].endswith(".jsonl"):
                file_paths.append((cat, f["path"]))

    if not file_paths:
        print("SKIPPED API-Bank: could not list any dialogs from GitHub")
        write_questions(here, [], meta={"name": "apibank", "status": "skipped"})
        return

    print(f"  fetching {len(file_paths)} dialog files from GitHub...")

    # 2) download each file (parallel)
    all_examples: list[dict] = []

    def fetch_one(item):
        cat, path = item
        try:
            text = fetch_raw(path)
        except Exception as e:
            return (cat, path, None, str(e))
        return (cat, path, parse_jsonl(text), None)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch_one, item) for item in file_paths]
        for fut in as_completed(futures):
            cat, path, turns, err = fut.result()
            if err or not turns:
                continue
            all_examples.extend(extract_examples(turns, cat))

    # Sort and assign question_ids deterministically.
    all_examples.sort(key=lambda e: (e["category"], e["prompt"][:60]))
    records = []
    for qid, ex in enumerate(all_examples):
        records.append({
            "question_id": qid,
            "category": ex["category"],
            "turns": [ex["prompt"]],
            "reference": ex["reference"],
        })

    n = write_questions(here, records, meta={
        "name": "apibank",
        "hf_path": "github:AlibabaResearch/DAMO-ConvAI/api-bank",
        "n_shots": 0,
        "prompting_style": "tool_use",
        "has_chat_variant": True,
    })
    print(f"wrote {n} API-Bank examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
