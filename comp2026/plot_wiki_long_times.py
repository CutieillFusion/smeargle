import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = ["long_context_1k", "long_context_2k", "long_context_4k", "long_context_8k"]
CATEGORY_LABELS = {"long_context_1k": "1k", "long_context_2k": "2k", "long_context_4k": "4k", "long_context_8k": "8k"}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_questions(question_file):
    mapping = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mapping[obj["question_id"]] = obj["category"]
    return mapping


def label_from_file(f):
    name = f.stem
    return name.replace("llama_3_1_8b_instruct_", "").split("_temperature")[0]


def compute_avg_draft_ratio_by_category(data, qid_to_cat):
    """Compute average draft_time / target_time ratio per category."""
    cat_ratios = defaultdict(list)

    for dp in data:
        qid = dp["question_id"]
        cat = qid_to_cat.get(qid)
        if cat is None:
            continue
        choice = dp["choices"][0]
        draft = sum(choice["draft_model_time"])
        target = sum(choice["target_model_time"])
        if target > 0:
            cat_ratios[cat].append(draft / target)

    return {cat: np.mean(vals) for cat, vals in cat_ratios.items()}


def main():
    parser = argparse.ArgumentParser(description="Plot average draft/target time per category for wiki_long")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--output", type=str, default="wiki_long/avg_times_per_category.png")
    args = parser.parse_args()

    qid_to_cat = load_questions(args.question_file)
    data_dir = Path(args.data_dir)
    spec_files = sorted(f for f in data_dir.glob("*.jsonl") if "baseline" not in f.name and "temperature" in f.name)

    categories = [c for c in CATEGORY_ORDER if c in set(qid_to_cat.values())]
    x = np.arange(len(categories))

    colors = ["#DD8452", "#55A868", "#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, f in enumerate(spec_files):
        data = load_jsonl(f)
        model_label = label_from_file(f)
        ratios = compute_avg_draft_ratio_by_category(data, qid_to_cat)

        vals = [ratios.get(c, 0.0) for c in categories]
        ax.plot(x, vals, marker="o", label=model_label, color=colors[i % len(colors)], linewidth=2)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="target (1.0)")
    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Draft / Target Time Ratio")
    ax.set_title("Draft-to-Target Time Ratio (wiki_long benchmark)")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories])
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
