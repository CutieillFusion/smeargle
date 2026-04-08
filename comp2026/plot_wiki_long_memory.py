import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = ["long_context_1k", "long_context_2k", "long_context_4k", "long_context_8k", "long_context_16k", "long_context_32k", "long_context_64k"]
CATEGORY_LABELS = {"long_context_1k": "1k", "long_context_2k": "2k", "long_context_4k": "4k", "long_context_8k": "8k", "long_context_16k": "16k", "long_context_32k": "32k", "long_context_64k": "64k"}


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


def compute_avg_peak_memory_by_category(data, qid_to_cat):
    """Compute average draft peak memory (MB) per category.

    Returns (mem_dict, oom_counts_dict).
    """
    cat_mem = defaultdict(list)
    cat_oom = defaultdict(int)

    for dp in data:
        qid = dp["question_id"]
        cat = qid_to_cat.get(qid)
        if cat is None:
            continue
        if dp.get("skipped") == "OOM":
            cat_oom[cat] += 1
            continue
        choice = dp["choices"][0]
        mem_bytes = choice.get("draft_peak_memory_bytes")
        if mem_bytes is not None:
            cat_mem[cat].append(mem_bytes / 1024 / 1024)  # -> MB

    return {cat: np.mean(vals) for cat, vals in cat_mem.items()}, dict(cat_oom)


def main():
    parser = argparse.ArgumentParser(description="Plot average draft peak memory per category for WikiLong")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--output", type=str, default="wiki_long/avg_memory_per_category.png")
    args = parser.parse_args()

    qid_to_cat = load_questions(args.question_file)
    data_dir = Path(args.data_dir)
    spec_files = sorted(f for f in data_dir.glob("*.jsonl") if "baseline" not in f.name and "temperature" in f.name)

    categories = [c for c in CATEGORY_ORDER if c in set(qid_to_cat.values())]
    x = np.arange(len(categories))

    colors = ["#DD8452", "#55A868", "#4C72B0", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    oom_vlines = []

    for i, f in enumerate(spec_files):
        data = load_jsonl(f)
        model_label = label_from_file(f)
        mem, oom_counts = compute_avg_peak_memory_by_category(data, qid_to_cat)
        color = colors[i % len(colors)]

        if not mem:
            print(f"  {model_label}: no draft_peak_memory_bytes data, skipping")
            continue

        plot_x, plot_y = [], []
        for j, c in enumerate(categories):
            if c in mem:
                plot_x.append(j)
                plot_y.append(mem[c])

        ax.plot(plot_x, plot_y, marker="o", label=model_label, color=color, linewidth=2)
        # Track where model fully OOMs for vertical line
        full_oom_cats = [c for c in categories if oom_counts.get(c, 0) > 0 and c not in mem]
        if full_oom_cats:
            first_full_oom_idx = categories.index(full_oom_cats[0])
            oom_vlines.append((5, model_label, color))

    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Draft Peak Memory (MB)")
    ax.set_title("Average Draft Peak Memory per Question (WikiLong)")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories])
    ax.legend()
    ax.set_ylim(bottom=0)

    # Draw OOM vertical lines after axis limits are set
    for vline_x, label, color in oom_vlines:
        ax.axvline(x=vline_x, color=color, linestyle=":", linewidth=1.5)
        ax.text(vline_x + 0.05, ax.get_ylim()[1] * 0.95, f"{label} OOM",
                color=color, fontsize=8, ha="left", va="top", rotation=90)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
