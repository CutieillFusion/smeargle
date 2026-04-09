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
    if "_window_" in name:
        return name.split("_window_")[-1]
    return name.replace("llama_3_1_8b_instruct_", "").split("_temperature")[0]


def _window_sort_key(f):
    if "_window_" in f.name:
        return int(f.name.split("_window_")[-1].split(".")[0])
    return 0


def compute_avg_draft_ratio_by_category(data, qid_to_cat):
    """Compute average draft_time / target_time ratio per category.

    Returns (ratios_dict, oom_counts_dict).
    """
    cat_ratios = defaultdict(list)
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
        draft = sum(choice["draft_model_time"])
        target = sum(choice["target_model_time"])
        if target > 0:
            cat_ratios[cat].append(draft / target)

    return {cat: np.mean(vals) for cat, vals in cat_ratios.items()}, dict(cat_oom)


def main():
    parser = argparse.ArgumentParser(description="Plot average draft/target time per category for WikiLong")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--output", type=str, default="wiki_long/avg_times_per_category.png")
    parser.add_argument("--file-filter", type=str, default=None,
                        help="Only include spec files whose filename contains this substring")
    args = parser.parse_args()

    qid_to_cat = load_questions(args.question_file)
    data_dir = Path(args.data_dir)
    spec_files = sorted((f for f in data_dir.glob("*.jsonl") if "baseline" not in f.name and "temperature" in f.name), key=_window_sort_key)
    if args.file_filter:
        spec_files = [f for f in spec_files if args.file_filter in f.name]

    categories = [c for c in CATEGORY_ORDER if c in set(qid_to_cat.values())]
    x = np.arange(len(categories))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]

    fig, ax = plt.subplots(figsize=(8, 5))
    oom_vlines = []

    for i, f in enumerate(spec_files):
        data = load_jsonl(f)
        model_label = label_from_file(f)
        ratios, oom_counts = compute_avg_draft_ratio_by_category(data, qid_to_cat)
        color = colors[i % len(colors)]

        plot_x, plot_y = [], []
        for j, c in enumerate(categories):
            if c in ratios:
                plot_x.append(j)
                plot_y.append(ratios[c])

        ax.plot(plot_x, plot_y, marker="o", label=model_label, color=color, linewidth=2)
        # Track where model fully OOMs for vertical line
        full_oom_cats = [c for c in categories if oom_counts.get(c, 0) > 0 and c not in ratios]
        if full_oom_cats:
            first_full_oom_idx = categories.index(full_oom_cats[0])
            oom_vlines.append((5, model_label, color))

    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Empirical c")
    ax.set_title("Empirical c (WikiLong)")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories])
    ax.legend()
    ax.set_ylim(bottom=0)

    # Draw OOM vertical lines after axis limits are set (one per x-position)
    seen_oom_x = set()
    for vline_x, label, color in oom_vlines:
        if vline_x in seen_oom_x:
            continue
        seen_oom_x.add(vline_x)
        ax.axvline(x=vline_x, color="red", linestyle=":", linewidth=1.5)
        ax.text(vline_x + 0.05, ax.get_ylim()[1] * 0.95, "OOM",
                color="red", fontsize=8, ha="left", va="top", rotation=90)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
