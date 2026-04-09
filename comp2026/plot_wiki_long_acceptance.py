import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_questions(question_file):
    """Build question_id -> category mapping."""
    mapping = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mapping[obj["question_id"]] = obj["category"]
    return mapping


def aggregate_acceptance_lengths(data, max_len=20):
    """Sum acceptance_lengths dicts across all questions into a single array."""
    counts = np.zeros(max_len)
    for dp in data:
        al = dp["choices"][0].get("acceptance_lengths", {})
        for pos_str, count in al.items():
            pos = int(pos_str)
            if pos < max_len:
                counts[pos] += count
    return counts


def cumulative_acceptance_rate(counts):
    """Reverse cumulative sum, normalized."""
    cumulative = np.cumsum(counts[::-1])[::-1]
    if cumulative[0] > 0:
        cumulative = cumulative / cumulative[0]
    return cumulative


def label_from_file(f):
    name = f.stem
    if "_window_" in name:
        return name.split("_window_")[-1]
    return name.replace("llama_3_1_8b_instruct_", "").split("_temperature")[0]


def _window_sort_key(f):
    if "_window_" in f.name:
        return int(f.name.split("_window_")[-1].split(".")[0])
    return 0


CATEGORY_ORDER = ["long_context_1k", "long_context_2k", "long_context_4k", "long_context_8k", "long_context_16k", "long_context_32k", "long_context_64k"]
CATEGORY_LABELS = {"long_context_1k": "1k", "long_context_2k": "2k", "long_context_4k": "4k", "long_context_8k": "8k", "long_context_16k": "16k", "long_context_32k": "32k", "long_context_64k": "64k"}
CATEGORY_COLORS = {"long_context_1k": "#4C72B0", "long_context_2k": "#DD8452", "long_context_4k": "#55A868", "long_context_8k": "#C44E52", "long_context_16k": "#8172B2", "long_context_32k": "#937860", "long_context_64k": "#DA8BC3"}


def main():
    parser = argparse.ArgumentParser(description="Plot per-category cumulative acceptance rate for WikiLong")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--output", type=str, default="wiki_long/acceptance_rate_per_position.png")
    parser.add_argument("--file-filter", type=str, default=None,
                        help="Only include spec files whose filename contains this substring")
    args = parser.parse_args()

    qid_to_cat = load_questions(args.question_file)
    data_dir = Path(args.data_dir)
    spec_files = sorted((f for f in data_dir.glob("*.jsonl") if "baseline" not in f.name and "temperature" in f.name), key=_window_sort_key)
    if args.file_filter:
        spec_files = [f for f in spec_files if args.file_filter in f.name]

    fig, axes = plt.subplots(1, len(spec_files), figsize=(8 * len(spec_files), 5), squeeze=False)

    for col, f in enumerate(spec_files):
        ax = axes[0, col]
        data = load_jsonl(f)
        model_label = label_from_file(f)

        # Group data by category, count OOMs
        cat_data = defaultdict(list)
        cat_oom = defaultdict(int)
        for dp in data:
            qid = dp["question_id"]
            cat = qid_to_cat.get(qid)
            if cat is None:
                continue
            if dp.get("skipped") == "OOM":
                cat_oom[cat] += 1
            else:
                cat_data[cat].append(dp)

        for cat in CATEGORY_ORDER:
            if cat not in cat_data:
                # Full OOM — add legend entry noting OOM
                if cat_oom.get(cat, 0) > 0:
                    ax.plot([], [], marker="None", color=CATEGORY_COLORS[cat], linestyle="None",
                            label=f"{CATEGORY_LABELS[cat]} (OOM)")
                continue
            counts = aggregate_acceptance_lengths(cat_data[cat])
            last_nonzero = np.max(np.nonzero(counts)) + 1 if np.any(counts) else 1
            counts = counts[:last_nonzero]
            rate = cumulative_acceptance_rate(counts)
            ax.plot(range(len(rate)), rate, marker="o",
                    label=CATEGORY_LABELS[cat], color=CATEGORY_COLORS[cat])

        ax.set_xlabel("Position")
        ax.set_ylabel("Acceptance Rate")
        ax.set_title(f"Acceptance Rate Per Position ({model_label})")
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
