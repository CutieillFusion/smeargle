import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = ["long_context_1k", "long_context_2k", "long_context_4k", "long_context_8k", "long_context_16k", "long_context_32k", "long_context_64k"]
CATEGORY_LABELS = {"long_context_1k": "1k", "long_context_2k": "2k", "long_context_4k": "4k", "long_context_8k": "8k", "long_context_16k": "16k", "long_context_32k": "32k", "long_context_64k": "64k"}


def load_questions(question_file):
    mapping = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mapping[obj["question_id"]] = obj["category"]
    return mapping


def collect_tokens(answers_file, qid_to_cat):
    """Return {category: [new_tokens, ...]} for every answerable question."""
    cat_tokens = defaultdict(list)
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dp = json.loads(line)
            qid = dp["question_id"]
            cat = qid_to_cat.get(qid)
            if cat is None or not dp["choices"]:
                continue
            new_tokens = dp["choices"][0]["new_tokens"][0]
            cat_tokens[cat].append(new_tokens)
    return cat_tokens


def main():
    parser = argparse.ArgumentParser(description="Plot new tokens generated per question per category (WikiLong)")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--answers-file", type=str, default=None,
                        help="Plot a single answers file instead of all spec files")
    parser.add_argument("--output", type=str, default="wiki_long/tokens_per_question.png")
    args = parser.parse_args()

    qid_to_cat = load_questions(args.question_file)
    categories = [c for c in CATEGORY_ORDER if c in set(qid_to_cat.values())]

    # Decide which files to plot
    if args.answers_file:
        spec_files = [Path(args.answers_file)]
    else:
        data_dir = Path(args.data_dir)
        spec_files = sorted(f for f in data_dir.glob("*.jsonl")
                            if "question" not in f.name and "baseline" not in f.name and "temperature" in f.name)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Accumulate tokens across all draft model files, then average
    combined_tokens = defaultdict(list)
    for f in spec_files:
        cat_tokens = collect_tokens(f, qid_to_cat)
        for c in categories:
            if c in cat_tokens and cat_tokens[c]:
                combined_tokens[c].append(np.mean(cat_tokens[c]))

    means = []
    plot_x = []
    for j, c in enumerate(categories):
        if c in combined_tokens and combined_tokens[c]:
            plot_x.append(j)
            means.append(np.mean(combined_tokens[c]))

    ax.plot(plot_x, means, marker="o", label="draft model average", color=colors[0], linewidth=2)

    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("New Tokens Generated")
    ax.set_title("New Tokens per Question by Category (WikiLong)")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in categories])
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
