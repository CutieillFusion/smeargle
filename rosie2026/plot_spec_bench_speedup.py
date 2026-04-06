import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


# The 8 MT-bench categories are all part of the "multi-turn conversation" subtask
SUBTASK_MAP = {
    "writing": "multi_turn_conv",
    "roleplay": "multi_turn_conv",
    "reasoning": "multi_turn_conv",
    "math": "multi_turn_conv",
    "coding": "multi_turn_conv",
    "extraction": "multi_turn_conv",
    "stem": "multi_turn_conv",
    "humanities": "multi_turn_conv",
    "summarization": "summarization",
    "rag": "rag",
    "translation": "translation",
    "qa": "qa",
    "math_reasoning": "math_reasoning",
}


def load_questions(question_file):
    """Build question_id -> subtask mapping."""
    mapping = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cat = obj["category"]
            mapping[obj["question_id"]] = SUBTASK_MAP.get(cat, cat)
    return mapping


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_speeds_by_category(data, qid_to_cat, tokenizer=None):
    """Compute aggregate tokens/sec per category.

    If tokenizer is provided (baseline), count tokens from answer text.
    Otherwise (spec models), use new_tokens field.
    """
    cat_tokens = defaultdict(float)
    cat_time = defaultdict(float)

    for dp in data:
        qid = dp["question_id"]
        cat = qid_to_cat.get(qid)
        if cat is None:
            continue
        wall_time = sum(dp["choices"][0]["wall_time"])
        if tokenizer is not None:
            tokens = 0
            for turn in dp["choices"][0]["turns"]:
                tokens += len(tokenizer(turn).input_ids) - 1
        else:
            tokens = sum(dp["choices"][0]["new_tokens"])
        cat_tokens[cat] += tokens
        cat_time[cat] += wall_time

    speeds = {}
    for cat in cat_tokens:
        speeds[cat] = cat_tokens[cat] / cat_time[cat]
    return speeds


def main():
    parser = argparse.ArgumentParser(description="Plot per-category speedup for spec_bench benchmark")
    parser.add_argument("--data-dir", type=str, default="spec_bench/")
    parser.add_argument("--question-file", type=str, default="spec_bench/question.jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="../models/llama_3_1_8b_instruct")
    parser.add_argument("--output", type=str, default="spec_bench/category_speedup.png")
    args = parser.parse_args()

    qid_to_cat = load_questions(args.question_file)
    categories = sorted(set(qid_to_cat.values()))

    data_dir = Path(args.data_dir)
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    # Identify baseline and spec model files
    baseline_file = None
    spec_files = []
    for f in jsonl_files:
        if "baseline" in f.name:
            baseline_file = f
        elif "temperature" in f.name:
            spec_files.append(f)

    if baseline_file is None:
        raise FileNotFoundError("No baseline JSONL file found in " + str(data_dir))

    # Load tokenizer for baseline token counting
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Compute baseline speeds per category
    baseline_data = load_jsonl(baseline_file)
    baseline_speeds = compute_speeds_by_category(baseline_data, qid_to_cat, tokenizer=tokenizer)

    # Compute speeds for all models (including baseline)
    model_speeds = {}

    def label_from_file(f):
        name = f.stem
        parts = name.replace("llama_3_1_8b_instruct_", "").split("_temperature")[0]
        return parts

    model_speeds[label_from_file(baseline_file)] = baseline_speeds
    for f in spec_files:
        data = load_jsonl(f)
        speeds = compute_speeds_by_category(data, qid_to_cat, tokenizer=None)
        model_speeds[label_from_file(f)] = speeds

    # Compute speedup ratios relative to baseline
    model_names = list(model_speeds.keys())
    speedups = {}
    for model in model_names:
        speedups[model] = {}
        for cat in categories:
            base = baseline_speeds.get(cat, 1.0)
            speedups[model][cat] = model_speeds[model].get(cat, 0.0) / base

    # Plot grouped bar chart
    x = np.arange(len(categories))
    n_models = len(model_names)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, model in enumerate(model_names):
        vals = [speedups[model].get(cat, 0.0) for cat in categories]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}x", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="baseline (1x)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Speedup Factor")
    ax.set_title("Per-Category Speedup (SpecBench)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
