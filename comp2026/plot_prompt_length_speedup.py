import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def load_questions(question_file):
    """Build question_id -> category mapping."""
    mapping = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mapping[obj["question_id"]] = obj["category"]
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
    parser = argparse.ArgumentParser(description="Plot per-category speedup for wiki_long benchmark")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="../models/llama_3_1_8b_instruct")
    parser.add_argument("--output", type=str, default="wiki_long/category_speedup.png")
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

    # Compute speeds for all models (including baseline for verification)
    model_speeds = {}
    # Extract a nice label from filename
    def label_from_file(f):
        name = f.stem
        # e.g. llama_3_1_8b_instruct_eagle3_temperature_0.0 -> eagle3
        parts = name.replace("llama_3_1_8b_instruct_", "").split("_temperature")[0]
        return parts

    model_speeds[label_from_file(baseline_file)] = baseline_speeds
    for f in spec_files:
        data = load_jsonl(f)
        speeds = compute_speeds_by_category(data, qid_to_cat, tokenizer=None)
        model_speeds[label_from_file(f)] = speeds

    # Compute speedup ratios for spec models only (exclude baseline)
    spec_names = [m for m in model_speeds if m != label_from_file(baseline_file)]
    speedups = {}
    for model in spec_names:
        speedups[model] = {}
        for cat in categories:
            base = baseline_speeds.get(cat, 1.0)
            speedups[model][cat] = model_speeds[model].get(cat, 0.0) / base

    # Plot line chart (similar to acceptance rate plots)
    short_labels = [c.replace("long_context_", "") for c in categories]
    x = range(len(categories))
    colors = ["#DD8452", "#55A868"]

    plt.figure(figsize=(8, 5))
    for i, model in enumerate(spec_names):
        vals = [speedups[model].get(cat, 0.0) for cat in categories]
        plt.plot(x, vals, marker="o", label=model, color=colors[i % len(colors)])

    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="baseline (1x)")
    plt.xlabel("Prompt Length")
    plt.ylabel("Speedup Factor")
    plt.title("Per-Prompt-Length Speedup (wiki_long benchmark)")
    plt.xticks(list(x), short_labels)
    plt.legend()
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
