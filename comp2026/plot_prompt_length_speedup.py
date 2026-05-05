import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


CATEGORY_ORDER = ["long_context_1k", "long_context_2k", "long_context_4k", "long_context_8k", "long_context_16k", "long_context_32k", "long_context_64k"]


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


def compute_avg_acceptance_length(data, qid_to_cat):
    """Compute average acceptance length per category."""
    cat_total = defaultdict(float)
    cat_count = defaultdict(float)
    for dp in data:
        qid = dp["question_id"]
        cat = qid_to_cat.get(qid)
        if cat is None or dp.get("skipped") == "OOM":
            continue
        al = dp["choices"][0].get("acceptance_lengths", {})
        for pos_str, count in al.items():
            cat_total[cat] += int(pos_str) * count
            cat_count[cat] += count
    return {cat: cat_total[cat] / cat_count[cat] for cat in cat_total if cat_count[cat] > 0}


def compute_speeds_by_category(data, qid_to_cat, tokenizer=None):
    """Compute aggregate tokens/sec per category.

    If tokenizer is provided (baseline), count tokens from answer text.
    Otherwise (spec models), use new_tokens field.
    Returns (speeds_dict, oom_counts_dict).
    """
    cat_tokens = defaultdict(float)
    cat_time = defaultdict(float)
    cat_oom = defaultdict(int)

    for dp in data:
        qid = dp["question_id"]
        cat = qid_to_cat.get(qid)
        if cat is None:
            continue
        if dp.get("skipped") == "OOM":
            cat_oom[cat] += 1
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
    return speeds, dict(cat_oom)


def main():
    parser = argparse.ArgumentParser(description="Plot per-category speedup for WikiLong")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="../models/llama_3_1_8b_instruct")
    parser.add_argument("--output", type=str, default="wiki_long/category_speedup.png")
    parser.add_argument("--file-filter", type=str, default=None,
                        help="Only include spec files whose filename contains this substring")
    args = parser.parse_args()

    qid_to_cat = load_questions(args.question_file)
    categories = [c for c in CATEGORY_ORDER if c in set(qid_to_cat.values())]

    data_dir = Path(args.data_dir)
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    # Identify baseline and spec model files
    baseline_file = None
    # Extract a nice label from filename
    def label_from_file(f):
        name = f.stem
        if "_window_" in name:
            return name.split("_window_")[-1]
        # e.g. llama_3_1_8b_instruct_eagle3_temperature_0.0 -> eagle3
        parts = name.replace("llama_3_1_8b_instruct_", "").split("_temperature")[0]
        return parts

    def _window_sort_key(f):
        if "_window_" in f.name:
            return int(f.name.split("_window_")[-1].split(".")[0])
        return 0

    spec_files = []
    for f in jsonl_files:
        if "baseline" in f.name:
            baseline_file = f
        elif "temperature" in f.name:
            spec_files.append(f)
    spec_files.sort(key=_window_sort_key)
    if args.file_filter:
        spec_files = [f for f in spec_files if args.file_filter in f.name]

    if baseline_file is None:
        raise FileNotFoundError("No baseline JSONL file found in " + str(data_dir))

    # Load tokenizer for baseline token counting
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Compute baseline speeds per category
    baseline_data = load_jsonl(baseline_file)
    baseline_speeds, _ = compute_speeds_by_category(baseline_data, qid_to_cat, tokenizer=tokenizer)

    # Compute speeds for all models (including baseline for verification)
    model_speeds = {}
    model_ooms = {}

    model_speeds[label_from_file(baseline_file)] = baseline_speeds
    model_ooms[label_from_file(baseline_file)] = {}
    model_acc_len = {}
    for f in spec_files:
        data = load_jsonl(f)
        speeds, oom_counts = compute_speeds_by_category(data, qid_to_cat, tokenizer=None)
        model_speeds[label_from_file(f)] = speeds
        model_ooms[label_from_file(f)] = oom_counts
        model_acc_len[label_from_file(f)] = compute_avg_acceptance_length(data, qid_to_cat)

    # Compute speedup ratios for spec models only (exclude baseline)
    spec_names = [m for m in model_speeds if m != label_from_file(baseline_file)]
    speedups = {}
    for model in spec_names:
        speedups[model] = {}
        for cat in categories:
            if cat not in baseline_speeds or cat not in model_speeds[model]:
                continue
            speedups[model][cat] = model_speeds[model][cat] / baseline_speeds[cat]

    # Plot line chart (similar to acceptance rate plots)
    short_labels = [c.replace("long_context_", "") for c in categories]
    x = range(len(categories))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]

    # Precompute per-model line data and OOM x positions
    model_lines = {}
    oom_positions = set()
    global_ymax = 1.0
    for i, model in enumerate(spec_names):
        color = colors[i % len(colors)]
        plot_x, plot_y = [], []
        for j, cat in enumerate(categories):
            if cat in speedups[model]:
                plot_x.append(j)
                plot_y.append(speedups[model][cat])
        model_lines[model] = (plot_x, plot_y, color)
        if plot_y:
            global_ymax = max(global_ymax, max(plot_y))
        full_oom_cats = [cat for cat in categories if model_ooms[model].get(cat, 0) > 0 and cat not in speedups[model]]
        if full_oom_cats:
            oom_positions.add(5)

    ordered_models = ["eagle3", "smeargle"]
    # Frame contents: which models have their real line drawn, and whether OOM marker is shown.
    # Order per user: baseline -> +eagle -> +OOM -> +smeargle
    frames = [
        {"models": set(), "oom": False},
        {"models": {"eagle3"}, "oom": False},
        {"models": {"eagle3"}, "oom": True},
        {"models": {"eagle3", "smeargle"}, "oom": True},
    ]

    ylim_top = global_ymax * 1.05
    output_path = Path(args.output)

    for step_idx, frame in enumerate(frames, start=1):
        fig, ax = plt.subplots(figsize=(8, 5))

        for model in ordered_models:
            if model not in model_lines:
                continue
            plot_x, plot_y, color = model_lines[model]
            if model in frame["models"]:
                ax.plot(plot_x, plot_y, marker="o", label=model, color=color)
            else:
                # Placeholder so legend stays identical across frames
                ax.plot([], [], marker="o", color=color, label=model)

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="baseline (1x)")
        ax.set_xlabel("Prompt Length")
        ax.set_ylabel("Speedup Factor")
        ax.set_title("Per-Prompt-Length Speedup (WikiLong)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_labels)
        ax.set_ylim(0, ylim_top)
        ax.grid(True)

        if frame["oom"]:
            for vline_x in oom_positions:
                ax.axvline(x=vline_x, color="red", linestyle=":", linewidth=1.5)
                ax.text(vline_x + 0.05, ylim_top * 0.95, "OOM",
                        color="red", fontsize=8, ha="left", va="top", rotation=90)

        ax.legend()
        plt.tight_layout()
        step_output = output_path.with_name(f"{output_path.stem}_step{step_idx}{output_path.suffix}")
        plt.savefig(step_output, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {step_output}")

    # Write speedup and tokens/s to a JSON file
    json_output = Path(args.output).with_suffix(".json")
    results = {}
    results["baseline"] = {cat: baseline_speeds.get(cat) for cat in categories if cat in baseline_speeds}
    for model in spec_names:
        results[model] = {}
        for cat in categories:
            results[model][cat] = {
                "tokens_per_sec": model_speeds[model].get(cat),
                "speedup": speedups[model].get(cat),
                "avg_acceptance_length": model_acc_len.get(model, {}).get(cat),
            }
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved data to {json_output}")


if __name__ == "__main__":
    main()
