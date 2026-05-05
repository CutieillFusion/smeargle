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
    parser = argparse.ArgumentParser(description="Plot per-category tokens/sec for WikiLong")
    parser.add_argument("--data-dir", type=str, default="wiki_long/")
    parser.add_argument("--question-file", type=str, default="wiki_long/question.jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="../models/llama_3_1_8b_instruct")
    parser.add_argument("--output", type=str, default="wiki_long/category_tokens_per_sec.png")
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

    # Compute speeds for all models (including baseline)
    model_speeds = {}
    model_ooms = {}

    baseline_label = label_from_file(baseline_file)
    model_speeds[baseline_label] = baseline_speeds
    model_ooms[baseline_label] = {}
    for f in spec_files:
        data = load_jsonl(f)
        speeds, oom_counts = compute_speeds_by_category(data, qid_to_cat, tokenizer=None)
        model_speeds[label_from_file(f)] = speeds
        model_ooms[label_from_file(f)] = oom_counts

    # Plot order: baseline first, then spec models
    spec_names = [m for m in model_speeds if m != baseline_label]
    all_names = [baseline_label] + spec_names

    # Plot line chart
    short_labels = [c.replace("long_context_", "") for c in categories]
    x = range(len(categories))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]

    display_names = {"eagle3": "Eagle", "smeargle": "Smeargle", "baseline": "Baseline"}

    # Precompute per-model line data, OOM x positions, and global y-axis
    model_lines = {}
    oom_positions = set()
    global_ymax = 0.0
    for i, model in enumerate(all_names):
        color = colors[i % len(colors)]
        plot_x, plot_y = [], []
        for j, cat in enumerate(categories):
            if cat in model_speeds[model]:
                plot_x.append(j)
                plot_y.append(model_speeds[model][cat])
        linestyle = "--" if model == baseline_label else "-"
        model_lines[model] = (plot_x, plot_y, color, linestyle)
        if plot_y:
            global_ymax = max(global_ymax, max(plot_y))
        if "eagle" in model:
            last_valid_idx = max(plot_x) if plot_x else None
            if last_valid_idx is not None and last_valid_idx < len(categories) - 1:
                oom_positions.add(last_valid_idx)

    # Frame contents per user order: baseline -> +eagle -> +OOM -> +smeargle
    frames = [
        {"models": {baseline_label}, "oom": False},
        {"models": {baseline_label, "eagle3"}, "oom": False},
        {"models": {baseline_label, "eagle3"}, "oom": True},
        {"models": {baseline_label, "eagle3", "smeargle"}, "oom": True},
    ]

    ylim_top = global_ymax * 1.05
    output_path = Path(args.output)

    for step_idx, frame in enumerate(frames, start=1):
        fig, ax = plt.subplots(figsize=(8, 5))

        for model in all_names:
            if model not in model_lines:
                continue
            plot_x, plot_y, color, linestyle = model_lines[model]
            display = display_names.get(model, model)
            if model in frame["models"]:
                ax.plot(plot_x, plot_y, marker="o", linestyle=linestyle,
                        label=display, color=color)
            else:
                # Placeholder so legend stays identical across frames
                ax.plot([], [], marker="o", linestyle=linestyle,
                        label=display, color=color)

        ax.set_xlabel("Prompt Length")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Per-Prompt-Length Tokens/sec (WikiLong)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_labels)
        ax.set_ylim(0, ylim_top)
        ax.grid(True)

        if frame["oom"]:
            for vline_x in oom_positions:
                ax.axvline(x=vline_x, color="red", linestyle=":", linewidth=1.5)
                ax.text(vline_x + 0.05, ylim_top * 0.95, "EAGLE OOM",
                        color="red", fontsize=8, ha="left", va="top", rotation=90)

        ax.legend()
        plt.tight_layout()
        step_output = output_path.with_name(f"{output_path.stem}_step{step_idx}{output_path.suffix}")
        plt.savefig(step_output, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {step_output}")


if __name__ == "__main__":
    main()
