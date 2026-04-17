import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    parser = argparse.ArgumentParser(description="Plot speedup vs sliding window size per category")
    parser.add_argument("--data-dir", type=str, default="sliding_window/")
    parser.add_argument("--question-file", type=str, default="sliding_window/question.jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="../models/llama_3_1_8b_instruct")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output PNGs (defaults to --data-dir)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir)
    data_dir = Path(args.data_dir)

    qid_to_cat = load_questions(args.question_file)
    categories = [c for c in CATEGORY_ORDER if c in set(qid_to_cat.values())]

    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    # Identify baseline and spec model files
    baseline_file = None
    spec_files = []
    for f in jsonl_files:
        if "baseline" in f.name:
            baseline_file = f
        elif "temperature" in f.name and "_window_" in f.name:
            spec_files.append(f)

    if baseline_file is None:
        raise FileNotFoundError("No baseline JSONL file found in " + str(data_dir))

    # Extract window size from filename and sort
    def window_size_from_file(f):
        return int(f.name.split("_window_")[-1].split(".")[0])

    spec_files.sort(key=window_size_from_file)

    # Load tokenizer for baseline token counting
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Compute baseline speeds per category
    baseline_data = load_jsonl(baseline_file)
    baseline_speeds, _ = compute_speeds_by_category(baseline_data, qid_to_cat, tokenizer=tokenizer)

    # Compute speeds and speedups for each window size
    # window_speedups[window_size][category] = speedup
    window_sizes = []
    window_speedups = {}
    window_ooms = {}
    for f in spec_files:
        ws = window_size_from_file(f)
        window_sizes.append(ws)
        data = load_jsonl(f)
        speeds, oom_counts = compute_speeds_by_category(data, qid_to_cat, tokenizer=None)
        speedups = {}
        for cat in categories:
            if cat in baseline_speeds and cat in speeds:
                speedups[cat] = speeds[cat] / baseline_speeds[cat]
        window_speedups[ws] = speedups
        window_ooms[ws] = oom_counts

    # Generate one plot per category
    color = "#4C72B0"

    for cat in categories:
        short_label = cat.replace("long_context_", "")

        plot_ws = []
        plot_speedup = []
        oom_ws = []
        for ws in window_sizes:
            if cat in window_speedups[ws]:
                plot_ws.append(ws)
                plot_speedup.append(window_speedups[ws][cat])
            elif window_ooms[ws].get(cat, 0) > 0:
                oom_ws.append(ws)

        fig, ax = plt.subplots(figsize=(8, 5))

        if plot_ws:
            ax.plot(plot_ws, plot_speedup, marker="o", color=color, label="Eagle3")

        # Mark OOM window sizes
        if oom_ws:
            for oom_x in oom_ws:
                ax.axvline(x=oom_x, color="red", linestyle=":", linewidth=1.5)
            # Single text label for OOM
            ax.text(oom_ws[0], ax.get_ylim()[1] * 0.95, "OOM",
                    color="red", fontsize=8, ha="center", va="top")

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="baseline (1x)")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Sliding Window Size")
        ax.set_ylabel("Speedup Factor")
        ax.set_title(f"Speedup vs Window Size ({short_label} prompt)")
        ax.set_xticks(window_sizes)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.grid(True)

        plt.tight_layout()
        out_path = output_dir / f"window_speedup_{cat}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
