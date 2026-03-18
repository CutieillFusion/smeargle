import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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
    return name.replace("llama_3_1_8b_instruct_", "").split("_temperature")[0]


def main():
    parser = argparse.ArgumentParser(description="Plot cumulative acceptance rate for spec_bench")
    parser.add_argument("--data-dir", type=str, default="spec_bench/")
    parser.add_argument("--output", type=str, default="spec_bench/acceptance_rate_per_position.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    spec_files = sorted(f for f in data_dir.glob("*.jsonl") if "baseline" not in f.name and "temperature" in f.name)

    plt.figure(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, f in enumerate(spec_files):
        data = load_jsonl(f)
        counts = aggregate_acceptance_lengths(data)
        # Trim trailing zeros
        last_nonzero = np.max(np.nonzero(counts)) + 1 if np.any(counts) else 1
        counts = counts[:last_nonzero]
        rate = cumulative_acceptance_rate(counts)
        label = label_from_file(f)
        plt.plot(range(len(rate)), rate, marker="o", label=label, color=colors[i % len(colors)])

    plt.xlabel("Position")
    plt.ylabel("Acceptance Rate")
    plt.title("Cumulative Acceptance Rate Per Position (SpecBench)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
