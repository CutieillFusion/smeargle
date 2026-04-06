import json
import matplotlib.pyplot as plt

CONFIG = {
    "eagle_data": "eagle_training_10_epoch.jsonl",
    "smeargle_data": "smeargle_training_10_epoch.jsonl",
    "grid_enabled": True,
    "grid_alpha": 0.3,
    "grid_style": "--",
    "output_file": "figure_loss_curves.png",
    "dpi": 300,
    "format": "png",
}


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def avg_loss_by_epoch_split(records):
    """Average pLoss across all positions for each (epoch, split)."""
    sums = {}
    counts = {}
    for r in records:
        key = (r["epoch"], r["split"])
        sums[key] = sums.get(key, 0.0) + r["pLoss"]
        counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def generate_loss_curves():
    eagle = load_jsonl(CONFIG["eagle_data"])
    smeargle = load_jsonl(CONFIG["smeargle_data"])

    eagle_avg = avg_loss_by_epoch_split(eagle)
    smeargle_avg = avg_loss_by_epoch_split(smeargle)

    epochs = sorted({r["epoch"] for r in eagle})

    EAGLE_TRAIN = "#2B4F8C"
    EAGLE_TEST = "#7BA3D4"
    SMEARGLE_TRAIN = "#C06A2E"
    SMEARGLE_TEST = "#F0B88A"

    plt.figure()
    for label, avg, style, color, marker in [
        ("EAGLE3 train", eagle_avg, "-", EAGLE_TRAIN, "o"),
        ("EAGLE3 test", eagle_avg, "--", EAGLE_TEST, "o"),
        ("SMEARGLE train", smeargle_avg, "-", SMEARGLE_TRAIN, "s"),
        ("SMEARGLE test", smeargle_avg, "--", SMEARGLE_TEST, "s"),
    ]:
        split = "train" if "train" in label else "test"
        vals = [avg[(e, split)] for e in epochs]
        plt.plot(epochs, vals, linestyle=style, color=color, marker=marker, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)

    if CONFIG["grid_enabled"]:
        plt.grid(True, alpha=CONFIG["grid_alpha"], linestyle=CONFIG["grid_style"])

    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG["output_file"], dpi=CONFIG["dpi"], format=CONFIG["format"])
    plt.close()
    print(f"Plot saved to {CONFIG['output_file']}")


if __name__ == "__main__":
    generate_loss_curves()
