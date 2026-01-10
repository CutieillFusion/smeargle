import json
from pathlib import Path


def load_last_records(path: str, n: int):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records[-n:] if n > 0 else records


def accuracy(records):
    max_t, max_p = 0, 0
    matches = 0
    total = 0
    for rec in records:
        tgt = rec.get("target_ids", [])
        pred = rec.get("pred_ids", [])
        for t, p in zip(tgt, pred):
            matches += int(t == p)
            if t > max_t:
                max_t = t
            if p > max_p:
                max_p = p
        total += min(len(tgt), len(pred))
    print(f"max_t: {max_t}, max_p: {max_p}")
    return matches / total if total else 0.0


if __name__ == "__main__":
    log_path = Path("train_debug.jsonl")
    last_n = 100
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")
    recs = load_last_records(str(log_path), last_n)
    acc = accuracy(recs)
    print(f"Accuracy over last {last_n} lines: {acc:.4f}")
 
