import json
from pathlib import Path
from collections import Counter


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


def compute_stats(records):
    """Compute inference statistics from debug records."""
    total_steps = len(records)
    if total_steps == 0:
        return {}
    
    accept_lengths = []
    accept_length_dist = Counter()
    steps_with_acceptance = 0  # Steps where accept_length > 0 (draft tokens accepted)
    all_accepted_ids = []  # Collect all accepted token IDs
    
    for rec in records:
        accept_len = rec.get("accept_length", 0)
        accept_lengths.append(accept_len)
        accept_length_dist[accept_len] += 1
        if accept_len > 0:
            steps_with_acceptance += 1
        
        # Collect accepted IDs
        accepted_ids = rec.get("accepted_ids", [])
        if accepted_ids:
            all_accepted_ids.extend(accepted_ids)
    
    avg_accept_length = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
    acceptance_rate = steps_with_acceptance / total_steps if total_steps > 0 else 0.0
    
    min_id = min(all_accepted_ids) if all_accepted_ids else None
    max_id = max(all_accepted_ids) if all_accepted_ids else None
    
    return {
        "total_steps": total_steps,
        "avg_accept_length": avg_accept_length,
        "acceptance_rate": acceptance_rate,  # Fraction of steps where draft tokens were accepted
        "accept_length_distribution": dict(sorted(accept_length_dist.items())),
        "min_accepted_id": min_id,
        "max_accepted_id": max_id,
    }


if __name__ == "__main__":
    import sys
    
    log_path = Path("inference_debug.jsonl")
    last_n = 0
    
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        last_n = int(sys.argv[2])
    
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")
    
    recs = load_last_records(str(log_path), last_n)
    stats = compute_stats(recs)
    
    print(f"Statistics over last {last_n if last_n > 0 else 'all'} steps:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Average accept length: {stats['avg_accept_length']:.4f}")
    print(f"  Acceptance rate (steps with accept_length > 0): {stats['acceptance_rate']:.4f}")
    print(f"  Accept length distribution: {stats['accept_length_distribution']}")
    if stats['min_accepted_id'] is not None:
        print(f"  Min accepted ID: {stats['min_accepted_id']}")
        print(f"  Max accepted ID: {stats['max_accepted_id']}")

