"""BIG-Bench Hard grader — average exact-match across subtasks (model-card 3-shot em)."""

import argparse
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _grade_utils import default_paths, load_answers, load_gold, normalize_text, print_score

MODEL_CARD_TARGET = 64.2  # base 3-shot CoT average/em


_FINAL_ANSWER_PAT = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:|final\s+answer\s*:)\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)


def extract_bbh_prediction(output: str) -> str:
    """BBH CoT outputs usually end with 'So the answer is X.' — strip to X."""
    if not output:
        return ""
    m = _FINAL_ANSWER_PAT.search(output)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    # Fall back to first non-empty line.
    for line in output.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    args = p.parse_args()

    q_file, ans_file = default_paths(__file__, args.answers)
    gold = load_gold(q_file)
    answers = load_answers(ans_file)

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    n = 0
    for qid, out in answers.items():
        if qid not in gold:
            continue
        g = gold[qid]
        subtask = g["category"]
        pred = extract_bbh_prediction(out)
        ref = g.get("reference") or ""
        total[subtask] += 1
        if normalize_text(pred) == normalize_text(ref):
            correct[subtask] += 1
        n += 1

    per_sub = [correct[s] / total[s] for s in total if total[s]]
    avg = sum(per_sub) / len(per_sub) if per_sub else 0.0
    print_score("BBH", "avg_em", avg, n, target=MODEL_CARD_TARGET)


if __name__ == "__main__":
    main()
