"""Shared helpers for grading Llama-3.1-8B benchmark outputs."""

import json
import os
import re
import string
import unicodedata
from collections import Counter
from typing import Iterable


LETTER_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_answers(path: str) -> dict[int, str]:
    """Read an answer JSONL produced by gen_answer_llama_3_1_8b_bench.py.

    Returns {question_id: first_turn_output} and skips rows with no choices
    (OOM / too-long skips).
    """
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("choices"):
                continue
            turns = d["choices"][0].get("turns") or []
            if turns:
                out[d["question_id"]] = turns[0]
    return out


def load_gold(path: str) -> dict[int, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["question_id"]] = d
    return out


def default_paths(grade_script: str, answers: str | None) -> tuple[str, str]:
    """Resolve (question_file, answer_file) from the grade.py location."""
    here = os.path.dirname(os.path.abspath(grade_script))
    q_file = os.path.join(here, "question.jsonl")
    if answers is None:
        raise SystemExit(
            "Pass --answers <path_to_answer_jsonl> (produced by "
            "gen_answer_llama_3_1_8b_bench.py)."
        )
    return q_file, answers


# ─── text normalisation ─────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(r"[%s]" % re.escape(string.punctuation))
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles/punct/extra ws."""
    s = s or ""
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def exact_match(pred: str, refs: Iterable[str]) -> int:
    p = normalize_text(pred)
    return int(any(p == normalize_text(r) for r in refs))


def token_f1(pred: str, ref: str) -> float:
    p_toks = normalize_text(pred).split()
    r_toks = normalize_text(ref).split()
    if not p_toks or not r_toks:
        return float(p_toks == r_toks)
    common = Counter(p_toks) & Counter(r_toks)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec = ns / len(p_toks)
    rec = ns / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def best_f1(pred: str, refs: Iterable[str]) -> float:
    refs = list(refs) or [""]
    return max(token_f1(pred, r) for r in refs)


# ─── answer extraction ─────────────────────────────────────────────────

_THE_ANSWER_IS = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:|final\s+answer\s*:)[\s\*]*([A-Za-z])",
    re.IGNORECASE,
)
_FIRST_LETTER = re.compile(r"\b([A-Z])\b")


def extract_letter(output: str, valid: str = "ABCD") -> str | None:
    """Extract an MC answer letter from model output (supports CoT + direct)."""
    if not output:
        return None
    valid = valid.upper()

    # 1) Explicit "The answer is X" / "Answer: X" pattern.
    for m in _THE_ANSWER_IS.finditer(output):
        cand = m.group(1).upper()
        if cand in valid:
            return cand

    # 2) First standalone capital letter in the first non-empty line.
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _FIRST_LETTER.search(line)
        if m and m.group(1) in valid:
            return m.group(1)
        # 3) Fall back: first character if it's a valid letter (e.g. output just "A").
        if line[0].upper() in valid:
            return line[0].upper()
        break

    return None


_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(output: str) -> str | None:
    """Extract the contents of the LAST \\boxed{...} in output."""
    if not output:
        return None
    matches = _BOXED.findall(output)
    return matches[-1] if matches else None


_LAST_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_last_number(output: str) -> str | None:
    """Return the last number-looking substring in output (normalized, no commas)."""
    if not output:
        return None
    nums = _LAST_NUMBER.findall(output)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def extract_gsm8k_gold(reference: str) -> str | None:
    """GSM8K gold answers are '...explanation...\\n#### 42'. Extract the number."""
    if not reference:
        return None
    m = re.search(r"####\s*(-?[\d,\.]+)", reference)
    if m:
        return m.group(1).replace(",", "").strip()
    return extract_last_number(reference)


# ─── reporting ─────────────────────────────────────────────────────────

def print_score(name: str, metric: str, score: float, n: int, target: float | None = None,
                extra: dict | None = None) -> None:
    pct = score * 100.0
    line = f"{name:<18} {metric:<10} {pct:>6.2f}%  n={n}"
    if target is not None:
        line += f"   (model-card target: {target:.1f}%)"
    print(line)
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
