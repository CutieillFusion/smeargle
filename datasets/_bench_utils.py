"""Shared helpers for rendering Llama-3.1-8B benchmark prompts into MT-Bench JSONL."""

import json
import os
import random
import string
from typing import Iterable, Sequence


LETTERS = list(string.ascii_uppercase)


def sample_shots(shot_pool: Sequence[dict], k: int, seed: int = 0) -> list[dict]:
    """Deterministically pick k few-shot examples from shot_pool."""
    if k <= 0 or not shot_pool:
        return []
    rng = random.Random(seed)
    pool = list(shot_pool)
    rng.shuffle(pool)
    return pool[:k]


def format_mc_prompt(
    question: str,
    choices: Sequence[str],
    shots: Sequence[dict] = (),
    instruction: str | None = None,
    answer_letter_key: str = "answer_letter",
    question_key: str = "question",
    choices_key: str = "choices",
) -> str:
    """Render a multiple-choice prompt with optional few-shot prefix.

    Each shot dict must have question_key, choices_key, answer_letter_key.
    """
    parts: list[str] = []
    if instruction:
        parts.append(instruction.strip())

    def render_block(q: str, chs: Sequence[str], ans: str | None) -> str:
        lines = [f"Question: {q.strip()}"]
        for i, ch in enumerate(chs):
            lines.append(f"{LETTERS[i]}. {ch}")
        lines.append("Answer:" + (f" {ans}" if ans else ""))
        return "\n".join(lines)

    for shot in shots:
        parts.append(render_block(
            shot[question_key],
            shot[choices_key],
            shot[answer_letter_key],
        ))
    parts.append(render_block(question, choices, None))
    return "\n\n".join(parts)


def format_open_prompt(
    question: str,
    shots: Sequence[dict] = (),
    instruction: str | None = None,
    question_key: str = "question",
    answer_key: str = "answer",
    q_prefix: str = "Question:",
    a_prefix: str = "Answer:",
) -> str:
    """Render an open-ended QA prompt with optional few-shot prefix."""
    parts: list[str] = []
    if instruction:
        parts.append(instruction.strip())
    for shot in shots:
        parts.append(
            f"{q_prefix} {shot[question_key].strip()}\n"
            f"{a_prefix} {shot[answer_key].strip()}"
        )
    parts.append(f"{q_prefix} {question.strip()}\n{a_prefix}")
    return "\n\n".join(parts)


def write_questions(
    out_dir: str,
    records: Iterable[dict],
    meta: dict,
) -> int:
    """Write question.jsonl + meta.json into out_dir. Returns line count."""
    os.makedirs(out_dir, exist_ok=True)
    q_path = os.path.join(out_dir, "question.jsonl")
    n = 0
    with open(q_path, "w") as fout:
        for rec in records:
            assert "question_id" in rec and "turns" in rec and "category" in rec, rec
            fout.write(json.dumps(rec) + "\n")
            n += 1

    meta = {**meta, "n_examples": n}
    with open(os.path.join(out_dir, "meta.json"), "w") as fout:
        json.dump(meta, fout, indent=2)
    return n


def safe_load_dataset(*args, **kwargs):
    """Wrap datasets.load_dataset with a helpful message on network/auth errors."""
    from datasets import load_dataset
    try:
        return load_dataset(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"load_dataset({args}, {kwargs}) failed: {type(e).__name__}: {e}. "
            "Check HF auth (`huggingface-cli login`) and network access."
        ) from e
