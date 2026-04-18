"""QuAC (1-shot) prep. Fetches the official JSON directly from AllenAI S3,
since the HF dataset is a legacy loader script and `datasets` 4.x dropped
`trust_remote_code`.
"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bench_utils import write_questions


VAL_URL = "https://s3.amazonaws.com/my89public/quac/val_v0.2.json"
TRAIN_URL = "https://s3.amazonaws.com/my89public/quac/train_v0.2.json"


def download_json(url: str, dest: str) -> dict:
    if not os.path.exists(dest):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    with open(dest) as f:
        return json.load(f)


def render_dialog(background, section_title, context, qa_history, current_q, current_a):
    lines = [
        f"Background: {background.strip()}",
        f"Section: {section_title.strip()}",
        f"Context: {context.strip()}",
    ]
    for q, a in qa_history:
        lines.append(f"Q: {q.strip()}")
        lines.append(f"A: {a.strip()}")
    lines.append(f"Q: {current_q.strip()}")
    lines.append("A:" + (f" {current_a.strip()}" if current_a is not None else ""))
    return "\n".join(lines)


def iter_quac_turns(quac_obj, limit_dialogs: int | None = None):
    for d_idx, entry in enumerate(quac_obj["data"]):
        if limit_dialogs is not None and d_idx >= limit_dialogs:
            break
        for para in entry["paragraphs"]:
            context = para["context"]
            background = entry.get("background", "")
            section_title = entry.get("section_title", "")
            pairs: list[tuple[str, str]] = []
            for qa in para["qas"]:
                q = qa["question"]
                answers = qa.get("answers") or []
                a = answers[0]["text"] if answers else ""
                yield {
                    "background": background,
                    "section_title": section_title,
                    "context": context,
                    "history": list(pairs),
                    "question": q,
                    "answer": a,
                }
                pairs.append((q, a))


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    val = download_json(VAL_URL, os.path.join(here, "_val.json"))
    train = download_json(TRAIN_URL, os.path.join(here, "_train.json"))

    train_flat = list(iter_quac_turns(train, limit_dialogs=2))
    shot = train_flat[0] if train_flat else None
    shot_text = ""
    if shot:
        shot_text = render_dialog(
            shot["background"], shot["section_title"], shot["context"],
            shot["history"], shot["question"], shot["answer"],
        )

    records = []
    for qid, ex in enumerate(iter_quac_turns(val)):
        current = render_dialog(
            ex["background"], ex["section_title"], ex["context"],
            ex["history"], ex["question"], None,
        )
        prompt = (shot_text + "\n\n" + current) if shot_text else current
        records.append({
            "question_id": qid,
            "category": "quac",
            "turns": [prompt],
            "reference": ex["answer"],
        })

    n = write_questions(here, records, meta={
        "name": "quac",
        "hf_path": VAL_URL,
        "split": "validation",
        "n_shots": 1,
        "prompting_style": "dialog_qa",
        "has_chat_variant": True,
    })
    print(f"wrote {n} QuAC examples to {here}/question.jsonl")


if __name__ == "__main__":
    main()
