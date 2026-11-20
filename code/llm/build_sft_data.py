"""Build a 24k ChatML SFT set from a public Chinese instruction dataset.

  python build_sft_data.py --source firefly --n 24000 --out data/sft

Sources (streaming, no full download):
  firefly    YeungNLP/firefly-train-1.1M      fields: input, target
  alpaca_zh  shibing624/alpaca-zh             fields: instruction, input, output
  coig       m-a-p/COIG-CQIA                  fields vary by subset

Output:
  data/sft/train.jsonl (n - 500)
  data/sft/val.jsonl   (500)
  data/sft/card.json   (source, license note, counts, filters)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

SOURCES = {
    "firefly": "YeungNLP/firefly-train-1.1M",
    "alpaca_zh": "shibing624/alpaca-zh",
    "coig": "m-a-p/COIG-CQIA",
}

SYSTEM = "你是一个简洁、诚实的中文助手。不确定时直接说不确定，不要编造。"


def pick(ex: dict, keys):
    for k in keys:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def to_pair(ex: dict, source: str):
    if source == "firefly":
        user, assistant = pick(ex, ["input"]), pick(ex, ["target"])
    elif source == "alpaca_zh":
        inst, inp, out = pick(ex, ["instruction"]), pick(ex, ["input"]), pick(ex, ["output"])
        user = inst + ("\n" + inp if inp else "")
        assistant = out
    else:
        user = pick(ex, ["instruction", "question", "prompt"])
        assistant = pick(ex, ["output", "answer", "response"])
    return user.strip(), assistant.strip()


def clean(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ok(user: str, assistant: str) -> bool:
    if not (10 <= len(user) <= 2000 and 10 <= len(assistant) <= 4000):
        return False
    if len(assistant) < 0.2 * len(user):
        return False
    if "作为一个AI" in assistant or "作为 AI 语言模型" in assistant:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="firefly")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(SOURCES[args.source], split="train", streaming=True)
    seen = set()
    rows = []
    scanned = 0
    for ex in ds:
        scanned += 1
        user, assistant = to_pair(ex, args.source)
        user, assistant = clean(user), clean(assistant)
        if not ok(user, assistant):
            continue
        h = hashlib.sha1((user + "\x00" + assistant).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})
        if len(rows) >= args.n:
            break
        if scanned % 5000 == 0:
            print(f"scanned={scanned} kept={len(rows)}", flush=True)

    if len(rows) < args.n * 0.8:
        raise SystemExit(f"only {len(rows)} usable rows after scanning {scanned}; "
                         f"try --source alpaca_zh or raise --n")

    random.Random(args.seed).shuffle(rows)
    n_val = min(500, max(1, len(rows) // 20))
    val, train = rows[:n_val], rows[n_val:]

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out / "train.jsonl", train)
    dump(out / "val.jsonl", val)
    card = {
        "source": SOURCES[args.source],
        "source_key": args.source,
        "requested": args.n,
        "scanned": scanned,
        "kept": len(rows),
        "train": len(train),
        "val": len(val),
        "filters": "10<=user<=2000, 10<=assistant<=4000, assistant>=0.2*user, dedup sha1, drop AI-template lines",
        "note": "Check the upstream dataset license before redistribution; this script only samples for local SFT.",
    }
    (out / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
