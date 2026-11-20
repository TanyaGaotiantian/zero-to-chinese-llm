"""Chinese pretraining corpus pipeline: dump -> clean -> dedup -> split -> tokenize -> bin.

Subcommands (run in order):
  extract   wikiextractor over *.xml.bz2
  clean     wiki markup removal + quality filter (writes funnel stats)
  dedup     exact hash + optional MinHash near-dup
  split     98/1/1 by document, seeded
  tokenizer train sentencepiece BPE
  pack      encode to uint16 flat bins (with EOS between docs)
  stats     dataset report -> stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HTML_RE = re.compile(r"<[^>]{1,200}>")
REF_RE = re.compile(r"\[\d{1,3}\]")
TEMPLATE_RE = re.compile(r"\{\{[^{}]{0,500}\}\}")
TABLE_RE = re.compile(r"^\s*(\{\||\|\}|\||!)")
HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+$")
MULTISPACE_RE = re.compile(r"[ \t\u00a0]+")

BAD_PREFIX = ("Category:", "分类:", "File:", "文件:", "Image:", "图像:",
              "Wikipedia:", "维基百科:", "Template:", "模板:", "Help:", "帮助:")


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_text(text: str) -> str:
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith(BAD_PREFIX) or TABLE_RE.match(s):
            continue
        m = HEADING_RE.match(s)
        if m:
            s = m.group(1)
        s = HTML_RE.sub("", s)
        s = REF_RE.sub("", s)
        s = TEMPLATE_RE.sub("", s)
        s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", s)  # [[a|b]] -> b
        s = re.sub(r"\[https?://[^\]]*\]", "", s)
        s = s.replace("'''", "").replace("''", "")
        s = MULTISPACE_RE.sub(" ", s).strip()
        if s:
            out.append(s)
    return "\n".join(out)


def quality_ok(text: str, min_chars: int, max_chars: int, min_zh: float):
    n = len(text)
    if n < min_chars:
        return False, "too_short"
    if n > max_chars:
        return False, "too_long"
    zh = sum("\u4e00" <= c <= "\u9fff" for c in text)
    if zh / max(1, n) < min_zh:
        return False, "low_zh_ratio"
    if len(set(text)) / max(1, n) < 0.05:
        return False, "low_diversity"
    lines = text.splitlines()
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.7:
        return False, "repeated_lines"
    return True, ""


# ---------------------------------------------------------------- commands
def cmd_extract(args):
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dumps = sorted(raw.glob("*.xml.bz2"))
    if not dumps:
        sys.exit(f"no *.xml.bz2 under {raw}")
    for dump in dumps:
        name = dump.name.split("-latest")[0]
        sub = out / name
        sub.mkdir(exist_ok=True)
        print(f"[extract] {dump.name} -> {sub}")
        subprocess.run(
            [sys.executable, "-m", "wikiextractor.WikiExtractor", str(dump),
             "-o", str(sub), "--json", "--no-templates", "-q"],
            check=True,
        )


def cmd_clean(args):
    src, out = Path(args.input), Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    funnel = Counter()
    kept = 0
    def _shards(root: Path):
        # wikiextractor 3.0.x writes wiki_00.., 3.1.x writes AA..; accept any data file.
        return sorted(p for p in root.rglob("*")
                      if p.is_file() and not p.name.startswith("."))

    with open(out, "w", encoding="utf-8") as fo:
        for path in _shards(src):
            for doc in iter_jsonl(path):
                funnel["raw"] += 1
                text = clean_text(doc.get("text", ""))
                ok, reason = quality_ok(text, args.min_chars, args.max_chars, args.min_zh)
                if not ok:
                    funnel[reason] += 1
                    continue
                fo.write(json.dumps({"id": doc.get("id", ""), "title": doc.get("title", ""),
                                     "url": doc.get("url", ""), "text": text},
                                    ensure_ascii=False) + "\n")
                kept += 1
                funnel["kept"] += 1
    funnel["kept_ratio"] = round(kept / max(1, funnel["raw"]), 4)
    print(json.dumps(funnel, ensure_ascii=False, indent=2))
    Path(args.funnel).write_text(json.dumps(funnel, ensure_ascii=False, indent=2))


def cmd_dedup(args):
    from datasketch import MinHash, MinHashLSH

    seen_exact = set()
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    kept = dropped_exact = dropped_near = 0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def shingles(text: str, k: int = 5, cap: int = 0):
        """Character k-shingles. `cap` keeps a deterministic uniform sample so
        that very long documents do not dominate the MinHash cost."""
        text = re.sub(r"\s+", "", text)
        n = max(1, len(text) - k + 1)
        if cap and n > cap:
            step = n / cap
            return [text[int(i * step):int(i * step) + k] for i in range(cap)]
        return [text[i:i + k] for i in range(n)]

    with open(out, "w", encoding="utf-8") as fo:
        for doc in iter_jsonl(Path(args.input)):
            h = hashlib.sha1(doc["text"].encode()).hexdigest()
            if h in seen_exact:
                dropped_exact += 1
                continue
            m = MinHash(num_perm=args.num_perm)
            sh = [x.encode() for x in shingles(doc["text"], cap=args.max_shingles)]
            if hasattr(m, "update_batch"):
                m.update_batch(sh)
            else:
                for x in sh:
                    m.update(x)
            if lsh.query(m):
                dropped_near += 1
                continue
            seen_exact.add(h)
            lsh.insert(str(kept), m)
            fo.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"kept": kept, "dropped_exact": dropped_exact,
                      "dropped_near": dropped_near}, ensure_ascii=False))


def cmd_split(args):
    docs = list(iter_jsonl(Path(args.input)))
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n = len(docs)
    n_val = max(1, int(n * args.val))
    n_test = max(1, int(n * args.test))
    parts = {
        "train": docs[: n - n_val - n_test],
        "val": docs[n - n_val - n_test: n - n_test],
        "test": docs[n - n_test:],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in parts.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in items:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(out / f"{name}.txt", "w", encoding="utf-8") as f:
            for d in items:
                f.write(d["text"].replace("\n", " ") + "\n")
        print(name, len(items))


def cmd_tokenizer(args):
    import sentencepiece as spm
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=args.train,
        model_prefix=str(out / "zh_bpe"),
        vocab_size=args.vocab,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        num_threads=args.threads,
        train_extremely_large_corpus=True,
        pad_id=0, unk_id=1, bos_id=-1, eos_id=2,
    )
    print("tokenizer ->", out / "zh_bpe.model")


def cmd_pack(args):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eos = sp.eos_id()
    for name in ("train", "val", "test"):
        src = Path(args.splits) / f"{name}.jsonl"
        dst = out / f"{name}.bin"
        n_tokens = 0
        with open(dst, "wb") as f:
            for doc in iter_jsonl(src):
                ids = sp.encode(doc["text"], out_type=int) + [eos]
                np.asarray(ids, dtype=np.uint16).tofile(f)
                n_tokens += len(ids)
        print(f"{name}: {n_tokens:,} tokens -> {dst}")


def cmd_stats(args):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    stats = {}
    for name in ("train", "val", "test"):
        docs = list(iter_jsonl(Path(args.splits) / f"{name}.jsonl"))
        chars = sum(len(d["text"]) for d in docs)
        bin_path = Path(args.bin) / f"{name}.bin"
        tokens = bin_path.stat().st_size // 2
        stats[name] = {
            "docs": len(docs),
            "chars": chars,
            "tokens": tokens,
            "chars_per_token": round(chars / max(1, tokens), 3),
        }
    stats["vocab_size"] = sp.get_piece_size()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    Path(args.out).write_text(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("clean")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funnel", default="data/interim/funnel.json")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--min-zh", type=float, default=0.4)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("dedup")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--num-perm", type=int, default=128)
    p.add_argument("--max-shingles", type=int, default=512,
                   help="cap shingles per doc (0 = no cap); keeps MinHash cost linear")
    p.set_defaults(func=cmd_dedup)

    p = sub.add_parser("split")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.01)
    p.add_argument("--test", type=float, default=0.01)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("tokenizer")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--threads", type=int, default=16)
    p.set_defaults(func=cmd_tokenizer)

    p = sub.add_parser("pack")
    p.add_argument("--splits", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("stats")
    p.add_argument("--splits", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
