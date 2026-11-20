"""Convert wikimedia/wikipedia parquet shards to wikiextractor-style JSONL.

Output lines: {"id","revid","url","title","text"} so the downstream
clean/dedup/split pipeline is identical for parquet and dump sources.

  python parquet_to_jsonl.py --input data/raw/wiki_parquet --out data/interim/extracted/zhwiki
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--limit-docs", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    src = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {src}")

    total = 0
    for f in files:
        pf = pq.ParquetFile(f)
        target = out / (f.stem + ".jsonl")
        with open(target, "w", encoding="utf-8") as fo:
            for batch in pf.iter_batches(batch_size=args.batch,
                                         columns=["id", "url", "title", "text"]):
                cols = batch.to_pydict()
                for i in range(len(cols["id"])):
                    text = cols["text"][i] or ""
                    fo.write(json.dumps({
                        "id": str(cols["id"][i]),
                        "revid": "",
                        "url": cols["url"][i] or "",
                        "title": cols["title"][i] or "",
                        "text": text,
                    }, ensure_ascii=False) + "\n")
                    total += 1
                    if args.limit_docs and total >= args.limit_docs:
                        break
                if args.limit_docs and total >= args.limit_docs:
                    break
        print(f"[parquet] {f.name} -> {target} (total {total})", flush=True)
        if args.limit_docs and total >= args.limit_docs:
            break
    print(f"[parquet] done docs={total}")


if __name__ == "__main__":
    main()
