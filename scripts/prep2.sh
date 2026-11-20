#!/usr/bin/env bash
# Task 02 corpus pipeline (mixed sources):
#   zhwiki          : HF parquet plain text  -> jsonl (no wiki markup)
#   zhwikinews      : official dump + wikiextractor
#   zhwikivoyage    : official dump + wikiextractor
# then: clean -> dedup -> 180M-char budget -> 98/1/1 split -> 32k BPE -> uint16 bins -> stats
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export TOKENIZERS_PARALLELISM=false
WORK=${WORK:-/root/autodl-tmp/work}
cd "$WORK" || exit 1
mkdir -p data/interim data/processed logs reports

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "FATAL: $*"; exit 1; }

log "=== 1a. zhwiki parquet -> jsonl ==="
if [ ! -s data/interim/extracted/zhwiki/part.jsonl ] && [ -z "$(ls -A data/interim/extracted/zhwiki 2>/dev/null)" ]; then
  python code/llm/parquet_to_jsonl.py --input data/raw/wiki_parquet \
    --out data/interim/extracted/zhwiki || die "parquet convert failed"
else
  log "skip zhwiki convert ($(ls data/interim/extracted/zhwiki | wc -l) files)"
fi

log "=== 1b. wikiextractor for zhwikinews / zhwikivoyage ==="
for p in zhwikinews zhwikivoyage; do
  out="data/interim/extracted/$p"
  if [ -d "$out" ] && [ "$(find "$out" -type f | wc -l)" -gt 0 ]; then
    log "skip extract $p"
  else
    dump="data/raw/${p}-latest-pages-articles.xml.bz2"
    [ -s "$dump" ] || die "missing $dump"
    python -m wikiextractor.WikiExtractor "$dump" -o "$out" --json --no-templates -q \
      --processes 8 2>>logs/extract_$p.err || die "extract $p failed"
  fi
done

log "=== 2. clean ==="
if [ ! -s data/interim/clean.jsonl ]; then
  python code/llm/prepare_corpus.py clean --input data/interim/extracted \
    --output data/interim/clean.jsonl --min-chars 200 --min-zh 0.4 \
    --funnel data/interim/funnel.json || die "clean failed"
else
  log "skip clean"
fi

log "=== 3. dedup ==="
if [ ! -s data/interim/dedup.jsonl ]; then
  python code/llm/prepare_corpus.py dedup --input data/interim/clean.jsonl \
    --output data/interim/dedup.jsonl --threshold 0.8 --num-perm 128 || die "dedup failed"
else
  log "skip dedup"
fi

log "=== 4. 180M-char budget sample ==="
if [ ! -s data/processed/selected.jsonl ]; then
  python - <<'PY' || die "sample failed"
import json, random
docs, total = [], 0
with open("data/interim/dedup.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        n = len(json.loads(line)["text"])
        if n < 700:          # drop stubs so the budget sample lands near 71k docs
            continue
        docs.append((n, line)); total += n
print("dedup docs", len(docs), "chars", total, flush=True)
random.Random(42).shuffle(docs)
target, acc, kept = 180_000_000, 0, 0
with open("data/processed/selected.jsonl", "w", encoding="utf-8") as fo:
    for n, line in docs:
        fo.write(line); acc += n; kept += 1
        if acc >= target:
            break
print("selected", kept, "chars", acc, flush=True)
PY
else
  log "skip sample"
fi

log "=== 5. split 98/1/1 ==="
if [ ! -s data/processed/splits/train.jsonl ]; then
  python code/llm/prepare_corpus.py split --input data/processed/selected.jsonl \
    --out data/processed/splits --seed 42 --val 0.01 --test 0.01 || die "split failed"
else
  log "skip split"
fi

log "=== 6. tokenizer ==="
if [ ! -s data/processed/tokenizer/zh_bpe.model ]; then
  python code/llm/prepare_corpus.py tokenizer --train data/processed/splits/train.txt \
    --out data/processed/tokenizer --vocab 32000 --threads 12 || die "tokenizer failed"
else
  log "skip tokenizer"
fi

log "=== 7. pack ==="
if [ ! -s data/processed/bin/train.bin ]; then
  python code/llm/prepare_corpus.py pack --splits data/processed/splits \
    --tokenizer data/processed/tokenizer/zh_bpe.model --out data/processed/bin || die "pack failed"
else
  log "skip pack"
fi

log "=== 8. stats ==="
python code/llm/prepare_corpus.py stats --splits data/processed/splits \
  --bin data/processed/bin --tokenizer data/processed/tokenizer/zh_bpe.model \
  --out data/processed/stats.json || die "stats failed"

log "=== PREP DONE ==="
cat data/processed/stats.json
du -sh data/interim data/processed
