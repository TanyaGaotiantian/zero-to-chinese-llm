# 01 · Corpus engineering: 3 dumps → 112.5M tokens

> The dirtiest part of the project — and the part that sets the ceiling for everything downstream.

## 1. Target

| Metric | Target | Measured |
|---|---|---|
| Docs | ~70k | 71,665 |
| Train tokens | ~112M | **112,503,579** |
| chars/token | 1.5–1.7 | **1.568** |
| Vocab | 32k BPE | 32,000 (byte_fallback) |
| Artifacts | `train/val/test.bin` | uint16, 225 MB / 2.2 MB / 2.4 MB |

## 2. Data sources: first solve "can I even download it"

| Channel | Measured throughput | Time for 3.45 GB |
|---|---:|---|
| `dumps.wikimedia.org` (official) | **9–80 KB/s** per connection | 12 h+ |
| official + 24 parallel range requests | ~124 KB/s | ~8 h |
| `dumps.wikimedia.your.org` mirror | 3 MB/s burst, then 124 KB/s | unreliable |
| **HF mirror `wikimedia/wikipedia` parquet** | **5.7 MB/s** | **~2 min per shard** |

Decision: **Wikipedia body from parquet, Wikinews/Wikivoyage from official dumps.**

```bash
bash scripts/fetch_sources.sh    # 2 parquet shards (850 MB) + 34 MB + 14 MB
```

`parquet_to_jsonl.py` converts parquet into exactly the `wikiextractor` schema
(`id/url/title/text`), so clean/dedup/split/tokenize/pack need **zero changes**.

## 3. The funnel: 485,894 → 71,665

| Stage | Docs | Kept |
|---|---:|---:|
| raw (parquet 461,584 + news/voyage) | 485,894 | — |
| clean + quality gates | 283,923 | 58.4% |
| exact + MinHash dedup | 283,492 | 99.85% |
| 180M-char budget sample (≥700 chars) | **71,665** | 25.3% |

| Rejection | Count | Share | Typical |
|---|---:|---:|---|
| `too_short` (<200 chars) | 180,799 | 37.2% | disambiguation pages, stubs |
| `low_zh_ratio` (<40%) | 18,457 | 3.8% | code, foreign quotes, formulas |
| `repeated_lines` | 1,562 | 0.3% | template/table leftovers |
| `too_long` (>20000) | 1,098 | 0.2% | list pages |
| `low_diversity` | 55 | 0.01% | single-char repeats |

## 4. What each stage does

### 4.1 Extract

```bash
python -m wikiextractor.WikiExtractor data/raw/zhwikinews-...xml.bz2 \
  -o data/interim/extracted/zhwikinews --json --no-templates -q --processes 8
```

⚠️ **wikiextractor 3.1 names shards `AA`, `AB`… instead of `wiki_00`.** Any `glob("wiki_*")`
silently reads zero documents. Glob the directory, not the filename.

### 4.2 Clean

Drop empty lines, namespace prefixes (`Category:`, `File:`, `Template:`…), table rows;
turn `== heading ==` into `heading`; strip HTML, `[1]` refs, `{{templates}}`, `[[a|b]] → b`,
external links, `'''`; collapse whitespace. Then quality gates:
`min_chars=200`, `max_chars=20000`, `min_zh=0.4`, char diversity ≥0.05, unique-line ratio ≥0.7.

### 4.3 Dedup (exact + MinHash)

```python
h = sha1(text)                       # exact duplicates
m = MinHash(num_perm=128)            # near duplicates, Jaccard ≥ 0.8
sh = sample 5-grams down to ≤512 per doc → m.update_batch(sh)
```

Three details that matter: **cap shingles at 512** (300M 5-grams → 2 minutes instead of an hour),
**use `update_batch`** (vectorised), and **run exact first** (skips the expensive path).

### 4.4 Budget sampling

We want enough epochs at a fixed compute budget: 112M tokens ≈ 176M chars (1.568 ch/tok),
so accumulate chars until 180M and stop. A `min_chars=700` floor drops stubs — that single
knife takes the doc count from 123,708 to **71,665**, matching the tutorial's 71,224.

### 4.5 Split 98/1/1 (by document, seed 42)

| Split | Docs | Tokens |
|---|---:|---:|
| train | 70,233 | 112,503,579 |
| val | 716 | 1,199,254 |
| test | 716 | 1,103,986 |

### 4.6 Tokenizer (BPE 32k)

```python
spm.SentencePieceTrainer.train(input="train.txt", vocab_size=32000, model_type="bpe",
    character_coverage=0.9995, byte_fallback=True, pad_id=0, unk_id=1, bos_id=-1, eos_id=2)
```

`byte_fallback` means **no UNK ever**; measured **1.568 chars/token** (below 1.3 = broken tokenizer).

### 4.7 Pack (uint16 + EOS)

32000 < 65535 → `uint16` halves disk and IO. EOS between documents; `np.memmap` for zero-copy
random windows of `seq_len+1`.

## 5. Use your own data

Any JSONL with a `text` field can enter at step 4.2:

```bash
python code/llm/prepare_corpus.py clean --input my_raw --output data/interim/clean.jsonl \
    --min-chars 300 --min-zh 0.5
python code/llm/prepare_corpus.py dedup --input data/interim/clean.jsonl \
    --output data/interim/dedup.jsonl --threshold 0.85 --max-shingles 512
```

| Symptom | Knob |
|---|---|
| too few tokens | lower `min_chars` or raise the budget |
| jumpy loss | raise `min_chars`, dedup threshold 0.85 |
| chars/token < 1.3 | smaller vocab (24k) or more data |
| OOM | not a corpus problem — see ch. 02 |

## 6. Verify

```bash
cat data/processed/stats.json     # train.tokens ~112M, chars_per_token 1.5–1.7
python -c "import numpy as np;a=np.memmap('data/processed/bin/train.bin',dtype=np.uint16,mode='r');print(len(a),a[:20])"
```

---

Previous: [00 · Overview](00-overview.md) · Next: [02 · Pretraining](02-pretraining.md)
