# 07 · Reproduction checklist

## Stage 0 · Environment

- [ ] `nvidia-smi` shows a 24 GB card
- [ ] `python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"` → `2.8.x (8, 9)`
- [ ] `/root/autodl-tmp` has ≥40 GB free

## Stage 1 · Data

- [ ] `bash scripts/fetch_sources.sh` ends with `SOURCES_DONE`
- [ ] parquet shard 0 is **587,109,027** bytes
- [ ] `bzip2 -t data/raw/zhwikinews-*.bz2` passes
- [ ] `du -sh data/raw` ≈ 650 MB

## Stage 2 · Corpus

- [ ] `bash scripts/prep2.sh` ends with `=== PREP DONE ===`
- [ ] `stats.json`: `train.tokens` ~112M, `chars_per_token` 1.5–1.7
- [ ] `funnel.json`: `raw` ≈ 485,000, `kept_ratio` ≈ 0.58
- [ ] `train.bin` ≈ 225 MB, first ids in 3..31999

## Stage 3 · Smoke

- [ ] `bash scripts/stages.sh smoke`
- [ ] step 1 loss ≈ 10.4–10.5 (`ln(32000)=10.37`)
- [ ] after 20 steps loss < 9.0
- [ ] `[mem] peak allocated` 11–13 GiB
- [ ] throughput > 80k tok/s

## Stage 4 · Resume

- [ ] `bash scripts/stages.sh resume`
- [ ] `logs/verify_resume.txt` ends with `PASS`
- [ ] `max |loss diff| < 0.05`
- [ ] `batch fingerprints identical on all overlapping steps`
- [ ] `logs/resume_killed.log` contains `Killed`

## Stage 5 · Pretraining

- [ ] `bash scripts/stages.sh 125m` → last step 699, val_loss < 4.8, allocated ≈ 11.15 GiB
- [ ] `bash scripts/stages.sh 360m` → last step 456, val_loss < 5.6
- [ ] `runs/*/samples.txt` contains plausible Chinese

## Stage 6 · SFT

- [ ] `bash scripts/stages.sh sft`
- [ ] `data/sft/card.json` has `kept = 24000`
- [ ] `grep "\[eval\]" logs/sft.log` shows val_loss **well above 1** (if it is <0.01, the shift bug is back)
- [ ] `runs/sft/latest.pt` exists

## Stage 7 · Memory

- [ ] `bash scripts/demo_memory.sh`
- [ ] `logs/memory_demo.log` contains `[diary]`
- [ ] `/recall <keyword>` returns the stored diary
- [ ] sqlite has rows in `messages` / `diaries` / `feedback`

## Stage 8 · Shut down (do not skip)

- [ ] copy `runs/*/train.csv`, `logs/*.log`, `stats.json`, `funnel.json` out
- [ ] `export_model.py` to strip optimizer state
- [ ] **shut the instance down** and watch the balance stop moving

> Idle GPU costs ¥1.88/h — one forgotten hour equals a full 125M pretraining run.

## Quick troubleshooting

| Symptom | Likely cause | Chapter |
|---|---|---|
| `raw=0` in funnel | wikiextractor 3.1 shard names | [01](01-corpus.md) |
| chars/token < 1.3 | tokenizer trained on too little text | [01](01-corpus.md) |
| `batch_sha1` mismatch | data order not stored | [03](03-resume.md) |
| SFT val_loss < 0.01 | missing label shift | [04](04-sft.md) |
| OOM at 360M | gradient checkpointing off | [02](02-pretraining.md) |

---

Previous: [06 · Cost and pitfalls](06-cost-and-pitfalls.md)
