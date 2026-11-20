# 06 · Cost and pitfalls: how much of ¥19 was tuition

## 1. The bill

| Stage | Wall clock | Cost |
|---|---:|---:|
| Fetch data | 30 min | ¥0.05 |
| Corpus pipeline | 8 min | ¥0.25 |
| 20-step smoke | 2 min | ¥0.06 |
| `kill -9` resume check | 5 min | ¥0.16 |
| 125M pretraining | 50.6 min | ¥1.59 |
| 360M pretraining | 114.4 min | ¥3.58 |
| SFT | ~15 min | ¥0.45 |
| Memory demo | 5 min | ¥0.16 |
| **Incident: idle instance** | **6 h** | **¥11.3** |
| **Total** | | **≈¥17.6** |

**Effective compute ≈¥6.3. Tuition ¥11.3.**

## 2. Seven pitfalls

### #1 · Six hours of idle GPU (¥11.3)

A tool call got interrupted mid-run; I never verified the background job and ended the turn.
The instance idled from 00:50 to 06:44 at 0% utilisation.

Fix: (a) after starting anything long, `pgrep -af train.py` + `tail -3 logs/*.log`;
(b) chain stages automatically (`chain.sh` waits for 125M, starts 360M, then SFT);
(c) always pass `--time-budget-min`.

### #2 · Wikimedia dumps throttled in a CN IDC

9–80 KB/s per connection; 24 parallel range requests only reach 124 KB/s. Fix: HF parquet
(5.7 MB/s) + schema conversion (chapter 01).

### #3 · "GPU-less mode" is not a free CPU box

0.5 vCPU / 2 GB. A 14 MB dump takes 45 s to extract; 3.45 GB would take 3 hours.
Use it for downloads only.

### #4 · GPU is gone after shutdown

`该主机空闲GPU不足` — pay-as-you-go releases the GPU when you stop. Fix: don't stop for short
gaps, or clone the instance to another host that has a free card.

### #5 · Cloning without the data disk

The clone dialog leaves the data-disk checkbox unchecked; Element-Plus only toggles when you
click the inner `input`. Result: an empty `/root/autodl-tmp`. Re-downloading was faster than
debugging the UI.

### #6 · wikiextractor 3.1 renamed shards

`wiki_00` → `AA`. Any `glob("wiki_*")` reads zero docs **without error**. Glob the directory.

### #7 · MinHash too slow

300M 5-grams at ~10 µs each is over an hour. Fix: cap at 512 shingles/doc + `update_batch`
→ 2 minutes, same detection quality.

### #8 · SFT loss collapsed to 0.0005

Forgot the one-token shift (`targets[t]` must follow `x[t]`). With tied embeddings the model
learns the identity map and CE → 0 while generation stays broken. Symptom: val_loss < 0.01
after 100 steps. Fix: shift labels by one; keep `sft_buggy.log` as a trophy.

## 3. Six rules that save money

1. Keep data on the data disk (`/root/autodl-tmp`), the system disk is 30 GB.
2. Downloads and `pip install` in CPU mode; only training on the GPU.
3. Always run a 20-step smoke test (2 min) before a long run.
4. Always cap with `--time-budget-min`.
5. Chain stages so nothing idles between them.
6. Make every stage idempotent so an interruption costs only the current stage.

## 4. If you only have ¥10

1. 125M only (skip 360M): −¥3.6
2. Halve the corpus budget to 90M chars: −¥0.8
3. One SFT epoch: −¥0.3
4. **Never** let the instance idle — that costs more than all of the above.

A minimal run (125M + 90M tokens + 1-epoch SFT) is ≈¥2.5 and still yields a Chinese-speaking
model plus a full resume verification.

---

Previous: [05 · Memory](05-memory.md) · Next: [07 · Checklist](07-checklist.md)
