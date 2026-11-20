# 00 · Overview: from random weights to a model that talks

> This chapter answers three questions: **what the project contains, why the stages are ordered this way, and what the acceptance criteria are.**

## 1. One-page pipeline

```text
                     ┌──────────────────────── Task 02: corpus engineering ──────────────────────┐
 zh.wikipedia (parquet) ─┐                                                                        │
 zh.wikinews (dump)     ─┼─▶ extract ─▶ clean ─▶ dedup ─▶ budget sample ─▶ 98/1/1 ─▶ BPE 32k ─▶ uint16
 zh.wikivoyage (dump)   ─┘   485,894   283,923  283,492      71,665        splits    1.568 ch/tok  112.5M tok
                                                                                                   │
                     ┌──────────────────────── Task 01: pretraining ─────────────────────────────┐│
                     │  125M (14L/768d) ──699 steps──▶ val_loss 4.54  ──┐                        ││
                     │      └── kill -9 resume check (60/60 fingerprints)│                        ││
                     │  360M (26L/1024d) ─456 steps─▶ val_loss 5.27  ──┴─▶ used by SFT              ││
                     └──────────────────────────────────────────────────────────────────────────┘│
                     ┌──────────────────────── Task 03: SFT + memory ────────────────────────────┐│
                     │  firefly 24k ─▶ ChatML SFT (assistant-only loss) ─▶ serve_model.py (OpenAI API)
                     │                                                        │                   │
                     │                       memory_chat.py ◀────────────────┘                   │
                     │                  (SQLite + FTS5 diary memory / feedback)                    │
                     └──────────────────────────────────────────────────────────────────────────┘
```

## 2. Why this order

| # | Stage | Why it cannot be skipped |
|---|---|---|
| 1 | Corpus | Every training step reads `train.bin`. Bad corpus = wasted compute. **Save the funnel stats here** or you cannot attribute a later failure |
| 2 | 20-step smoke | Real config + real data, 2 minutes: VRAM, throughput, and "does loss start at `ln(V)`" all at once |
| 3 | Resume check | Long runs always get interrupted (shutdown / preemption / OOM). **Prove recovery works before** the long run |
| 4 | 125M run | End-to-end validation at small scale + the base for SFT |
| 5 | 360M run | Same data, same code, only the config changes — proves scaling is smooth |
| 6 | SFT | Turns the base into an answering model |
| 7 | Memory prototype | Proves the model is a usable *component*, not a toy |

> The key decision: **verify resume before the long run**. Most tutorials put it last as a bonus, when failure is most expensive.

## 3. Acceptance (measurable)

| Task | Metric | Target | Measured |
|---|---|---|---|
| 02 | docs / tokens | 71,224 / 111.86M | 71,665 / 112.50M |
| 02 | chars/token | 1.5–1.7 | 1.568 |
| 01 | 125M throughput | 18.8k tok/s (5070 Ti) | **114.3k tok/s** (4090D) |
| 01 | peak VRAM | ≈12 GiB | 11.15 GiB (reserved 12.12) |
| 01 | resume drift | < 0.05 | **0.0100**, 60/60 fingerprints |
| 01 | two scales | 125M / 360M | 125.75M / 360.40M |
| 03 | SFT samples | ≈24k | 24,000 |
| 03 | memory | compress / persist / recall | SQLite + FTS5 (ch. 05) |

## 4. File map

| Looking for | Go to |
|---|---|
| Model definition | [`code/llm/model.py`](../../code/llm/model.py) (230 lines, no third-party deps) |
| Training + resume | [`code/llm/train.py`](../../code/llm/train.py) |
| Seven corpus stages | [`code/llm/prepare_corpus.py`](../../code/llm/prepare_corpus.py) |
| SFT | [`code/llm/sft.py`](../../code/llm/sft.py) + [`build_sft_data.py`](../../code/llm/build_sft_data.py) |
| Memory CLI | [`code/llm/memory_chat.py`](../../code/llm/memory_chat.py) |
| ckpt → endpoint | [`code/llm/serve_model.py`](../../code/llm/serve_model.py) |
| Real logs | [`results/`](../../results) |

## 5. Timeline (one 4090D, measured)

| Time | Stage | Wall clock | Cost |
|---|---|---|---|
| T+0:00 | fetch data | 30 min | ¥0.05 (CPU mode) |
| T+0:30 | corpus pipeline | 8 min | ¥0.25 |
| T+0:40 | 20-step smoke | 2 min | ¥0.06 |
| T+0:45 | kill -9 resume check | 5 min | ¥0.16 |
| T+0:50 | 125M pretraining | 50 min | ¥1.59 |
| T+1:40 | 360M pretraining | 114 min | ¥3.58 |
| T+3:35 | SFT | ~15 min | ¥0.45 |
| T+3:50 | memory demo | 5 min | ¥0.15 |
| **Total** | | **≈4 h of compute** | **≈¥6.3** |

> Budget 8 hours for the first attempt: debugging and re-runs are normal.

---

Next: [01 · Corpus engineering](01-corpus.md)
