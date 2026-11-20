# Model card · 青鸾 QingLuan 125M / 360M

## Overview

| | QingLuan-125M | QingLuan-360M |
|---|---|---|
| Type | decoder-only Transformer (LLaMA-style) | same |
| Parameters | 125,753,088 | 360,403,968 |
| Layers / d_model / heads | 14 / 768 / 12 | 26 / 1024 / 16 |
| Context | 1024 tokens | 1024 tokens |
| Vocab | 32,000 (sentencepiece BPE, byte_fallback) | same |
| Training tokens | 343.6M (3.05 epochs over 112.5M) | 224.1M (2 epochs) |
| Precision | bf16 autocast, AdamW | bf16 + gradient checkpointing |
| License | Apache-2.0 (code and weights) | same |

Both models were trained **from random initialisation** on a 112.5M-token Chinese corpus
(Chinese Wikipedia + Wikinews + Wikivoyage), not distilled or fine-tuned from another model.

## Intended use

- Teaching / research on the full pretraining pipeline (corpus → training → resume → SFT).
- A control baseline for small-model experiments (throughput, memory, tokenizer studies).
- A local component in prototypes that need a tiny Chinese LM (see `memory_chat.py`).

## Out of scope

- **Factual question answering.** The 125M model produces fluent Chinese with unreliable facts.
- **Instruction following beyond short answers.** SFT used 24k generic pairs; it is not an assistant.
- **Anything safety-critical.** No alignment stage, no content filtering.

## Training data

| Source | Volume | License |
|---|---|---|
| `wikimedia/wikipedia` 20231101.zh (official-dump derivative) | 461,584 docs raw | CC BY-SA 4.0 |
| `zh.wikinews` dump | 34 MB bz2 | CC BY 2.5 |
| `zh.wikivoyage` dump | 14 MB bz2 | CC BY-SA 4.0 |
| after cleaning/dedup/budget | **71,665 docs / 112.5M tokens** | |

SFT: `YeungNLP/firefly-train-1.1M`, 24,000 filtered pairs (check the upstream license before redistribution).

## Evaluation

| Metric | 125M | 360M |
|---|---:|---:|
| val_loss (wiki, 1.1M tokens) | **4.541** (ppl 93.8) | **5.271** (ppl 194.5) |
| SFT val_loss (assistant-only) | **3.807** (best) | — |
| Throughput (4090D, 491k tok/step) | **114.3k tok/s** | 33.3k tok/s |
| Peak VRAM | 11.15 GiB | 8.29 GiB |

No benchmark suite was run: at this scale, CEval/CMRC scores are dominated by noise, and the
project's claim is reproducibility, not quality.

## Sample generations

Base model, prompt `中国的首都是`（temperature 0.8, top-k 40）:

> 中国的首都是中国的"中国"。 中国自1856年《华尔街》创刊,当时中国内地已有"清华"二字,是现今中国各地民间用词的来源……

The syntax is Chinese; the content is not trustworthy. That is the expected behaviour of a
125M model trained on 112M tokens (≈1.9 tokens per parameter, vs. the ~20 used by
compute-optimal recipes).

## Limitations

1. **Undertrained by design** — the corpus is intentionally small so the whole pipeline runs in hours.
2. **1024-token context**, no RoPE scaling.
3. **No safety tuning.**
4. Reproduces biases present in Wikipedia and in the SFT dataset.

## Reproduce

```bash
bash scripts/fetch_sources.sh && bash scripts/prep2.sh
bash scripts/stages.sh smoke && bash scripts/stages.sh resume
bash scripts/stages.sh 125m && bash scripts/stages.sh 360m && bash scripts/stages.sh sft
```
