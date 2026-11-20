# 02 · Pretraining: 114.3k tok/s at 11.15 GiB

> What the model is, how the hyperparameters were chosen, **why a 4090D is 6× faster than a 5070 Ti**, and whether `torch.compile` pays off.

## 1. Architecture

LLaMA-style decoder-only, **no custom CUDA kernels**:

| Component | Choice | Why |
|---|---|---|
| Norm | RMSNorm (pre-norm) | one pass less than LayerNorm |
| Positions | RoPE (θ=10000, interleaved) | parameter-free, extrapolates |
| FFN | SwiGLU (≈8/3·d, rounded to 128 → 2112/2736) | better than GELU-MLP at equal params |
| Embedding | **tied** (`lm_head = tok_emb`) | 24.6M of 125M lives here |
| Attention | `F.scaled_dot_product_attention` | cuDNN/Flash backends, no build step |
| Dropout | 0 | data ≫ params |

| Config | Layers | d_model | heads | ffn | seq | **params** |
|---|---:|---:|---:|---:|---:|---:|
| 125m | 14 | 768 | 12 | 2112 | 1024 | **125,753,088** |
| 360m | 26 | 1024 | 16 | 2736 | 1024 | **360,403,968** |

## 2. Hyperparameters

| Item | 125M | 360M |
|---|---:|---:|
| seq_len | 1024 | 1024 |
| micro-batch | 16 | 8 |
| grad-accum | 30 | 60 |
| effective batch | 491,520 tok/step | 491,520 tok/step |
| steps | 699 (3.05 epochs) | 456 (2 epochs) |
| lr | 6e-4 cosine→10% | 3e-4 cosine→10% |
| warmup | 20 | 24 |
| precision | bf16 autocast | bf16 + **grad checkpoint** |
| optimizer | AdamW(0.9, 0.95), wd 0.1 on ≥2-D | same |

## 3. Run it

```bash
bash scripts/stages.sh smoke     # 20 steps, 2 min
bash scripts/stages.sh 125m      # 50 min
bash scripts/stages.sh 360m      # 2 h
```

## 4. Measured

### 125M — 699 steps / 343.6M tokens / 50.6 min

| step | loss | tok/s |
|---:|---:|---:|
| 1 | 10.5145 | 80.1k |
| 20 | 8.6147 | 115.7k |
| 100 | 7.4460 *(val)* | 114.5k |
| 300 | 5.7029 *(val)* | 114.3k |
| 500 | 4.6975 *(val)* | 114.3k |
| 600 | 4.5412 *(val)* | 111.8k |
| 680 (last CSV row) | 4.55 (train) | 112.2k |

> The run actually finished **699 steps** (`[done] steps=699 tokens=343572480`), but with
> `--log-interval 20` the last CSV row is step 680: the final 19 steps only exist inside the
> final checkpoint. Do not mistake 680 for the end of training.

Peak: **11.15 GiB allocated / 12.12 GiB reserved**. ppl 1713 → **93.8**.

### 360M — 456 steps / 224.1M tokens / 114.4 min

val_loss 7.764 → 6.777 → 6.186 → 5.752 → 5.445 → **5.271** (ppl 194.5); peak **8.29 / 8.68 GiB**.

Throughput is **29%** of the 125M run while parameters are only 2.87×: gradient checkpointing
recomputes activations, and 26 layers launch more, smaller kernels.

## 5. Why 114k tok/s

The bottleneck for small models is not HBM bandwidth, it is **keeping the SMs fed**.

| Factor | Effect |
|---|---|
| 4090D FP16 tensor ≈2.2× a 5070 Ti | more math per second |
| effective batch 491k tokens/step | large M dimension → high kernel efficiency |
| `torch.compile` | fuses RMSNorm/SwiGLU elementwise ops, ~20% faster steady state |
| tied embeddings | skips one 24.6M×768 gradient |
| uint16 + memmap | data loading never blocks |

`torch.compile`: first step ~40 s (compile) vs ~6 s; steady state ~92k → **114k tok/s**.
Worth it for 699 steps, and `train.py` skips it on CPU so smoke tests still run on a laptop.

## 6. Scaling further

| Target | Change | Note |
|---|---|---|
| 1B | `n_layer=24, n_embd=2048, n_head=16, ffn=5504` | needs 8-bit Adam / ZeRO; 24 GB is not enough |
| 4K context | `seq_len=4096`, RoPE θ=5e4 | VRAM ×4, drop micro-batch to 4 |
| Faster | micro-batch 32 for 125M | recompute the effective batch |

## 7. Verify

```bash
tail -3 runs/125m/train.csv                     # 699 rows, ~110k tok/s
grep -E "\[mem\]|\[done\]" logs/125m.log        # 11.15 GiB, 343572480 tokens, 50.6 min
```

---

Previous: [01 · Corpus](01-corpus.md) · Next: [03 · Resume verification](03-resume.md)
