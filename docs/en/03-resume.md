# 03 · Crash recovery: does training really continue after `kill -9`?

> The chapter most projects fake. Most code "supports resume" by restoring model weights only —
> the loss curve breaks at the restart point and nobody compares.

## 1. Why it is hard

One training step carries far more state than the weights:

| State | If not restored |
|---|---|
| Model params | starts over |
| Optimizer state (AdamW moments) | loss jumps for tens of steps |
| Global step | LR schedule misaligned |
| tokens_seen | budget/throughput stats wrong |
| **Data order** (epoch + pos + permutation) | silently re-trains the same examples |
| **CPU/CUDA RNG** | not reproducible |

Data order is the hard one: a `shuffle=True` dataloader with no explicit state can never be
restored, and every other fix is then pointless.

## 2. What this project does

```python
self.order = np.random.default_rng(seed + epoch).permutation(self.n)   # pure function of (seed, epoch)
st = {"epoch": e, "pos": p, "n": n, "order_sha1": sha1(order)}
# on resume:
self.order = np.random.default_rng(seed + st["epoch"]).permutation(self.n)
assert sha1(self.order) == st["order_sha1"], "data order mismatch"
```

The checkpoint also stores `optim`, `step`, `tokens`, `config`, `args`, `torch_rng`, `cuda_rng`.
And because a crash can leave CSV rows *past* the last checkpoint, resume truncates them:

```python
keep = [rows[0]] + [r for r in rows[1:] if int(r[0]) <= start_step]
```

## 3. Protocol

- **A (baseline)**: 60 uninterrupted steps.
- **B (crash)**: same config, `kill -9` around step 25, resume from `latest.pt` (step 20) to 60.
- **Compare**: per-step `loss` and `batch_sha1`.

```bash
bash scripts/stages.sh resume
```

## 4. Result

```text
overlap steps: 60  range: 1..60
max |loss diff| = 0.009987  mean = 0.002670
batch fingerprints identical on all overlapping steps
PASS
```

| Metric | Result |
|---|---|
| overlap | 60 / 60 steps |
| `batch_sha1` | **identical on all 60** → same data order |
| max abs(Δloss) | **0.009987** (tolerance 0.05) |

`batch_sha1` covers the first micro-batch of each step: identical ⇒ the model saw the same data.
The residual loss delta is bf16 accumulation non-determinism, not lost state (two identical
uninterrupted runs also differ at the 1e-3 level).

## 5. Five common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| save only `model.state_dict()` | loss jumps 0.3+ on resume | store optimizer state |
| `shuffle=True` without state | loss looks fine, data repeats (invisible) | `(seed, epoch)` permutation + sha1 |
| LR computed from the new run's step | kink in the curve | store the global step |
| no RNG state | A/B comparison never matches | save `torch_rng` / `cuda_rng` |
| append CSV without truncation | duplicate step rows | truncate to `start_step` |

---

Previous: [02 · Pretraining](02-pretraining.md) · Next: [04 · SFT](04-sft.md)
