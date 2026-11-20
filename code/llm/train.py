"""Train a small GPT on packed uint16 token bins, with exact-resume support.

Usage (cloud, 5070 Ti):
  python train.py --config 125m --data-dir ../data/processed/bin --run-dir runs/125m \
      --micro-batch 16 --grad-accum 32 --max-steps 700

The script writes:
  run_dir/config.json, train.csv, latest.pt, ckpt_*.pt, samples.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import CONFIGS, GPT


# ---------------------------------------------------------------- data
class BinDataset:
    """Flat uint16 token stream -> fixed-length (seq_len+1) samples.

    Epoch order is a deterministic permutation of sample indices, so a resume
    can regenerate the exact same order from (seed, epoch) and skip `pos`.
    """

    def __init__(self, path: str, seq_len: int, seed: int):
        self.path = path
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.sample_len = seq_len + 1
        self.n = len(self.arr) // self.sample_len
        if self.n == 0:
            raise ValueError(f"{path}: not enough tokens for seq_len={seq_len}")
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self.order = np.random.default_rng(seed).permutation(self.n)

    def _new_epoch(self):
        self.epoch += 1
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.n)
        self.pos = 0

    def next_batch(self, batch_size: int):
        if self.pos + batch_size > self.n:
            self._new_epoch()
        idx = self.order[self.pos:self.pos + batch_size]
        self.pos += batch_size
        chunk = np.stack([
            self.arr[i * self.sample_len:(i + 1) * self.sample_len] for i in idx
        ])
        x = torch.from_numpy(chunk[:, :-1].astype(np.int64))
        y = torch.from_numpy(chunk[:, 1:].astype(np.int64))
        return x, y

    def reset(self):
        self.epoch = 0
        self.pos = 0
        self.order = np.random.default_rng(self.seed).permutation(self.n)

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "pos": self.pos,
            "n": self.n,
            "order_sha1": hashlib.sha1(self.order.tobytes()).hexdigest(),
        }

    def load_state_dict(self, st: dict):
        assert st["n"] == self.n, "dataset size changed since checkpoint"
        self.epoch = st["epoch"]
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.n)
        sha = hashlib.sha1(self.order.tobytes()).hexdigest()
        assert sha == st["order_sha1"], "data order mismatch: numpy version changed?"
        self.pos = st["pos"]


# ---------------------------------------------------------------- helpers
def lr_at(step: int, max_steps: int, warmup: int, peak: float, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak * (min_ratio + (1.0 - min_ratio) * coeff)


def unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_optimizer(model, lr: float, wd: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, val_ds: BinDataset, batches: int, device: str, amp_dtype):
    model.eval()
    val_ds.reset()
    losses = []
    for _ in range(batches):
        x, y = val_ds.next_batch(4)  # small batch keeps eval peak memory flat
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_ckpt(path, model, optim, step, tokens, cfg, ds, args):
    ckpt = {
        "model": unwrap(model).state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "tokens": tokens,
        "config": asdict(cfg),
        "data": ds.state_dict(),
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_save(ckpt, path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="125m")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=200)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--resume", default="auto", help="auto | none | path/to.pt")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--time-budget-min", type=float, default=0)
    ap.add_argument("--sample-prompt", default="")
    ap.add_argument("--tokenizer", default="", help="sentencepiece model path")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] CUDA not available: running a slow CPU smoke test only")

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    cfg = CONFIGS[args.config]
    cfg.grad_checkpoint = args.grad_checkpoint
    model = GPT(cfg).to(device)
    print("param breakdown:", json.dumps(model.param_breakdown(), indent=2))

    optim = make_optimizer(model, args.lr, args.weight_decay)

    train_ds = BinDataset(os.path.join(args.data_dir, "train.bin"), cfg.seq_len, args.seed)
    val_ds = BinDataset(os.path.join(args.data_dir, "val.bin"), cfg.seq_len, args.seed + 1)

    start_step, tokens_seen = 0, 0
    resume_path = ""
    if args.resume == "auto":
        cand = os.path.join(args.run_dir, "latest.pt")
        resume_path = cand if os.path.exists(cand) else ""
    elif args.resume != "none":
        resume_path = args.resume

    if resume_path:
        print(f"[resume] {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens"]
        train_ds.load_state_dict(ckpt["data"])
        torch.set_rng_state(ckpt["torch_rng"])
        if device == "cuda" and ckpt.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        print(f"[resume] step={start_step} tokens={tokens_seen} epoch={train_ds.epoch} pos={train_ds.pos}")

    if args.compile and device == "cuda":
        print("[compile] torch.compile(model) ... first step will be slow")
        model = torch.compile(model)

    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({"model": asdict(cfg), "args": vars(args),
                   "params": unwrap(model).param_breakdown()}, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.run_dir, "train.csv")
    # On resume, drop log rows after the checkpoint step so the file has no
    # duplicated steps (the crashed run may have logged past the last ckpt).
    if resume_path and start_step > 0 and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        if rows:
            keep = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= start_step]
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerows(keep)
    new_file = not os.path.exists(csv_path) or start_step == 0
    csv_f = open(csv_path, "a" if not new_file else "w", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(["step", "wall_s", "lr", "loss", "grad_norm", "tokens", "tok_per_s", "batch_sha1"])

    amp_dtype = torch.bfloat16
    t0 = time.time()
    last_log_t = t0
    last_log_tokens = tokens_seen
    running_loss = 0.0
    step = start_step - 1
    model.train()

    for step in range(start_step, args.max_steps):
        if args.time_budget_min and (time.time() - t0) / 60 > args.time_budget_min:
            print("[stop] time budget reached")
            break

        lr = lr_at(step, args.max_steps, args.warmup, args.lr, args.min_lr_ratio)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        first_batch_sha = ""
        for micro in range(args.grad_accum):
            x, y = train_ds.next_batch(args.micro_batch)
            if micro == 0:
                first_batch_sha = hashlib.sha1(x.numpy().tobytes()).hexdigest()[:16]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            loss_sum += loss.detach().float().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        step_tokens = args.micro_batch * cfg.seq_len * args.grad_accum
        tokens_seen += step_tokens
        running_loss = loss_sum / args.grad_accum

        if (step + 1) % args.log_interval == 0 or step == start_step:
            now = time.time()
            dt = max(1e-9, now - last_log_t)
            tok_s = (tokens_seen - last_log_tokens) / dt
            wall = now - t0
            print(f"step {step+1:5d} | loss {running_loss:.4f} | lr {lr:.2e} | "
                  f"gn {float(grad_norm):.3f} | {tok_s/1000:.1f}k tok/s | {wall/60:.1f} min")
            writer.writerow([step + 1, f"{wall:.1f}", f"{lr:.6e}", f"{running_loss:.6f}",
                             f"{float(grad_norm):.4f}", tokens_seen, f"{tok_s:.1f}", first_batch_sha])
            csv_f.flush()
            last_log_t, last_log_tokens = now, tokens_seen

        if (step + 1) % args.save_interval == 0:
            save_ckpt(os.path.join(args.run_dir, "latest.pt"), model, optim,
                      step + 1, tokens_seen, cfg, train_ds, args)
            print(f"[ckpt] step {step+1} saved")

        if args.eval_interval and (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_ds, args.eval_batches, device, amp_dtype)
            print(f"[eval] step {step+1} val_loss {val_loss:.4f} ppl {math.exp(val_loss):.2f}")

    save_ckpt(os.path.join(args.run_dir, "latest.pt"), model, optim,
              step + 1, tokens_seen, cfg, train_ds, args)
    csv_f.close()

    if args.sample_prompt and args.tokenizer:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
        ids = torch.tensor([sp.encode(args.sample_prompt, out_type=int)], device=device)
        out = unwrap(model).generate(ids, max_new_tokens=120, temperature=0.8, top_k=50)
        text = sp.decode(out[0].tolist())
        with open(os.path.join(args.run_dir, "samples.txt"), "a") as f:
            f.write(f"PROMPT: {args.sample_prompt}\n{text}\n\n")
        print(text)

    if device == "cuda":
        print(f"[mem] peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
              f"reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
    print(f"[done] steps={step+1} tokens={tokens_seen} wall={(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
