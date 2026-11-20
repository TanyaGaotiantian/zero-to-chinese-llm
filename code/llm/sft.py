"""Supervised fine-tuning for the 125M base model (ChatML, assistant-only loss).

Data format (one JSON object per line):
  {"messages":[{"role":"system","content":"..."},
               {"role":"user","content":"..."},
               {"role":"assistant","content":"..."}]}

Special ids are appended above the 32k sentencepiece vocab:
  32000 = <|im_start|>, 32001 = <|im_end|>, pad = 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from train import atomic_save, lr_at, make_optimizer

IM_START, IM_END = 32000, 32001
BASE_VOCAB = 32000


def load_pretrained(model: GPT, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    msd = model.state_dict()
    for k, v in sd.items():
        if k not in msd:
            continue
        if k == "tok_emb.weight":
            msd[k][: v.shape[0]].copy_(v)
        elif k == "lm_head.weight":
            continue
        elif v.shape == msd[k].shape:
            msd[k].copy_(v)
    model.load_state_dict(msd)
    print(f"[init] loaded {path} (new special-token rows are randomly initialised)")


class SFTDataset:
    def __init__(self, path: str, sp, max_len: int):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line)["messages"])
        self.sp = sp
        self.max_len = max_len
        self.order = list(range(len(self.rows)))
        random.Random(42).shuffle(self.order)
        self.pos = 0

    def __len__(self):
        return len(self.rows)

    def _encode(self, messages):
        ids, labels = [], []
        for m in messages:
            role, content = m["role"], m["content"]
            ids.append(IM_START)
            labels.append(-100)
            prefix = self.sp.encode(f"{role}\n", out_type=int)
            ids += prefix
            labels += [-100] * len(prefix)
            cids = self.sp.encode(content, out_type=int)
            ids += cids
            labels += (cids if role == "assistant" else [-100] * len(cids))
            ids.append(IM_END)
            labels.append(IM_END if role == "assistant" else -100)
        if len(ids) > self.max_len:
            ids, labels = ids[: self.max_len], labels[: self.max_len]
        return ids, labels

    def next_batch(self, batch_size):
        if self.pos + batch_size > len(self.order):
            random.Random(43).shuffle(self.order)
            self.pos = 0
        idx = self.order[self.pos:self.pos + batch_size]
        self.pos += batch_size
        items = [self._encode(self.rows[i]) for i in idx]
        maxlen = max(len(x) for x, _ in items)
        x = torch.zeros((len(items), maxlen), dtype=torch.long)
        y = torch.full((len(items), maxlen), -100, dtype=torch.long)
        for i, (ids, labels) in enumerate(items):
            x[i, :len(ids)] = torch.tensor(ids)
            y[i, :len(labels)] = torch.tensor(labels)
        return x, y


@torch.no_grad()
def evaluate(model, ds: SFTDataset, batches: int, device: str):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = ds.next_batch(4)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pretrained latest.pt")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-layer", type=int, default=14)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--ffn", type=int, default=2112)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    os.makedirs(args.run_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    cfg = GPTConfig(vocab_size=BASE_VOCAB + 2, n_layer=args.n_layer, n_head=args.n_head,
                    n_kv_head=args.n_head, n_embd=args.n_embd, ffn_hidden=args.ffn,
                    seq_len=args.seq_len)
    model = GPT(cfg).to(device)
    load_pretrained(model, args.init)

    train_ds = SFTDataset(args.train, sp, args.seq_len)
    val_ds = SFTDataset(args.val, sp, args.seq_len)
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch * args.grad_accum))
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(f"[data] train={len(train_ds)} val={len(val_ds)} steps/epoch={steps_per_epoch} total={total_steps}")

    optim = make_optimizer(model, args.lr, 0.01)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for _ in range(steps_per_epoch):
            if step >= total_steps:
                break
            lr = lr_at(step, total_steps, args.warmup, args.lr, 0.1)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _m in range(args.grad_accum):
                x, y = train_ds.next_batch(args.batch)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                    _, loss = model(x, y)
                (loss / args.grad_accum).backward()
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss_sum/args.grad_accum:.4f} lr {lr:.2e}")
            if step % args.eval_interval == 0:
                vl = evaluate(model, val_ds, 20, device)
                print(f"[eval] step {step} val_loss {vl:.4f} ppl {math.exp(vl):.2f}")
            if step % args.save_interval == 0:
                atomic_save({"model": model.state_dict(), "step": step,
                             "config": asdict(cfg)}, os.path.join(args.run_dir, "latest.pt"))

    atomic_save({"model": model.state_dict(), "step": step,
                 "config": asdict(cfg)}, os.path.join(args.run_dir, "latest.pt"))
    print("[done]", step)


if __name__ == "__main__":
    main()
