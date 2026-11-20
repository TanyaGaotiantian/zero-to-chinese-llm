"""Minimal GPT for Chinese pretraining experiments.

Target: RTX 5070 Ti 16GB / RTX 4090 24GB / RTX 5090 32GB.
Design choices:
  - Pre-norm RMSNorm + RoPE + SwiGLU + tied embeddings (LLaMA-style)
  - F.scaled_dot_product_attention (cuDNN backend on Blackwell, no FlashAttention build needed)
  - optional gradient checkpointing for the 360M config
  - no custom CUDA kernels: installs cleanly on sm_120 with torch>=2.7/cu128
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    n_layer: int = 14
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 768
    ffn_hidden: int = 2112       # SwiGLU hidden; 3*d*h params
    seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    grad_checkpoint: bool = False

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in fp32 for stability, cast back
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: torch.device, dtype: torch.dtype):
    """cos/sin of shape (seq_len, head_dim), interleaved pairs."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (T, D/2)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D)
    return x * cos[None, None, :, :] + rotate_half(x) * sin[None, None, :, :]


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)  # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # scaled init for residual output projections (GPT-2 style)
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for blk in self.blocks:
            blk.attn.wo.weight.data.mul_(scale)
            blk.mlp.w2.weight.data.mul_(scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        # tied weights are stored once, so sum(parameters()) is already correct
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def param_breakdown(self) -> dict:
        emb = self.tok_emb.weight.numel()
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        return {
            "embedding": emb,
            "per_block": per_block,
            "blocks": per_block * self.cfg.n_layer,
            "final_norm": sum(p.numel() for p in self.norm_f.parameters()),
            "total": self.num_params(),
        }

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"sequence {T} > seq_len {self.cfg.seq_len}"
        device = idx.device
        x = self.drop(self.tok_emb(idx))
        # build RoPE in the activation dtype: under autocast the embedding
        # output is bf16, and an fp32 cos/sin would silently upcast attention
        cos, sin = build_rope_cache(T, self.cfg.n_embd // self.cfg.n_head,
                                    self.cfg.rope_theta, device, x.dtype)
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


CONFIGS = {
    # tiny config used by smoke_test.py so the whole loop runs on CPU in ~1 min
    "smoke": GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=128,
                       ffn_hidden=352, seq_len=128),
    "125m": GPTConfig(n_layer=14, n_head=12, n_kv_head=12, n_embd=768,
                      ffn_hidden=2112, seq_len=1024),
    "360m": GPTConfig(n_layer=26, n_head=16, n_kv_head=16, n_embd=1024,
                      ffn_hidden=2736, seq_len=1024),
}


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        m = GPT(cfg)
        print(name, m.param_breakdown())
