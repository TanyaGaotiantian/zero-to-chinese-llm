#!/usr/bin/env python3
"""Local playground for the 125M SFT model: a Gradio UI, a CLI, and auto-download.

  python demo/app.py                 # web UI on http://127.0.0.1:7860
  python demo/app.py --cli           # terminal REPL
  python demo/app.py --once "你好"    # one-shot generation

Weights are downloaded from the GitHub release on first run (~287 MiB):
  https://github.com/TanyaGaotiantian/zero-to-chinese-llm/releases/tag/v1.0-models

Honest expectation setting: this is a 125M model trained on 112.5M tokens. It writes
grammatical Chinese and gets facts wrong. The point of the demo is to let you poke at
the *pipeline output*, not to replace an assistant.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "code" / "llm"))

from model import GPT, GPTConfig  # noqa: E402

MODEL_URL = ("https://github.com/TanyaGaotiantian/zero-to-chinese-llm/releases/download/"
             "v1.0-models/gpt125m-sft-bf16.pt")
TOKENIZER_URL = ("https://github.com/TanyaGaotiantian/zero-to-chinese-llm/releases/download/"
                 "v1.0-models/zh_bpe.model")
CACHE = Path(os.environ.get("ZHLLM_CACHE", Path.home() / ".cache" / "zero-to-chinese-llm"))
IM_START, IM_END = 32000, 32001

CFG = None
SP = None
MODEL = None
DEVICE = "cpu"


def _download(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 1024:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}\n        -> {dst}", flush=True)

    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct:3d}%  ")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, reporthook=hook)
    print()
    return dst


def load(ckpt: str | None = None, tokenizer: str | None = None, device: str | None = None):
    global CFG, SP, MODEL, DEVICE
    if MODEL is not None:
        return
    import sentencepiece as spm

    ckpt_path = Path(ckpt) if ckpt else _download(MODEL_URL, CACHE / "gpt125m-sft-bf16.pt")
    tok_path = Path(tokenizer) if tokenizer else _download(TOKENIZER_URL, CACHE / "zh_bpe.model")

    SP = spm.SentencePieceProcessor(model_file=str(tok_path))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    CFG = GPTConfig(**{k: v for k, v in (ck.get("config") or {}).items()
                       if k in GPTConfig.__annotations__})
    DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = GPT(CFG).to(DEVICE).eval()
    MODEL.load_state_dict(ck["model"], strict=False)
    print(f"[model] 125M SFT · dtype={next(MODEL.parameters()).dtype} · device={DEVICE} "
          f"· step={ck.get('step')}", flush=True)


def build_prompt(messages):
    ids = []
    for m in messages:
        ids.append(IM_START)
        ids += SP.encode(f"{m['role']}\n", out_type=int)
        ids += SP.encode(m["content"], out_type=int)
        ids.append(IM_END)
    ids.append(IM_START)
    ids += SP.encode("assistant\n", out_type=int)
    return ids


@torch.no_grad()
def generate(messages, max_new_tokens=120, temperature=0.7, top_p=0.9,
             repetition_penalty=1.3, window=64):
    import torch.nn.functional as F

    load()
    ids = build_prompt(messages)
    idx = torch.tensor([ids[-CFG.seq_len:]], device=DEVICE)
    recent = list(idx[0].tolist())[-window:]
    out = []
    for _ in range(int(max_new_tokens)):
        logits, _ = MODEL(idx[:, -CFG.seq_len:])
        logits = logits[:, -1, :].float()
        if repetition_penalty > 1.0 and recent:
            uniq = torch.tensor(sorted(set(recent)), device=DEVICE)
            vals = logits[0, uniq]
            logits[0, uniq] = torch.where(vals > 0, vals / repetition_penalty,
                                          vals * repetition_penalty)
        logits = logits / max(temperature, 1e-4)
        if top_p and top_p < 1.0:
            srt, idxs = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(srt, dim=-1).cumsum(dim=-1)
            cut = probs > top_p
            cut[..., 1:] = cut[..., :-1].clone()
            cut[..., 0] = False
            srt[cut] = -float("inf")
            logits = torch.full_like(logits, -float("inf")).scatter(1, idxs, srt)
        nid = int(torch.multinomial(F.softmax(logits, dim=-1), num_samples=1))
        if nid == IM_END:
            break
        out.append(nid)
        recent = (recent + [nid])[-window:]
        idx = torch.cat([idx, torch.tensor([[nid]], device=DEVICE)], dim=1)
    return SP.decode(out).strip()


def chat(history, message, temperature, top_p, rep, max_new):
    """history: list of [user, assistant] pairs (Gradio 4 style)."""
    msgs = []
    for u, a in history:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": message})
    try:
        reply = generate(msgs, max_new_tokens=max_new, temperature=temperature,
                         top_p=top_p, repetition_penalty=rep)
    except Exception as e:                      # keep the UI alive
        reply = f"[error] {e}"
    return reply or "(空)"


DISCLAIMER = (
    "**这是 125M 模型**，只用 112.5M tokens 从零训练（计算最优的 ~1/10）。"
    "它能把中文写通顺，但**事实性不可靠**；它不能替代任何助手。\n\n"
    "参数量 125.75M · val_loss 3.807 · 24k ChatML SFT"
)

EXAMPLES = [
    "用一句话解释什么是机器学习",
    "把下面这句话改成更礼貌的说法：你错了。",
    "列出三个学习编程的建议",
    "给一家咖啡店起三个名字",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true", help="terminal REPL instead of web UI")
    ap.add_argument("--once", default="", help="one-shot prompt, print and exit")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--share", action="store_true", help="gradio public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    load(args.ckpt or None, args.tokenizer or None, args.device or None)

    if args.once:
        print(generate([{"role": "user", "content": args.once}], max_new_tokens=150))
        return

    if args.cli:
        print("输入问题，Ctrl-C 退出。")
        msgs = []
        while True:
            try:
                q = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            msgs.append({"role": "user", "content": q})
            a = generate(msgs, max_new_tokens=150)
            print("模型>", a)
            msgs.append({"role": "assistant", "content": a})
        return

    try:
        import gradio as gr
    except ImportError:
        sys.exit("需要 gradio：pip install gradio   （或加 --cli 用命令行）")

    with gr.Blocks(title="Zero to Chinese LLM · 125M demo") as demo:
        gr.Markdown("# Zero to Chinese LLM · 125M demo\n" + DISCLAIMER)
        chatbot = gr.Chatbot(height=380, label="125M SFT model")
        with gr.Row():
            msg = gr.Textbox(placeholder="问点什么……", scale=4, label="你的问题")
            send = gr.Button("发送", scale=1, variant="primary")
        with gr.Row():
            temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="temperature")
            topp = gr.Slider(0.3, 1.0, value=0.9, step=0.05, label="top-p")
            rep = gr.Slider(1.0, 1.6, value=1.3, step=0.05, label="repetition penalty")
            mx = gr.Slider(32, 300, value=120, step=8, label="max new tokens")
        gr.Examples(EXAMPLES, inputs=msg)

        def respond(message, history, temperature, top_p, rep, max_new):
            history = history or []
            if not message.strip():
                return history, ""
            reply = chat(history, message, temperature, top_p, rep, max_new)
            history = history + [[message, reply]]
            return history, ""

        send.click(respond, [msg, chatbot, temp, topp, rep, mx], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, temp, topp, rep, mx], [chatbot, msg])

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
