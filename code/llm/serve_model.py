"""Minimal OpenAI-compatible server for a local checkpoint (no vLLM needed).

  python serve_model.py --ckpt runs/sft/latest.pt --tokenizer data/processed/tokenizer/zh_bpe.model --port 8000

Then any OpenAI client can talk to http://127.0.0.1:8000/v1 (model name: local-sft).
Only /v1/chat/completions and /v1/models are implemented; it is a single-request
debug server, not a production endpoint.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from model import GPT, GPTConfig

IM_START, IM_END = 32000, 32001
MODEL = None
SP = None
CFG = None
DEVICE = "cpu"
LOCK = threading.Lock()


def build_prompt(messages):
    ids = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        ids.append(IM_START)
        ids += SP.encode(f"{role}\n", out_type=int)
        ids += SP.encode(content, out_type=int)
        ids.append(IM_END)
    ids.append(IM_START)
    ids += SP.encode("assistant\n", out_type=int)
    return ids


@torch.no_grad()
def generate(ids, max_new_tokens=200, temperature=0.8, top_k=40):
    x = torch.tensor([ids[-CFG.seq_len:]], device=DEVICE)
    out = MODEL.generate(x, max_new_tokens=max_new_tokens,
                         temperature=max(temperature, 1e-4), top_k=top_k)
    new = out[0, len(x[0]):].tolist()
    # cut at the first end-of-turn id
    if IM_END in new:
        new = new[: new.index(IM_END)]
    return SP.decode(new).strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": "local-sft", "object": "model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]
        t0 = time.time()
        with LOCK:
            text = generate(build_prompt(msgs),
                            max_new_tokens=int(req.get("max_tokens", 200)),
                            temperature=float(req.get("temperature", 0.8)))
        self._json({
            "id": f"chatcmpl-{int(t0)}", "object": "chat.completion", "created": int(t0),
            "model": req.get("model", "local-sft"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(text), "total_tokens": len(text)},
        })


def main():
    global MODEL, SP, CFG, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import sentencepiece as spm
    SP = spm.SentencePieceProcessor(model_file=args.tokenizer)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    CFG = GPTConfig(**{k: v for k, v in cfg.items() if k in GPTConfig.__annotations__})
    DEVICE = args.device
    MODEL = GPT(CFG).to(DEVICE).eval()
    MODEL.load_state_dict(ck["model"], strict=False)
    print(f"[serve] loaded {args.ckpt} params={sum(p.numel() for p in MODEL.parameters())/1e6:.2f}M "
          f"device={DEVICE} port={args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
