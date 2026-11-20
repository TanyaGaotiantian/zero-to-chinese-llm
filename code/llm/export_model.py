"""Strip optimizer state from a training checkpoint for handoff.

  python export_model.py runs/125m/latest.pt export/125m_model.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ckpt.get("model") or ckpt
    # unwrap torch.compile prefix if present
    model = {k.replace("_orig_mod.", ""): v for k, v in model.items()}
    payload = {
        "model": model,
        "config": ckpt.get("config"),
        "step": ckpt.get("step"),
        "tokens": ckpt.get("tokens"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    size = out.stat().st_size / 2**20
    print(f"saved {out} ({size:.0f} MiB, step={payload['step']})")


if __name__ == "__main__":
    main()
