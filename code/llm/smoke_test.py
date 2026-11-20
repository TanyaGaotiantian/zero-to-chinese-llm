#!/usr/bin/env python3
"""CPU smoke test: 5-minute end-to-end check of the whole training loop.

  python smoke_test.py                      # model + 20 steps on synthetic data
  python smoke_test.py --resume-check       # + kill -9 resume determinism check

It uses the tiny "smoke" config in model.py (2 layers, d=128, seq=128) and
synthetic uint16 bins, so it runs on a laptop with no GPU and no corpus.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent / ".smoke"


def make_bins(n_train=400_000, n_val=40_000, vocab=32000, seed=0):
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.asarray(rng.integers(3, vocab - 1, n_train, dtype=np.uint16)).tofile(data / "train.bin")
    np.asarray(rng.integers(3, vocab - 1, n_val, dtype=np.uint16)).tofile(data / "val.bin")
    print(f"[data] synthetic bins -> {data}  (train {n_train:,} tok, val {n_val:,} tok)")


def run(args, run_dir, log=None, timeout=None):
    cmd = [sys.executable, str(HERE / "train.py"), "--config", "smoke",
           "--data-dir", str(WORK / "data"), "--run-dir", str(run_dir)] + args
    print("$ " + " ".join(cmd[1:]))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    try:
        for line in p.stdout:
            out.append(line)
            if log:
                log.write(line)
            sys.stdout.write(line)
    finally:
        p.wait(timeout=timeout)
    return p.returncode, "".join(out)


def read_csv(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["step"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--resume-check", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    make_bins()

    base = ["--micro-batch", "4", "--grad-accum", "2", "--warmup", "2",
            "--eval-interval", "10", "--eval-batches", "2", "--log-interval", "5"]

    run_dir = WORK / "runs" / "smoke"
    rc, out = run(base + ["--max-steps", str(args.steps)], run_dir)
    if rc != 0:
        sys.exit("smoke run failed")
    rows = read_csv(run_dir / "train.csv")
    assert rows, "no train.csv rows"
    print(f"\n[ok] {len(rows)} logged steps, final train loss {float(list(rows.values())[-1]['loss']):.4f}")
    print("     loss 应接近 ln(32000)=10.37（随机初始化）并开始下降")

    if args.resume_check:
        print("\n=== resume determinism check ===")
        ref, test = WORK / "runs" / "resume_ref", WORK / "runs" / "resume_test"
        for d in (ref, test):
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
        steps = str(args.steps)
        run(base + ["--max-steps", steps, "--save-interval", "5"], ref)

        cmd = [sys.executable, str(HERE / "train.py"), "--config", "smoke",
               "--data-dir", str(WORK / "data"), "--run-dir", str(test)] + \
              base + ["--max-steps", steps, "--save-interval", "5", "--log-interval", "1"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(max(2.0, 0.35 * args.steps))
        p.send_signal(signal.SIGKILL)
        p.wait()
        ck = test / "latest.pt"
        if not ck.exists():
            sys.exit("killed before any checkpoint was written; rerun with a larger --steps")
        print(f"[kill] SIGKILL after ~{max(2.0, 0.35*args.steps):.1f}s, ckpt kept: step "
              f"{sorted(read_csv(test / 'train.csv'))[-1]}")

        run(base + ["--max-steps", steps, "--save-interval", "5", "--log-interval", "1"], test)
        r = subprocess.run([sys.executable, str(HERE / "verify_resume.py"),
                            str(ref / "train.csv"), str(test / "train.csv"), "--tol", "0.05"],
                           text=True, capture_output=True)
        print(r.stdout)
        if r.returncode != 0:
            sys.exit("resume check FAILED")
        print("[ok] resume check PASSED")

    if not args.keep:
        import shutil
        shutil.rmtree(WORK, ignore_errors=True)
        print(f"[clean] removed {WORK}")


if __name__ == "__main__":
    main()
