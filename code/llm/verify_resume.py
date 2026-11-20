"""Compare two training logs to prove checkpoint resume is faithful.

  python verify_resume.py runs/125m/train.csv runs/125m-resumed/train.csv
Exit code 0 if overlapping steps match within tolerance.
"""
import argparse
import csv
import sys


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["step"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("resumed")
    ap.add_argument("--tol", type=float, default=0.05, help="max |loss diff|")
    args = ap.parse_args()

    a, b = load(args.baseline), load(args.resumed)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("no overlapping steps: did the resumed run start at step 0?")

    diffs = []
    for s in common:
        diffs.append(abs(float(a[s]["loss"]) - float(b[s]["loss"])))
    worst = max(diffs)
    print(f"overlap steps: {len(common)}  range: {common[0]}..{common[-1]}")
    print(f"max |loss diff| = {worst:.6f}  mean = {sum(diffs)/len(diffs):.6f}")

    sha_mismatch = [s for s in common
                    if a[s].get("batch_sha1") and b[s].get("batch_sha1")
                    and a[s]["batch_sha1"] != b[s]["batch_sha1"]]
    if sha_mismatch:
        print(f"WARNING: batch fingerprint differs at {len(sha_mismatch)} steps, "
              f"first={sha_mismatch[0]}")
    else:
        print("batch fingerprints identical on all overlapping steps")

    if worst > args.tol:
        sys.exit(f"FAIL: loss diff {worst:.4f} > tol {args.tol}")
    print("PASS")


if __name__ == "__main__":
    main()
