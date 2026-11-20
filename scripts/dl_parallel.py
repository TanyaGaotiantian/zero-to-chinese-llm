#!/usr/bin/env python3
"""Parallel ranged downloader (resumable per chunk) for large files over a slow-per-connection link."""
import os, sys, subprocess, concurrent.futures as cf, shutil

CHUNK = int(os.environ.get("CHUNK_MB", "64")) * 1024 * 1024
WORKERS = int(os.environ.get("WORKERS", "24"))


def head_size(url):
    out = subprocess.run(["curl", "-sIL", "--max-time", "60", url],
                         capture_output=True, text=True).stdout
    size = 0
    ok = False
    for line in out.splitlines():
        if line.lower().startswith("http/") and ("200" in line or "206" in line):
            ok = True
        if line.lower().startswith("content-length:"):
            size = int(line.split(":")[1].strip())
    if not ok or size <= 0:
        raise SystemExit(f"cannot determine size for {url}\n{out[:500]}")
    return size


def fetch(url, start, end, path, tries=6):
    want = end - start + 1
    for t in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == want:
            return path
        cmd = ["curl", "-sL", "--max-time", "900", "-r", f"{start + have}-{end}", url,
               "-o", "-"]
        r = subprocess.run(cmd, capture_output=True)
        data = r.stdout
        if data:
            with open(path, "ab") as f:
                f.write(data)
        if os.path.exists(path) and os.path.getsize(path) == want:
            return path
        print(f"  retry {t+1} {os.path.basename(path)} {os.path.getsize(path) if os.path.exists(path) else 0}/{want}", flush=True)
    raise SystemExit(f"failed chunk {path}")


def main():
    url, dst = sys.argv[1], sys.argv[2]
    size = head_size(url)
    os.makedirs(dst + ".parts", exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(f"[skip] {dst} already complete ({size})", flush=True)
        return
    chunks = [(i, s, min(s + CHUNK - 1, size - 1))
              for i, s in enumerate(range(0, size, CHUNK))]
    print(f"[dl] {dst} size={size/1e9:.2f}GB chunks={len(chunks)} workers={WORKERS}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch, url, s, e, f"{dst}.parts/{i:05d}") for i, s, e in chunks]
        for n, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if n % 5 == 0 or n == len(chunks):
                done = sum(os.path.getsize(f"{dst}.parts/{i:05d}")
                           for i in range(len(chunks)) if os.path.exists(f"{dst}.parts/{i:05d}"))
                print(f"  {n}/{len(chunks)} chunks, {done/1e9:.2f}GB", flush=True)
    with open(dst, "wb") as out:
        for i, s, e in chunks:
            with open(f"{dst}.parts/{i:05d}", "rb") as f:
                shutil.copyfileobj(f, out)
    got = os.path.getsize(dst)
    print(f"[ok] {dst} {got} bytes (expect {size})", flush=True)
    if got != size:
        raise SystemExit("size mismatch")
    shutil.rmtree(dst + ".parts")


if __name__ == "__main__":
    main()
