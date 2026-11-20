#!/usr/bin/env bash
WORK="${WORK:-/root/autodl-tmp/work}"
cd "$WORK" || exit 1
echo "== time =="; date
echo "== gpu =="; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "== tmux =="; tmux ls 2>/dev/null || echo "no tmux"
echo "== log tail =="; tail -25 logs/run_all.log 2>/dev/null
echo "== last train lines =="; tail -5 logs/125m.log 2>/dev/null; tail -5 logs/360m.log 2>/dev/null; tail -5 logs/sft.log 2>/dev/null
echo "== checkpoints =="; find runs -name latest.pt -printf '%p %k KB\n' 2>/dev/null
echo "== disk =="; df -h /root/autodl-tmp | tail -1
