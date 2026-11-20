#!/usr/bin/env bash
# Stage driver for task 01 (125M -> 360M pretraining) + 03 (SFT).
# Usage: ./stages.sh smoke|resume|125m|360m|sft|memory|all
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
WORK=${WORK:-/root/autodl-tmp/work}
cd "$WORK" || exit 1
mkdir -p logs runs reports
DATA="$WORK/data/processed/bin"
TOK="$WORK/data/processed/tokenizer/zh_bpe.model"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "FATAL: $*"; exit 1; }
gpu() { nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; }

stage_smoke() {
  log "=== smoke 20 steps (125M) ==="
  ( cd code/llm && python train.py --config 125m --data-dir "$DATA" --run-dir "$WORK/runs/smoke" \
      --micro-batch 16 --grad-accum 30 --max-steps 20 --warmup 5 \
      --eval-interval 20 --eval-batches 5 --save-interval 20 --log-interval 5 --compile ) \
    2>&1 | tee logs/smoke.log
  grep -E "tok/s|\[mem\]" logs/smoke.log | tail -4 || true
}

stage_resume() {
  log "=== resume determinism test ==="
  rm -rf "$WORK/runs/resume_ref" "$WORK/runs/resume_test"
  ( cd code/llm && python train.py --config 125m --data-dir "$DATA" --run-dir "$WORK/runs/resume_ref" \
      --micro-batch 8 --grad-accum 30 --max-steps 60 --warmup 4 --eval-interval 0 \
      --save-interval 10 --log-interval 1 ) > logs/resume_ref.log 2>&1 || die "ref run failed"
  log "ref done: $(tail -1 logs/resume_ref.log)"

  ( cd code/llm && timeout -s KILL 100 python train.py --config 125m --data-dir "$DATA" \
      --run-dir "$WORK/runs/resume_test" --micro-batch 8 --grad-accum 30 --max-steps 40 \
      --warmup 4 --eval-interval 0 --save-interval 10 --log-interval 1 ) > logs/resume_killed.log 2>&1
  log "killed at: $(tail -1 logs/resume_killed.log)"
  ls -l "$WORK/runs/resume_test/latest.pt" || die "no ckpt before kill"

  ( cd code/llm && python train.py --config 125m --data-dir "$DATA" --run-dir "$WORK/runs/resume_test" \
      --micro-batch 8 --grad-accum 30 --max-steps 60 --warmup 4 --eval-interval 0 \
      --save-interval 10 --log-interval 1 ) > logs/resume_after.log 2>&1 || die "resume run failed"
  log "resume done: $(tail -1 logs/resume_after.log)"

  ( cd code/llm && python verify_resume.py "$WORK/runs/resume_ref/train.csv" \
      "$WORK/runs/resume_test/train.csv" --tol 0.05 ) 2>&1 | tee logs/verify_resume.txt
}

stage_125m() {
  log "=== 125M pretraining ==="
  ( cd code/llm && python train.py --config 125m --data-dir "$DATA" --run-dir "$WORK/runs/125m" \
      --micro-batch 16 --grad-accum 30 --max-steps 699 --warmup 20 \
      --lr 6e-4 --min-lr-ratio 0.1 --weight-decay 0.1 \
      --eval-interval 100 --eval-batches 20 --save-interval 50 --log-interval 20 \
      --time-budget-min "${BUDGET:-170}" --compile \
      --tokenizer "$TOK" --sample-prompt "中国的首都是" ) 2>&1 | tee logs/125m.log
}

stage_360m() {
  log "=== 360M pretraining ==="
  ( cd code/llm && python train.py --config 360m --data-dir "$DATA" --run-dir "$WORK/runs/360m" \
      --micro-batch 8 --grad-accum 60 --max-steps 456 --warmup 24 \
      --lr 3e-4 --min-lr-ratio 0.1 --weight-decay 0.1 --grad-checkpoint \
      --eval-interval 76 --eval-batches 12 --save-interval 38 --log-interval 10 \
      --time-budget-min "${BUDGET:-300}" --compile \
      --tokenizer "$TOK" --sample-prompt "中国的首都是" ) 2>&1 | tee logs/360m.log
}

stage_sft() {
  log "=== SFT data (24k) ==="
  if [ ! -s "$WORK/data/sft/train.jsonl" ]; then
    ( cd code/llm && python build_sft_data.py --source firefly --n 24000 --out "$WORK/data/sft" ) \
      2>&1 | tee logs/sft_data.log || die "SFT data build failed"
  fi
  log "=== SFT training ==="
  ( cd code/llm && python sft.py --init "$WORK/runs/125m/latest.pt" --tokenizer "$TOK" \
      --train "$WORK/data/sft/train.jsonl" --val "$WORK/data/sft/val.jsonl" \
      --run-dir "$WORK/runs/sft" --batch 8 --grad-accum 4 --epochs 2 \
      --lr 1e-5 --warmup 50 --eval-interval 100 --save-interval 100 ) 2>&1 | tee logs/sft.log
}

case "${1:-all}" in
  smoke) stage_smoke ;;
  resume) stage_resume ;;
  125m) stage_125m ;;
  360m) stage_360m ;;
  sft) stage_sft ;;
  all) stage_smoke; stage_resume; stage_125m; stage_360m; stage_sft ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
log "stage ${1:-all} finished; $(gpu)"
