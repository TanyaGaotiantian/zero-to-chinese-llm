#!/usr/bin/env bash
# Sequential unattended chain: wait for 125M -> run 360M -> run SFT.
set -uo pipefail
cd /root/autodl-tmp/work || exit 1
log() { echo "[$(date '+%F %T')] $*"; }

log "chain start; waiting for 125m to finish"
while pgrep -f "train[.]py --config 125m" > /dev/null; do sleep 20; done
log "125m done -> 360m"
./stages.sh 360m

log "360m done -> sft"
while pgrep -f "train[.]py --config 360m" > /dev/null; do sleep 20; done
./stages.sh sft
log "chain finished"
