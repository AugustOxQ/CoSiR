#!/usr/bin/env bash
# r14 queue, local GPU parts (ticket 12a): the GPU gates of stage 0 (extraction parity i, trainer checks iv and v, export
# check ix), under `flock -o -w` on /tmp/gpu0.lock (the lock is released when this script's child exits). Check first:
#   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# Usage: bash bs_run_gpu.sh [stage0 args...]   e.g. --only iv v   (default: every GPU gate). Not on weekdays 10:00-19:00
# Amsterdam time and not while the extraction queue holds the lock longer than BS_LOCK_WAIT (default 3600 s: then exit 1).
# One summary line: "QUEUE stage0-gpu OK|FAILED rc=<rc> seconds=<s>" (also appended to $OUT/queue_summary.txt).
# The X-c* and X photo training of part a is on node402: scripts/run_bs_node.sh (ticket 10), not here.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT="${BS_OUTPUTS:-/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs}"
mkdir -p "$OUT"
t0=$SECONDS
flock -o -w "${BS_LOCK_WAIT:-3600}" /tmp/gpu0.lock "$PY" "$HERE/bs_run.py" stage0 --parts gpu "$@"
rc=$?
line="QUEUE stage0-gpu $([[ $rc -eq 0 ]] && echo OK || echo FAILED) rc=$rc seconds=$((SECONDS - t0))"
echo "$line" | tee -a "$OUT/queue_summary.txt"
exit $rc
