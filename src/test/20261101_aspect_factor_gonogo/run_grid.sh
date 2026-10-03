#!/usr/bin/env bash
# E3 training grid (CVPR plan Task 12): the 8 runs of PREREGISTRATION.md §3 at model seed 42, at most 3 at once.
# The controller launches it from the repo root, under the shared GPU lock:
#   flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261101_aspect_factor_gonogo/run_grid.sh
# Each run logs to logs/<RUN>_seed42.log. A run whose checkpoint already exists is skipped (run_gonogo.py refuses to
# overwrite it), so an identical relaunch after an infrastructure failure trains only the missing runs.
# Exits non-zero if any run failed or any checkpoint is missing at the end.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
SEED=42
RUNS="A1 A2 A3 A4 A5 A6 H1 S1"
PARALLEL=3
cd "$ROOT" || exit 1
mkdir -p "$HERE/logs"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4          # 3 runs at once: 12 CPU threads in all
export HERE PY SEED

one_run() {
  local run="$1"
  local ckpt="$HERE/checkpoints/${run}_seed${SEED}.pt"
  local log="$HERE/logs/${run}_seed${SEED}.log"
  if [ -e "$ckpt" ]; then
    echo "[$(date +%T)] $run: checkpoint exists, skipped"
    return 0
  fi
  [ -e "$log" ] && mv "$log" "$HERE/logs/${run}_seed${SEED}_prev$(date +%s).log"
  echo "[$(date +%T)] $run: start"
  "$PY" "$HERE/run_gonogo.py" --train "$run" --seed "$SEED" > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[$(date +%T)] $run: done ($(grep -v 'factor epoch' "$log" | grep 'steps in' | tail -1))"
    return 0
  fi
  echo "[$(date +%T)] $run: FAILED (exit $rc; see $log)"
  return 1
}
export -f one_run

echo "[$(date +%T)] grid start: $RUNS (at most $PARALLEL in parallel), GPU: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"
printf '%s\n' $RUNS | xargs -P "$PARALLEL" -I{} bash -c 'one_run "$1"' _ {}
status=$?
missing=""
for run in $RUNS; do
  [ -e "$HERE/checkpoints/${run}_seed${SEED}.pt" ] || missing="$missing $run"
done
if [ "$status" -ne 0 ] || [ -n "$missing" ]; then
  echo "[$(date +%T)] grid FAILED: xargs status $status; missing checkpoints:${missing:- none}"
  exit 1
fi
echo "[$(date +%T)] grid done: 8 checkpoints in $HERE/checkpoints"
