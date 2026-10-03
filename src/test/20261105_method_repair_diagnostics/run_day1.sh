#!/usr/bin/env bash
# Day-1 trainings of the method-repair diagnostics (PREREGISTRATION.md §4). The CONTROLLER runs this under the GPU lock:
#   flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261105_method_repair_diagnostics/run_day1.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=/root/miniconda3/envs/CoSiR/bin/python
DIR=src/test/20261105_method_repair_diagnostics
mkdir -p "$DIR/results"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4      # four processes share the 32 cores with other sessions
for run in L3 L5 LT MK3; do
  $PY "$DIR/train_runs.py" --train "$run" --seed 42 > "$DIR/results/train_${run}_seed42.log" 2>&1 &
done
wait
echo "day-1 trainings done"
