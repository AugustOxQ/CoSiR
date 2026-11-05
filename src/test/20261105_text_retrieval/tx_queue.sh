#!/usr/bin/env bash
# r22 explore: one queue job per step. Run from anywhere; each step waits in res-queue.
set -euo pipefail
R=/project/CoSiR/src/test/20261105_text_retrieval
PY=/root/miniconda3/envs/CoSiR/bin/python
L=/data/SSD2/cosir_r22/20261105_text_retrieval/logs
mkdir -p "$L"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$R"
for job in teacher mmart semart words; do
  res-queue gpu --short --label r22:text -- "$PY" tx_extract.py --job "$job" > "$L/extract_$job.log" 2>&1
done
res-queue gpu --est 45m --label r22:core -- "$PY" tx_core.py > "$L/core.log" 2>&1
res-queue gpu --short --label r22:extras -- "$PY" tx_extras.py --part all > "$L/extras.log" 2>&1
"$PY" tx_summary.py > "$L/summary.log" 2>&1
echo "r22 queue done: $(tail -1 "$L/summary.log")"
