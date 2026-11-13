#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
job_dir=/project/CoSiR/src/test/20261112_siglip_only/rederive
mode=${1:-}
if [[ -z "$mode" ]]; then
  echo 'Usage: r29rd_run_s2e.sh {val|held|replay|test} [--device cpu|cuda] [script options]' >&2
  exit 2
fi
shift
cd "$job_dir"
if [[ "$mode" == test ]]; then
  exec /root/miniconda3/envs/CoSiR/bin/python -m unittest -v test_r29rd_s2e "$@"
fi
exec /root/miniconda3/envs/CoSiR/bin/python "$job_dir/r29rd_s2e.py" "$mode" "$@"
