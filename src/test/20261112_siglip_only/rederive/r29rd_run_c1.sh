#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: r29rd_run_c1.sh validate|replay|held [--device cpu|cuda] [other r29rd_c1.py options]' >&2
  exit 2
fi
r29rd_mode="$1"
shift
case "$r29rd_mode" in validate|replay|held) ;; *) echo 'Expected validate, replay or held' >&2; exit 2 ;; esac
cd /project/CoSiR/src/test/20261112_siglip_only/rederive
exec env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  CUDA_CACHE_PATH=/project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_c1_cuda_cache \
  /root/miniconda3/envs/CoSiR/bin/python \
  /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_c1.py "$r29rd_mode" "$@" \
  >> "/project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_c1_${r29rd_mode}.log" 2>&1
