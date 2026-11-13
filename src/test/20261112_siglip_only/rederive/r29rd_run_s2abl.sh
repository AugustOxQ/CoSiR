#!/usr/bin/env bash
set -euo pipefail
cd /project/CoSiR/src/test/20261112_siglip_only/rederive
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
exec /root/miniconda3/envs/CoSiR/bin/python -B r29rd_s2abl.py "$@"
