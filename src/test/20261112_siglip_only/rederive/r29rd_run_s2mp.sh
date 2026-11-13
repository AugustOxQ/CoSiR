#!/usr/bin/env bash
set -euo pipefail
cd /project/CoSiR/src/test/20261112_siglip_only/rederive
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
# The controller owns resource scheduling. CPU by default; pass --device cuda
# after the subcommand only when res-queue has assigned a GPU slot.
exec /root/miniconda3/envs/CoSiR/bin/python r29rd_s2mp.py "$@"
