#!/usr/bin/env bash
set -euo pipefail
cd /project/CoSiR/src/test/20261112_siglip_only/rederive
exec env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/CoSiR/bin/python \
  /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_sc1.py \
  > /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_full.log 2>&1
