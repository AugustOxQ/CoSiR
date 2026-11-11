#!/usr/bin/env bash
# r26 GPU stage 1: the two extractions through res-queue (ModernBERT first: Stage 1 needs it). One summary line at the end.
set -u
cd "$(dirname -- "${BASH_SOURCE[0]}")"
logs=/data/SSD2/cosir_r26/20261111_percept_original/outputs/logs
py=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
res-queue gpu --est 10m --label r26:extract_mb -- "$py" -u po_extract.py modernbert --device cuda > "$logs/extract_mb.log" 2>&1; mb=$?
res-queue gpu --est 20m --label r26:extract_patch -- "$py" -u po_extract.py clip_patch --device cuda > "$logs/extract_patch.log" 2>&1; patch=$?
echo "R26_EXTRACT modernbert=$mb clip_patch=$patch $(TZ=Europe/Amsterdam date '+%F %H:%M')"
