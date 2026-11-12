#!/bin/bash
# r28 GPU stage: settings check rows (seed 0) and the four ablation removals (five seeds, with held exports) on
# SigLIP 1. Each step queues itself through res-queue; the reference c* is r14's and is not retrained.
set -u
cd /project/CoSiR
ROOT=/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1
PY=(env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python
    src/test/20261112_siglip_main/r28_trainer.py)
fails=()
for row in k11=10 k11=20 k11=80 k6=0.0005; do
  res-queue gpu --short --label r28:settings -- "${PY[@]}" train --encoder siglip1 --row "$row" --seeds 0 \
      --keep all --root "$ROOT" || fails+=("settings:$row")
done
for row in instructed_graphs content_correction buddy relational; do
  res-queue gpu --est 35m --label r28:ablation -- "${PY[@]}" train --encoder siglip1 --row "$row" \
      --seeds 0 1 2 3 4 --keep all --root "$ROOT" || { fails+=("train:$row"); continue; }
  res-queue gpu --short --label r28:ablation-export -- "${PY[@]}" export --encoder siglip1 --row "$row" \
      --seeds 0 1 2 3 4 --kind held --root "$ROOT" || fails+=("export:$row")
done
echo "r28 trainer stage done $(TZ=Europe/Amsterdam date '+%F %H:%M'); failed steps: ${fails[*]:-none}"
