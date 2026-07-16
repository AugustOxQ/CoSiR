#!/bin/bash
# Full held-out encoder grid: 6 encoders x 2 datasets, then score.
# DINOv2 is reused from existing caches; only 5 encoders actually extract.
# Heavy step = the two RedCaps vision passes (~22 min); everything else is minutes.
# Total ~30-40 min. Safe to re-run: existing caches are skipped (--force to redo).
#
#   bash src/test/20260708_heldout_grid/run_full.sh
#
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
cd "$(dirname "$0")"

LOG="run_full_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG"

for DS in impressions redcaps; do
  for M in dinov2 siglip_v vit_sup minilm bge e5; do
    echo "===== extract $DS / $M =====" | tee -a "$LOG"
    python extract_heldout.py --dataset "$DS" --model "$M" 2>&1 | tee -a "$LOG"
  done
done

for DS in impressions redcaps; do
  echo "===== score $DS =====" | tee -a "$LOG"
  python score_grid.py --dataset "$DS" 2>&1 | tee -a "$LOG"
done

echo "DONE. Fill the _tbd_ tables in docs/reports/2026-07-08_heldout_grid.md from:"
echo "  docs/reports/assets/heldout_grid/{impressions,redcaps}_grid.{json,png}"
