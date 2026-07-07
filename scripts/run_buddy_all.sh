#!/bin/bash
set -euo pipefail
# Fire all THREE buddy-family focused ablations in one shot, for a unified wandb view.
#
# Each family keeps its own wandb GROUP (buddy-{reg,con,refresh} ablation) so
# scripts/analyze_buddy_families.py can filter by group; the shared WANDB_TAG lets you
# select all runs at once in the wandb workspace (filter by tag → see all; group-by
# 'group' → split by family).
#
# Every arm — including the no-term baselines — logs buddy_knn_preservation@k at eval
# cadence: the cross-family "is the buddy neighbourhood surviving into retrieval space?"
# metric. That makes the baseline arm the reference and each family's active arm the test.
#
# Defaults are tuned for a 250-epoch FIRST LOOK on Impressions. Override any knob:
#   EPOCHS=500 EVAL_INTERVAL=100 bash scripts/run_buddy_all.sh
#   WANDB_TAG=my-run LOG_BUDDY_PRESERVATION= bash scripts/run_buddy_all.sh   # (empty = off)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Shared knobs (exported so the child runners inherit them) ────────────────
export EPOCHS="${EPOCHS:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
export WANDB_TAG="${WANDB_TAG:-buddy-families-250ep}"
export LOG_BUDDY_PRESERVATION="${LOG_BUDDY_PRESERVATION:-1}"   # non-empty = on; empty = off

# #3's schedule must fit the epoch budget or refresh never fires (blend=1 ≡ blend=0).
# At EPOCHS=250 the defaults (warmup=50, period=50) refresh at 50/100/150/200 — aligned
# with EVAL_INTERVAL=50 so every refreshed graph gets evaluated.
export BUDDY_REFRESH_WARMUP="${BUDDY_REFRESH_WARMUP:-50}"
export BUDDY_REFRESH_PERIOD="${BUDDY_REFRESH_PERIOD:-50}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==================================================================="
echo "Buddy families ablation"
echo "  EPOCHS=$EPOCHS  EVAL_INTERVAL=$EVAL_INTERVAL  tag=$WANDB_TAG"
echo "  preservation=${LOG_BUDDY_PRESERVATION:-<off>}  refresh warmup/period=$BUDDY_REFRESH_WARMUP/$BUDDY_REFRESH_PERIOD"
echo "==================================================================="

echo ">>> Family #3  (self-refresh: static blend=0  vs  live blend=1.0)"
bash "$HERE/run_buddyrefresh_full.sh"

echo ">>> Family #2  (contrastive: lambda_buddy_con 0  vs  0.3)"
bash "$HERE/run_buddycon_full.sh"

echo ">>> Family #1  (smoothness: lambda_buddy 0  vs  {0.1,0.3,1.0})"
bash "$HERE/run_buddyreg_full.sh"

echo "==================================================================="
echo "All three families launched (8 runs). Analyse with:"
echo "  python scripts/analyze_buddy_families.py --entity augustoxq --project <your-wandb-project>"
echo "==================================================================="
