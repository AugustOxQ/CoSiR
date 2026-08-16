#!/bin/bash
set -euo pipefail
# Experiment 1 on Impressions — see scripts/run_init_ablation.sh for the mechanism.
# 250 epochs matches the Impressions operating point used throughout the family sweeps
# (e.g. scripts/run_buddy_seeds.sh).
export DATASET="impressions"
# Only apply the dataset-specific defaults outside SMOKE mode — setting EPOCHS/EVAL_INTERVAL
# here unconditionally would pre-empt run_init_ablation.sh's own SMOKE=1 defaults (2 / 1),
# since ${VAR:-default} no-ops once VAR is already set.
if [ -z "${SMOKE:-}" ]; then
  export EPOCHS="${EPOCHS:-250}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
fi
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/impressions}"
export WANDB_TAG="${WANDB_TAG:-init-ablation-impressions}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_init_ablation.sh"
