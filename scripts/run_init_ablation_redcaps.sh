#!/bin/bash
set -euo pipefail
# Experiment 1 on RedCaps-150k — see scripts/run_init_ablation.sh for the mechanism.
# EPOCHS=100 (not 250) for the same documented reason as scripts/run_buddycon_redcaps.sh: a
# deadline-driven, NOT data-size-scaled schedule that can only shrink a true delta, never
# inflate one. TEST_RATIO=0.2 for the same eval-cost reason (redcaps_test is 25k pairs).
export DATASET="redcaps_150k"
# Only apply the dataset-specific defaults outside SMOKE mode — setting EPOCHS/EVAL_INTERVAL
# here unconditionally would pre-empt run_init_ablation.sh's own SMOKE=1 defaults (2 / 1),
# since ${VAR:-default} no-ops once VAR is already set.
if [ -z "${SMOKE:-}" ]; then
  export EPOCHS="${EPOCHS:-100}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi
export TEST_RATIO="${TEST_RATIO:-0.2}"
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/redcaps_150k}"
export WANDB_TAG="${WANDB_TAG:-init-ablation-redcaps}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_init_ablation.sh"
