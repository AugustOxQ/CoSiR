#!/bin/bash
set -euo pipefail
# Experiment 16.2 on RedCaps-300k's corrected _diverse training store — see
# scripts/run_buddy_k_ablation.sh for the mechanism. redcaps_full supplies the shared test
# set/image paths; EXTRA_OVERRIDES replaces only its training annotation and feature store.
export DATASET="redcaps_full"
export K_SWEEP="${K_SWEEP:-30 35 50}"
if [ -z "${SMOKE:-}" ]; then
  export EPOCHS="${EPOCHS:-100}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi
export TEST_RATIO="${TEST_RATIO:-0.2}"
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_buddy_k_ablation/redcaps_300k_diverse}"
export WANDB_TAG="${WANDB_TAG:-buddy-k-ablation-redcaps_300k_diverse}"
export EXTRA_OVERRIDES="data.train_annotation_path=/data/PDD/redcaps/redcaps_plus/redcaps_300k_diverse.json featuremanager.storage_dir=/data/SSD2/pre_extract/redcaps_300k_diverse/features ${EXTRA_OVERRIDES:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_buddy_k_ablation.sh"
