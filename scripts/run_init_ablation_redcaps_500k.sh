#!/bin/bash
set -euo pipefail
# Experiment 1 on RedCaps-500k (first 500,000 rows of redcaps_train.json) — see
# scripts/run_init_ablation_redcaps_300k.sh for the pattern this mirrors.
export DATASET="redcaps_full"
if [ -z "${SMOKE:-}" ]; then
  export EPOCHS="${EPOCHS:-100}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi
export TEST_RATIO="${TEST_RATIO:-0.2}"
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/redcaps_500k}"
export WANDB_TAG="${WANDB_TAG:-init-ablation-redcaps-500k}"
export EXTRA_OVERRIDES="data.train_annotation_path=/data/PDD/redcaps/redcaps_plus/redcaps_train_500000.json featuremanager.storage_dir=/data/SSD2/pre_extract/redcaps_500k/features ${EXTRA_OVERRIDES:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_init_ablation.sh"
