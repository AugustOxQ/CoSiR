#!/bin/bash
set -euo pipefail
# Experiment 1 on RedCaps-300k (first 300,000 rows of redcaps_train.json) — see
# scripts/run_init_ablation.sh for the mechanism. Same operating point and
# epoch/interval schedule as scripts/run_init_ablation_redcaps.sh (the 150k
# version) for direct comparability; only the training-set size, annotation
# slice, and feature-store location differ. Shares dataset=redcaps_full's
# test set/image paths (data.test_annotation_path etc.) via EXTRA_OVERRIDES on
# top of that dataset group, rather than a dedicated dataset=redcaps_300k
# config — the only thing that changes is the training slice.
export DATASET="redcaps_full"
if [ -z "${SMOKE:-}" ]; then
  export EPOCHS="${EPOCHS:-100}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi
export TEST_RATIO="${TEST_RATIO:-0.2}"
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/redcaps_300k}"
export WANDB_TAG="${WANDB_TAG:-init-ablation-redcaps-300k}"
export EXTRA_OVERRIDES="data.train_annotation_path=/data/PDD/redcaps/redcaps_plus/redcaps_train_300000.json featuremanager.storage_dir=/data/SSD2/pre_extract/redcaps_300k/features ${EXTRA_OVERRIDES:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_init_ablation.sh"
