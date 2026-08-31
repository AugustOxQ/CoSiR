#!/bin/bash
set -euo pipefail
# Experiment 1 on RedCaps-full (dataset=redcaps_full, ~3.1M training pairs) — see
# scripts/run_init_ablation.sh for the mechanism. Same operating point and epoch/interval
# schedule as scripts/run_init_ablation_redcaps.sh (the 150k version) for direct
# comparability; only the training-set size and feature-store location differ
# (configs/dataset/redcaps_full.yaml -> /data/SSD2/pre_extract/redcaps_full/features).
#
# First invocation (any strategy) triggers a from-scratch feature extraction over ~3.1M
# image/caption pairs (src/hook/train_cosir.py:_extract_or_load_features) — expect this to
# take hours, not minutes. Subsequent invocations reuse the cached feature store.
export DATASET="redcaps_full"
if [ -z "${SMOKE:-}" ]; then
  export EPOCHS="${EPOCHS:-100}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi
export TEST_RATIO="${TEST_RATIO:-0.2}"
export BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/redcaps_full}"
export WANDB_TAG="${WANDB_TAG:-init-ablation-redcaps-full}"

HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_init_ablation.sh"
