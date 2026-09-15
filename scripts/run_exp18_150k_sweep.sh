#!/bin/bash
# scripts/run_exp18_150k_sweep.sh
# Experiment 18: architecture-fix-only baseline vs. prototype-conditioning arm.
# One arm per DAS6 node (run concurrently): node411 = baseline, node412 = prototype_pooled.
#
#   ARM=baseline bash scripts/run_exp18_150k_sweep.sh         # on node411
#   ARM=prototype_pooled bash scripts/run_exp18_150k_sweep.sh # on node412
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="${ARM:?Set ARM=baseline or ARM=prototype_pooled}"
DATASET="${DATASET:-redcaps_150k_cluster}"
SEED_SWEEP="${SEED_SWEEP:-1 2 3}"
EPOCHS="${EPOCHS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-16}"  # Task 6's chosen value — override if smoke test picked differently
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-/local/wding/res/CoSiR_Experiment/exp18_${ARM}}"
WANDB_GROUP="${WANDB_GROUP:-exp18 buddy prototype conditioning}"

if [ "$ARM" = "baseline" ]; then
  CONDITIONING_MODE="free_vector"
  COMBINER_TYPE="lowrank"
elif [ "$ARM" = "prototype_pooled" ]; then
  CONDITIONING_MODE="prototype_pooled"
  COMBINER_TYPE="lowrank"
else
  echo "ARM must be 'baseline' or 'prototype_pooled', got '$ARM'" >&2
  exit 1
fi

for SEED in $SEED_SWEEP; do
  RD="${BASE_RESULTS_DIR}/seed${SEED}"
  TAG="exp18-150k-${ARM}-seed${SEED}"
  echo ">>> arm=${ARM} seed=${SEED} -> results_dir=${RD}"
  python main_cosir.py \
    dataset="$DATASET" \
    model=clip_base \
    model.combiner_type="$COMBINER_TYPE" \
    model.conditioning_mode="$CONDITIONING_MODE" \
    model.num_prototypes="$NUM_PROTOTYPES" \
    train.initialization_strategy=buddies \
    train.epochs="$EPOCHS" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    seed="$SEED" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    ++wandb.tags=[$TAG]
done
