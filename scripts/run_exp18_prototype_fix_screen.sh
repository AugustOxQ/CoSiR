#!/bin/bash
# scripts/run_exp18_prototype_fix_screen.sh
# Experiment 18 follow-up: single-seed 3x2 screen of the two candidate fixes
# for the prototype-pooled collapse (1 effective PCA dim, near-null probe
# selectivity, argmax concentration on 1-2 prototypes in all 3 seeds):
#   - lr_prototype: prototype_bank's own optimizer lr (was tied to the base
#     model lr=1e-5; free_vector's condition table gets lr_label=1e-2)
#   - prototype_temperature_init: softmax temperature at init (was hardcoded
#     to 1.0, not previously configurable)
#
# One node holds TEMP fixed and sweeps LR_PROTO_SWEEP; run the other node
# with a different TEMP to cover the 3x2 grid across both nodes:
#   TEMP=1.0 bash scripts/run_exp18_prototype_fix_screen.sh   # node411
#   TEMP=0.3 bash scripts/run_exp18_prototype_fix_screen.sh   # node412
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="${DATASET:-redcaps_150k_cluster}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-16}"
TEMP="${TEMP:?Set TEMP=1.0 or TEMP=0.3 (prototype_temperature_init for this node)}"
LR_PROTO_SWEEP="${LR_PROTO_SWEEP:-1e-5 1e-3 1e-2}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-/local/wding/res/CoSiR_Experiment/exp18_prototype_fix_screen}"
WANDB_GROUP="${WANDB_GROUP:-exp18 prototype fix screen}"

for LR_PROTO in $LR_PROTO_SWEEP; do
  RD="${BASE_RESULTS_DIR}/temp${TEMP}_lrproto${LR_PROTO}"
  TAG="exp18-protofix-temp${TEMP}-lrproto${LR_PROTO}"
  echo ">>> temp=${TEMP} lr_prototype=${LR_PROTO} -> results_dir=${RD}"
  python main_cosir.py \
    dataset="$DATASET" model=clip_base \
    model.combiner_type=lowrank \
    model.conditioning_mode=prototype_pooled \
    model.num_prototypes="$NUM_PROTOTYPES" \
    model.prototype_temperature_init="$TEMP" \
    optimizer.lr_prototype="$LR_PROTO" \
    train.initialization_strategy=buddies \
    train.epochs="$EPOCHS" eval.evaluation_interval="$EVAL_INTERVAL" \
    seed="$SEED" experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" ++wandb.tags=[$TAG]
done
