#!/bin/bash
# scripts/run_exp18_prototype_fix_3seed.sh
# Experiment 18 follow-up: 3-seed confirmation sweep of the winning combo from
# run_exp18_prototype_fix_screen.sh's 3x2 single-seed grid — temperature_init=0.3,
# lr_prototype=1e-2 — the only combo to break past 1 effective PCA dimension, with
# the highest silhouette (0.559) and the first-ever positive register-axis probe
# selectivity (0.200) for prototype_pooled across this whole investigation.
#
# Seed 1 for this exact combo is already done (it's the screen's own
# temp0.3_lrproto1e-2 run) — this script only runs the two remaining seeds, split
# one per node:
#   SEED=2 bash scripts/run_exp18_prototype_fix_3seed.sh   # node411
#   SEED=3 bash scripts/run_exp18_prototype_fix_3seed.sh   # node412
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="${DATASET:-redcaps_150k_cluster}"
SEED="${SEED:?Set SEED=2 or SEED=3 (seed=1 for this combo already exists from the screen sweep)}"
EPOCHS="${EPOCHS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-16}"
TEMP="${TEMP:-0.3}"
LR_PROTO="${LR_PROTO:-1e-2}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-/local/wding/res/CoSiR_Experiment/exp18_prototype_fix_3seed}"
WANDB_GROUP="${WANDB_GROUP:-exp18 prototype fix 3seed}"

RD="${BASE_RESULTS_DIR}/seed${SEED}"
TAG="exp18-protofix-3seed-seed${SEED}"
echo ">>> temp=${TEMP} lr_prototype=${LR_PROTO} seed=${SEED} -> results_dir=${RD}"
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
