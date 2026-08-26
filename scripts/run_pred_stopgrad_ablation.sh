#!/bin/bash
set -euo pipefail
# Experiment 11.3 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md,
# Experiment 11.3 subsection): does removing the stop-gradient on the condition-predictor
# distillation term (loss.pred_stopgrad=false) change test_oracle (the table's own held-out
# codebook quality) or test_pre_diff (the predictor's standalone usefulness vs. raw CLIP),
# relative to Experiment 11.1's existing trained/frozen arms?
#
# loss.pred_stopgrad is NOT a template-compatibility key (src/hook/train_cosir.py's _extra
# dict only reads k/alpha/method/b_weight/encoder_pair/distance_mode -- the same class of
# non-template-affecting knob as train.em_interval, per 11.1's own script). This script
# therefore points at the SAME results_dir as scripts/run_condition_freeze_ablation.sh so this
# arm's 3 runs load the exact same cached buddy-init template as 11.1's existing trained/frozen
# runs, and reuses the SAME wandb group so scripts/analyze_pred_stopgrad_ablation.py can pull
# all three arms from one query. Do NOT give this arm its own results_dir or wandb group --
# that would break the paired comparison against 11.1's already-completed runs.
#
# One new arm, added on top of 11.1's existing {trained, frozen}:
#   pred_coupled: em_interval=-1 (trained regime, same as 11.1's "trained" arm)
#                 + loss.pred_stopgrad=false (Experiment 11.3's bidirectional coupling)
#
#   SMOKE=1 bash scripts/run_pred_stopgrad_ablation.sh   # 2 epochs, seed=1
#   bash scripts/run_pred_stopgrad_ablation.sh           # full 3-run sweep

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
# Deliberately the SAME results_dir as scripts/run_condition_freeze_ablation.sh -- see header.
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_condition_freeze_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-pred-stopgrad-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-condition freeze ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, pred_coupled arm only -- pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Pred-stopgrad ablation ($DATASET): pred_coupled x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "  results_dir (SHARED with Experiment 11.1, by design): $RESULTS_DIR"
echo "==================================================================="

python main_cosir.py -m \
  dataset="$DATASET" \
  eval.evaluation_interval="$EVAL_INTERVAL" \
  eval.oracle_aggregation=max \
  eval.test_ratio="$TEST_RATIO" \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim="$EMBEDDING_DIM" \
  optimizer.lr="$LR_SWEEP" \
  optimizer.lr_label="$LR_LABEL_SWEEP" \
  seed="$SEED_SWEEP" \
  train.initialization_strategy=buddies \
  train.buddies.alpha="$ALPHA" \
  train.em_interval=-1 \
  train.epochs="$EPOCHS" \
  loss.pred_stopgrad=false \
  experiment.results_dir="$RESULTS_DIR" \
  wandb.group="$WANDB_GROUP" \
  +train.arm="pred_coupled" \
  ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
  ${EXTRA_OVERRIDES:-}

echo "==================================================================="
echo "Done. Analyse (pred_coupled vs 11.1's trained/frozen, mean delta +/- std) with:"
echo "  python scripts/analyze_pred_stopgrad_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
