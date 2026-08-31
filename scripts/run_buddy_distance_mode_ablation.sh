#!/bin/bash
set -euo pipefail
# Experiment 10 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does modality-provenance-aware distance mixing ("typed") beat the current fixed-alpha
# blend ("blend") on retrieval, and does it narrow C6's still-open gap to raw CLIP
# (test_pre_diff)? Same operating point and isolation discipline as Experiment 1:
# initialization_strategy=buddies fixed, all training-time buddy terms OFF, only the
# init-construction's distance_mode varies.
#
# train.buddies.distance_mode is a TEMPLATE-COMPATIBILITY key, exactly like
# initialization_strategy/encoder_pair: each mode gets its OWN results_dir so its own
# template_embeddings/, avoiding template-reuse races.
#
#   SMOKE=1 bash scripts/run_buddy_distance_mode_ablation.sh   # 2 epochs, seed=1, both modes
#   bash scripts/run_buddy_distance_mode_ablation.sh           # full sweep (6 runs)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DISTANCE_MODE_SWEEP="${DISTANCE_MODE_SWEEP:-blend typed}"
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_buddy_distance_mode_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-buddy-distance-mode-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-buddy distance-mode ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, both modes — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Buddy distance-mode ablation ($DATASET): {$DISTANCE_MODE_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "==================================================================="

for MODE in $DISTANCE_MODE_SWEEP; do
  RD="${BASE_RESULTS_DIR}/mode_${MODE}"
  echo ">>> distance_mode=${MODE}  ->  results_dir=${RD}"
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
    +train.buddies.distance_mode="$MODE" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${EXTRA_OVERRIDES:-}
done

echo "==================================================================="
echo "Done. Analyse (paired, typed vs. blend, mean delta +/- std, plus test_pre_diff gap-to-CLIP) with:"
echo "  python scripts/analyze_buddy_distance_mode_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
