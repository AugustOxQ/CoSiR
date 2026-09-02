#!/bin/bash
set -euo pipefail
# Combiner-architecture ablation Phase 4 (.planning/2026-09-01-combiner-architecture-ablation/task_plan.md):
# small-scale smoke sweep comparing fusion families at fixed buddy_dim=16, to pick a winner
# (or narrow to 2) before the real buddy_dim/rank/depth sweeps in Phase 5-6. Not paper numbers.
#
# combiner_type is NOT a template-compatibility key (unlike buddies.k/alpha/method): the buddy
# graph init (K, alpha) is fixed and shared across all four arms, so one buddy-init template
# covers the whole sweep. Only the neural fusion module (model.combiner_type) varies.
#
#   SMOKE=1 bash scripts/run_combiner_architecture_smoke.sh   # 2 epochs, one arm, pipeline sanity
#   bash scripts/run_combiner_architecture_smoke.sh           # full smoke sweep, all 4 arms

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Architecture axis (bash loop; each combiner_type gets its own results_dir) ─
COMBINER_SWEEP="${COMBINER_SWEEP:-legacy residual_control lowrank film}"

# ── Fixed operating point (matches Experiment 16.2's K=30 anchor) ─────────────
DATASET="${DATASET:-redcaps_150k}"
K="${K:-30}"
ALPHA="${ALPHA:-0.5}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
SEED_SWEEP="${SEED_SWEEP:-1}"
TEST_RATIO="${TEST_RATIO:-0.2}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_combiner_architecture_smoke}"
WANDB_GROUP="${WANDB_GROUP:-combiner architecture smoke}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  COMBINER_SWEEP="${COMBINER_SWEEP_SMOKE:-lowrank}"
  echo ">>> SMOKE: 2 epochs, combiner_type=${COMBINER_SWEEP}, pipeline sanity only"
else
  EPOCHS="${EPOCHS:-30}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
fi

echo "==================================================================="
echo "Combiner architecture smoke sweep ($DATASET): {$COMBINER_SWEEP}"
echo "  fixed: K=$K alpha=$ALPHA buddy_dim=$EMBEDDING_DIM lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP seed=$SEED_SWEEP"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL group=$WANDB_GROUP"
echo "==================================================================="

for COMBINER in $COMBINER_SWEEP; do
  RD="${BASE_RESULTS_DIR}/${COMBINER}"
  TAG="combiner-arch-smoke-${COMBINER}"
  echo ">>> combiner_type=${COMBINER}  ->  results_dir=${RD}"
  python main_cosir.py -m \
    dataset="$DATASET" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    eval.oracle_aggregation=max \
    eval.test_ratio="$TEST_RATIO" \
    model=clip_base \
    model.num_layers=6 \
    model.embedding_dim="$EMBEDDING_DIM" \
    model.combiner_type="$COMBINER" \
    optimizer.lr="$LR_SWEEP" \
    optimizer.lr_label="$LR_LABEL_SWEEP" \
    seed="$SEED_SWEEP" \
    train.initialization_strategy=buddies \
    train.buddies.alpha="$ALPHA" \
    train.buddies.k="$K" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ++wandb.tags=[$TAG]
done

echo "==================================================================="
echo "Done. Analyse per-combiner_type retrieval with:"
echo "  python scripts/analyze_combiner_architecture_smoke.py"
echo "==================================================================="
