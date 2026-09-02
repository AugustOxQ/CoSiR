#!/bin/bash
set -euo pipefail
# Combiner-architecture ablation Phase 5 (.planning/2026-09-01-combiner-architecture-ablation/task_plan.md):
# buddy_dim sweep for the Phase 4 winner (lowrank, rank=16) at the same fixed operating point
# (K=30, alpha=0.5, redcaps_150k). residual_control/film were dropped from the main track after
# Phase 4 (reproducible oracle-vs-predicted-condition fragility, not a capacity question a
# buddy_dim sweep would address) — see findings.md 2026-09-02 entry.
#
# embedding_dim (buddy_dim) IS a template-compatibility key, same as buddies.k/alpha/method: the
# per-sample buddy vector is a spectral embedding of the buddy graph into `embedding_dim`
# dimensions, so each dimension needs its own results_dir/template, same pattern as
# scripts/run_buddy_k_ablation.sh's K axis.
#
#   SMOKE=1 bash scripts/run_combiner_buddydim_sweep.sh   # 2 epochs, one dim, pipeline sanity
#   bash scripts/run_combiner_buddydim_sweep.sh           # full sweep, all 3 dims x 3 seeds

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Template-key axis (bash loop; each buddy_dim gets its own results_dir/template) ──
DIM_SWEEP="${DIM_SWEEP:-8 16 32}"

# ── Non-template axis (Hydra multirun; reuses each dim's template) ────────────
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

# ── Fixed operating point (matches Phase 4 smoke sweep, lowrank winner) ───────
COMBINER_TYPE="${COMBINER_TYPE:-lowrank}"
DATASET="${DATASET:-redcaps_150k}"
K="${K:-30}"
ALPHA="${ALPHA:-0.5}"
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
TEST_RATIO="${TEST_RATIO:-0.2}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_combiner_buddydim_sweep}"
WANDB_GROUP="${WANDB_GROUP:-combiner buddydim sweep}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  DIM_SWEEP="${DIM_SWEEP_SMOKE:-16}"
  echo ">>> SMOKE: 2 epochs, seed=1, buddy_dim=${DIM_SWEEP}, pipeline sanity only"
else
  EPOCHS="${EPOCHS:-30}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
fi

echo "==================================================================="
echo "Combiner buddy_dim sweep ($DATASET, combiner_type=$COMBINER_TYPE): {$DIM_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: K=$K alpha=$ALPHA lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL group=$WANDB_GROUP"
echo "==================================================================="

for DIM in $DIM_SWEEP; do
  RD="${BASE_RESULTS_DIR}/${COMBINER_TYPE}_dim_${DIM}"
  TAG="combiner-buddydim-${COMBINER_TYPE}-dim${DIM}"
  echo ">>> buddy_dim=${DIM}  ->  results_dir=${RD}"
  python main_cosir.py -m \
    dataset="$DATASET" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    eval.oracle_aggregation=max \
    eval.test_ratio="$TEST_RATIO" \
    model=clip_base \
    model.num_layers=6 \
    model.embedding_dim="$DIM" \
    model.combiner_type="$COMBINER_TYPE" \
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
echo "Done. Analyse per-buddy_dim retrieval with:"
echo "  python scripts/analyze_combiner_buddydim_sweep.py"
echo "==================================================================="
