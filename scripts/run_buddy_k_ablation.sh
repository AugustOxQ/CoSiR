#!/bin/bash
set -euo pipefail
# Experiment 16.2 Stage B (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does the K selected by Stage A's strict/union buddy-graph diagnostic change downstream
# retrieval, holding the buddy-init operating point fixed? This is the trained buddies arm
# only: no imgtxt baseline arm and no train.em_interval override.
#
# train.buddies.k is a TEMPLATE-COMPATIBILITY key (see src/hook/train_cosir.py:244-271),
# alongside alpha/method: a template built at one K is rejected and rebuilt at another. Give
# each K its OWN results_dir/template_embeddings/ to avoid concurrent Hydra multiruns racing
# on a template directory. Thus K is a bash-loop template-key axis, exactly like
# initialization_strategy in scripts/run_init_ablation.sh; seed is the inner Hydra multirun
# non-template axis.
#
#   SMOKE=1 bash scripts/run_buddy_k_ablation.sh   # 2 epochs, seed=1, all K values
#   bash scripts/run_buddy_k_ablation.sh           # full K sweep, 3 seeds each
#
# Normally called by the per-scale RedCaps wrappers, which set DATASET/K_SWEEP/
# BASE_RESULTS_DIR/WANDB_TAG and EXTRA_OVERRIDES for their corrected _diverse annotation and
# feature store. Safe to call directly for an ad-hoc rerun.
#
# EXTRA_OVERRIDES: raw space-separated Hydra overrides appended to every invocation, for
# wrappers that use the redcaps_full dataset group while replacing its training annotation and
# feature store with a scale-specific _diverse store.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Template-key axis (bash loop; each K = its own template + results_dir) ─────
K_SWEEP="${K_SWEEP:-30 35 50}"

# ── Non-template axis (Hydra multirun; reuses each K's template) ───────────────
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

# ── Fixed operating point (C5/C6/C9 matched cell) ─────────────────────────────
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

# ── Dataset + storage (set by per-scale wrappers; sane standalone defaults) ───
DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_buddy_k_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-buddy-k-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-buddy K ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, all K values — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Buddy K ablation ($DATASET): {$K_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  trained arm only: no train.em_interval override"
echo "==================================================================="

for K in $K_SWEEP; do
  RD="${BASE_RESULTS_DIR}/k_${K}"
  echo ">>> buddies.k=${K}  ->  results_dir=${RD}"
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
    train.buddies.k="$K" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${EXTRA_OVERRIDES:-}
done

echo "==================================================================="
echo "Done. Analyse per-(scale, K) means and paired deltas from K=30 with:"
echo "  python scripts/analyze_buddy_k_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
