#!/bin/bash
set -euo pipefail
# Experiment 1 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does buddy-graph spectral initialization actually beat the prior generic (imgtxt) init on
# retrieval, with every training-time buddy term held OFF (lambda_buddy=0, lambda_buddy_con=0,
# buddy_refresh=False — all default/absent, never added as an override below)?
#
# initialization_strategy is a TEMPLATE-COMPATIBILITY key (see
# src/hook/train_cosir.py:244-271): a template built under one strategy is REJECTED and
# silently rebuilt under another, so sharing one results_dir across strategies would still be
# "correct" but risks two multirun processes racing on the same template_embeddings/ dir. We
# avoid that entirely by giving each strategy its OWN results_dir, exactly like the b_weight
# sweep does in scripts/run_blean_impressions.sh — a bash loop over the template-key axis
# (strategy), with an inner Hydra multirun over the non-template axis (seed).
#
#   SMOKE=1 DATASET=impressions bash scripts/run_init_ablation.sh   # 2 epochs, seed=1, both strategies
#   DATASET=impressions bash scripts/run_init_ablation.sh           # full sweep for one dataset
#
# Normally called by the per-dataset wrappers (run_init_ablation_impressions.sh,
# run_init_ablation_redcaps.sh), which set DATASET/EPOCHS/EVAL_INTERVAL/TEST_RATIO/
# BASE_RESULTS_DIR/WANDB_TAG. Safe to call directly for ad-hoc reruns.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Template-key axis (bash loop; each value = its own template + results_dir) ──
INIT_STRATEGY_SWEEP="${INIT_STRATEGY_SWEEP:-imgtxt buddies}"

# ── Non-template axis (Hydra multirun; reuses each strategy's template) ─────────
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

# ── Fixed operating point (confirmed strong cell from the family sweeps) ────────
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

# ── Dataset + storage (set by the per-dataset wrapper; sane standalone defaults) ─
DATASET="${DATASET:-impressions}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_init_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-init-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-buddy-init ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, both strategies — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
fi

echo "==================================================================="
echo "Init-strategy ablation ($DATASET): {$INIT_STRATEGY_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "==================================================================="

for STRAT in $INIT_STRATEGY_SWEEP; do
  RD="${BASE_RESULTS_DIR}/init_${STRAT}"
  echo ">>> initialization_strategy=${STRAT}  ->  results_dir=${RD}"
  python main_cosir.py -m \
    dataset="$DATASET" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    eval.oracle_aggregation=max \
    model=clip_base \
    model.num_layers=6 \
    model.embedding_dim="$EMBEDDING_DIM" \
    optimizer.lr="$LR_SWEEP" \
    optimizer.lr_label="$LR_LABEL_SWEEP" \
    seed="$SEED_SWEEP" \
    train.initialization_strategy="$STRAT" \
    train.buddies.alpha="$ALPHA" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ${TEST_RATIO:+eval.test_ratio=$TEST_RATIO} \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]}
done

echo "==================================================================="
echo "Done. Analyse (paired within seed, imgtxt vs buddies, mean delta +/- std) with:"
echo "  python scripts/analyze_init_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
