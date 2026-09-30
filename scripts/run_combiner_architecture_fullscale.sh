#!/bin/bash
set -euo pipefail
# Full-scale validation of the combiner-architecture ablation's headline result
# (docs/reports/auto/buddy/2026-09-02_combiner_architecture_ablation.md): does the `lowrank`
# fusion family's oracle i2t R1 win over `legacy` survive at the project's real
# training scale/duration, not just the redcaps_150k/30-epoch smoke pass?
#
# Runs on DAS6 node411 via the cluster-run skill. Same operating point as the smoke
# sweep (K=30, alpha=0.5, buddy_dim=16) but redcaps_500k_diverse / 100 epochs, 1 seed
# per arm (pipeline + first-signal pass, not the smoke sweep's n=3 rigor).
#
#   SMOKE=1 bash scripts/run_combiner_architecture_fullscale.sh   # 2 epochs, one arm
#   bash scripts/run_combiner_architecture_fullscale.sh           # full run, both arms

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMBINER_SWEEP="${COMBINER_SWEEP:-legacy lowrank}"

DATASET="${DATASET:-redcaps_500k_diverse_cluster}"
K="${K:-30}"
ALPHA="${ALPHA:-0.5}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
SEED_SWEEP="${SEED_SWEEP:-1}"
TEST_RATIO="${TEST_RATIO:-0.2}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-/local/wding/res/CoSiR_combiner_architecture_fullscale}"
WANDB_GROUP="${WANDB_GROUP:-combiner architecture fullscale}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  COMBINER_SWEEP="${COMBINER_SWEEP_SMOKE:-lowrank}"
  echo ">>> SMOKE: 2 epochs, combiner_type=${COMBINER_SWEEP}, pipeline sanity only"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Combiner architecture full-scale validation ($DATASET): {$COMBINER_SWEEP}"
echo "  fixed: K=$K alpha=$ALPHA buddy_dim=$EMBEDDING_DIM lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP seed=$SEED_SWEEP"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL group=$WANDB_GROUP"
echo "==================================================================="

for COMBINER in $COMBINER_SWEEP; do
  RD="${BASE_RESULTS_DIR}/${COMBINER}"
  TAG="combiner-arch-fullscale-500k-${COMBINER}"
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
echo "Done. Pull results with ~/.claude/skills/cluster-run/cluster pull, then analyse with:"
echo "  python scripts/analyze_combiner_architecture_smoke.py --group '${WANDB_GROUP}'"
echo "==================================================================="
