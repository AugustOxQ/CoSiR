#! /bin/bash
set -euo pipefail

# QUICK B-lean (b_weight) sweep on Impressions — "does leaning the init on strict-B help?"
# Built from the Family #2 (buddy CONTRASTIVE) setup (scripts/run_buddycon_full.sh), adding
# an outer sweep over train.buddies.b_weight (the spectral-affinity multiplier for strict
# intersection buddies; 1.0 = today's union-only init, >1 pulls strict buddies tighter).
#
# WHY a bash loop, not a Hydra multirun, over b_weight: b_weight is a buddy-INIT/template
# key. Each value must rebuild its own template, so each gets its OWN results_dir. A single
# multirun would share one template_embeddings/ and silently reuse it → the sweep would be a
# no-op. Within each b_weight, Family #2's lambda_buddy_con axis (0 vs >0) is NOT a template
# key, so it reuses that b_weight's template (clean term ablation), exactly as run_buddycon_full.
#
#   SMOKE=1 bash scripts/run_blean_impressions.sh   # 2 epochs, bw=1.0 & con=0 — pipeline + template-rebuild sanity
#   bash scripts/run_blean_impressions.sh           # the quick sweep (bw={1,4,8} × con={0,0.3}, 1 seed)
#
# Override knobs, e.g.:  BWEIGHT_SWEEP="1.0 2.0 4.0 8.0" EPOCHS=250 SEED_SWEEP=1,2,3 bash scripts/run_blean_impressions.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Swept init axis (bash loop; each value = distinct template + results_dir) ─
BWEIGHT_SWEEP="${BWEIGHT_SWEEP:-1.0 4.0 8.0}"

# ── Family #2 term axis (Hydra multirun; reuses the per-b_weight template) ────
LAMBDA_BUDDYCON_SWEEP="${LAMBDA_BUDDYCON_SWEEP:-0, 0.3}"   # 0 = baseline arm
BUDDY_CON_SAMPLES="${BUDDY_CON_SAMPLES:-4}"
BUDDY_CON_TEMP="${BUDDY_CON_TEMP:-0.07}"

# ── Held constant (confirmed Impressions operating point) ────────────────────
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
LR_SWEEP="${LR_SWEEP:-${LR:-1e-3}}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-${LR_LABEL:-1e-4}}"
SEED_SWEEP="${SEED_SWEEP:-${SEED:-42}}"
ALPHA="${ALPHA:-0.5}"
DATASET="${DATASET:-impressions}"
WANDB_TAG="${WANDB_TAG:-blean-bweight-impressions}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_blean_ablation/impressions}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  BWEIGHT_SWEEP="${BWEIGHT_SWEEP_SMOKE:-1.0}"
  LAMBDA_BUDDYCON_SWEEP="0"
  SEED_SWEEP="42"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, b_weight=1.0, lambda_con=0 — pipeline + template build sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
fi

echo "==================================================================="
echo "B-lean b_weight sweep (Impressions, Family #2 setup)"
echo "  b_weight ∈ {$BWEIGHT_SWEEP}   lambda_con ∈ {$LAMBDA_BUDDYCON_SWEEP}"
echo "  dim=$EMBEDDING_DIM lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP alpha=$ALPHA seed=$SEED_SWEEP"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG"
echo "==================================================================="

for BW in $BWEIGHT_SWEEP; do
  RD="${BASE_RESULTS_DIR}/bw_${BW}"
  echo ">>> b_weight=${BW}  →  results_dir=${RD}"
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
    train.buddies.alpha="$ALPHA" \
    train.buddies.b_weight="$BW" \
    train.epochs="$EPOCHS" \
    +loss.lambda_buddy=0 \
    +loss.lambda_buddy_con="$LAMBDA_BUDDYCON_SWEEP" \
    +loss.buddy_con_samples="$BUDDY_CON_SAMPLES" \
    +loss.buddy_con_temperature="$BUDDY_CON_TEMP" \
    experiment.results_dir="$RD" \
    wandb.group="blean-bweight ablation" \
    ${TEST_RATIO:+eval.test_ratio=$TEST_RATIO} \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${LOG_BUDDY_PRESERVATION:++loss.log_buddy_preservation=true}
done

echo "==================================================================="
echo "Done. Compare retrieval across b_weight (each in its own results_dir):"
echo "  ${BASE_RESULTS_DIR}/bw_*/   (wandb tag: $WANDB_TAG, group: blean-bweight ablation)"
echo "==================================================================="
