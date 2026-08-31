#!/bin/bash
set -euo pipefail
# Experiment 8 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does the CHOICE of (vision, text) encoder pair used to BUILD the buddy graph/init matter
# for downstream retrieval, holding the frozen CLIP training backbone, gated combiner, and
# all training-time buddy terms OFF (same operating point as Experiment 1's 'buddies' arm)?
#
# train.buddies.encoder_pair is a TEMPLATE-COMPATIBILITY key, exactly like
# initialization_strategy in scripts/run_init_ablation.sh: each pair gets its OWN
# results_dir so its own template_embeddings/, avoiding template-reuse races. A bash loop
# over the template-key axis (encoder pair), Hydra multirun over the non-template axis
# (seed) — same pattern as run_init_ablation.sh.
#
#   SMOKE=1 bash scripts/run_buddy_init_encoder_ablation.sh                # 2 epochs, 1 pair
#   ENCODER_PAIR_SWEEP="clip_img:clip_txt dinov2:bge" bash scripts/run_buddy_init_encoder_ablation.sh
#   bash scripts/run_buddy_init_encoder_ablation.sh                        # full 16-pair sweep
#
# Requires the held-out feature cache for every non-CLIP encoder used to already exist:
#   python src/test/20260708_heldout_grid/extract_heldout.py --dataset redcaps --model <name>
# (already run for the C3 cross-VLM survival study — see docs/reports/2026-07-16_buddy_cross_vlm_survival.md)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Template-key axis (bash loop; each pair = its own template + results_dir) ───
if [ -z "${ENCODER_PAIR_SWEEP:-}" ]; then
  PAIRS=()
  for V in clip_img dinov2 siglip_v vit_sup; do
    for T in clip_txt minilm bge e5; do
      PAIRS+=("${V}:${T}")
    done
  done
  ENCODER_PAIR_SWEEP="${PAIRS[*]}"
fi

# ── Non-template axis (Hydra multirun; reuses each pair's template) ─────────────
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

# ── Fixed operating point (same as Experiment 1's confirmed strong cell) ────────
LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

# ── Dataset + storage (RedCaps-150k only, per spec §4 Experiment 8 scope) ───────
DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_buddy_encoder_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-buddy-encoder-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-buddy-init encoder-pair ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  ENCODER_PAIR_SWEEP="${ENCODER_PAIR_SWEEP_SMOKE:-clip_img:clip_txt}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, pair(s)={$ENCODER_PAIR_SWEEP} — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Buddy-init encoder-pair ablation ($DATASET): {$ENCODER_PAIR_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "==================================================================="

for PAIR in $ENCODER_PAIR_SWEEP; do
  SAFE_PAIR="${PAIR/:/_x_}"
  RD="${BASE_RESULTS_DIR}/pair_${SAFE_PAIR}"
  echo ">>> encoder_pair=${PAIR}  ->  results_dir=${RD}"
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
    +train.buddies.encoder_pair="$PAIR" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${EXTRA_OVERRIDES:-}
done

echo "==================================================================="
echo "Done. Analyse (paired vs. clip_img:clip_txt, mean delta +/- std) with:"
echo "  python scripts/analyze_buddy_init_encoder_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
