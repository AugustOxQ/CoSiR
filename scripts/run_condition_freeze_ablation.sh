#!/bin/bash
set -euo pipefail
# Experiment 11.1 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does post-init training of the conditions do anything, holding buddy-init geometry, the
# frozen CLIP backbone, and every other hyperparameter identical between arms?
#
# train.em_interval is NOT a template-compatibility key (see src/hook/train_cosir.py's
# _extra dict, which only reads k/alpha/method/b_weight/encoder_pair/distance_mode) — unlike
# every other axis this project has swept (initialization_strategy, encoder_pair,
# distance_mode), toggling it does NOT change the buddy-init graph/embedding at all. This
# script deliberately points BOTH arms at the SAME results_dir so they load the exact same
# cached template (template_dir = experiment.directory.parent / "template_embeddings" in
# _init_embedding_manager) — every one of this sweep's 6 runs starts from a byte-identical
# buddy-init, and only the em_interval value governs whether the condition table can move
# after that. Do NOT give the two arms separate results_dir/template dirs — that would weaken
# the pairing this experiment is designed around.
#
# Two arms, both via train.em_interval (existing EM-alternation mechanism):
#   trained: em_interval=-1        (default; conditions update every step, as in every prior
#                                    experiment in this plan)
#   frozen:  em_interval=EPOCHS+1  (epoch // em_interval stays 0 for the whole run -> "network"
#                                    phase forever -> embedding_manager.embeddings.requires_grad_(False)
#                                    from epoch 0 onward, per the epoch-0 _prev_em_phase transition)
#
#   SMOKE=1 bash scripts/run_condition_freeze_ablation.sh   # 2 epochs, seed=1, both arms
#   bash scripts/run_condition_freeze_ablation.sh           # full 6-run sweep

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
# Deliberately ONE results_dir for both arms — see header comment.
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_condition_freeze_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-condition-freeze-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-condition freeze ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, both arms — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

FROZEN_EM_INTERVAL="$((EPOCHS + 1))"

echo "==================================================================="
echo "Condition freeze ablation ($DATASET): {trained, frozen} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "  results_dir (SHARED across both arms, by design): $RESULTS_DIR"
echo "==================================================================="

for ARM in trained frozen; do
  if [ "$ARM" = "trained" ]; then
    EM="-1"
  else
    EM="$FROZEN_EM_INTERVAL"
  fi
  echo ">>> arm=${ARM} (em_interval=${EM})"
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
    train.em_interval="$EM" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RESULTS_DIR" \
    wandb.group="$WANDB_GROUP" \
    +train.arm="$ARM" \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${EXTRA_OVERRIDES:-}
done

echo "==================================================================="
echo "Done. Analyse retrieval (paired vs. trained, mean delta +/- std) with:"
echo "  python scripts/analyze_condition_freeze_ablation.py --tag $WANDB_TAG"
echo "Geometry diagnostic (per run, then compare):"
echo "  python scripts/analyze_condition_geometry.py --exp-dir <run_dir>"
echo "  python scripts/analyze_condition_geometry.py --compare <frozen_run_dir> <trained_run_dir>"
echo "==================================================================="
