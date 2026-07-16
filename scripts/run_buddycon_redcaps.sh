#!/bin/bash
set -euo pipefail
# CROSS-DATASET validation of the confirmed Family #2 win (buddy contrastive supervision)
# on RedCaps-150k — a genuinely 1:1 dataset (every image_id unique, no near-duplicate
# scaffolding), with IN-DOMAIN eval on redcaps_test (25k pairs). This is the one real
# threat to the Impressions result (814 source photos behind 12k records): does λ_con help
# when the buddy graph can't lean on same-photo near-dups?
#
# Confirmed operating point from Impressions peak-finding: λ_con=1.0 at lr=1e-3, lr_label=1e-4,
# dim=16, alpha=0.5. Here we run the minimal confirmation: λ_con ∈ {0, 1.0} × seeds {1,2,3}
# = 6 runs, paired within seed under one tag.
#
# RedCaps has NEVER been trained through this pipeline (prior redcaps work was graph analysis
# only). SMOKE FIRST:  SMOKE=1 bash scripts/run_buddycon_redcaps.sh
#   → 2 epochs, 1 seed, λ_con=0, eval every epoch. Confirms features load, the 150k buddy
#     template builds, and test_oracle/* is emitted. Shares RESULTS_DIR with the full run, so
#     the (expensive, one-time) 150k spectral template is built once and reused by the sweep.
# Then the full 6-run sweep:  bash scripts/run_buddycon_redcaps.sh
#
# EPOCHS default is 100 (Impressions used 250) — a deadline compromise, NOT scaled by data size.
# The per-sample condition embeddings z update once per epoch (only when their sample is drawn),
# so epochs = updates-per-z, and z is the buddy-con term's substrate. 100 gives z enough updates
# to engage (50 risked a FALSE null — both arms mute and Δ→0); 100 vs 250 just makes this a
# CONSERVATIVE test (a slightly-undertrained z can only shrink the true Δ, never inflate it).
# Combiner weights train every step and are fine with fewer epochs. Features are cached, so
# 150k steps/epoch is cheap combiner ops (the one-time cost is the 150k spectral template build).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Dataset + storage (in-domain redcaps_test eval) ──────────────────────────
export DATASET="${DATASET:-redcaps_150k}"
export RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddycon_ablation/redcaps_150k}"

# ── Fixed operating point (confirmed Impressions cell) ───────────────────────
export LR_SWEEP="${LR_SWEEP:-1e-3}"
export LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
export ALPHA="${ALPHA:-0.5}"

# ── Shared knobs ─────────────────────────────────────────────────────────────
export WANDB_TAG="${WANDB_TAG:-buddy-con-redcaps}"
export LOG_BUDDY_PRESERVATION="${LOG_BUDDY_PRESERVATION:-1}"
# redcaps_test is 25k pairs — too slow to extract/eval fully. 0.2 → first 5k (interleaved,
# covers 318/349 subreddits, representative). The eval Δ (λ_con 0 vs 1) is what we read.
export TEST_RATIO="${TEST_RATIO:-0.2}"

if [ -n "${SMOKE:-}" ]; then
  export EPOCHS="${EPOCHS:-2}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  export LAMBDA_BUDDYCON_SWEEP="${LAMBDA_BUDDYCON_SWEEP:-0}"
  export SEED_SWEEP="${SEED_SWEEP:-1}"
  export WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, 1 seed, λ_con=0 — pipeline + template + metric sanity"
else
  export EPOCHS="${EPOCHS:-100}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
  export LAMBDA_BUDDYCON_SWEEP="${LAMBDA_BUDDYCON_SWEEP:-0,1.0}"
  export SEED_SWEEP="${SEED_SWEEP:-1,2,3}"
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==================================================================="
echo "RedCaps-150k buddy-con validation  (λ_con={$LAMBDA_BUDDYCON_SWEEP} × seeds={$SEED_SWEEP})"
echo "  dataset=$DATASET  results=$RESULTS_DIR"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA"
echo "  EPOCHS=$EPOCHS  EVAL_INTERVAL=$EVAL_INTERVAL  test_ratio=$TEST_RATIO  tag=$WANDB_TAG"
echo "==================================================================="

bash "$HERE/run_buddycon_full.sh"

echo "==================================================================="
echo "Done. Analyse (per-seed paired, mean Δ ± std) with:"
echo "  python scripts/analyze_buddy_families.py --entity augustoxq --project cosir_image --tag $WANDB_TAG --only con"
echo "==================================================================="
