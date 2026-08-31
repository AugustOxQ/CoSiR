#!/bin/bash
set -euo pipefail
# Multi-SEED replication for the two families that showed a mild positive signal in the
# lr×lr_label grid (#1 smoothness, #2 contrastive). The grid put both at ~+0.5 R1 on t2i —
# right at the n=1 noise floor (measured ~0.1-0.7 R1 from the blend=0 / lambda_con=0.3
# duplicate). The ONLY way to tell a real +0.5 from seed jitter is replication, so this
# fixes the operating point and varies ONLY the seed.
#
# Fixed operating point (the strong cell from the grid):
#   lr=1e-3, lr_label=1e-4, dim=16, alpha=0.5, 250 epochs.
# Per family: term OFF (0) vs ON, replicated over SEED_SWEEP. Analyse with seed as a
# pairing coordinate → mean Δ ± std across seeds + win-rate (see analyze_buddy_families.py).
#
# Family #3 is intentionally DROPPED: the grid showed it dead-neutral (Δ≈0 both directions)
# even though the live graph provably disagreed with CLIP (new_edge_frac≈0.20) — replicating
# a confirmed null buys nothing. Re-add with RUN_REFRESH=1 if you want the null with error bars.
#
# Override anything:  SEED_SWEEP=1,2,3,4,5 EPOCHS=250 bash scripts/run_buddy_seeds.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Shared knobs (exported so child runners inherit) ─────────────────────────
export EPOCHS="${EPOCHS:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
export WANDB_TAG="${WANDB_TAG:-buddy-families-seeds}"          # fresh tag → analyse with --tag
export LOG_BUDDY_PRESERVATION="${LOG_BUDDY_PRESERVATION:-1}"

# ── Fixed operating point (the strong cell from the grid) ────────────────────
export LR_SWEEP="${LR_SWEEP:-1e-3}"
export LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"

# ── The replication axis ─────────────────────────────────────────────────────
export SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

# ── Term axes: OFF vs the single ON value the grid liked ─────────────────────
export LAMBDA_BUDDY_SWEEP="${LAMBDA_BUDDY_SWEEP:-0,0.3}"          # #1
export LAMBDA_BUDDYCON_SWEEP="${LAMBDA_BUDDYCON_SWEEP:-0,0.3}"    # #2

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==================================================================="
echo "Buddy families SEED replication  (seeds={$SEED_SWEEP})"
echo "  operating point: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP  EPOCHS=$EPOCHS  tag=$WANDB_TAG"
echo "  #1 lambda_buddy={$LAMBDA_BUDDY_SWEEP}   #2 lambda_buddy_con={$LAMBDA_BUDDYCON_SWEEP}"
echo "==================================================================="

echo ">>> Family #2  (contrastive: lambda_buddy_con 0 vs 0.3, × seeds)"
bash "$HERE/run_buddycon_full.sh"

echo ">>> Family #1  (smoothness: lambda_buddy 0 vs 0.3, × seeds)"
bash "$HERE/run_buddyreg_full.sh"

if [ -n "${RUN_REFRESH:-}" ]; then
  echo ">>> Family #3  (self-refresh: blend 0 vs 1.0, × seeds) [opt-in]"
  bash "$HERE/run_buddyrefresh_full.sh"
fi

echo "==================================================================="
echo "Seed replication launched. Analyse (seed-paired, mean Δ ± std) with:"
echo "  python scripts/analyze_buddy_families.py --entity augustoxq --project cosir_image --tag $WANDB_TAG --only con"
echo "  python scripts/analyze_buddy_families.py --entity augustoxq --project cosir_image --tag $WANDB_TAG --only reg"
echo "==================================================================="
