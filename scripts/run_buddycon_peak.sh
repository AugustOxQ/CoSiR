#!/bin/bash
set -euo pipefail
# Peak-finding for the CONFIRMED Family #2 win (contrastive supervision).
#
# The seed replication (tag buddy-families-seeds) established lambda_buddy_con=0.3 as a real
# +1.2 R1 t2i / +1.1 i2t win (mean/SEM 7.0 / 2.3 over 3 seeds) at the strong cell
# lr=1e-3, lr_label=1e-4. This sweep asks: is 0.3 the peak, or does more of the term help?
#
# Operating point is FIXED (lr, lr_label, dim, alpha) — the ONLY axes are the term strength
# and the seed. lambda_buddy_con=0 stays IN-BATCH so every arm pairs against a same-seed
# baseline under ONE tag (analyze_buddy_families.py pairs within a tag; a cross-tag baseline
# would be fragile). 4 term values × 3 seeds = 12 runs, no template rebuild.
#
# Override:  LAMBDA_BUDDYCON_SWEEP=0,0.3,0.5,1.0,2.0 SEED_SWEEP=1,2,3,4,5 bash scripts/run_buddycon_peak.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Shared knobs ─────────────────────────────────────────────────────────────
export EPOCHS="${EPOCHS:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
export WANDB_TAG="${WANDB_TAG:-buddy-con-peak}"
export LOG_BUDDY_PRESERVATION="${LOG_BUDDY_PRESERVATION:-1}"

# ── Fixed operating point (the strong, confirmed cell) ───────────────────────
export LR_SWEEP="${LR_SWEEP:-1e-3}"
export LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"

# ── The two axes: term strength (0 = in-batch baseline) × seed ───────────────
export LAMBDA_BUDDYCON_SWEEP="${LAMBDA_BUDDYCON_SWEEP:-0,0.3,0.5,1.0}"
export SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==================================================================="
echo "Family #2 peak-finding  (lambda_buddy_con={$LAMBDA_BUDDYCON_SWEEP} × seeds={$SEED_SWEEP})"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP  EPOCHS=$EPOCHS  tag=$WANDB_TAG"
echo "==================================================================="

bash "$HERE/run_buddycon_full.sh"

echo "==================================================================="
echo "Peak sweep launched (12 runs). Analyse (per-seed paired, mean Δ ± std) with:"
echo "  python scripts/analyze_buddy_families.py --entity augustoxq --project cosir_image --tag $WANDB_TAG --only con"
echo "==================================================================="
