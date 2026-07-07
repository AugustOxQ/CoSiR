#! /bin/bash
set -euo pipefail

# Smoke test for the Family #1 buddy-graph smoothness regularizer.
# Confirms end-to-end wiring: the buddy edge list is persisted at init, reloaded,
# and the lambda_buddy * L_buddy term is added during training.
#
# What to look for:
#   * console prints "[buddy-reg] enabled: lambda_buddy=..., samples/anchor=..., edges=..."
#     once, before the training loop
#   * wandb "loss" section shows a finite, non-zero "loss_buddy" that trends down
#
# NOTE: results_dir is intentionally FRESH. use_template_embeddings/save_as_template_embeddings
# are on, so an existing template_embeddings/ built BEFORE this feature would lack
# buddy_edges.npy and the run would print "[buddy-reg] ... disabling" and train without
# the term. A fresh dir forces _buddy_init to rebuild the template and write the edges.

# Reduce CUDA memory fragmentation (especially important for large-patch models like SigLIP)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Tunables (override from the environment, e.g. LAMBDA_BUDDY=0.3 bash scripts/run_buddyreg_smoke.sh)
LAMBDA_BUDDY="${LAMBDA_BUDDY:-0.1}"
BUDDY_REG_SAMPLES="${BUDDY_REG_SAMPLES:-4}"
EPOCHS="${EPOCHS:-2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddyreg_smoke/impressions}"

# initialization_strategy is already 'buddies' by default (configs/train/default.yaml).
# lambda_buddy / buddy_reg_samples are NOT in the YAML (read via getattr defaults), so
# they must be ADDED with a leading '+', not plain-overridden.
python main_cosir.py \
  dataset=impressions \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim="${EMBEDDING_DIM}" \
  train.epochs="${EPOCHS}" \
  +loss.lambda_buddy="${LAMBDA_BUDDY}" \
  +loss.buddy_reg_samples="${BUDDY_REG_SAMPLES}" \
  experiment.results_dir="${RESULTS_DIR}" \
  wandb.group="buddy-reg smoke"
