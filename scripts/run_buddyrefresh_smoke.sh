#!/bin/bash
set -euo pipefail
# Family #3 smoke: exercises the refresh code path end-to-end on a tiny run.
# warmup=1, period=1 so refresh fires at epochs 1 and 2; blend=1 = full refresh.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddyrefresh_smoke/impressions}"
python main_cosir.py \
  dataset=impressions \
  eval.evaluation_interval=100 \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim=16 \
  optimizer.lr=1e-4 \
  optimizer.lr_label=1e-4 \
  train.buddies.alpha=0.5 \
  train.epochs=3 \
  +loss.lambda_buddy=0 \
  +loss.lambda_buddy_con=0.3 \
  +loss.buddy_con_samples=4 \
  +loss.buddy_con_temperature=0.07 \
  +loss.buddy_refresh=true \
  +loss.buddy_refresh_warmup=1 \
  +loss.buddy_refresh_period=1 \
  +loss.buddy_refresh_blend=1.0 \
  +loss.buddy_refresh_k=30 \
  experiment.results_dir="${RESULTS_DIR}"
