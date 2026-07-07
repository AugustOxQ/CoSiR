#!/bin/bash
set -euo pipefail
# Full-scale FOCUSED ABLATION for Family #3 (self-refreshing buddy graph).
#
# Holds Family #1 OFF and the #2 contrastive term ON (lambda_buddy_con fixed),
# and sweeps ONLY +loss.buddy_refresh_blend. blend=0 = static Family #2 graph
# (baseline); blend=1.0 = full CLIP-anchored refresh. Because none of the
# buddy_refresh* keys are part of the buddy template key, every arm reuses the
# SAME buddy init template — so the arms differ only by the training-time graph.
#
# RESULTS_DIR is intentionally FRESH so each per-(dim,alpha) template is rebuilt
# with buddy_edges.npy present (needed by both #2 and the refresh union).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -r -a __cvd_arr <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_PROCS=${#__cvd_arr[@]}
  else
    NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
else
  NUM_PROCS=1
fi
[ -z "${NUM_PROCS}" ] && NUM_PROCS=1
[ "${NUM_PROCS}" -lt 1 ] && NUM_PROCS=1
echo "Using ${NUM_PROCS} processes"

# ── Swept axis ───────────────────────────────────────────────────────────────
BUDDY_REFRESH_BLEND_SWEEP="${BUDDY_REFRESH_BLEND_SWEEP:-0, 1.0}"  # 0 = static #2 baseline

# ── Held constant ────────────────────────────────────────────────────────────
LAMBDA_BUDDYCON="${LAMBDA_BUDDYCON:-0.3}"
BUDDY_CON_SAMPLES="${BUDDY_CON_SAMPLES:-4}"
BUDDY_CON_TEMP="${BUDDY_CON_TEMP:-0.07}"
BUDDY_REFRESH_WARMUP="${BUDDY_REFRESH_WARMUP:-50}"
BUDDY_REFRESH_PERIOD="${BUDDY_REFRESH_PERIOD:-50}"
BUDDY_REFRESH_K="${BUDDY_REFRESH_K:-30}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
LR="${LR:-1e-4}"
LR_LABEL="${LR_LABEL:-1e-4}"
ALPHA="${ALPHA:-0.5}"
EPOCHS="${EPOCHS:-500}"
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddyrefresh_ablation/impressions}"

python main_cosir.py -m \
  dataset=impressions \
  eval.evaluation_interval="${EVAL_INTERVAL:-100}" \
  eval.oracle_aggregation=max \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim="${EMBEDDING_DIM}" \
  optimizer.lr="${LR}" \
  optimizer.lr_label="${LR_LABEL}" \
  train.buddies.alpha="${ALPHA}" \
  train.epochs="${EPOCHS}" \
  +loss.lambda_buddy=0 \
  +loss.lambda_buddy_con="${LAMBDA_BUDDYCON}" \
  +loss.buddy_con_samples="${BUDDY_CON_SAMPLES}" \
  +loss.buddy_con_temperature="${BUDDY_CON_TEMP}" \
  +loss.buddy_refresh=true \
  +loss.buddy_refresh_warmup="${BUDDY_REFRESH_WARMUP}" \
  +loss.buddy_refresh_period="${BUDDY_REFRESH_PERIOD}" \
  +loss.buddy_refresh_blend="${BUDDY_REFRESH_BLEND_SWEEP}" \
  +loss.buddy_refresh_k="${BUDDY_REFRESH_K}" \
  experiment.results_dir="${RESULTS_DIR}" \
  wandb.group="buddy-refresh ablation" \
  ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
  ${LOG_BUDDY_PRESERVATION:++loss.log_buddy_preservation=true}
