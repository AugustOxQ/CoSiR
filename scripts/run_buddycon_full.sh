#! /bin/bash
set -euo pipefail

# Full-scale FOCUSED ABLATION for the Family #2 buddy CONTRASTIVE supervision term.
#
# Sweeps ONLY +loss.lambda_buddy_con (with 0 = baseline), holding the model/optimizer/
# buddy-init knobs fixed AND Family #1 off (lambda_buddy=0) so this isolates the new
# contrastive term by itself. Because lambda_buddy_con is NOT part of the buddy template
# key, every lambda_buddy_con value reuses the SAME buddy init/template — so
# lambda_buddy_con=0 vs >0 differ only by the training term. Clean ablation of the term.
#
# Set the HELD values below to your best-known config, e.g.:
#   EMBEDDING_DIM=16 LR=1e-4 LR_LABEL=1e-2 ALPHA=0.5 bash scripts/run_buddycon_full.sh
#
# NOTE: RESULTS_DIR is intentionally FRESH so each per-(dim,alpha) template is rebuilt by
# _buddy_init WITH buddy_edges.npy. Reusing a pre-feature template_embeddings/ would trip
# the "[buddy-con] ... disabling" fallback and train the lambda_buddy_con>0 arms without
# the term.

# Reduce CUDA memory fragmentation (especially important for large-patch models like SigLIP)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 自动检测进程数（优先使用 CUDA_VISIBLE_DEVICES，其次 nvidia-smi，否则 CPU=1）
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

# ── Swept axis (the experiment variable) ─────────────────────────────────────
LAMBDA_BUDDYCON_SWEEP="${LAMBDA_BUDDYCON_SWEEP:-0, 0.3}"   # 0 = baseline arm
BUDDY_CON_SAMPLES="${BUDDY_CON_SAMPLES:-4}"
BUDDY_CON_TEMP="${BUDDY_CON_TEMP:-0.07}"

# ── Held constant (set these to your best-known values) ──────────────────────
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
LR="${LR:-1e-4}"
LR_LABEL="${LR_LABEL:-1e-4}"
ALPHA="${ALPHA:-0.5}"
EPOCHS="${EPOCHS:-500}"
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddycon_ablation/impressions}"

# initialization_strategy is already 'buddies' by default (configs/train/default.yaml).
# lambda_buddy_con / buddy_con_samples / buddy_con_temperature are read via getattr
# defaults (not in the YAML), so they must be ADDED with a leading '+'.
# Family #1 is held OFF (+loss.lambda_buddy=0) so this isolates the contrastive term.
python main_cosir.py -m \
  dataset=impressions \
  eval.evaluation_interval=100 \
  eval.oracle_aggregation=max \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim="${EMBEDDING_DIM}" \
  optimizer.lr="${LR}" \
  optimizer.lr_label="${LR_LABEL}" \
  train.buddies.alpha="${ALPHA}" \
  train.epochs="${EPOCHS}" \
  +loss.lambda_buddy=0 \
  +loss.lambda_buddy_con="${LAMBDA_BUDDYCON_SWEEP}" \
  +loss.buddy_con_samples="${BUDDY_CON_SAMPLES}" \
  +loss.buddy_con_temperature="${BUDDY_CON_TEMP}" \
  experiment.results_dir="${RESULTS_DIR}" \
  wandb.group="buddy-con ablation"
