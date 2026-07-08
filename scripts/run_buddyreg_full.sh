#! /bin/bash
set -euo pipefail

# Full-scale FOCUSED ABLATION for the Family #1 buddy-graph smoothness regularizer.
#
# Sweeps ONLY +loss.lambda_buddy (with 0 = baseline), holding the model/optimizer/buddy
# init knobs fixed. Because lambda_buddy is NOT part of the template key, every
# lambda_buddy value reuses the SAME buddy init/template — so lambda_buddy=0 vs >0 differ
# only by the training term. That makes this a clean ablation of the regularizer itself.
#
# Set the HELD values below to your best-known config from prior buddy runs, e.g.:
#   EMBEDDING_DIM=16 LR=1e-4 LR_LABEL=1e-2 ALPHA=0.5 bash scripts/run_buddyreg_full.sh
#
# NOTE: RESULTS_DIR is intentionally FRESH so each per-(dim,alpha) template is rebuilt by
# _buddy_init WITH buddy_edges.npy. Reusing a pre-feature template_embeddings/ would trip
# the "[buddy-reg] ... disabling" fallback and train the lambda_buddy>0 arms without the term.

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
LAMBDA_BUDDY_SWEEP="${LAMBDA_BUDDY_SWEEP:-0,0.1,0.3,1.0}"   # 0 = baseline arm
BUDDY_REG_SAMPLES="${BUDDY_REG_SAMPLES:-4}"

# ── Held constant (set these to your best-known values) ──────────────────────
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
LR_SWEEP="${LR_SWEEP:-${LR:-1e-4}}"              # comma list ⇒ Hydra grid (multirun)
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-${LR_LABEL:-1e-4}}"
SEED_SWEEP="${SEED_SWEEP:-${SEED:-42}}"          # comma list ⇒ replicate over seeds
ALPHA="${ALPHA:-0.5}"
EPOCHS="${EPOCHS:-1000}"
RESULTS_DIR="${RESULTS_DIR:-res/CoSiR_buddyreg_ablation/impressions}"

# initialization_strategy is already 'buddies' by default (configs/train/default.yaml).
# lambda_buddy / buddy_reg_samples are read via getattr defaults (not in the YAML), so
# they must be ADDED with a leading '+'.
python main_cosir.py -m \
  dataset=impressions \
  eval.evaluation_interval="${EVAL_INTERVAL:-100}" \
  eval.oracle_aggregation=max \
  model=clip_base \
  model.num_layers=6 \
  model.embedding_dim="${EMBEDDING_DIM}" \
  optimizer.lr="${LR_SWEEP}" \
  optimizer.lr_label="${LR_LABEL_SWEEP}" \
  seed="${SEED_SWEEP}" \
  train.buddies.alpha="${ALPHA}" \
  train.epochs="${EPOCHS}" \
  +loss.lambda_buddy="${LAMBDA_BUDDY_SWEEP}" \
  +loss.buddy_reg_samples="${BUDDY_REG_SAMPLES}" \
  experiment.results_dir="${RESULTS_DIR}" \
  wandb.group="buddy-reg ablation" \
  ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
  ${LOG_BUDDY_PRESERVATION:++loss.log_buddy_preservation=true}
