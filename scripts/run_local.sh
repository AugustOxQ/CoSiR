#! /bin/bash
set -euo pipefail

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

python main_cosir.py -m \
  dataset=impressions \
  eval.evaluation_interval=100 \
  "eval.oracle_aggregation=mean, max" \ 
  loss.lambda_collapse=0.1 \
  loss.lambda_contrastive=0 \
  loss.lambda_laplacian=30 \
  loss.lambda_mixup=1 \
  model=siglip_base \
  model.num_layers=6 \
  optimizer.lr=1e-5 \
  optimizer.lr_label=1.0e-2 \
  train.epochs=1000 \
  train.normalize=False \
  train.imgtxt_factor=1 \
  train.initialization_strategy=imgtxt