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
  eval.oracle_aggregation=max \
  model=clip_base \
  experiment.results_dir="res/CoSiR_Experiment_new_oracle_guided/impressions" \
  model.num_layers=6 \
  model.embedding_dim=16 \
  "optimizer.lr=1e-3" \
  "optimizer.lr_label=1e-2" \
  train.epochs=1000 \
  train.normalize=False \
  wandb.group="new oracle guided setting" \