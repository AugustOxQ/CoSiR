#! /bin/bash
set -euo pipefail

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

# python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=5e-4 train.epochs=500 loss.lambda_2=0.3 loss.lambda_3=0 wandb.name="imgtxt5e-4 500 epochs"

python main_cosir.py dataset=impressions train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=300 loss.lambda_2=0.3 loss.lambda_3=0 wandb.name="txt1e-4 300 epochs normalizing text features in manifold loss knn10" optimizer.label_lr_multiplier=5.0e4

# python main_cosir_phase2.py dataset=impressions eval.perform_evaluation=false optimizer.lr_2=1e-5 model.num_layers=4 train.epochs_2=500 model.dropout=0.1 train.initialization_strategy="imgtxt" train.representative_number=36 wandb.project="cosir_seperate_dataset" wandb.name="imgtxt12 1e-5"