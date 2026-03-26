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

# python main_cosir_phase2.py dataset=impressions eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=500 model.dropout=0.1 train.initialization_strategy="imgtxt" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="imgtxt96 "

# python main_cosir.py dataset=redcaps train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-4 train.epochs=30 wandb.name="1e-4"

python main_cosir.py dataset=redcaps train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=1e-5 train.epochs=20 wandb.name="1e-5"

# python main_cosir.py dataset=redcaps train.imgtxt_factor=1 train.initialization_strategy="imgtxt" optimizer.lr=5e-4 train.epochs=30 wandb.name="5e-4"

# python main_cosir_phase2.py dataset=impressions eval.perform_evaluation=false optimizer.lr_2=1e-4 model.num_layers=4 train.epochs_2=500 model.dropout=0.1 train.initialization_strategy="imgtxt" train.representative_number=96 wandb.project="cosir_seperate_dataset" wandb.name="imgtxt96 "