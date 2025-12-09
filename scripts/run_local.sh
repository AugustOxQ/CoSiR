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

ENTRY="main_cosir.py"

# 这个是只比较loss参数，不比较模型大小
python main_cosir.py dataset=coco optimizer.lr=1e-5 model.num_layers=4 model.hidden_dim=128 loss.lambda_1=1.0 loss.lambda_2=0.3 loss.lambda_3=0.15 loss.lambda_4=0.1

python main_cosir.py dataset=coco optimizer.lr=1e-5 model.num_layers=4 model.hidden_dim=128 loss.lambda_1=1.0 loss.lambda_2=1.0 loss.lambda_3=0.5 loss.lambda_4=0.01

python main_cosir.py dataset=coco optimizer.lr=1e-5 model.num_layers=4 model.hidden_dim=128 loss.lambda_1=1.0 loss.lambda_2=0.5 loss.lambda_3=0.3 loss.lambda_4=0.01