#! /bin/bash
set -euo pipefail

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <learning_rate>"
    echo "Example: $0 1e-6"
    echo "Or with CUDA_VISIBLE_DEVICES: CUDA_VISIBLE_DEVICES=0 $0 1e-6"
    exit 1
fi

# 自动检测进程数（优先使用 CUDA_VISIBLE_DEVICES，其次 nvidia-smi，否则 CPU=1）
if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -r -a __cvd_arr <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_PROCS=${#__cvd_arr[@]}
    # 获取当前使用的GPU ID（CUDA_VISIBLE_DEVICES的第一个值）
    GPU_ID=${__cvd_arr[0]}
  else
    NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
    GPU_ID="auto"
  fi
else
  NUM_PROCS=1
  GPU_ID="cpu"
fi
[ -z "${NUM_PROCS}" ] && NUM_PROCS=1
[ "${NUM_PROCS}" -lt 1 ] && NUM_PROCS=1

ENTRY="main_cosir.py"
input_lr=$1

# 将学习率中的特殊字符替换为下划线（用于文件名和wandb name）
lr_safe=$(echo "$input_lr" | sed 's/[^a-zA-Z0-9]/_/g')

EXP_NOTE_PREFIX="Try img_txt amplify factor 10 and no boundary loss weight to see how is the case also re extract dataset to be with 4096 batch size"

echo "=========================================="
echo "Experiment Configuration:"
echo "  Learning Rate: $input_lr"
echo "  GPU ID: ${GPU_ID:-auto}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================="

# 注意：当设置了CUDA_VISIBLE_DEVICES时，PyTorch会将可见的GPU映射为cuda:0
# 所以device=cuda:0是正确的，它会使用CUDA_VISIBLE_DEVICES指定的GPU
python main_cosir.py dataset=coco_cluster optimizer.lr=$input_lr device=cuda:0 wandb.name="exp_lr_${lr_safe}" wandb.notes="$EXP_NOTE_PREFIX with lr $input_lr" &
echo "Started experiment (lr=$input_lr) on GPU ${GPU_ID:-auto}"