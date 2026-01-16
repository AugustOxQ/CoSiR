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

# Set dir for experiment logs
EXP_LOG_DIR="/local/wding/res/CoSiR_Experiment/logs"
EXP_NOTE_PREFIX="Try img_txt amplify factor 10 and no boundary loss weight to see how is the case also re extract dataset to be with 4096 batch size"
mkdir -p $EXP_LOG_DIR

echo "Experiment 1, from 1e-6 to 1e-6"

# 实验1：使用GPU0，设置唯一的wandb name避免冲突
# 注意：CUDA_VISIBLE_DEVICES=0时，物理GPU0映射为逻辑GPU0，所以device=cuda:0正确
CUDA_VISIBLE_DEVICES=0 python main_cosir.py dataset=coco_cluster optimizer.lr=1e-6 device=cuda:0 wandb.name="exp1_lr1e-6" wandb.notes="$EXP_NOTE_PREFIX with lr 1e-6" > $EXP_LOG_DIR/exp1.txt 2>&1 &
PID1=$!
echo "Started exp1 (lr=1e-6) on GPU0, PID=$PID1"

sleep 30

# 实验2：使用GPU1，设置唯一的wandb name避免冲突
# 注意：CUDA_VISIBLE_DEVICES=1时，物理GPU1映射为逻辑GPU0，所以device=cuda:0会使用物理GPU1
CUDA_VISIBLE_DEVICES=1 python main_cosir.py dataset=coco_cluster optimizer.lr=5e-6 device=cuda:0 wandb.name="exp2_lr5e-6" wandb.notes="$EXP_NOTE_PREFIX with lr 5e-6" > $EXP_LOG_DIR/exp2.txt 2>&1 &
PID2=$!
echo "Started exp2 (lr=5e-6) on GPU1, PID=$PID2"

sleep 30

# 实验3：使用GPU2，设置唯一的wandb name避免冲突
# 注意：CUDA_VISIBLE_DEVICES=2时，物理GPU2映射为逻辑GPU0，所以device=cuda:0会使用物理GPU2
CUDA_VISIBLE_DEVICES=2 python main_cosir.py dataset=coco_cluster optimizer.lr=1e-5 device=cuda:0 wandb.name="exp3_lr1e-5" wandb.notes="$EXP_NOTE_PREFIX with lr 1e-5" > $EXP_LOG_DIR/exp3.txt 2>&1 &
PID3=$!
echo "Started exp3 (lr=1e-5) on GPU2, PID=$PID3"

wait

echo "Experiment 2, from 5e-5 to 5e-4"

# 实验4：使用GPU0，设置唯一的wandb name避免冲突
CUDA_VISIBLE_DEVICES=0 python main_cosir.py dataset=coco_cluster optimizer.lr=5e-5 device=cuda:0 wandb.name="exp4_lr5e-5" wandb.notes="$EXP_NOTE_PREFIX with lr 5e-5" > $EXP_LOG_DIR/exp4.txt 2>&1 &
PID4=$!
echo "Started exp4 (lr=5e-5) on GPU0, PID=$PID4"

sleep 30

# 实验5：使用GPU1，设置唯一的wandb name避免冲突
CUDA_VISIBLE_DEVICES=1 python main_cosir.py dataset=coco_cluster optimizer.lr=1e-4 device=cuda:0 wandb.name="exp5_lr1e-4" wandb.notes="$EXP_NOTE_PREFIX with lr 1e-4" > $EXP_LOG_DIR/exp5.txt 2>&1 &
PID5=$!
echo "Started exp5 (lr=1e-4) on GPU1, PID=$PID5"

sleep 30

# 实验6：使用GPU2，设置唯一的wandb name避免冲突
CUDA_VISIBLE_DEVICES=2 python main_cosir.py dataset=coco_cluster optimizer.lr=5e-4 device=cuda:0 wandb.name="exp6_lr5e-4" wandb.notes="$EXP_NOTE_PREFIX with lr 5e-4" > $EXP_LOG_DIR/exp6.txt 2>&1 &
PID6=$!
echo "Started exp6 (lr=5e-4) on GPU2, PID=$PID6"

wait

echo "All finish"