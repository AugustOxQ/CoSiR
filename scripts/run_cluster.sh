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

echo "Experiment 1, from 1e-6 to 1e-6"

CUDA_VISIBLE_DEVICES=0 python main_cosir.py dataset=cc3m_cluster train.lr=1e-6 > exp1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_cosir.py dataset=cc3m_cluster train.lr=5e-6 > exp2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main_cosir.py dataset=cc3m_cluster train.lr=1e-5 > exp3.log 2>&1 &

wait

echo "Experiment 2, from 5e-5 to 5e-4"

CUDA_VISIBLE_DEVICES=0 python main_cosir.py dataset=cc3m_cluster train.lr=5e-5 > exp4.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_cosir.py dataset=cc3m_cluster train.lr=1e-4 > exp5.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main_cosir.py dataset=cc3m_cluster train.lr=5e-4 > exp6.log 2>&1 &

wait

echo "All finish