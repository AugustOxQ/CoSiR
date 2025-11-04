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

python main_cosir.py dataset=