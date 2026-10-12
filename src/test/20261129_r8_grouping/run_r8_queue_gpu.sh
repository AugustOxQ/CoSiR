#!/usr/bin/env bash
# Ticket 13: the GPU queue (views, FCMI FC3 then FC1 FC2 FC4-FC6, encoders AL DI SG). Options: --dry-run, --smoke.
# Start in the background: nohup ./run_r8_queue_gpu.sh > /dev/null 2>&1 &   (logs in results/logs/, smoke: results/smoke/logs/)
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec /root/miniconda3/envs/CoSiR/bin/python run_r8_queue.py gpu "$@"
