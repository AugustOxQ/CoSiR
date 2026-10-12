#!/usr/bin/env bash
# Ticket 13: the CPU queue (SC1 check, heads, groupings, placement, readers, scoring, C13, trigger, NBC, gate, descriptive). Options: --dry-run, --smoke.
# Start in the background: nohup ./run_r8_queue_cpu.sh > /dev/null 2>&1 &   (logs in results/logs/, smoke: results/smoke/logs/)
set -u
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec /root/miniconda3/envs/CoSiR/bin/python run_r8_queue.py cpu "$@"
