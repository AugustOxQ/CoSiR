#! /bin/bash
set -euo pipefail

# One in-process W&B agent for the head-to-head sweeps (run one per GPU).
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_h2h_agent.sh <entity/project/sweep_id> [count]

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here.

export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_SILENT=true

SWEEP="${1:?usage: run_h2h_agent.sh <entity/project/sweep_id> [count]}"
COUNT_ARGS=()
if [ -n "${2:-}" ]; then COUNT_ARGS=(--count "$2"); fi

exec python scripts/buddy_percept_sweep/h2h_agent.py --sweep "$SWEEP" "${COUNT_ARGS[@]}"
