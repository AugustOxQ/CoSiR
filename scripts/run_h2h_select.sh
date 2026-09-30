#! /bin/bash
set -euo pipefail

# Select / stress / test / summarize for the head-to-head sweeps on a DAS6 node.
# Usage (from a reserved srun shell, via `cluster launch`):
#   bash scripts/run_h2h_select.sh <select|run|reference|summarize> [args...]
# cluster-run's job.sh already activates the CoSiR env; do not source conda here.

export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1

if [ "$#" -lt 1 ]; then
    echo "usage: run_h2h_select.sh <select|run|reference|summarize> [args...]" >&2
    exit 2
fi

exec python scripts/buddy_percept_sweep/h2h_select.py "$@"
