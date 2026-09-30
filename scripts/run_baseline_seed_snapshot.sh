#! /bin/bash
set -euo pipefail

# Fit the pilots' unchanged Attention-h1 baseline at one seed on a DAS6 GPU
# node, keep the snapshot, and score it with both the independent re-clustering
# and the sweep harness's transfer/merge method.
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_baseline_seed_snapshot.sh <seed>    e.g. 42

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here.

# This script only ever runs on a DAS6 node, where the ArtELingo data lives
# under /local/wding/..., not this container's /data/PDD, /data/SSD2 paths;
# the snapshot runner reads these two env vars (defaulting to the local-only paths).
export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${1:?usage: run_baseline_seed_snapshot.sh <seed, e.g. 42>}"

exec python src/test/20260930_harness_confirmation/run_baseline_seed_snapshot.py \
    --seed "$1" \
    --out-dir src/test/20260930_harness_confirmation/snapshots/seed"$1"
