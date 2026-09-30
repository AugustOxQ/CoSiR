#! /bin/bash
set -euo pipefail

# Early V2 check: run the buddy pilot port at one seed and compare it with the
# pilot snapshot QC2 wrote on this node.
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_h2h_early_v2.sh <seed> <qc2 job tag, e.g. qc2-s42>

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here.

# This script only ever runs on a DAS6 node, where the ArtELingo data lives
# under /local/wding/..., not this container's /data/PDD, /data/SSD2 paths.
export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${1:?usage: run_h2h_early_v2.sh <seed> <qc2 job tag>}"
: "${2:?usage: run_h2h_early_v2.sh <seed> <qc2 job tag>}"

exec python src/test/20260930_matched_h2h/early_v2_check.py \
    --seed "$1" \
    --snapshot "/local/wding/jobs/$2/code/src/test/20260930_harness_confirmation/snapshots/seed$1/snapshot_seed$1.npz"
