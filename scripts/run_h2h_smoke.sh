#! /bin/bash
set -euo pipefail

# Smoke-test one matched head-to-head trial on real data on a DAS6 node.
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_h2h_smoke.sh <preset> [seeds]   e.g. buddy_pilot_k16 1001,1002

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here.

# This script only ever runs on a DAS6 node, where the ArtELingo data lives
# under /local/wding/..., not this container's /data/PDD, /data/SSD2 paths.
export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# The PercepT port replays the pilot's GoEmotions model loads (they consume
# torch RNG); the model is already in the node's Hugging Face cache.
export HF_HUB_OFFLINE=1

: "${1:?usage: run_h2h_smoke.sh <preset> [seeds]}"

exec python src/test/20260930_matched_h2h/smoke_trial.py --preset "$1" --seeds "${2:-1001}"
