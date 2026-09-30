#! /bin/bash
set -euo pipefail

# Early V3 check: run the PercepT Stage 1 port at one seed and score it with
# the pilots' own Stage 2 against the published §6g numbers.
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_h2h_early_v3.sh [seed]

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

exec python src/test/20260930_matched_h2h/early_v3_check.py --seed "${1:-42}"
