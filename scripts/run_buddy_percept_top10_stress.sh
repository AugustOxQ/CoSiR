#! /bin/bash
set -euo pipefail

# Stress-test buddy-percept sweep finalists at multiple seeds on a DAS6 GPU
# node. Reads the frozen leaderboard written by the `select` subcommand.
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_buddy_percept_top10_stress.sh <ranks>    e.g. 1,10 or 3,8

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here.

# This script only ever runs on a DAS6 node, where the ArtELingo data lives
# under /local/wding/..., not this container's /data/PDD, /data/SSD2 paths;
# real_data.py reads these two env vars (defaulting to the local-only paths).
export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo

: "${1:?usage: run_buddy_percept_top10_stress.sh <ranks, e.g. 1,10>}"

exec python src/test/20260928_buddy_percept_sweep/run_top10_stress.py stress \
    --finalists src/test/20260928_buddy_percept_sweep/finalists.json \
    --ranks "$1"
