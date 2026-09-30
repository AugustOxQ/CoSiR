#! /bin/bash
set -euo pipefail

# Confirmation re-run of a buddy-percept sweep finalist on a DAS6 GPU node:
# forwards all arguments to the `stress` subcommand (e.g. --set transfer_k=20,
# --tag, --independent-ami), reading the frozen finalists.json.
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_buddy_percept_confirm.sh --ranks 1 --seeds 11,22 \
#       --set transfer_k=20 --tag k20 --independent-ami

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here.

# This script only ever runs on a DAS6 node, where the ArtELingo data lives
# under /local/wding/..., not this container's /data/PDD, /data/SSD2 paths;
# real_data.py and pilot_metrics.py read these two env vars (defaulting to the
# local-only paths).
export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo

[ "$#" -ge 1 ] || { echo "usage: run_buddy_percept_confirm.sh <stress args, e.g. --ranks 1 --set transfer_k=20 --tag k20 --independent-ami>" >&2; exit 2; }

exec python src/test/20260928_buddy_percept_sweep/run_top10_stress.py stress \
    --finalists src/test/20260928_buddy_percept_sweep/finalists.json "$@"
