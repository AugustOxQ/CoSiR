#! /bin/bash
set -euo pipefail

# Launch a W&B sweep agent for the buddy-percept comprehensive sweep
# (docs/superpowers/plans/2026-09-28-buddy-percept-sweep.md).
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_buddy_percept_sweep_agent.sh <entity/project/sweep_id> [--count N]

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here (the node's
# conda lives under /var/scratch/wding/miniconda3, not ~/miniconda3).

# This script only ever runs on a DAS6 node (via `cluster launch`), where
# the ArtELingo data lives under REMOTE_ROOT (/local/wding/...), not this
# container's /data/PDD, /data/SSD2 paths. run_pipeline.py and
# run_percept_stage2_pilot.py already read these two env vars (defaulting
# to the local-only paths when unset), and real_data.py routes the
# held-out split's otherwise-hardcoded constants through them too.
export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo

SWEEP_ID="${1:?usage: run_buddy_percept_sweep_agent.sh <sweep_id> [wandb agent args...]}"
shift

exec wandb agent "$SWEEP_ID" "$@"
