#! /bin/bash
set -euo pipefail

# Launch a W&B sweep agent for the buddy-percept comprehensive sweep
# (docs/superpowers/plans/2026-09-28-buddy-percept-sweep.md).
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_buddy_percept_sweep_agent.sh <entity/project/sweep_id> [--count N]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoSiR

SWEEP_ID="${1:?usage: run_buddy_percept_sweep_agent.sh <sweep_id> [wandb agent args...]}"
shift

exec wandb agent "$SWEEP_ID" "$@"
