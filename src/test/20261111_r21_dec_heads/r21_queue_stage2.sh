#!/usr/bin/env bash
# Stage 2 (spec 3.5): dec-off and the three per-form picks of the picks file on seeds 0 to 4 (seed 0 runs exist from stages
# 0c and 1: train skips verified ones), then their held exports (on the GPU, the weights' training device, as c*), then
# the table (1 to 2 h of one CPU slot) and the diagnostics.  --picks FILE: the picks file.
source "$(dirname "${BASH_SOURCE[0]}")/r21_queue_lib.sh"
r21_args "$@"
if [ "$R21_DRY" = 1 ] && [ ! -f "$R21_PICKS" ]; then echo "no picks file: $R21_PICKS" >&2; exit 1; fi
ARMS=(dec-off "$(r21_pick stage1b plain)" "$(r21_pick stage1b start)" "$(r21_pick stage1b teacher)")
# the teacher form's cache is verified by train (it reads the cache the stage 1b `teacher` step wrote)
for arm in "${ARMS[@]}"; do for seed in 0 1 2 3 4; do r21_gpu 2 15m "${R21_RUN[@]}" train --arm "$arm" --seeds "$seed" --device cuda; done; done
for arm in "${ARMS[@]}"; do r21_gpu 2 10m "${R21_RUN[@]}" held --arm "$arm" --seeds 0 1 2 3 4; done
r21_cpu 2 2h "${R21_TABLE[@]}" table --picks "$R21_PICKS"
r21_cpu 2 - "${R21_TABLE[@]}" diag --picks "$R21_PICKS"
r21_done stage2
