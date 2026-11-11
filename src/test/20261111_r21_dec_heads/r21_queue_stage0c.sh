#!/usr/bin/env bash
# Stage 0c (spec 3.5): the dec-off rerun of c* seed 0, then its validation scores and the sheet (prints gate 0c's status).
source "$(dirname "${BASH_SOURCE[0]}")/r21_queue_lib.sh"
r21_args "$@"
r21_gpu 0c 15m "${R21_RUN[@]}" train --arm dec-off --seeds 0 --device cuda
r21_cpu 0c - "${R21_SCORE[@]}" val --arm dec-off --seed 0
r21_cpu 0c - "${R21_SCORE[@]}" sheet
r21_done stage0c
