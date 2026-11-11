#!/usr/bin/env bash
# Stage 1a (spec 3.5): the 8 plain runs, lambda in {1, 10} x W in {1, 5} x stop rule {S1, S2}, seed 0; then val per arm, sheet.
source "$(dirname "${BASH_SOURCE[0]}")/r21_queue_lib.sh"
r21_args "$@"
ARMS=()
for lam in 1 10; do for w in 1 5; do for stop in S1 S2; do ARMS+=("dec-plain_l${lam}_w${w}_${stop}"); done; done; done
for arm in "${ARMS[@]}"; do r21_gpu 1a 15m "${R21_RUN[@]}" train --arm "$arm" --seeds 0 --device cuda; done
for arm in "${ARMS[@]}"; do r21_cpu 1a - "${R21_SCORE[@]}" val --arm "$arm" --seed 0; done
r21_cpu 1a - "${R21_SCORE[@]}" sheet
r21_done stage1a
