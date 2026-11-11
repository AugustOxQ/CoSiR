#!/usr/bin/env bash
# Stage 1b (spec 3.5): at 1a's pick (lambda, W, stop rule) the three forms x balance off/on = 6 runs, one shared with 1a
# (the plain, balance-off arm is the pick itself: train skips it as verified). The teacher cache is written first
# (the teacher form needs it before its first run). Then val per arm, sheet.  --picks FILE: the picks file.
source "$(dirname "${BASH_SOURCE[0]}")/r21_queue_lib.sh"
r21_args "$@"
if [ "$R21_DRY" = 1 ] && [ ! -f "$R21_PICKS" ]; then echo "no picks file: $R21_PICKS" >&2; exit 1; fi
BASE="$(r21_pick stage1a)"                       # dec-plain_l{lam}_w{W}_{stop}
SUFFIX="${BASE#dec-plain_}"
ARMS=()
for form in plain start teacher; do ARMS+=("dec-${form}_${SUFFIX}" "dec-${form}_${SUFFIX}_bal"); done
r21_cpu 1b - "${R21_RUN[@]}" teacher
for arm in "${ARMS[@]}"; do
  if [ "$arm" != "$BASE" ]; then r21_gpu 1b 15m "${R21_RUN[@]}" train --arm "$arm" --seeds 0 --device cuda; fi
done
for arm in "${ARMS[@]}"; do r21_cpu 1b - "${R21_SCORE[@]}" val --arm "$arm" --seed 0; done
r21_cpu 1b - "${R21_SCORE[@]}" sheet
r21_done stage1b
