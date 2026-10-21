#!/usr/bin/env bash
stage=1
source "$(dirname -- "${BASH_SOURCE[0]}")/ab_queue_common.sh"
if ! step require_stage0 cpu ab_stage0.py --require-pass --outputs "$outputs"; then
    finish
fi
main_round=/project/CoSiR/src/test/20261021_d2_ablation
if [[ ! -f $main_round/graphs/manifest.json ]]; then
    if ! step graphs cpu ab_graphs.py --out "$main_round"; then finish; fi
fi
rows=(R1 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12)
if [[ ${AB_SMOKE:-0} == 1 ]]; then rows=(R3 R5 R7 R9 R10); fi
for row in "${rows[@]}"; do
    step "train_${row}" gpu --rows "$row" --seeds 0 1 2
done
finish
