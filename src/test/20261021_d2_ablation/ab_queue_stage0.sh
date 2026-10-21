#!/usr/bin/env bash
stage=0
source "$(dirname -- "${BASH_SOURCE[0]}")/ab_queue_common.sh"
for row in R0 R2; do
    step "train_${row}" gpu --rows "$row" --seeds 0 1 2
done
step check cpu ab_stage0.py --outputs "$outputs"
finish
