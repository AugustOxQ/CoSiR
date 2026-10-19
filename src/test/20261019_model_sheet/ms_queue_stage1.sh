#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/ms_queue_common.sh"
stage=1
status=queue_stage1_status.txt
printf 'MS_STAGE1_START %s\n' "$(date -Iseconds)" >> "$status"
if ! step require_stage0 env CUDA_VISIBLE_DEVICES='' "$python_bin" -u ms_stage0.py --require-pass; then
    finish
fi
for row in A1 T1 T2 T3 T4 T5 D1 D2 D3 D4 D5; do
    step "train_${row}" gpu --rows "$row" --seeds 0
done
step score cpu --score --sheet
step picks cpu --pick-seeds
finish
