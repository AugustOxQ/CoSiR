#!/usr/bin/env bash
# Sourced by the three GPU-stage queues.
set -u
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python_bin=/root/miniconda3/envs/CoSiR/bin/python
codes=''
failed=0
step() {
    local label=$1 code
    shift
    "$@" > "queue_stage${stage}_${label}.log" 2>&1
    code=$?
    printf '%s=%s\n' "$label" "$code" >> "$status"
    codes="$codes $label=$code"
    if (( code != 0 )); then failed=1; fi
    return "$code"
}
gpu() {
    flock -o -w 7200 /tmp/gpu0.lock env CUDA_VISIBLE_DEVICES=0 "$python_bin" -u ms_train.py "$@" --device cuda
}
cpu() {
    env CUDA_VISIBLE_DEVICES='' "$python_bin" -u ms_sheet.py "$@"
}
finish() {
    printf 'MS_STAGE%s_DONE%s\n' "$stage" "$codes" >> "$status"
    exit "$failed"
}
