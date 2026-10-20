#!/usr/bin/env bash
# Sourced by the stage 2 queues (copy of ct_queue_common.sh; stage 1's file is hashed, so not edited).
# Outputs go to outputs/stage2 (CT2_OUTPUTS overrides, for the queue test); the GPU lock file is CT_GPU_LOCK.
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin=${CT_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}
lock_wait=${CT_LOCK_WAIT:-14400}
lock_file=${CT_GPU_LOCK:-/tmp/gpu0.lock}
outputs=${CT2_OUTPUTS:-"$here/outputs/stage2"}
mkdir -p "$outputs/logs"
status="$outputs/queue_${queue}_status.txt"
printf 'CT2_%s_START %s\n' "$queue" "$(date -Iseconds)" >> "$status"
failed=0
step() {
    local label=$1 code start=$SECONDS
    shift
    "$@" > "$outputs/logs/${queue}_${label}.log" 2>&1
    code=$?
    printf '%s=%s seconds=%s\n' "$label" "$code" "$((SECONDS - start))" | tee -a "$status"
    if (( code != 0 )); then failed=1; fi
    return "$code"
}
gpu() {
    flock -o -w "$lock_wait" "$lock_file" env CUDA_VISIBLE_DEVICES=0 "$python_bin" -u "$@"
}
cpu() {
    env CUDA_VISIBLE_DEVICES='' "$python_bin" -u "$@"
}
# Run python steps of ct2_run.py with the queue's output folder; extra arguments follow the step name.
run_args=(--root "$outputs")
finish() {
    printf 'CT2_%s_DONE failed=%s %s\n' "$queue" "$failed" "$(date -Iseconds)" >> "$status"
    exit "$failed"
}
