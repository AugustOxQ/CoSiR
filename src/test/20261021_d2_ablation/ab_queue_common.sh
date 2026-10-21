#!/usr/bin/env bash
# Sourced by the stage queues; follows the model-sheet queue helpers.
set -u
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=''
python_bin=/root/miniconda3/envs/CoSiR/bin/python
outputs=${AB_OUTPUTS:-/project/CoSiR/src/test/20261021_d2_ablation/outputs}
if [[ ${AB_SMOKE:-0} == 1 ]]; then
    if [[ -z ${AB_OUTPUTS:-} || $(realpath -m -- "$outputs") != /tmp/claude-0/* ]]; then
        printf 'AB_SMOKE requires AB_OUTPUTS under /tmp/claude-0/\n' >&2
        exit 1
    fi
fi
mkdir -p -- "$outputs/logs"
status="$outputs/logs/stage${stage}_status.txt"
printf 'AB_STAGE%s_START %s\n' "$stage" "$(date -Iseconds)" >> "$status"
codes=''
failed=0
gpu_check_failed=0
step() {
    local label=$1 code
    shift
    "$@" > "$outputs/logs/stage${stage}_${label}.log" 2>&1
    code=$?
    printf '%s=%s\n' "$label" "$code" >> "$status"
    codes="$codes $label=$code"
    if (( code != 0 )); then failed=1; fi
    if (( gpu_check_failed )); then finish; fi
    return "$code"
}
finish() {
    printf 'AB_STAGE%s_DONE%s\n' "$stage" "$codes" | tee -a "$status"
    exit "$failed"
}
gpu_check() {
    local apps
    if ! nvidia-smi --query-gpu=driver_version --format=csv,noheader; then
        printf 'GPU_CHECK_FAIL\n' >> "$status"
        codes="$codes gpu_check=1"
        gpu_check_failed=1
        return 1
    fi
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
    if [[ -n $apps ]]; then
        printf 'GPU_COMPUTE_APPS_NOTE pid,used_memory:\n%s\n' "$apps" >> "$status"
    fi
}
gpu() {
    gpu_check || return 1
    local smoke_args=()
    if [[ $stage == 1 && ${AB_SMOKE:-0} == 1 ]]; then smoke_args=(--smoke); fi
    flock -o -w 7200 /tmp/gpu0.lock env CUDA_VISIBLE_DEVICES=0 "$python_bin" -u ab_train.py "$@" --device cuda --outputs "$outputs" "${smoke_args[@]}"
}
cpu() {
    env CUDA_VISIBLE_DEVICES='' "$python_bin" -u "$@"
}
