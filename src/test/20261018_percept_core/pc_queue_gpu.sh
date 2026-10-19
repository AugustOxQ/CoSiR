#!/usr/bin/env bash
# Submit one locked background job; --foreground waits for its final exit code.
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smoke=''
foreground=0
worker=0
lock_wait=${PC_LOCK_WAIT:-14400}
while (( $# )); do
    case "$1" in
        --smoke) smoke=$(realpath -m -- "$2"); shift 2 ;;
        --foreground) foreground=1; shift ;;
        --worker) worker=1; shift ;;
        --lock-wait) lock_wait=$2; shift 2 ;;
        *) echo "Usage: $0 [--smoke DIR] [--foreground] [--lock-wait SECONDS]" >&2; exit 2 ;;
    esac
done
root=${smoke:-$here}
run_locked() {
    local here=$1 root=$2 lock_wait=$3 code
    shift 3
    flock -o -w "$lock_wait" /tmp/gpu0.lock bash "$here/pc_queue_gpu.sh" "$@"
    code=$?
    if ! awk '
        /^PC_gpu_SUBMIT / {done=0}
        /^PC_gpu_DONE / {done=1}
        END {exit !done}
    ' "$root/pc_queue_status.txt"; then
        printf 'PC_gpu_DONE lock_or_worker=%s failed=1 %s\n' "$code" "$(date -Iseconds)" >> "$root/pc_queue_status.txt"
    fi
    return "$code"
}
if (( ! worker )); then
    mkdir -p "$root"
    printf 'PC_gpu_SUBMIT %s\n' "$(date -Iseconds)" >> "$root/pc_queue_status.txt"
    args=(--worker)
    if [[ -n "$smoke" ]]; then args+=(--smoke "$smoke"); fi
    if (( foreground )); then
        run_locked "$here" "$root" "$lock_wait" "${args[@]}"
        exit "$?"
    fi
    export -f run_locked
    nohup bash -c 'run_locked "$@"' _ "$here" "$root" "$lock_wait" "${args[@]}" > "$root/pc_queue_gpu.log" 2>&1 < /dev/null &
    printf 'GPU queue pid=%s log=%s\n' "$!" "$root/pc_queue_gpu.log"
    exit 0
fi
queue=gpu
source "$here/pc_queue_common.sh"
cd "$here/../../.." || exit 1
stage0_args=()
if [[ -n "$smoke" ]]; then stage0_args=(--smoke "$smoke"); fi
step stage0 "$python_bin" -u "$here/pc_stage0.py" "${stage0_args[@]}"
stage0_code=$?
# Extraction still runs after a failed regression. A finished full store is immutable.
roberta_args=()
if [[ -n "$smoke" ]]; then
    roberta_args=(--smoke --count 512 --outdir "$smoke/roberta")
fi
if [[ -z "$smoke" ]] && "$python_bin" -c 'import json; from pathlib import Path; p=Path("/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions/done.json"); raise SystemExit(0 if p.exists() and json.loads(p.read_text())["count"] == 99999 else 1)'; then
    skip roberta skip
    roberta_code=0
else
    step roberta "$python_bin" -u "$here/pc_roberta.py" "${roberta_args[@]}"
    roberta_code=$?
fi
seeds=(0 1 2)
run_args=()
if [[ -n "$smoke" ]]; then seeds=(0); run_args=(--smoke "$smoke"); fi
declare -A stage1_codes
for version in clip_roberta qwen_qwen qwen_roberta; do
    for seed in "${seeds[@]}"; do
        name="stage1_${version}_s${seed}"
        if (( stage0_code == 0 && roberta_code == 0 )); then
            step "$name" "$python_bin" -u "$here/pc_stage1.py" --version "$version" --seed "$seed" "${run_args[@]}"
            stage1_codes["${version}_${seed}"]=$?
        else
            skip "$name" blocked
            stage1_codes["${version}_${seed}"]=1
        fi
    done
done
for arm in P1 P2 P3 P4 P5; do
    case "$arm" in P1|P2) version=clip_roberta ;; P3|P4) version=qwen_qwen ;; P5) version=qwen_roberta ;; esac
    for seed in "${seeds[@]}"; do
        name="stage2_${arm}_s${seed}"
        if (( ${stage1_codes["${version}_${seed}"]} == 0 )); then
            step "$name" "$python_bin" -u "$here/pc_stage2.py" --arm "$arm" --seed "$seed" "${run_args[@]}"
        else
            skip "$name" blocked
        fi
    done
done
if (( ${stage1_codes["clip_roberta_0"]} == 0 )); then
    first_root="$here/outputs"
    repeat_args=(--outroot "$here/outputs/repeat")
    if [[ -n "$smoke" ]]; then
        first_root="$smoke"
        repeat_args=(--smoke "$smoke/repeat")
    fi
    step stage1_repeat "$python_bin" -u "$here/pc_stage1.py" --version clip_roberta --seed 0 "${repeat_args[@]}"
    step repeat_check "$python_bin" -u "$here/pc_repeat.py" --first "$first_root" --second "$first_root/repeat" --out "$first_root/stage1_repeat.json"
else
    skip stage1_repeat blocked
    skip repeat_check blocked
fi
finish
