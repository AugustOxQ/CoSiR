#!/usr/bin/env bash
# Shared queue logging. Source after setting root, queue and smoke.
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python_bin=${PC_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}
status="$root/pc_queue_status.txt"
mkdir -p "$root/logs" "$root/outputs/queue_tmp"
export TMPDIR="$root/outputs/queue_tmp"
ledger="$root/logs/pc_queue_${queue}_steps.txt"
: > "$ledger"
failed=0
codes=''
printf 'PC_%s_START %s\n' "$queue" "$(date -Iseconds)" >> "$status"
record() {
    printf '%s=%s seconds=%s\n' "$1" "$2" "$3" >> "$status"
    printf '%s=%s seconds=%s\n' "$1" "$2" "$3" >> "$ledger"
}
step() {
    local name=$1 start=$EPOCHREALTIME code elapsed
    shift
    "$@" > "$root/logs/${name}.log" 2>&1
    code=$?
    elapsed=$(awk -v start="$start" -v end="$EPOCHREALTIME" 'BEGIN {printf "%.3f", end-start}')
    record "$name" "$code" "$elapsed"
    codes="$codes $name=$code"
    if (( code != 0 )); then failed=1; fi
    return "$code"
}
skip() {
    record "$1" "$2" 0
    codes="$codes $1=$2"
}
finish() {
    if [[ -n "$smoke" ]]; then
        "$python_bin" "$here/pc_queue.py" times --status "$ledger" --outdir "$root" --queue "$queue" || failed=1
    fi
    printf 'PC_%s_DONE%s failed=%s %s\n' "$queue" "$codes" "$failed" "$(date -Iseconds)" >> "$status"
    return "$failed"
}
