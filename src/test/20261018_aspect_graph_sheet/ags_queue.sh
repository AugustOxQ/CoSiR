#!/usr/bin/env bash
# Controller-run CPU queue. Each worker records its exit as soon as it finishes.
set -u
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
python_bin=/root/miniconda3/envs/CoSiR/bin/python
status=queue_status.txt
codes=''
failed=0
record() {
    printf '%s=%s\n' "$1" "$2" >> "$status"
}
step() {
    local label=$1
    shift
    "$@" > "queue_${label}.log" 2>&1
    local code=$?
    record "$label" "$code"
    codes="$codes $label=$code"
    if (( code != 0 )); then failed=1; fi
}
parallel_train() {
    local phase=$1
    shift
    local aspect code index
    local -a pids=() names=()
    for aspect in emotion style genre; do
        names+=("${phase}_${aspect}")
        (
            "$python_bin" -u ags_head.py --aspects "$aspect" "$@" > "queue_${phase}_${aspect}.log" 2>&1
            code=$?
            # One short append per worker; status lines cannot interleave.
            record "${phase}_${aspect}" "$code"
            exit "$code"
        ) &
        pids+=("$!")
    done
    for index in "${!pids[@]}"; do
        wait "${pids[$index]}"
        code=$?
        codes="$codes ${names[$index]}=$code"
        if (( code != 0 )); then failed=1; fi
    done
}
# Append a run boundary, preserving any previous controller run.
printf 'AGS_START %s\n' "$(TZ=Europe/Amsterdam date '+%F %H:%M')" >> "$status"
step graphs "$python_bin" -u ags_graphs.py
parallel_train seed0 --seeds 0
step score0 "$python_bin" -u ags_head.py --score
step pick_seeds "$python_bin" -u ags_head.py --pick-seeds
parallel_train seeds12 --seed-rows --seeds 1 2
step score012 "$python_bin" -u ags_head.py --score
printf 'AGS_DONE%s %s\n' "$codes" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" >> "$status"
exit "$failed"
