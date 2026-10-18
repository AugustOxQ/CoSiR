#!/usr/bin/env bash
# Three aspect workers, each with eight CPU threads, after the independent head gate.
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smoke=''
if (( $# )); then
    if [[ $# != 2 || $1 != --smoke ]]; then echo "Usage: $0 [--smoke DIR]" >&2; exit 2; fi
    smoke=$(realpath -m -- "$2")
fi
root=${smoke:-$here}
queue=cpu
source "$here/pc_queue_common.sh"
export CUDA_VISIBLE_DEVICES=''
cd "$here/../../.." || exit 1
rows=("$python_bin" -u "$here/pc_rows.py")
smoke_args=()
if [[ -n "$smoke" ]]; then
    rows=("$python_bin" -u "$here/pc_queue.py" rows "$smoke")
    smoke_args=(--smoke)
fi
step build_graphs "${rows[@]}" --build-graphs
graph_code=$?
step gate_train "${rows[@]}" --train --set gate "${smoke_args[@]}"
gate_train_code=$?
step gate_check "${rows[@]}" --gate-check "${smoke_args[@]}"
gate_code=$?
if (( graph_code == 0 && gate_train_code == 0 && gate_code == 0 )); then
    pids=()
    for aspect in emotion style genre; do
        (
            failed=0
            for row in row6 row10 row13; do
                if [[ $row == row10 && $aspect == genre ]]; then continue; fi
                step "${row}_${aspect}" "${rows[@]}" --train --set "$row" --aspects "$aspect" "${smoke_args[@]}"
            done
            exit "$failed"
        ) &
        pids+=("$!")
    done
    for index in "${!pids[@]}"; do
        wait "${pids[$index]}"
        code=$?
        codes="$codes worker${index}=$code"
        if (( code != 0 )); then failed=1; fi
    done
else
    skip head_rows blocked
fi
step rows_score "${rows[@]}" --score "${smoke_args[@]}"
score_args=()
if [[ -n "$smoke" ]]; then score_args=(--outdir "$smoke"); fi
step sheet "$python_bin" -u "$here/pc_score.py" "${score_args[@]}"
finish
