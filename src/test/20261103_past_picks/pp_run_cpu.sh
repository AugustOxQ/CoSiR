#!/usr/bin/env bash
# Each CPU unit has its own queue reservation. A failed unit does not stop other encoders.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
stage=${1:?usage: pp_run_cpu.sh 0|1|2|3|4 [--step NAME ...] [--enc E ...]}; shift
plan=$("$PY" "$HERE/pp_run.py" "$stage" --list "$@") || exit $?
rc=0
while IFS=$'\t' read -r step est; do
    [[ -n "$step" ]] || continue
    # Remove the selection argument: each reservation executes exactly one step.
    args=(); selecting=0
    for arg in "$@"; do
        if [[ "$arg" == --step ]]; then selecting=1; continue; fi
        [[ "$arg" == --* ]] && selecting=0
        [[ "$selecting" == 0 ]] && args+=("$arg")
    done
    res-queue cpu --label "r19:$step" --est "$est" -- "$PY" "$HERE/pp_run.py" "$stage" --step "$step" "${args[@]}"
    r=$?; [[ "$r" == 0 ]] || rc=$r
done <<< "$plan"
exit "$rc"
