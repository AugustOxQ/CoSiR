#!/usr/bin/env bash
# Each CPU step has its own `res-queue cpu` reservation; PP_CPU_JOBS (default 3) are open at once.
# A barrier step waits for all earlier steps and holds back all later ones. A failed step never stops the others.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
stage=${1:?usage: pp_run_cpu.sh 0|1|2|3|4 [--step NAME ...] [--enc E ...]}; shift
plan=$("$PY" "$HERE/pp_run.py" "$stage" --list "$@") || exit $?
# Each reservation executes exactly one step: drop the --step selection from the arguments.
args=(); selecting=0
for arg in "$@"; do
    if [[ "$arg" == --step ]]; then selecting=1; continue; fi
    [[ "$arg" == --* ]] && selecting=0
    [[ "$selecting" == 0 ]] && args+=("$arg")
done
jobs_max=${PP_CPU_JOBS:-3}
rcf=$(mktemp); trap 'rm -f "$rcf"' EXIT
while IFS=$'\t' read -r step est; do
    [[ -n "$step" ]] || continue
    barrier=0
    [[ "$step" =~ (^|_)(merge|merge_encoders|collect|decide|diagnostics)$ ]] && barrier=1
    if (( barrier )); then wait; else while (( $(jobs -rp | wc -l) >= jobs_max )); do wait -n; done; fi
    ( res-queue cpu --label "r19:$step" --est "$est" -- "$PY" "$HERE/pp_run.py" "$stage" --step "$step" "${args[@]}"
      r=$?; [[ "$r" == 0 ]] || echo "$r" >> "$rcf" ) &
    if (( barrier )); then wait; fi
done <<< "$plan"
wait
rc=0; [[ -s "$rcf" ]] && rc=$(tail -n1 "$rcf")
exit "$rc"
