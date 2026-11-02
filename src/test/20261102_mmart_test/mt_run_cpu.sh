#!/usr/bin/env bash
# r15 CPU queue (ticket 07). No GPU anywhere. Usage: bash mt_run_cpu.sh <part> [extra mt_run.py args]
#   first      stage 0 (gates, exports, cut sets, status), then stage 1 (ranges, picks, records); then STOP: the controller
#              commits records/stage1_*.json, stage1_teacher.json, status.json, cutsets.json (the runner prints the command)
#   stage2     scorers per encoder, then mt_outcome decide -> OUT/stage2/outcomes.json (refuses unless stage 1 is committed)
#   verdicts   copies outcomes.json to records/verdicts.json, only after the committed records/c13.json says pass (C13)
#   stage3     mt_diag.py all (after the verdicts)
#   stage0, stage1   one stage alone
#   smoke      all five in order, no commit and no C13 (needs MT_SMOKE=1 MT_OUTPUTS=<scratch>)
# Extra arguments go to mt_run.py (for `first`: to stage 0 and stage 1 both, so only --enc is useful there).
# Ends with one line "QUEUE <part> OK|FAILED rc=<rc> seconds=<s>" (also appended to $OUT/queue_summary.txt).
# Env: MT_OUTPUTS as everywhere; MT_RUN_TRAILER = trailer lines for the printed commit message.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT="$(cd "$HERE" && "$PY" -c 'import mt_paths; print(mt_paths.OUT)')"
part="${1:?part: first|stage0|stage1|stage2|verdicts|stage3|smoke}"; shift
mkdir -p "$OUT"
t0=$SECONDS
run() { "$PY" "$HERE/mt_run.py" "$@"; }
case "$part" in
    first) run stage0 "$@"; rc0=$?; run stage1 "$@"; rc=$?; [[ $rc0 -ne 0 && $rc -eq 0 ]] && rc=$rc0 ;;
    stage0|stage1|stage2|stage3|verdicts) run "$part" "$@"; rc=$? ;;
    smoke)
        [[ "${MT_SMOKE:-0}" == 1 && -n "${MT_OUTPUTS:-}" ]] || { echo "smoke needs MT_SMOKE=1 and MT_OUTPUTS=<scratch>" >&2; exit 2; }
        rc=0
        for p in stage0 stage1 stage2 verdicts stage3; do run "$p" "$@"; r=$?; [[ $r -ne 0 ]] && rc=$r; done ;;
    *) echo "unknown part $part" >&2; exit 2 ;;
esac
line="QUEUE $part $([[ $rc -eq 0 ]] && echo OK || echo FAILED) rc=$rc seconds=$((SECONDS - t0))"
echo "$line" | tee -a "$OUT/queue_summary.txt"
exit $rc
