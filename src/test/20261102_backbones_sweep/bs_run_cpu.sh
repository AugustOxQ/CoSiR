#!/usr/bin/env bash
# r14 queue, local CPU parts (ticket 12a). Usage: bash bs_run_cpu.sh <part>, part one of:
#   stage0-cpu   the CPU gates of stage 0 (ii, iii, vi, vii, x, xi, xii)
#   stage1       seed ranges and picks of the encoders still in part a; commits RECORDS/stage1_<enc>.json (after the node's part a)
#   stage2       EmoSet r_E and picks; commits RECORDS/stage2_emoset.json
#   stage3       cells and outcomes (needs status.json: stage 0, part a and stage 2 in)
#   stage4       the three part b sheets
#   stage5       diagnostics (bs_diag, ticket 12b)
# Extra arguments go to bs_run.py. One summary line at the end: "QUEUE <part> OK|FAILED rc=<rc> seconds=<s>" (also appended to
# $OUT/queue_summary.txt). Env: BS_OUTPUTS as everywhere; BS_RUN_NO_COMMIT=1 for a smoke.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT="${BS_OUTPUTS:-/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs}"
part="${1:?part: stage0-cpu|stage1|stage2|stage3|stage4|stage5}"; shift
case "$part" in
    stage0-cpu) args=(stage0 --parts cpu) ;;
    stage1|stage2|stage3|stage4|stage5) args=("$part") ;;
    *) echo "unknown part $part" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
t0=$SECONDS
"$PY" "$HERE/bs_run.py" "${args[@]}" "$@"
rc=$?
line="QUEUE $part $([[ $rc -eq 0 ]] && echo OK || echo FAILED) rc=$rc seconds=$((SECONDS - t0))"
echo "$line" | tee -a "$OUT/queue_summary.txt"
exit $rc
