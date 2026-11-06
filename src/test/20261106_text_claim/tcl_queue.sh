#!/usr/bin/env bash
# r23 text claim: one queue job per step, three chains. Usage: tcl_queue.sh val|held|post
#   val   stage 0 (extraction, gates i to iv) and stage 1 (validation scoring, ranges, picks); ends with `tcl_run.py status`.
#         Then the run chat commits records/stage1_*.json and status.json.
#   held  stage 2 (the bar rows of art_held and mm_test narrative, then decide). Then C13 (run chat), then `tcl_run.py verdicts`.
#   post  stage 3 (side rows, the other four MMArt kinds, diagnostics); the runner refuses without records/verdicts.json.
# The first failed step stops the chain (set -e): trace it, fix, rerun; every step resumes (extraction by done.json, scoring
# by its done files). D13: a job with an estimate waits for 19:00 if it would overlap 10:00 to 19:00 on a weekday.
# Durations are estimates, not measurements.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
OUT="${TCL_OUTPUTS:-/data/SSD2/cosir_r23/20261106_text_claim}"
L="$OUT/logs"
mkdir -p "$L"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$R"
chain="${1:?usage: tcl_queue.sh val|held|post}"

wait_for_19() {   # $1 = estimate in minutes
  local s; s="$("$PY" tcl_run.py _wait --est "$1")"
  if [ "$s" -gt 0 ]; then echo "$(date '+%F %T') r23: waiting ${s}s for 19:00 (D13)"; sleep "$s"; fi
}
gpu_short() {     # stage step
  res-queue gpu --short --label "r23:$2" -- "$PY" tcl_run.py "$1" --only "$2" > "$L/$2.log" 2>&1
}
gpu_est() {       # stage step minutes
  wait_for_19 "$3"
  res-queue gpu --est "${3}m" --label "r23:$2" -- "$PY" tcl_run.py "$1" --only "$2" > "$L/$2.log" 2>&1
}
cpu_short() {     # stage step (small CPU steps)
  res-queue cpu --est 10m --label "r23:$2" -- "$PY" tcl_run.py "$1" --only "$2" > "$L/$2.log" 2>&1
}

case "$chain" in
val)
  gpu_est stage0 extract_art_teacher 40
  gpu_est stage0 extract_mm_plain 20
  for k in narrative formal emotional historical unified; do
    gpu_short stage0 "extract_mm_teacher_${k}_val"
    gpu_est stage0 "extract_mm_teacher_${k}_test" 20
  done
  gpu_short stage0 extract_roberta
  gpu_short stage0 gate_i
  gpu_short stage0 gate_ii
  gpu_est stage0 gate_iii 20
  cpu_short stage0 gate_iv
  gpu_est stage1 score_art_val 15
  for k in narrative formal emotional historical unified; do gpu_short stage1 "score_mm_val_$k"; done
  cpu_short stage1 stage1_art
  cpu_short stage1 stage1_mm
  "$PY" tcl_run.py status > "$L/status.log" 2>&1
  echo "r23 val chain done: $(tr -d '\n ' < "$L/status.log"). Commit records/stage1_*.json and status.json (git add -f), then run: tcl_queue.sh held"
  ;;
held)
  gpu_est stage2 score_art_held 30
  gpu_short stage2 score_mm_test_narrative
  wait_for_19 45
  res-queue cpu --est 45m --label r23:decide -- "$PY" tcl_run.py stage2 --only decide > "$L/decide.log" 2>&1
  echo "r23 held chain done: $(grep -h 'TCL_RUN stage2 decide' "$L/decide.log" | tail -1). Next: C13, then tcl_run.py verdicts"
  ;;
post)
  gpu_est stage3 score_art_held_side 40
  for k in formal emotional historical unified; do gpu_short stage3 "score_mm_test_$k"; done
  gpu_est stage3 diag 30
  echo "r23 post chain done: $(grep -h 'TCL_RUN stage3 diag' "$L/diag.log" | tail -1)"
  ;;
*)
  echo "usage: tcl_queue.sh val|held|post" >&2; exit 2 ;;
esac
