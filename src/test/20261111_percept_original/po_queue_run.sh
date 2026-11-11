#!/usr/bin/env bash
# r26 run: Stage 1, Stage 2 and held (GPU, each step through res-queue), then score and the exports (CPU). One summary
# line at the end; a failed step stops the steps after it. Start it with run_in_background.
#
#   po_queue_run.sh --smoke <scratch dir>     seed 0, smoke sizes, short GPU jobs; no records/, no exports
#   po_queue_run.sh [--at HH:MM]              the full run (seeds 0-4, both cuts); --at waits until that Amsterdam time
#                                             today before it queues Stage 1 (weekday rule: heavy runs after 19:00)
set -u
cd "$(dirname -- "${BASH_SOURCE[0]}")"
at=""; smoke=""
while [ $# -gt 0 ]; do
  case "$1" in
    --at) at=$2; shift 2 ;;
    --smoke) smoke=$2; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
py=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [ -n "$smoke" ]; then
  export PO_OUTPUTS=$smoke; mode=(--smoke); tag=smoke_
  est_s1=(--short); est_s2=(--short); est_held=(--short)
else
  mode=(); tag=""
  est_s1=(--est 60m); est_s2=(--est 45m); est_held=(--est 15m)
fi
out=${PO_OUTPUTS:-/data/SSD2/cosir_r26/20261111_percept_original/outputs}
logs=$out/logs; mkdir -p "$logs"
if [ -n "$at" ]; then
  now=$(date +%s); target=$(TZ=Europe/Amsterdam date -d "today $at" +%s)
  if [ "$target" -gt "$now" ]; then
    echo "waiting until $at ($(( (target - now) / 60 )) min)"; sleep $(( target - now ))
  fi
fi
rc=0; done_steps=""
step() {  # step <name> <res-queue args...> -- <cmd...>
  local name=$1; shift
  [ $rc -ne 0 ] && return
  echo "$(TZ=Europe/Amsterdam date '+%H:%M') start $name"
  res-queue "$@" > "$logs/${tag}${name}.log" 2>&1; rc=$?
  echo "$(TZ=Europe/Amsterdam date '+%H:%M') end $name rc=$rc (log $logs/${tag}${name}.log)"
  [ $rc -eq 0 ] && done_steps="$done_steps $name"
}
step stage1 gpu "${est_s1[@]}" --label "r26:${tag}stage1" -- "$py" -u po_run.py stage1 "${mode[@]}" --device cuda
step stage2 gpu "${est_s2[@]}" --label "r26:${tag}stage2" -- "$py" -u po_run.py stage2 "${mode[@]}" --device cuda
step held gpu "${est_held[@]}" --label "r26:${tag}held" -- "$py" -u po_run.py held "${mode[@]}" --device cuda
if [ -n "$smoke" ]; then
  step score cpu --short --label "r26:smoke_score" -- "$py" -u po_run.py score --smoke
else
  step score cpu --est 20m --label "r26:score" -- "$py" -u po_run.py score --record
  step export cpu --short --label "r26:export" -- "$py" -u po_export_val.py --test
fi
echo "R26_RUN ${smoke:+smoke }rc=$rc done:${done_steps:- none} $(TZ=Europe/Amsterdam date '+%F %H:%M')"
exit $rc
