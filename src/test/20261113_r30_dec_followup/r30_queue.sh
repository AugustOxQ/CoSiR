#!/usr/bin/env bash
# r30: wait until r31's runs end, then run the three r30 steps on the CPU queue.
# "r31 ended": a job labelled r29:* or r31:* (r31 runs r29's runner) was seen in res-queue, and then no such job ran
# or waited for QUIET_MIN minutes in a row. Gives up when no such job was seen by DEADLINE (Amsterdam time).
# Ends with one summary line.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/data/SSD2/cosir_r21/20261111_dec_heads/r30_followup
LOG=$OUT/logs
PY=/root/miniconda3/envs/CoSiR/bin/python
QUIET_MIN=${QUIET_MIN:-60}
POLL_S=${POLL_S:-120}
DEADLINE=${DEADLINE:-"2026-11-14 11:00"}
mkdir -p "$LOG"

now() { TZ=Europe/Amsterdam date '+%F %H:%M'; }
r31_jobs() {
  res-queue status --json 2>/dev/null | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
print(sum(1 for k in ("gpu", "cpu") for j in d[k]["jobs"] if str(j.get("label", "")).startswith(("r29:", "r31:"))))'
}

seen=0; quiet=0
echo "$(now) wait start (quiet ${QUIET_MIN} min, deadline $DEADLINE)" >> "$LOG/wait.log"
while :; do
  n=$(r31_jobs)
  if [ -z "$n" ]; then n=1; echo "$(now) status read failed; counted as busy" >> "$LOG/wait.log"; fi
  if [ "$n" -gt 0 ]; then
    [ "$seen" -eq 0 ] && echo "$(now) r31 job seen" >> "$LOG/wait.log"
    seen=1; quiet=0
  elif [ "$seen" -eq 1 ]; then
    quiet=$((quiet + POLL_S))
    if [ "$quiet" -ge $((QUIET_MIN * 60)) ]; then echo "$(now) r31 quiet for ${QUIET_MIN} min" >> "$LOG/wait.log"; break; fi
  elif [[ "$(now)" > "$DEADLINE" ]]; then
    echo "r30 queue: NOT RUN, no r29:/r31: job seen by $DEADLINE ($(now))"; exit 3
  fi
  sleep "$POLL_S"
done

fails=()
for step in umap valence plot; do
  echo "$(now) $step start" >> "$LOG/wait.log"
  res-queue cpu --short --label "r30:$step" -- env PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/r30_followup.py" "$step" \
    > "$LOG/$step.log" 2>&1 || fails+=("$step")
  echo "$(now) $step end (exit status logged; fails so far: ${fails[*]:-none})" >> "$LOG/wait.log"
  [ "${#fails[@]}" -gt 0 ] && break
done
if [ "${#fails[@]}" -eq 0 ]; then echo "r30 queue: OK, 3 steps done ($(now)); outputs in $OUT"
else echo "r30 queue: FAILED at ${fails[*]} ($(now)); see $LOG"; exit 1; fi
