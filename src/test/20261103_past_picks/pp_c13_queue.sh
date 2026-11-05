#!/usr/bin/env bash
# r19 C13 full derive: one `res-queue cpu` reservation per section, at most 3 at once, in four dependency levels.
# Level 1: the set and the ranges. Level 2: picks, and the SemArt units of the older encoders. Level 3: the SemArt units
# of the new three, which need their picks. Level 4: cells and cost. A failed section never stops the others; a section
# that needs it refuses by itself. Rerun to finish: complete sections are skipped.
# usage: pp_c13_queue.sh [HHMM]    wait until that Amsterdam time first (0 = start now; default 1900).
# Logs: <OUT>/logs/<section>.log. Last line: `C13 derive done sections=<n> failed=<m>`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
OUT=/data/SSD2/cosir_r19/20261103_past_picks/c13
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PP_SMOKE PP_OUTPUTS
start=${1:-1900}
if [[ "$start" != 0 ]]; then
    until (( 10#$(TZ=Europe/Amsterdam date +%H%M) >= 10#$start )); do sleep 30; done
fi
mkdir -p "$OUT/logs"
plan=$("$PY" "$HERE/pp_c13.py" sections) || exit $?
level() {
    case "$1" in
        set|ranges:*) echo 1 ;;
        picks:*|semval:qwen|semval:siglip2|semval:siglip1|semval:clip_l14) echo 2 ;;
        semval:*) echo 3 ;;
        *) echo 4 ;;
    esac
}
rcf=$(mktemp); trap 'rm -f "$rcf"' EXIT
n=0
for want in 1 2 3 4; do
    while IFS=$'\t' read -r name est; do
        [[ -n "$name" && "$(level "$name")" == "$want" ]] || continue
        n=$((n + 1))
        minutes=${est%m}; minutes=${minutes%.*}; (( 10#${minutes:-0} >= 1 )) || minutes=1
        while (( $(jobs -rp | wc -l) >= 3 )); do wait -n; done
        ( res-queue cpu --label "r19:c13_${name//:/_}" --est "${minutes}m" -- \
              "$PY" "$HERE/pp_c13.py" derive --sections "$name" > "$OUT/logs/$name.log" 2>&1
          r=$?; [[ "$r" == 0 ]] || echo "$name rc=$r" >> "$rcf" ) &
    done <<< "$plan"
    wait
done
[[ -s "$rcf" ]] && cat "$rcf"
failed=$(wc -l < "$rcf")
echo "C13 derive done sections=$n failed=$failed $(TZ=Europe/Amsterdam date '+%F %H:%M')"
[[ "$failed" == 0 ]]
