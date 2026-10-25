#!/usr/bin/env bash
# Stage 1x (exploration, decides nothing): four-view sheet, part 1 reported readings (after VP1 agrees), ceiling probe.
# CPU except the probe (GPU, short; CPU 8 to 25 min if the GPU is not given). Runs beside the GPU stages; one CPU job at a time.
queue=stage1x
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
if [[ "$dry" != 1 ]]; then exec 8> "${RB_CPU_LOCK:-/tmp/rb_cpu.lock}"; flock -w "$lock_wait" 8 || { echo "RB_STAGE1X_STOP cpu lock wait"; exit 1; }; fi
gate vp1 || finish
# A missing 1x reading is reported as missing (spec §3.5): these steps do not stop each other.
step sheet cpu "$here/rb_sheet.py" --outputs "$outputs" "${smoke[@]}"
step part1_report cpu "$here/rb_part1_report.py" --outputs "$outputs" "${smoke[@]}"
step probe gpu "$here/rb_probe.py" --outputs "$outputs" "${smoke[@]}"
if (( failed )); then echo "RB_STAGE1X_NOTE a 1x step failed: reported as missing (spec §3.5), not a stop"; fi
printf 'RB_STAGE1X_DONE failed=%s %s\n' "$failed" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee -a "$status"
exit 0
