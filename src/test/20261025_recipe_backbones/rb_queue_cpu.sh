#!/usr/bin/env bash
# Part 2 reported diagnostics (CPU only, long: 30 to 40 min per candidate per stage at 8 workers, estimated).
# Usage: rb_queue_cpu.sh 3|4   Runs beside the next GPU stage, never inside the GPU lock; one CPU job at a time.
# They need the verdict file and V2's agreement (the reported files are written only after the outcome, rule §4.2).
stage=${1:?usage: rb_queue_cpu.sh 3|4}
queue=cpu$stage
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
[[ "$stage" == 3 || "$stage" == 4 ]] || { echo "RB_CPU_STOP stage must be 3 or 4"; exit 2; }
if [[ "$dry" != 1 ]]; then exec 8> "${RB_CPU_LOCK:-/tmp/rb_cpu.lock}"; flock -w "$lock_wait" 8 || { echo "RB_CPU${stage}_STOP cpu lock wait"; exit 1; }; fi
if [[ "$dry" != 1 && ! -f "$outputs/part2/verdict_stage$stage.json" && "$stage" == 4 ]]; then
    note "no verdict_stage4.json (no candidate ran stage 4): nothing to report"; finish
fi
gate "v2_$stage" || finish
# Default candidates of rb_part2_report: every block of the verdict file with an outcome.
step "report$stage" cpu "$here/rb_part2_report.py" --stage "$stage" --outputs "$outputs" --workers 8 "${smoke[@]}" || finish
finish
