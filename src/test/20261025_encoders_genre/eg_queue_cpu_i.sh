#!/usr/bin/env bash
# Stage i reported readings (CPU only, long: about 30 to 40 min per candidate per stage at 8 workers, 12-24's figure).
# Usage: eg_queue_cpu_i.sh 3|4|5   Runs beside the GPU stages, never inside the GPU lock; one CPU job at a time.
# eg_report_i needs the verdict file and V2's agreement (reported files are written only after the outcome, rule §4.1).
stage=${1:?usage: eg_queue_cpu_i.sh 3|4|5}
queue=cpu_i$stage
source "$(dirname -- "${BASH_SOURCE[0]}")/eg_queue_common.sh"
[[ "$stage" == 3 || "$stage" == 4 || "$stage" == 5 ]] || { echo "EG_CPU_I${stage}_STOP stage must be 3, 4 or 5"; exit 2; }
if [[ "$dry" != 1 ]]; then exec 8> "${EG_CPU_LOCK:-/tmp/eg_cpu.lock}"; flock -w "$lock_wait" 8 || { echo "EG_CPU_I${stage}_STOP cpu lock wait"; exit 1; }; fi
if [[ "$dry" != 1 && ! -f "$outputs/part2/verdict_stage$stage.json" ]]; then
    note "no verdict_stage$stage.json (no candidate ran that stage): nothing to report"; finish
fi
if [[ "$dry" != 1 && "$(jget "$outputs/part2/verdict_stage$stage.json" "any(b.get('outcome_id') is not None for b in d['blocks'].values())")" == False ]]; then
    note "no candidate has an outcome: report the causes and stores without a V2 re-derivation"
else
    gate v2 --stage "$stage" || { printf 'EG_CPU_I%s_STOP V2 does not agree (or is older than the verdict): no reported readings\n' "$stage" | tee -a "$status"; failed=1; finish; }
fi
# Default candidates of eg_report_i: every block with an outcome and no reported block yet.
step "report$stage" cpu "$here/eg_report_i.py" --stage "$stage" --outputs "$outputs" --workers 8 "${smoke[@]}" || finish
finish
