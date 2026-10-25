#!/usr/bin/env bash
# Stage 0 (V0): R0-5, R6-3, R7-3 at seed 0 on the old 5,000 against 12-22's exports; GPU. Exit 1 stops everything.
queue=stage0
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
step budget cpu "$run" budget --outputs "$outputs" "${smoke[@]}"
[[ "$dry" == 1 ]] || tail -n 6 "$outputs/logs/${queue}_budget.log" | tee -a "$status"
if [[ "${RB_SMOKE:-0}" != 1 && "$dry" != 1 ]] && cpu "$run" gate stage1 --outputs "$outputs" > /dev/null 2>&1; then
    note "stage0.json already passed; skipped"; finish
fi
if ! step stage0 gpu "$here/rb_stage0.py" --outputs "$outputs" --device "$device" "${smoke[@]}"; then
    # A smoke run never passes by design (rb_stage0): exit 1 is fine when it wrote a smoke record.
    if [[ "${RB_SMOKE:-0}" == 1 ]] && gate_out=$(cpu "$run" gate stage1 --outputs "$outputs" --smoke 2>&1); then
        failed=0; note "smoke: exit 1 is by design; $gate_out"
    else
        finish
    fi
fi
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_stage0.log" | tee -a "$status"
finish
