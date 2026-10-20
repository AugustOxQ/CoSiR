#!/usr/bin/env bash
# Stage 2 foreground queue: extraction, graphs (CPU), stage 2a, full training, R-COL, held inference.
queue=gpu
source "$(dirname -- "${BASH_SOURCE[0]}")/ct2_queue_common.sh"
run="$here/ct2_run.py"
step budget cpu "$run" budget || finish
tail -n 1 "$outputs/logs/${queue}_budget.log" | tee -a "$status"
step extract gpu "$run" extract "${run_args[@]}" --device cuda || finish
step graphs cpu "$run" graphs "${run_args[@]}" || finish
step stage2a gpu "$run" stage2a "${run_args[@]}" --device cuda || finish
step d2 gpu "$run" d2 "${run_args[@]}" --device cuda || finish
step core gpu "$run" core "${run_args[@]}" --device cuda || finish
step paper gpu "$run" paper "${run_args[@]}" --device cuda || finish
step rcol cpu "$run" rcol "${run_args[@]}" --device cuda || finish
step held gpu "$run" held "${run_args[@]}" --device cuda || finish
finish
