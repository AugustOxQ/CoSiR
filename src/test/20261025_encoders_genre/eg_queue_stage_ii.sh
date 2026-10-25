#!/usr/bin/env bash
# Stage ii (rule §4.2; GPU, then CPU): P inference of the seven encoders' arms and VL1 (b), (c), VL3; VL1 (a); the gate;
# eg_lift (lifts, intervals, L outcomes); eg_vl2; the encoder statistics and correlations (the P table); then the held
# table, which needs the stage i reports (eg_queue_cpu_i.sh) and always comes after the P table.
# Usage: eg_queue_stage_ii.sh [held]    `held` runs only the held table (the gpu queue calls it after the reports).
# Stage ii starts when every candidate not deferred has its stage 3 block, or stage i stopped at X0 or X1 (then it runs
# with K2 to K4 and the missing candidates are reported as such). VL1 failed: stage ii stops.
queue=stage_ii
source "$(dirname -- "${BASH_SOURCE[0]}")/eg_queue_common.sh"
stop() { printf 'EG_STAGE_II_STOP %s\n' "$*" | tee -a "$status"; failed=1; finish; }
held() {
    if [[ -f "$outputs/stage_ii/held_table.json" && "$dry" != 1 ]]; then note "held_table.json exists, skipped"; return 0; fi
    if [[ "$dry" != 1 ]]; then
        [[ -f "$outputs/stage_ii/stats.json" ]] || stop "the P table (stats.json) must exist before the held table"
        if [[ ! -f "$outputs/part2/reported_stage3.json" ]] || { [[ -f "$outputs/part2/verdict_stage4.json" && ! -f "$outputs/part2/reported_stage4.json" ]]; }; then
            note "held table waits for the stage i reports (part2/reported_stage3.json, _stage4.json): run eg_queue_cpu_i.sh 3 and 4, then 'bash eg_queue_stage_ii.sh held'"
            return 0
        fi
    fi
    step held cpu "$here/eg_stats.py" held --outputs "$outputs" || stop "held table failed"
}
if [[ "${1:-}" == held ]]; then held; finish; fi
if [[ "$dry" != 1 ]]; then
    if gate x0 && gate x1; then
        todo=$(filter_deferred $(driver candidates 3))
        [[ -z "$todo" ]] || stop "stage 3 has no block yet for: $todo (run eg_queue_stage_i.sh; or defer them with eg_run.py defer / EG_DEFER)"
    else
        note "stage i stopped at X0 or X1: stage ii runs with K2 to K4 only (rule §4.1)"
    fi
fi
[[ -n "${EG_DEFER:-}" ]] && note "deferred for time (EG_DEFER, logged): $EG_DEFER"
if [[ -f "$outputs/stage_ii/vl1.json" && "$dry" != 1 ]]; then
    note "vl1.json exists, P inference skipped"
else
    step pinfer gpu "$here/eg_pinfer.py" all --device "$device" --outputs "$outputs" "${smoke[@]}" || stop "P inference or VL1 (b), (c), VL3 failed (a VL3 failure may be rerun once: eg_pinfer.py infer --forced-vl3 CAUSE)"
fi
if [[ -f "$outputs/stage_ii/vl1_scorer.json" && "$dry" != 1 ]]; then
    note "vl1_scorer.json exists, VL1 (a) skipped"
else
    step vl1a cpu "$here/eg_lift.py" check --outputs "$outputs" || stop "VL1 (a) failed"
fi
gate vl1 || stop "VL1 does not hold: stage ii stops (rule §4.2)"
if [[ -f "$outputs/stage_ii/lift.json" && "$dry" != 1 ]]; then
    note "lift.json exists, scoring skipped"
else
    step lift cpu "$run" lift-score --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "lift scoring failed"
fi
step vl2 cpu "$here/eg_vl2.py" --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "VL2 disagrees: trace before any outcome is stated"
gate vl2 || stop "VL2 gate does not hold"
if [[ -f "$outputs/stage_ii/stats.json" && "$dry" != 1 ]]; then
    note "stats.json exists, skipped"
else
    step stats cpu "$here/eg_stats.py" stats --outputs "$outputs" "${smoke[@]}" || stop "encoder statistics failed"
fi
held
finish
