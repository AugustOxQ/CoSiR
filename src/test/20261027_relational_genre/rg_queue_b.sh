#!/usr/bin/env bash
# Stage B (spec §3.5, rule §4.2; GPU and CPU): needs stage A's outcome (the last rg_queue_a.sh run ended without a stop and
# stage_a/verdict.json exists). VB1 (`rg_score check vb1`; if it does not hold B and C stop); GR-2 and GR-4, and v*_A when
# stage A says A-NEW (read from stage_a/verdict.json), at seeds 1 to 3; `rg_score b`; `rg_v2 --phase b` (a disagreement: no
# outcome, stops B and C); `rg_score report` (a soft step: it decides nothing, so a failure is noted and does not stop C).
# A fresh-seed run that fails leaves its arm not confirmed (rule "Failed runs"); the scorer reads its run.json. All fresh-seed weights are kept (rule §4.3 Weights): nothing goes on the pending list here.
queue=b
source "$(dirname -- "${BASH_SOURCE[0]}")/rg_queue_common.sh"
a=$outputs/stage_a
b=$outputs/stage_b
w=(--workers "$workers")
if [[ "$dry" != 1 ]]; then
    queue_ok a || stop "stage A did not end cleanly (last line: '$(last_done a)'): no stage B until VA1, VA2 and the stage A outcome hold"
    [[ -f "$a/verdict.json" ]] || stop "$a/verdict.json missing"
fi
# --- VB1 ------------------------------------------------------------------------------------------------------------
softstep vb1 cpu "$here/rg_score.py" check vb1 --outputs "$outputs" "${w[@]}" "${smoke[@]}"
vb1_code=$?
if (( vb1_code != 0 )); then   # the check failed or crashed: an older vb1.json must not be read as its result
    if [[ "${RG_SMOKE:-0}" == 1 ]]; then
        note "smoke: the vb1 step exited $vb1_code (expected at 2 epochs and 20 draws); the chain goes on"
    else
        stop "VB1: the vb1 step exited $vb1_code (log $outputs/logs/${queue}_vb1.log): stages B and C stop (rule §4.2); the cause goes in the report"
    fi
else
    regression_gate VB1 "$b/vb1.json" || stop "VB1 does not hold: stages B and C stop (rule §4.2); the cause goes in the report"
fi
# --- fresh seeds -----------------------------------------------------------------------------------------------------
arms=(GR-2 GR-4)
if [[ "$dry" == 1 ]]; then
    arms+=(V_STAR_A_IF_A-NEW)
elif [[ "$(jget "$a/verdict.json" "d['outcome']")" == A-NEW ]]; then
    arms+=("$(jget "$a/verdict.json" "d['winner']")")
    note "A-NEW: v*_A ${arms[2]} trains on seeds 1 to 3 beside GR-2 and GR-4"
else
    note "A-SAME: GR-2 and GR-4 only (GR-8 and R6-5 are 12-25's frozen runs)"
fi
train_arms "1 2 3" "${arms[@]}"
# --- scoring, VB2, report ----------------------------------------------------------------------------------------------
step score_b cpu "$here/rg_score.py" b --outputs "$outputs" "${w[@]}" "${smoke[@]}" || stop "scoring stage B failed"
step vb2 cpu "$here/rg_v2.py" --phase b --outputs "$outputs" --workers "$workers" "${smoke[@]}" \
    || stop "VB2 disagrees with the scorer: no stage B outcome is stated; trace (stages B and C stop)"
[[ "$dry" == 1 ]] || note "stage B outcome: $(jget "$b/verdict.json" "d['outcome']"), v_held $(jget "$b/verdict.json" "d['v_held']")"
# the report decides nothing (spec §3.5, §3.6): its failure is noted and does not stop stage C
softstep report_ab cpu "$here/rg_score.py" report --outputs "$outputs" "${smoke[@]}" \
    || note "the stage A and B report failed (log $outputs/logs/${queue}_report_ab.log); it decides nothing, stage C is not held back; rerun \`rg_score.py report\` after the fix"
finish
