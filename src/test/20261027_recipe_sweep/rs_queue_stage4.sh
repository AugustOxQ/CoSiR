#!/usr/bin/env bash
# Stage 4 (spec §3.5, rule §4.4; GPU then CPU), only under PICK: c* (the `pick` of stage3/verdict.json) trained at full scale,
# seeds 0 to 4, held inference, the verdict (`rs_stage4 verdict`), V4b (`rs_vh stage4`), the reported margins, then c*'s analysis.
# Gates checked here: V1a, V4a (training half gates/v4a_train.json and scorer half gates/v4a_scorer.json; `rs_full train` and `held`
# also refuse without them), V3c (stage3/rederive_stage3.json passes: c* is stated). KEEP-G, or no stage 3 file because stage 1
# gave SWEEP-NONE or stage 2 TOP-NONE: stage 4 does not run (a note, exit 0). G's outcomes in that case are the stage 4 scorer on
# 12-27's saved held outputs, a by-hand `rs_stage4 verdict --candidate G` the queue does not run (spec §3.5, rule §4.4 last paragraph).
# V4d: c*'s five seeds run once; a failed seed is rerun once by hand (same seed and code) and listed.
# The held ablation does not run when V4a fails. Pending deletions: the V4a replay's weights (c*'s five seeds are kept).
queue=stage4
source "$(dirname -- "${BASH_SOURCE[0]}")/rs_queue_common.sh"
w=(--workers "$workers")
v3=$outputs/stage3/verdict.json
if [[ "$dry" == 1 ]]; then
    c=C_STAR
elif [[ ! -f "$v3" ]]; then
    m=$(sweep_machine)
    ended=$(jget "$outputs/sweep/$m/stage1.json" "d['outcome']")
    [[ "$ended" == SWEEP-NONE || "$(jget "$outputs/sweep/$m/stage2.json" "d['outcome']")" == TOP-NONE ]] \
        || stop "$v3 missing and stage 1 or 2 on $m did not end without candidates: run the stage 3 queue first"
    note "no stage 3 file ($ended / $(jget "$outputs/sweep/$m/stage2.json" "d['outcome']")): G stays; stage 4 does not run (rule §4.4)"
    finish
else
    outcome=$(jget "$v3" "d['outcome']")
    if [[ "$outcome" != PICK ]]; then note "stage 3 outcome '$outcome' (KEEP-G): G stays the recipe; stage 4 does not run (rule §4.4)"; finish; fi
    c=$(jget "$v3" "d['pick']")
    [[ -n "$c" && "$c" != None ]] || stop "PICK without a pick in $v3"
    note "c* = $c (stage 3 pick)"
fi
gate V1a-train "$outputs/gates/v1a_train.json" "no run of stage 4 starts (rule §4.2 V1a)"
gate V1a-scorer "$outputs/gates/v1a_scorer.json" "no run of stage 4 starts (rule §4.2 V1a)"
gate V4a-train "$outputs/gates/v4a_train.json" "no held output of c* is computed (rule §4.4 V4a)"
gate V4a-scorer "$outputs/gates/v4a_scorer.json" "no held output of c* is computed (rule §4.4 V4a)"
gate V3c "$outputs/stage3/rederive_stage3.json" "c* is not stated until the stage 3 re-derivation agrees (rule §4.3 V3c)"
pending_add "$outputs/full/regression/v4a_run" "V4a regression run weights (G's full-scale seed 0 replay)" "regression run of the gate; gates/v4a_train.json is the record"
seeds="0 1 2 3 4"
step train_c gpu "$here/rs_full.py" train --config "$c" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}" \
    || stop "c*'s full-scale training failed (V4d: a seed with a non-finite loss is not rerun; no stage 4 outcome); see the log"
step held_c gpu "$here/rs_full.py" held --config "$c" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}" \
    || stop "c*'s held inference failed (the guard, or V4c twice; no stage 4 outcome); see the log"
step verdict cpu "$here/rs_stage4.py" --outputs "$outputs" "${w[@]}" verdict --candidate "$c" || stop "the stage 4 verdict failed"
[[ "$dry" == 1 ]] || note "stage 4 outcomes written (not stated until V4b agrees): $(jget "$outputs/stage4/verdict.json" "d.get('outcomes') or d.get('outcome')")"
step v4b cpu "$here/rs_vh.py" stage4 --outputs "$outputs" "${w[@]}" --candidate "$c" \
    || stop "V4b (rs_vh stage4) disagrees with the verdict: no stage 4 outcome is stated; c*'s analysis does not run; trace"
softstep report cpu "$here/rs_stage4.py" --outputs "$outputs" "${w[@]}" report \
    || note "the reported margins failed (log $outputs/logs/${queue}_report.log); they decide nothing; rerun rs_stage4 report after the fix"
step analysis_c cpu "$here/rs_analysis.py" --model "$c" --outputs "$outputs" "${smoke[@]}" || stop "c*'s analysis failed"
finish
