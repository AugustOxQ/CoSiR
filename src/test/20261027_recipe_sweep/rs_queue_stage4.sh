#!/usr/bin/env bash
# Stage 4 (spec §3.5, rule §4.4; GPU then CPU), under PICK: c* (the `pick` of stage3/verdict.json) trained at full scale,
# seeds 0 to 4, held inference, the verdict (`rs_stage4 verdict`), V4b (`rs_vh stage4`), the reported margins, then c*'s analysis.
# Gates checked here: V1a, V4a (training half gates/v4a_train.json and scorer half gates/v4a_scorer.json; `rs_full train` and `held`
# also refuse without them), V3c (stage3/rederive_stage3.json passes: c* is stated).
# KEEP-G, or no stage 3 file because stage 1 gave SWEEP-NONE or stage 2 TOP-NONE (F = G): no training and no held inference. The
# queue runs two CPU steps on G's saved held outputs (12-27's, equal to this round's replay by V4a): `rs_stage4 verdict --candidate G`
# and then `rs_vh stage4 --candidate G` (V4b; exit 1 stops: no outcome stated). Gates there: V4a scorer half, and the re-derivation
# behind F = G (V3c under KEEP-G; V1c of the sweep machine, stage 1 and, when stage 2 ran, stage 2, with no stage 3 file). Rule §4.4,
# last paragraph: G's outcomes are this round's stage 4 verdict on G's saved held outputs, checked by V4b. No report, no analysis
# (G's analysis is stage 0's).
# V4d: c*'s five seeds run once; a failed seed is rerun once by hand (same seed and code) and listed.
# The held ablation does not run when V4a fails. Pending deletions: the V4a replay's weights (c*'s five seeds are kept).
queue=stage4
source "$(dirname -- "${BASH_SOURCE[0]}")/rs_queue_common.sh"
w=(--workers "$workers")
v3=$outputs/stage3/verdict.json
m=$(sweep_machine)
g_only=
if [[ -f "$v3" ]]; then
    outcome=$(jget "$v3" "d['outcome']")
    if [[ "$outcome" == PICK ]]; then
        c=$(jget "$v3" "d['pick']")
        [[ -n "$c" && "$c" != None ]] || stop "PICK without a pick in $v3"
        note "c* = $c (stage 3 pick)"
    else
        g_only="stage 3 outcome '$outcome' (KEEP-G)"
    fi
else
    ended=$(jget "$outputs/sweep/$m/stage1.json" "d['outcome']")
    if [[ "$ended" == SWEEP-NONE || "$(jget "$outputs/sweep/$m/stage2.json" "d['outcome']")" == TOP-NONE ]]; then
        g_only="no stage 3 file ($ended / $(jget "$outputs/sweep/$m/stage2.json" "d['outcome']"))"
    elif [[ "$dry" == 1 ]]; then
        c=C_STAR     # nothing to read: the dry run shows the PICK path with a placeholder
    else
        stop "$v3 missing and stage 1 or 2 on $m did not end without candidates: run the stage 3 queue first"
    fi
fi
if [[ -n "$g_only" ]]; then
    note "$g_only: G stays the recipe; no training and no held inference. G's outcomes are this round's stage 4 verdict on G's saved held outputs, checked by V4b (rule §4.4, last paragraph)"
    gate V4a-scorer "$outputs/gates/v4a_scorer.json" "G's held outcomes are not computed (rule §4.4 V4a, scorer half)"
    f_basis_gates
    step verdict cpu "$here/rs_stage4.py" --outputs "$outputs" "${w[@]}" verdict --candidate G || stop "the stage 4 verdict on G failed"
    [[ "$dry" == 1 ]] || note "stage 4 outcomes of G written (not stated until V4b agrees): $(jget "$outputs/stage4/verdict.json" "d.get('outcomes') or d.get('outcome')")"
    step v4b cpu "$here/rs_vh.py" stage4 --outputs "$outputs" "${w[@]}" --candidate G \
        || stop "V4b (rs_vh stage4) disagrees with the verdict on G: no stage 4 outcome is stated; trace"
    finish
fi
gate V1a-train "$outputs/gates/v1a_train.json" "no run of stage 4 starts (rule §4.2 V1a)"
gate V1a-scorer "$outputs/gates/v1a_scorer.json" "no run of stage 4 starts (rule §4.2 V1a)"
gate V4a-train "$outputs/gates/v4a_train.json" "no held output of c* is computed (rule §4.4 V4a)"
gate V4a-scorer "$outputs/gates/v4a_scorer.json" "no held output of c* is computed (rule §4.4 V4a)"
gate V3c "$outputs/stage3/rederive_stage3.json" "c* is not stated until the stage 3 re-derivation agrees (rule §4.3 V3c)"
pending_add "$outputs/regression/v4a_run" "V4a regression run weights (G's full-scale seed 0 replay)" "regression run of the gate; gates/v4a_train.json is the record"
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
