#!/usr/bin/env bash
# Stage 3 (spec §3.5, rule §4.3; GPU then CPU): TOP's configurations of stage 2 at seeds 1 to 3, the verdict, V3c.
# Gates checked here before any training: V1a (both halves; `rs_train train` does not check them), V3a (the runner `rs_stage3
# verdict` also refuses without it), and the stage 1 and 2 re-derivations (V1c part (a) and (b)) of the machine that chose TOP.
# SWEEP-NONE, TOP-NONE: no stage 3 file is written, G stays: a note and exit 0 (not a failure). The machine is local when the
# fallback ran, else node402 (RS_SWEEP_MACHINE overrides). TOP's weights are kept (rule §4.5 Weights): nothing on the pending list.
queue=stage3
source "$(dirname -- "${BASH_SOURCE[0]}")/rs_queue_common.sh"
w=(--workers "$workers")
m=$(sweep_machine)
s1=$outputs/sweep/$m/stage1.json
s2=$outputs/sweep/$m/stage2.json
note "stages 1 and 2 machine: $m"
gate V1a-train "$outputs/gates/v1a_train.json" "no run of stage 3 starts (rule §4.2 V1a)"
gate V1a-scorer "$outputs/gates/v1a_scorer.json" "no run of stage 3 starts (rule §4.2 V1a)"
gate V3a "$outputs/gates/v3a.json" "stage 3 stops (rule §4.3 V3a)"
if [[ "$dry" == 1 ]]; then
    top="TOP_1 TOP_2 TOP_3"
elif [[ "${RS_SMOKE:-0}" == 1 && ! -f "$s2" ]]; then
    stop "smoke: write a smoke $s2 first (rs_queue.md, Smoke)"
else
    [[ -f "$s1" ]] || stop "$s1 missing: stage 1 on $m did not finish"
    [[ "$(jget "$s1" "d['outcome']")" != SWEEP-NONE ]] || { note "stage 1 gave SWEEP-NONE: stages 2, 3 and 4 do not run; G stays the recipe"; finish; }
    [[ -f "$s2" ]] || stop "$s2 missing: stage 2 on $m did not finish"
    [[ "$(jget "$s2" "d['outcome']")" == TOP ]] || { note "stage 2 gave $(jget "$s2" "d['outcome']"): stages 3 and 4 do not run; G stays the recipe"; finish; }
    top=$(jget "$s2" "' '.join(t['name'] for t in sorted(d['top'], key=lambda t: t['rank']))")
    note "TOP: $top"
fi
if [[ "${RS_SMOKE:-0}" == 1 ]]; then
    softstep v3a_smoke cpu "$here/rs_stage3.py" check v3a --outputs "$outputs" --workers "$workers" "${smoke[@]}" \
        || note "smoke: the V3a check did not pass at 20 draws; the chain goes on"
else
    gate V1c-a "$outputs/scores/$m/rederive_partitions.json" "stage 3 does not start before the stage 1 and 2 partitions are re-derived (rule §4.2 V1c)"
    gate V1c-b "$outputs/sweep/$m/rederive_stage2.json" "stage 3 does not start before TOP is re-derived (rule §4.2 V1c)"
fi
# --- TOP at seeds 1 to 3 (one call: the data load once; a failed run is a recorded outcome) -------------------------------------
step train_top gpu "$here/rs_train.py" train --configs $top --seeds 1 2 3 --device "$device" --outputs "$outputs" "${smoke[@]}" \
    || stop "TOP's training crashed or was refused (log $outputs/logs/${queue}_train_top.log)"
if [[ "$dry" != 1 ]]; then
    seeds="1 2 3" check_runs $top
    (( unfinished == 0 )) || stop "a TOP configuration has no run.json (V3d): trace"
fi
step verdict cpu "$here/rs_stage3.py" verdict --top-from "sweep/$m/stage2.json" --outputs "$outputs" "${w[@]}" "${smoke[@]}" \
    || stop "the stage 3 verdict failed"
[[ "$dry" == 1 ]] || note "stage 3 outcome: $(jget "$outputs/stage3/verdict.json" "d['outcome']"), c* $(jget "$outputs/stage3/verdict.json" "d['pick']") (not stated until V3c agrees)"
if [[ "${RS_SMOKE:-0}" == 1 ]]; then
    note "smoke: rs_vp stage3 has no smoke mode; it reads real files; the queue ends after the verdict"
    finish
fi
step v3c cpu "$here/rs_vp.py" stage3 --outputs "$outputs" --sweep-machine "$m" --workers "$workers" \
    || stop "V3c (rs_vp stage3) disagrees with the verdict: no stage 3 outcome is stated; stage 4 does not run until it is fixed"
finish
