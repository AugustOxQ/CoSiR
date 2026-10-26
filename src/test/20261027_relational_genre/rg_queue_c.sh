#!/usr/bin/env bash
# Stage C (spec §3.5, rule §4.3; GPU and CPU), only on B-PICK: stage B ended without a stop, stage_b/verdict.json `outcome`
# is B-PICK and names `v_held` (G). B-NONE ends the queue with nothing to run (exit 0; no held test). Order: `rg_full regress
# vc1` (VC1 training half), `rg_held check vc1` (scorer half); the gate: no held output of G unless both pass; G's five
# seeds at full scale; `rg_full held` (guarded; VC3 with its one rerun is the runner's); `rg_held score`; `rg_vc2` (VC2; a
# disagreement: no outcome); `rg_held report`. A seed that stops on a non-finite loss or still fails VC3 leaves no stage C
# outcome (VC4: the runner refuses, the queue stops, the cause is in the step's log). Pending deletions: the VC1 run's
# weights (G's five seeds are kept, rule §4.3 Weights).
queue=c
source "$(dirname -- "${BASH_SOURCE[0]}")/rg_queue_common.sh"
b=$outputs/stage_b
c=$outputs/stage_c
if [[ "$dry" == 1 ]]; then
    g=G
else
    queue_ok b || stop "stage B did not end cleanly (last line: '$(last_done b)'): no stage C"
    outcome=$(jget "$b/verdict.json" "d['outcome']")
    g=$(jget "$b/verdict.json" "d['v_held']")
    if [[ "$outcome" != B-PICK ]]; then note "stage B outcome '$outcome': no held test (rule §4.2); stage C does not run"; finish; fi
    [[ -n "$g" && "$g" != None ]] || stop "B-PICK without v_held in $b/verdict.json"
    note "G = $g (stage B pick, v_held)"
fi
seeds="0 1 2 3 4"
# --- VC1 -----------------------------------------------------------------------------------------------------------------
softstep vc1_train gpu "$here/rg_full.py" regress vc1 --device "$device" --outputs "$outputs" "${smoke[@]}"
pending_add "$c/regression/vc1_run" "VC1 regression run weights (R6-5 at full scale, switches off)" "regression run of the gate; vc1_train.json is the record"
# a regression check is never shrunk: rg_held check takes no --smoke (it reads the frozen K4 and T files only)
softstep vc1_scorer cpu "$here/rg_held.py" --outputs "$outputs" --workers "$workers" check vc1
regression_gate VC1 "$c/regression/vc1_train.json" "$c/regression/vc1_scorer.json" \
    || stop "VC1 does not hold: no held output of G is computed (rule §4.3); the cause goes in the report"
# --- G at full scale, held outputs ------------------------------------------------------------------------------------------
step train_g gpu "$here/rg_full.py" train --arm "$g" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}" \
    || stop "G's full-scale training failed (VC4: a seed with a non-finite loss is not rerun; no stage C outcome); see the log"
step held_g gpu "$here/rg_full.py" held --arm "$g" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}" \
    || stop "G's held inference failed (the guard, or VC3 twice; VC4: no stage C outcome); see the log"
step score_c cpu "$here/rg_held.py" --outputs "$outputs" --workers "$workers" "${smoke[@]}" score --arm "$g" \
    || stop "the held scorer failed"
step vc2 cpu "$here/rg_vc2.py" --outputs "$outputs" --arm "$g" --workers "$workers" "${smoke[@]}" \
    || stop "VC2 disagrees with the held scorer: no stage C outcome is stated; trace"
[[ "$dry" == 1 ]] || note "stage C outcome: $(jget "$c/verdict.json" "d.get('outcome_id')")"
step report_c cpu "$here/rg_held.py" --outputs "$outputs" --workers "$workers" "${smoke[@]}" report --arm "$g" \
    || stop "the stage C report failed"
finish
