#!/usr/bin/env bash
# Stage 4 (claim test, full scale): candidates with an S1 outcome other than S1-NO (D13), with their matched arms.
queue=stage4
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
gate stage4 || finish
cands=$(driver candidates 4) || finish
[[ -n "${RB_DEFER:-}" ]] && note "deferred for time (logged): $RB_DEFER"
# A rerun (after a stop or a deferral) trains and scores only candidates without a block (rb_run candidates).
if [[ -z "$cands" ]]; then
    [[ -f "$outputs/part2/verdict_stage4.json" ]] || { note "no candidate has an S1 outcome other than S1-NO: stage 4 does not run"; finish; }
    note "every stage 4 candidate already has a block: nothing to train or score (check and V2 run again)"
else
    # A failed candidate leaves only itself without an outcome (V4; the scorer records the cause); the others go on
    # (spec §9: the only stops are a failed stage 0, 2 or 3a).
    for k in $cands; do
        soft=1 candidate full "$k" || note "$k failed (step line above, log in $outputs/logs): no outcome for $k; the other candidates go on"
    done
    soft=1 step verdict4 cpu "$here/rb_score.py" --stage 4 --candidates $cands --outputs "$outputs" --workers 8 "${smoke[@]}"
    failed=0
fi
step check4 cpu "$run" check-verdict 4 --outputs "$outputs" "${smoke[@]}" || finish
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_check4.log" | tee -a "$status"
step v2_4 cpu "$here/rb_v2.py" --stage 4 --outputs "$outputs" --workers 8 "${smoke[@]}" || { echo "RB_STAGE4_STOP V2 disagrees: trace before any outcome is stated (rule §4.2)"; finish; }
finish
