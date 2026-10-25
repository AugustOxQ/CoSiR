#!/usr/bin/env bash
# Stage 1 (rule §4.1): the six recipes on P (GPU), the pick (CPU), then VP1; the pick is stated only after VP1 agrees.
# VP2 (spec D15, ticket 16): a candidate failing VP2 is rerun once, forced and logged; failing again it goes to
# rb_pick --vp2-failed and the pick goes on without it. R0-5 failing stops part 1 (VP3).
queue=stage1
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
gate stage1 || finish
# A restart after stage 1 finished (pick written, VP1 agrees) skips it: VP3 refuses to retrain a finished recipe.
if [[ "$dry" != 1 ]] && cpu "$run" gate vp1 --outputs "$outputs" "${smoke[@]}" > /dev/null 2>&1; then
    note "stage 1 already done (VP1 agrees); skipped"
    cpu "$run" pick --outputs "$outputs" "${smoke[@]}" | tee -a "$status"; finish
fi
vp2_args=()
gate_vp3() { step "vp3_$1" cpu "$run" vp3 --recipe "$r" --outputs "$outputs" "${smoke[@]}" "${@:2}"; }   # uses the loop's $r
for r in $("$python_bin" "$run" recipes); do
    if [[ "$dry" != 1 ]] && ls "$outputs/part1/train/$r/"{emotion,style,genre}__s0__P.npz > /dev/null 2>&1; then
        note "$r finished before (P exports of all three aspects): not trained again (VP3)"; continue
    fi
    gate_vp3 "$r" || finish
    if ! step "part1_$r" gpu "$here/rb_part1.py" --recipes "$r" --outputs "$outputs" --device "$device" "${smoke[@]}"; then
        failed=0
        cause=$(cpu "$run" cause "$outputs/logs/${queue}_part1_$r.log" --outputs "$outputs" "${smoke[@]}") || {
            echo "$cause"; echo "RB_STAGE1_STOP $r failed, not a VP2 failure"; failed=1; finish; }
        note "VP2 failed for $r ($cause); forced rerun, logged (VP3)"
        gate_vp3 "${r}_forced" --forced || finish
        if ! step "part1_${r}_rerun" gpu "$here/rb_part1.py" --recipes "$r" --outputs "$outputs" --device "$device" "${smoke[@]}"; then
            failed=0
            cause=$(cpu "$run" cause "$outputs/logs/${queue}_part1_${r}_rerun.log" --outputs "$outputs" "${smoke[@]}") || {
                echo "$cause"; echo "RB_STAGE1_STOP $r failed again, not a VP2 failure"; failed=1; finish; }
            if [[ "$r" == R0-5 ]]; then
                echo "RB_STAGE1_STOP R0-5 fails VP2 after the forced rerun: part 1 stops as under V0 ($cause)"; failed=1; finish
            fi
            note "VP2 fails again for $r: passed to rb_pick --vp2-failed; part 1 goes on without it"
            vp2_args+=(--vp2-failed "$r=$cause")
        fi
    fi
done
step pick cpu "$here/rb_pick.py" --outputs "$outputs" "${smoke[@]}" "${vp2_args[@]}" || finish
# No pick line is printed from here until VP1 agrees (the pick log is not echoed).
step vp1 cpu "$here/rb_vp1.py" --outputs "$outputs" "${smoke[@]}" || { echo "RB_STAGE1_STOP VP1 disagrees or failed: trace and fix before any pick is stated (rule §4.1)"; finish; }
gate vp1 || finish
[[ "$dry" == 1 ]] || { cpu "$run" pick --outputs "$outputs" "${smoke[@]}" | tee -a "$status"; }
finish
