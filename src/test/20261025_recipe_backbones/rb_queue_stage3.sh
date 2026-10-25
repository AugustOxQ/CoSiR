#!/usr/bin/env bash
# Stage 3 (claim test, sample scale): K2, K3 (K4, K5 if triggered), matched arms, seeds 0 to 4; verdict, then V2.
# RB_DEFER="K5" defers a candidate for time (spec D32; never dropped: rerun later without RB_DEFER).
queue=stage3
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
gate stage3 || finish
cands=$(driver candidates 3) || finish
[[ -n "${RB_DEFER:-}" ]] && note "deferred for time (logged): $RB_DEFER"
# A failed candidate leaves only itself without an outcome (V4; the scorer records the cause); the others go on
# (spec §9: the only stops are a failed stage 0, 2 or 3a).
for k in $cands; do
    soft=1 candidate sample "$k" || note "$k failed (step line above, log in $outputs/logs): no outcome for $k; the other candidates go on"
done
failed=0
soft=1 step verdict3 cpu "$here/rb_score.py" --stage 3 --candidates $cands --outputs "$outputs" --workers 8 "${smoke[@]}"
# Exit 1 of the scorer means a candidate without an outcome (V failure: reported with the cause), a crash too: the
# verdict file must hold a block for every candidate.
failed=0
step check3 cpu "$run" check-verdict 3 --candidates $cands --outputs "$outputs" "${smoke[@]}" || finish
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_check3.log" | tee -a "$status"
step v2_3 cpu "$here/rb_v2.py" --stage 3 --outputs "$outputs" --workers 8 "${smoke[@]}" || { echo "RB_STAGE3_STOP V2 disagrees: trace before any outcome is stated (rule §4.2)"; finish; }
finish
