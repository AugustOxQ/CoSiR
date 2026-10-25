#!/usr/bin/env bash
# Stage 4 (claim test, full scale): candidates with an S1 outcome other than S1-NO (D13), with their matched arms.
queue=stage4
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
gate stage4 || finish
cands=$(driver candidates 4) || finish
[[ -n "${RB_DEFER:-}" ]] && note "deferred for time (logged): $RB_DEFER"
if [[ -z "$cands" ]]; then note "no candidate has an S1 outcome other than S1-NO: stage 4 does not run"; finish; fi
for k in $cands; do candidate full "$k" || finish; done
soft=1 step verdict4 cpu "$here/rb_score.py" --stage 4 --candidates $cands --outputs "$outputs" --workers 8 "${smoke[@]}"
failed=0
step check4 cpu "$run" check-verdict 4 --candidates $cands --outputs "$outputs" "${smoke[@]}" || finish
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_check4.log" | tee -a "$status"
step v2_4 cpu "$here/rb_v2.py" --stage 4 --outputs "$outputs" --workers 8 "${smoke[@]}" || { echo "RB_STAGE4_STOP V2 disagrees: trace before any outcome is stated (rule §4.2)"; finish; }
finish
