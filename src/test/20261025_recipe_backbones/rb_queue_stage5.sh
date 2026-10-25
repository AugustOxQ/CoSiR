#!/usr/bin/env bash
# Stage 5 (ArtELingo-28): K1 at full scale always; any other model at full scale only when its stage 4 ran to an outcome
# (outcome_id not None), else sample. B1 and B6 outputs at every scale used. Verdict and reported files, then V2.
queue=stage5
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
gate stage5 || finish
if [[ "$dry" == 1 ]]; then plan="K1 K2 K3 K4 K5|full sample|{}"; else plan=$(driver a28) || finish; fi
models=${plan%%|*}; rest=${plan#*|}; bscales=${rest%%|*}
note "models: $models; B1/B6 scales: $bscales; scales: ${rest#*|}"
step k1_a28 gpu "$here/rb_student.py" infer --scale full --arm K1 --kinds a28 --outputs "$outputs" "${smoke[@]}" || finish
for s in $bscales; do
    for b in B1 B6; do
        step "${b}_a28_$s" gpu "$here/rb_matched.py" --scale "$s" --arm "$b" --device "$device" --outputs "$outputs" "${smoke[@]}" || finish
    done
done
soft=1 step a28 cpu "$here/rb_a28.py" --models $models --part both --outputs "$outputs" --workers 8 "${smoke[@]}"
failed=0
step check5 cpu "$run" check-verdict 5 --candidates $models --outputs "$outputs" "${smoke[@]}" || finish
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_check5.log" | tee -a "$status"
step v2_5 cpu "$here/rb_v2.py" --stage 5 --outputs "$outputs" --workers 8 "${smoke[@]}" || { echo "RB_STAGE5_STOP V2 disagrees (rule §4.3)"; finish; }
finish
