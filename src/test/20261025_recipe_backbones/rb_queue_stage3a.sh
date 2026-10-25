#!/usr/bin/env bash
# Stage 3a (full-scale regression, spec §3.5): the same code at full scale, Qwen input, seed 0. Stage 4 needs it.
queue=stage3a
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
gate part2 || finish
step reg_student gpu "$here/rb_student.py" regress stage3a --device cuda --outputs "$outputs" "${smoke[@]}" || finish
step reg_matched gpu "$here/rb_matched.py" --regress stage3a --device cuda --outputs "$outputs" || finish
step reg_scorer cpu "$here/rb_score.py" --check stage3a --outputs "$outputs" || finish
gate stage3a || { echo "RB_STAGE3A_STOP stage 3a regression failed: stage 4 does not run; trace, report (spec §3.5)"; finish; }
finish
