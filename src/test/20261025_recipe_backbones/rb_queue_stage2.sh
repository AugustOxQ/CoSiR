#!/usr/bin/env bash
# Stage 2 (sample-scale regression, spec §3.5): Qwen + R0-5 student, matched code and the part 2 scorer reproduce the CT
# bit for bit / within 1e-6. Gate: every expected item of student/, matched/, scorer/ present and passing.
queue=stage2
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
gate part2 || finish
step reg_student gpu "$here/rb_student.py" regress stage2 --device cuda --outputs "$outputs" "${smoke[@]}" || finish
step reg_matched gpu "$here/rb_matched.py" --regress stage2 --device cuda --outputs "$outputs" || finish   # never shrunk: no --smoke
step reg_scorer cpu "$here/rb_score.py" --check stage2 --outputs "$outputs" || finish                        # never shrunk
gate stage3 || { echo "RB_STAGE2_STOP stage 2 regression failed: stop part 2, trace, report (spec §3.5)"; finish; }
finish
