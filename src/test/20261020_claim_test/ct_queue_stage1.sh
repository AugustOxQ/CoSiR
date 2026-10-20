#!/usr/bin/env bash
queue=stage1
source "$(dirname -- "${BASH_SOURCE[0]}")/ct_queue_common.sh"
step require_stage0 cpu "$here/ct_stage0.py" --require-pass --device cuda || finish
step budget cpu "$here/ct_run.py" budget || finish
step frozen cpu "$here/ct_run.py" frozen || finish
step held_list cpu "$here/ct_run.py" held-list || finish
step d2 gpu "$here/ct_run.py" d2 || finish
step core gpu "$here/ct_run.py" core || finish
step paper gpu "$here/ct_run.py" paper || finish
step rcol cpu "$here/ct_run.py" rcol || finish
step held gpu "$here/ct_run.py" held || finish
step instructed_images gpu "$here/ct_run.py" images || finish
finish
