#!/usr/bin/env bash
# Stage 2 CPU queue: run after the GPU queue returned success.
queue=cpu
source "$(dirname -- "${BASH_SOURCE[0]}")/ct2_queue_common.sh"
run="$here/ct2_run.py"
step require_stage2a cpu "$here/ct2_stage2a.py" --require-pass --device cuda || finish
step verdict cpu "$run" verdict "${run_args[@]}" --device cuda || finish
step diagnostics cpu "$run" diagnostics "${run_args[@]}" --device cuda || finish
finish
