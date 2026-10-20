#!/usr/bin/env bash
queue=cpu
source "$(dirname -- "${BASH_SOURCE[0]}")/ct_queue_common.sh"
step require_stage0 cpu "$here/ct_stage0.py" --require-pass --device cuda || finish
# Run after the foreground Stage 1 queue has returned success.
step verdict cpu "$here/ct_run.py" verdict || finish
step diagnostics cpu "$here/ct_run.py" diagnostics || finish
finish
