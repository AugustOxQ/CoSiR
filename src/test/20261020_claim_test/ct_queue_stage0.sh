#!/usr/bin/env bash
queue=stage0
source "$(dirname -- "${BASH_SOURCE[0]}")/ct_queue_common.sh"
# A traced explanation can be supplied after inspecting a failed comparison.
causes=()
if (( $# )); then
    if [[ $# != 2 || $1 != --causes ]]; then echo "Usage: $0 [--causes JSON]" >&2; exit 2; fi
    causes=(--causes "$2")
fi
step d2 gpu "$here/ct_run.py" d2 --stage0-training || finish
step core gpu "$here/ct_run.py" core --stage0-training || finish
step check gpu "$here/ct_stage0.py" --device cuda "${causes[@]}" || finish
finish
