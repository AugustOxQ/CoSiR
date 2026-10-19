#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/ms_queue_common.sh"
stage=3
status=queue_stage3_status.txt
printf 'MS_STAGE3_START %s\n' "$(date -Iseconds)" >> "$status"
if (( $# < 1 )); then
    step arguments false
    finish
fi
base=$1
shift
# Record the exact derived definition before launching the GPU subprocess.
if step manifest env CUDA_VISIBLE_DEVICES='' "$python_bin" -c '
import sys
from pathlib import Path
from ms_score import atomic_json
from ms_rows import ROWS
base=sys.argv[1]
assert base in ROWS
settings=["inputs=instructed",*sys.argv[2:]]
assert all("=" in s for s in settings)
assert all(s.split("=",1)[0] != "inputs" for s in sys.argv[2:]), "I1 inputs must be instructed"
atomic_json(Path("outputs/i1.json"), dict(name="I1",base=base,set=settings))
' "$base" "$@"; then
    step train_I1 gpu --row "$base" --set inputs=instructed "$@" --name I1 --seeds 0
fi
step score cpu --score --sheet
finish
