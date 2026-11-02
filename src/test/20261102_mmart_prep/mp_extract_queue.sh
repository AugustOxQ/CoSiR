#!/usr/bin/env bash
# Controller launches this at 19:00. No scheduling or GPU work occurs on import.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PYTHON=/root/miniconda3/envs/CoSiR/bin/python
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SMOKE=()
if [[ ${1:-} == --smoke && $# == 3 && $2 =~ ^[1-9][0-9]*$ ]]; then
    export MP_STORE_ROOT="$3" MP_OUTPUTS="$3"
    SMOKE=(--smoke "$2")
elif [[ $# != 0 ]]; then
    echo 'usage: mp_extract_queue.sh [--smoke N SCRATCH_DIRECTORY]' >&2
    exit 2
fi
ROOT=${MP_STORE_ROOT:-/data/SSD2/pre_extract/mmart}
mkdir -p -- "$ROOT/logs"
LOG="$ROOT/logs/queue_$(date +%Y%m%d_%H%M%S).log"
STATUS="$HERE/mp_queue_status.json"
START=$SECONDS
STEP=setup
finish() {
    local rc=$?
    trap - EXIT
    "$PYTHON" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(dict(step=sys.argv[2], rc=int(sys.argv[3]), seconds=int(sys.argv[4]), log=sys.argv[5]))+"\n")' \
        "$STATUS" "$STEP" "$rc" "$((SECONDS - START))" "$LOG"
    if (( rc == 0 )); then
        echo "MP QUEUE DONE ok in $((SECONDS - START))s"
    else
        echo "MP QUEUE FAILED at $STEP rc=$rc"
    fi
    exit "$rc"
}
exec > >(tee -a "$LOG") 2>&1
trap finish EXIT
STEP=extract-all
flock -n -o -E 75 /tmp/gpu0.lock "$PYTHON" "$HERE/mp_extract.py" extract-all "${SMOKE[@]}"
RC=$?
if (( RC == 75 )); then echo 'GPU busy: /tmp/gpu0.lock is held; extraction did not start'; fi
if (( RC != 0 )); then exit "$RC"; fi
for STEP in parity nearcopy check; do
    "$PYTHON" "$HERE/mp_extract.py" "$STEP"
    RC=$?
    if (( RC != 0 )); then exit "$RC"; fi
done
STEP=done
