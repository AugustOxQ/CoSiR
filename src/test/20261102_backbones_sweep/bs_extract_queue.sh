#!/usr/bin/env bash
# r14 ticket 01 queue: unpack EmoSet, then (under the GPU lock) parity, the extractions (Affection first, then EmoSet),
# then the store check. Run from anywhere:
#   bash bs_extract_queue.sh            (real run: stores under /data/SSD2/pre_extract, outputs under BS_OUTPUTS/default)
# Env: BS_LOCK_WAIT seconds to wait for /tmp/gpu0.lock (default 3600; the lock is released if the wait ends: exit 1).
# Status file: $OUT/stage0/extract_queue_status.txt, one line per step "<step>\t<seconds>\t<exit code>", then one summary
# line "QUEUE DONE ..." or "QUEUE FAILED at <step> ...". Stops at the first failing step. Re-running skips sealed
# stores; a leftover <folder>.pending stops the extraction (remove it by hand after looking at it).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT="${BS_OUTPUTS:-/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs}"
STATUS="$OUT/stage0/extract_queue_status.txt"
mkdir -p "$OUT/stage0"

if [[ "${BS_QUEUE_LOCKED:-}" != "1" ]]; then
    mkdir -p "$OUT/stage0"
    echo "# queue started $(TZ=Europe/Amsterdam date '+%F %H:%M')" >> "$STATUS"
    t0=$SECONDS
    "$PY" "$HERE/bs_extract.py" unpack
    rc=$?
    printf 'unpack\t%d\t%d\n' $((SECONDS - t0)) $rc >> "$STATUS"
    if [[ $rc -ne 0 ]]; then
        echo "QUEUE FAILED at unpack exit=$rc" | tee -a "$STATUS"
        exit 1
    fi
    export BS_QUEUE_LOCKED=1
    exec flock -o -w "${BS_LOCK_WAIT:-3600}" /tmp/gpu0.lock bash "${BASH_SOURCE[0]}"
fi

T_ALL=$SECONDS
step() {   # step <name> <command...>
    local name="$1"; shift
    local t=$SECONDS
    "$@"
    local rc=$?
    printf '%s\t%d\t%d\n' "$name" $((SECONDS - t)) $rc >> "$STATUS"
    if [[ $rc -ne 0 ]]; then
        echo "QUEUE FAILED at $name exit=$rc after $((SECONDS - T_ALL))s" | tee -a "$STATUS"
        exit 1
    fi
}

step parity "$PY" "$HERE/bs_extract.py" parity
step affection_siglip2 "$PY" "$HERE/bs_extract.py" extract --dataset affection --view siglip2 --skip-sealed
step affection_siglip1 "$PY" "$HERE/bs_extract.py" extract --dataset affection --view siglip1 --skip-sealed
step affection_clip_l14 "$PY" "$HERE/bs_extract.py" extract --dataset affection --view clip_l14 --skip-sealed
for v in qwen_default emotion_focus emotion_focus_photo clip_l14 siglip2 siglip1; do
    step "emoset_$v" "$PY" "$HERE/bs_extract.py" extract --dataset emoset --view "$v" --skip-sealed
done
step check "$PY" "$HERE/bs_extract.py" check
echo "QUEUE DONE ok in $((SECONDS - T_ALL))s (parity, 3 Affection stores, 6 EmoSet stores, check)" | tee -a "$STATUS"
