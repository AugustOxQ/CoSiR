#!/usr/bin/env bash
# New stores (spec §3.7, D26, D27; GPU). Needs the pick (K5 and the DINOv2 stores only when K5 is triggered). Run from the
# repo root of the merged main checkout: scripts/extract_features.py lives there. A finished store is skipped.
# Smoke: extract_features --limit 16 --out-root <outputs>/stores (writes under _smoke/), rb_extract_a28 --limit 16 --dest.
queue=stores
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
root=$(cd -- "$here/../../.." && pwd)
gate part2 || finish
pre=/data/SSD2/pre_extract
k5=0
for c in $(driver triggers); do [[ "$c" == K5 ]] && k5=1; done
for c in ${RB_DEFER:-}; do [[ "$c" == K5 && "$k5" == 1 ]] && k5=deferred; done   # built by the rerun without RB_DEFER
extract() {  # extract LABEL DONEFILE ARGS...
    local label=$1 done_file=$2; shift 2
    if [[ "${RB_SMOKE:-0}" == 1 ]]; then
        step "$label" gpu "$root/scripts/extract_features.py" "$@" --limit 16 --out-root "$outputs/stores" --device "$device"
    elif [[ -f "$done_file" && "$dry" != 1 ]]; then
        note "$label: $done_file exists, skipped"
    else
        step "$label" gpu "$root/scripts/extract_features.py" "$@" --device "$device"
    fi
}
extract clip_full "$pre/artelingo_full/clip_b32/index.json" --dataset artelingo_full --backbone clip_b32 || finish
if [[ "$k5" == 1 ]]; then
    extract dino_full "$pre/artelingo_full/dinov2_large/index.json" --dataset artelingo_full --backbone dinov2_large --image-only || finish
    extract dino_a28 "$pre/artelingo28/dinov2_large/index.json" --dataset artelingo28 --backbone dinov2_large --image-only || finish
elif [[ "$k5" == deferred ]]; then
    note "K5 deferred (RB_DEFER): no DINOv2 stores in this run"
else
    note "K5 not triggered: no DINOv2 stores"
fi
a28_root=$pre/artelingo28/qwen3vl_emb_2b_aspect_prompts
if [[ "${RB_SMOKE:-0}" == 1 ]]; then
    step a28_qwen gpu "$here/rb_extract_a28.py" --limit 16 --dest "$outputs/stores/a28" --device "$device" || finish
else
    missing=()
    for k in emotion_focus style_word genre_word; do [[ -f "$a28_root/$k/done.json" ]] || missing+=("$k"); done
    if [[ ${#missing[@]} == 0 && "$dry" != 1 ]]; then note "a28_qwen: three stores exist, skipped"
    else step a28_qwen gpu "$here/rb_extract_a28.py" --keys "${missing[@]}" --device "$device" || finish; fi
fi
finish
