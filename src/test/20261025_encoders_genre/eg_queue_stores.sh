#!/usr/bin/env bash
# New stores (rule §4.1, spec §3.5; GPU): X0, the eight extractions (CLIP B/16, CLIP L/14, SigLIP 1, DINOv2-large; each
# for ArtELingo and ArtELingo-28), then V5 (calibration, then each new store). Run from the merged main checkout:
# scripts/extract_features.py and the X0 git archive live there. A finished store, a written x0.json and a passed V5
# entry are skipped. A failed X0 writes no store and ends the queue (stage i stops; stages ii and iii go on).
# A failed extraction or V5 of one store costs only that candidate (no outcome); the others go on.
# Candidates deferred (part2/deferral.json, EG_DEFER) get no store in this run: run the queue again without it.
# Smoke: extract_features --limit 16 --out-root <outputs>/stores (writes under _smoke/); V5 only for stores whose real
# index.json exists (V5 reads /data/SSD2/pre_extract, never a smoke store).
queue=stores
source "$(dirname -- "${BASH_SOURCE[0]}")/eg_queue_common.sh"
if [[ -f "$outputs/stores/x0.json" && "$dry" != 1 ]]; then
    note "x0.json exists, X0 skipped"
else
    step x0 gpu "$here/eg_x0.py" --outputs "$outputs" --device "$device" || finish
fi
gate x0 || { printf 'EG_STORES_STOP X0 failed: no new store is written; stage i stops, stages ii (K2 to K4) and iii go on\n' | tee -a "$status"; failed=1; finish; }
cands=$(all_cands)
[[ -n "${EG_DEFER:-}" ]] && note "deferred for time (EG_DEFER): $EG_DEFER"
note "stores for: $cands"
extract() {  # extract LABEL DONEFILE ARGS...   returns 1 on a failed extraction (the caller notes it)
    local label=$1 done_file=$2; shift 2
    if [[ "${EG_SMOKE:-0}" == 1 ]]; then
        soft=1 step "$label" gpu "$root/scripts/extract_features.py" "$@" --limit 16 --out-root "$outputs/stores" --device "$device"
    elif [[ -f "$done_file" && "$dry" != 1 ]]; then
        note "$label: $done_file exists, skipped"
    else
        soft=1 step "$label" gpu "$root/scripts/extract_features.py" "$@" --device "$device"
    fi
}
for k in $cands; do
    s=$(store_of "$k")
    img=(); [[ "$s" == dinov2_large ]] && img=(--image-only)
    extract "${s}_full" "$pre/artelingo_full/$s/index.json" --dataset artelingo_full --backbone "$s" "${img[@]}" \
        || note "$k: extraction of artelingo_full/$s failed (log in $outputs/logs); $k has no outcome unless fixed"
    extract "${s}_a28" "$pre/artelingo28/$s/index.json" --dataset artelingo28 --backbone "$s" "${img[@]}" \
        || note "$k: extraction of artelingo28/$s failed (log in $outputs/logs); $k has no A outcome unless fixed"
done
# V5. The calibration (CLIP B/32 and SigLIP 2) first; it decides whether the 1% bar stands.
if [[ "$(jget "$outputs/stores/v5.json" "set(d.get('calibration',{})) >= {'clip_b32','siglip2_so400m'}")" == True && "$dry" != 1 ]]; then
    note "v5.json holds the calibration, skipped"
else
    soft=1 step v5_calibrate cpu "$here/eg_v5.py" --calibrate --outputs "$outputs" || { note "V5 calibration failed: no store is checked"; failed=1; finish; }
fi
for k in $cands; do
    s=$(store_of "$k")
    if [[ "${EG_SMOKE:-0}" == 1 && ! -f "$pre/artelingo_full/$s/index.json" ]]; then
        note "$k: no real store yet (smoke extracts into _smoke/): V5 skipped"; continue
    fi
    both=$(jget "$outputs/stores/v5.json" "d['stores'].get('artelingo_full/$s',{}).get('pass') is True and d['stores'].get('artelingo28/$s',{}).get('pass') is True")
    if [[ "$both" == True && "$dry" != 1 ]]; then note "$k: V5 passed on both datasets, skipped"; continue; fi
    soft=1 step "v5_$s" cpu "$here/eg_v5.py" --store "$s" --dataset artelingo_full --dataset artelingo28 --outputs "$outputs" \
        || note "$k: V5 failed for $s (see log; a fix may be checked again; else $k has no outcome, and the others go on)"
done
failed=0   # a failed store or V5 costs only that candidate; stage i's gates report it
finish
