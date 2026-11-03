#!/bin/sh
# r19 stage 0 (i): the extraction queue. One `res-queue gpu` step per store; pp_extract.py takes no lock itself.
# Order: parity of the writing code first (short), Affection's three (the photo models need them), MMArt's three,
# SemArt's five. `extract` runs the store check (OUT/stage0/gate_i/store_<dataset>_<view>.json) after each store.
# A sealed store is not rewritten (pp_extract refuses it): rerun after a failure by deleting only the .pending folder
# you know is yours. Status lines (step, seconds, exit code) go to $OUT/stage0/gate_i/extract_queue_status.txt.
# Estimates (--est) are from earlier per-image rates (ms: B/32 5, B/16 5.5, DINOv2 8.6, L/14 9, SigLIP 1 20, Qwen 90);
# the smoke run of ticket 02 measured the ones in its hand-back.
# Usage: PP_OUTPUTS unset for the real run.   sh pp_extract_queue.sh [parity|affection|mmart|semart|all]
cd "$(dirname "$0")" || exit 1
PY=${PY:-/root/miniconda3/envs/CoSiR/bin/python}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=${PP_OUTPUTS:-/data/SSD2/cosir_r19/20261103_past_picks/outputs}
STATUS=$OUT/stage0/gate_i/extract_queue_status.txt
mkdir -p "$OUT/stage0/gate_i"
WHICH=${1:-all}

step() {  # step <est> <label-suffix> <short|long> <args of pp_extract.py>
    est=$1; label=$2; kind=$3; shift 3
    flag=""; [ "$kind" = short ] && flag="--short"
    t0=$(date +%s)
    res-queue gpu --est "$est" $flag --label "r19:$label" -- "$PY" pp_extract.py "$@"
    rc=$?
    printf '%s\t%s\t%s\n' "$label" "$(( $(date +%s) - t0 ))" "$rc" >> "$STATUS"
    [ "$rc" -eq 0 ] || echo "step $label failed (exit $rc)" >&2
}

if [ "$WHICH" = parity ] || [ "$WHICH" = all ]; then
    for v in dinov2_large clip_b32 clip_b16 clip_l14 siglip1 qwen siglip2 genre_word; do
        step 12m "parity-$v" short parity --view "$v"
    done
fi
if [ "$WHICH" = affection ] || [ "$WHICH" = all ]; then
    step 20m "extract-affection-clip_b32" long extract --dataset affection --view clip_b32
    step 20m "extract-affection-clip_b16" long extract --dataset affection --view clip_b16
    step 25m "extract-affection-dinov2_large" long extract --dataset affection --view dinov2_large
fi
if [ "$WHICH" = mmart ] || [ "$WHICH" = all ]; then
    step 6m "extract-mmart-clip_b32" short extract --dataset mmart --view clip_b32
    step 6m "extract-mmart-clip_b16" short extract --dataset mmart --view clip_b16
    step 8m "extract-mmart-dinov2_large" short extract --dataset mmart --view dinov2_large
fi
if [ "$WHICH" = semart ] || [ "$WHICH" = all ]; then
    step 12m "extract-semart-clip_b16" short extract --dataset semart --view clip_b16
    step 12m "extract-semart-dinov2_large" short extract --dataset semart --view dinov2_large
    step 12m "extract-semart-clip_l14" short extract --dataset semart --view clip_l14
    step 15m "extract-semart-siglip1" long extract --dataset semart --view siglip1
    step 40m "extract-semart-genre_word" long extract --dataset semart --view genre_word
fi
if grep -qP '\t[1-9][0-9]*$' "$STATUS" 2>/dev/null; then
    echo "some steps failed: see $STATUS" >&2
    exit 1
fi
