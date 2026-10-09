#!/usr/bin/env bash
# Round 6 fine-tuned CLIP features for FT-LB and FT-LoRA (src/test/20261125_artelingo_held_test/DECISION_RULE.md section 2
# and section 8 item 3) on a DAS6 node, or on the local GPU under its lock. Launch from the CoSiR checkout:
#   cluster launch --node node401 -- bash scripts/run_r6_ftfeat.sh <job> [--variants LB,LoRA] [--check-only]
# <job> is a job-input folder written by `r6_gpu_inputs.py --job ft`, under $R6_JOB_ROOT (default /local/wding/r6_jobs):
# rows_manifest.npz and ft_rows.npz (row ids, neutral image names, captions). Images come from the uint8 cache
# $R6_FT_CACHE (r6_ft_cache.py; optional) or are decoded from the neutral images in $R6_IMAGE_DIR (default
# /local/wding/r6_jobs/images; scripts/das6_sync_r6.py --images ships them). The checkpoints are the selected clipft
# runs' best_params.pt: $R6_FT_CKPT_DIR/LB_lr3e-5.pt and $R6_FT_CKPT_DIR/LoRA_lr1e-4.pt (default
# /local/wding/r6_jobs/ckpt; das6_sync_r6.py does not ship them: copy them there first, 42 MB and 7.6 MB).
# Outputs go to outputs/r6_ft/<job> (R6_OUT overrides) and are fetched with `cluster pull --tag`: features_<variant>.npz
# with rows, img, txt (raw projections, float32). `--check-only` checks the inputs and checkpoints and encodes 4 rows.
# The job reads no label, aspect name or candidate order and prints no metric.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-/var/scratch/wding/cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
R6_JOB_ROOT="${R6_JOB_ROOT:-/local/wding/r6_jobs}"
R6_IMAGE_DIR="${R6_IMAGE_DIR:-/local/wding/r6_jobs/images}"
R6_FT_CKPT_DIR="${R6_FT_CKPT_DIR:-/local/wding/r6_jobs/ckpt}"
PY="${R6_PYTHON:-python}"
F=src/test/20261125_artelingo_held_test
MODEL=openai/clip-vit-base-patch32

fail() { echo "run_r6_ftfeat: $*" >&2; exit 2; }

(( $# >= 1 )) || fail "usage: run_r6_ftfeat.sh <job> [--variants LB,LoRA] [--check-only]"
JOB="$1"
shift
[[ "$JOB" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "bad job name: $JOB"
JOB_DIR="$R6_JOB_ROOT/$JOB"
OUT="${R6_OUT:-outputs/r6_ft/$JOB}"

echo "host $(hostname), cwd $(pwd), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
for f in rows_manifest.npz ft_rows.npz images.txt; do
    [[ -f "$JOB_DIR/$f" ]] || fail "job input missing: $JOB_DIR/$f"
done

"$PY" -c "import torch, peft, transformers; print('torch', torch.__version__, 'peft', peft.__version__, 'transformers', transformers.__version__)" \
    || fail "this env cannot import torch, peft and transformers"
repo="$HF_HUB_CACHE/models--${MODEL//\//--}"
[[ -f "$repo/refs/main" ]] || fail "CLIP not in the hub cache: $repo/refs/main"
snapshot="$repo/snapshots/$(cat "$repo/refs/main")"
[[ -f "$snapshot/config.json" ]] || fail "incomplete CLIP snapshot: $snapshot"
[[ -e "$snapshot/model.safetensors" || -e "$snapshot/pytorch_model.bin" ]] || fail "CLIP snapshot without weights: $snapshot"
broken=0
for entry in "$snapshot"/*; do [[ -e "$entry" ]] || { echo "dangling: $entry" >&2; broken=$((broken + 1)); }; done
(( broken == 0 )) || fail "$broken CLIP snapshot file(s) point at missing blobs in $snapshot"
echo "CLIP snapshot ok: $snapshot"

ARGS=(--ckpt "LB=$R6_FT_CKPT_DIR/LB_lr3e-5.pt" --ckpt "LoRA=$R6_FT_CKPT_DIR/LoRA_lr1e-4.pt")
for c in LB_lr3e-5 LoRA_lr1e-4; do
    [[ -f "$R6_FT_CKPT_DIR/$c.pt" ]] || fail "checkpoint missing: $R6_FT_CKPT_DIR/$c.pt"
done
if [[ -n "${R6_FT_CACHE:-}" ]]; then
    for f in images_uint8.npy paintings.json cache_record.json; do
        [[ -f "$R6_FT_CACHE/$f" ]] || fail "image cache incomplete, missing: $R6_FT_CACHE/$f"
    done
    ARGS+=(--cache-dir "$R6_FT_CACHE")
    echo "image source: cache $R6_FT_CACHE"
else
    total=0 missing=0
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        total=$((total + 1))
        [[ "$name" =~ ^[0-9a-f]{20}\.[a-z0-9]{1,5}$ ]] || fail "images.txt holds a name that is not neutral: $name"
        if [[ ! -f "$R6_IMAGE_DIR/$name" ]]; then
            missing=$((missing + 1))
            (( missing <= 5 )) && echo "missing image: $R6_IMAGE_DIR/$name" >&2
        fi
    done < "$JOB_DIR/images.txt"
    echo "images: $total listed, $missing missing under $R6_IMAGE_DIR"
    (( total > 0 && missing == 0 )) || fail "$missing of $total job images missing under $R6_IMAGE_DIR"
fi
echo "inputs ok"

exec "$PY" "$F/r6_gpu_ft_features.py" --job-dir "$JOB_DIR" --out "$OUT" --image-dir "$R6_IMAGE_DIR" "${ARGS[@]}" "$@"
