#!/usr/bin/env bash
# Pre-registered Qwen3-VL-8B MLLM probe (src/test/20261106_mllm_probe_8b/PREREGISTRATION.md) on a DAS6 node.
# Launch from the CoSiR worktree:  cluster launch --node node404 -- bash scripts/run_mllm_probe_8b.sh
# The job's cwd is its code worktree; results land in outputs/mllm_probe_8b_seed46 (`cluster pull --tag` fetches them).
# `--check-only` verifies the inputs and exits without running the probe (no GPU).
# Data reach the node with scripts/das6_sync_mllm_probe_8b.py. Node paths below can be overridden for a local check.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export COSIR_ARTELINGO_FEATURES="${COSIR_ARTELINGO_FEATURES:-/local/wding/pre_extract/artelingo/features}"
export COSIR_ARTELINGO_ANNOTATIONS="${COSIR_ARTELINGO_ANNOTATIONS:-/local/wding/Dataset/artelingo/artelingo_train.json}"
export COSIR_WIKIART_GENRE_DIR="${COSIR_WIKIART_GENRE_DIR:-/local/wding/Dataset/wikiart_genre}"
export COSIR_WIKIART_DIR="${COSIR_WIKIART_DIR:-/local/wding/Dataset/wikiart_proj/wikiart}"
export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-/var/scratch/wding/cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

MODEL=Qwen/Qwen3-VL-8B-Instruct
IMAGE_LIST=scripts/mllm_probe_8b_images.txt
OUT=outputs/mllm_probe_8b_seed46

fail() { echo "run_mllm_probe_8b: $*" >&2; exit 2; }

echo "host $(hostname), cwd $(pwd), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
[[ -f "$COSIR_ARTELINGO_FEATURES/metadata.json" ]] || fail "features missing: $COSIR_ARTELINGO_FEATURES/metadata.json"
[[ -f "$COSIR_ARTELINGO_ANNOTATIONS" ]] || fail "annotations missing: $COSIR_ARTELINGO_ANNOTATIONS"
for csv in genre_train.csv genre_val.csv; do
    [[ -f "$COSIR_WIKIART_GENRE_DIR/$csv" ]] || fail "genre labels missing: $COSIR_WIKIART_GENRE_DIR/$csv"
done

repo="$HF_HUB_CACHE/models--${MODEL//\//--}"
[[ -f "$repo/refs/main" ]] || fail "model not in the hub cache: $repo/refs/main"
snapshot="$repo/snapshots/$(cat "$repo/refs/main")"
[[ -f "$snapshot/config.json" && -f "$snapshot/model.safetensors.index.json" ]] || fail "incomplete snapshot: $snapshot"
broken=0
for entry in "$snapshot"/*; do [[ -e "$entry" ]] || { echo "dangling: $entry" >&2; broken=$((broken + 1)); }; done
(( broken == 0 )) || fail "$broken snapshot file(s) point at missing blobs in $snapshot"
echo "model snapshot ok: $snapshot ($(ls "$snapshot" | wc -l) files)"

[[ -f "$IMAGE_LIST" ]] || fail "image list missing: $IMAGE_LIST"
total=0 missing=0
while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    total=$((total + 1))
    if [[ ! -f "$COSIR_WIKIART_DIR/$rel" ]]; then
        missing=$((missing + 1))
        (( missing <= 5 )) && echo "missing image: $COSIR_WIKIART_DIR/$rel" >&2
    fi
done < "$IMAGE_LIST"
echo "images: $total listed, $missing missing under $COSIR_WIKIART_DIR"
(( total > 0 && missing == 0 )) || fail "$missing of $total probe images missing under $COSIR_WIKIART_DIR"

python -c "import transformers; from transformers import Qwen3VLForConditionalGeneration; print('transformers', transformers.__version__)" \
    || fail "this env's transformers has no Qwen3VLForConditionalGeneration"
echo "inputs ok"
[[ "${1:-}" == "--check-only" ]] && exit 0

exec python src/test/20261102_mllm_probe/run_probe.py --n 600 --seed 46 --model "$MODEL" --out "$OUT"
