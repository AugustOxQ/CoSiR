#!/usr/bin/env bash
# Frozen r9 features on a DAS6 node. --check-only validates inputs and encodes four images.
#   bash scripts/run_r9_extract.sh <artelingo_full|artelingo28|semart> <backbone> [--check-only] [extra args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-/var/scratch/wding/cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY="${R9_PYTHON:-python}"
ANN=/var/scratch/wding/Dataset/artelingo/artelingo_train.json
IMAGES=/var/scratch/wding/Dataset/wikiart_proj/wikiart
IDS=/var/scratch/wding/Dataset/artelingo/r9_sample_ids.npy
SEMART=/var/scratch/wding/Dataset/semart/SemArt
A28=/var/scratch/wding/Dataset/artelingo/ArtELingo/semeval-artelingo28
A28_CSV="$A28/artelingo28_public_noworkerid_split.csv"
A28_ZIP="$A28/images.zip"
A28_TRANS="$A28/nllb200_3.3b_en/translations.parquet"
A28_READ=src/test/20261202_r9_scope/results/read_set_a28.parquet
OUT="${R9_OUT_ROOT:-/local/wding/res/r9}/pre_extract"
CHECK=0
EXTRA=()
fail() { echo "run_r9_extract: $*" >&2; exit 2; }
(( $# >= 2 )) || fail "usage: run_r9_extract.sh <dataset> <backbone> [--check-only] [extra args]"
DATASET="$1" BACKBONE="$2"
shift 2
case "$DATASET/$BACKBONE" in
    artelingo_full/siglip2_so400m|artelingo28/siglip2_so400m|semart/siglip2_so400m) MODEL=google/siglip2-so400m-patch14-384 ;;
    artelingo28/clip_b32|semart/clip_b32) MODEL=openai/clip-vit-base-patch32 ;;
    artelingo_full/qwen3vl_emb_2b|artelingo28/qwen3vl_emb_2b|semart/qwen3vl_emb_2b) MODEL=Qwen/Qwen3-VL-Embedding-2B ;;
    *) fail "unsupported dataset/backbone: $DATASET/$BACKBONE" ;;
esac
while (( $# )); do
    case "$1" in
        --a28-csv=*|--a28-images-zip=*|--a28-translations=*|--a28-read-set=*|--artelingo-ann=*|--artelingo-images=*|--sample-ids=*|--semart-dir=*|--out-root=*)
            set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
    esac
    case "$1" in
        --check-only) CHECK=1; shift ;;
        --a28-csv|--a28-images-zip|--a28-translations|--a28-read-set|--artelingo-ann|--artelingo-images|--sample-ids|--semart-dir|--out-root)
            (( $# >= 2 )) || fail "missing value for $1"
            case "$1" in
                --artelingo-ann) ANN="$2" ;;
                --artelingo-images) IMAGES="$2" ;;
                --sample-ids) IDS="$2" ;;
                --semart-dir) SEMART="$2" ;;
                --out-root) OUT="$2" ;;
                --a28-csv) A28_CSV="$2" ;;
                --a28-images-zip) A28_ZIP="$2" ;;
                --a28-translations) A28_TRANS="$2" ;;
                --a28-read-set) A28_READ="$2" ;;
            esac
            shift 2 ;;
        --dataset|--dataset=*|--backbone|--backbone=*|--verify|--verify=*)
            fail "dataset/backbone are positional; --verify belongs to extract_features.py" ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
echo "host $(hostname), cwd $(pwd), commit $(git rev-parse --short HEAD)"
ARGS=(--dataset "$DATASET" --backbone "$BACKBONE" --out-root "$OUT")
if [[ "$DATASET" == artelingo_full ]]; then
    [[ -f "$ANN" && -d "$IMAGES" && -f "$IDS" ]] || fail "missing ArtELingo annotations, images or sample IDs"
    ARGS+=(--artelingo-ann "$ANN" --artelingo-images "$IMAGES" --sample-ids "$IDS")
elif [[ "$DATASET" == artelingo28 ]]; then
    for input in "$A28_CSV" "$A28_ZIP" "$A28_TRANS" "$A28_READ"; do
        [[ -f "$input" ]] || fail "missing ArtELingo-28 input: $input"
    done
    ARGS+=(--a28-csv "$A28_CSV" --a28-images-zip "$A28_ZIP" --a28-translations "$A28_TRANS" --a28-read-set "$A28_READ")
else
    [[ -d "$SEMART/Images" ]] || fail "missing SemArt Images: $SEMART"
    for split in train val test; do
        [[ -f "$SEMART/semart_$split.csv" ]] || fail "missing SemArt split: $split"
    done
    ARGS+=(--semart-dir "$SEMART")
fi
repo="$HF_HUB_CACHE/models--${MODEL//\//--}"
[[ -f "$repo/refs/main" ]] || fail "model not cached: $repo/refs/main"
snapshot="$repo/snapshots/$(cat "$repo/refs/main")"
for f in config.json preprocessor_config.json tokenizer_config.json; do
    [[ -f "$snapshot/$f" ]] || fail "incomplete snapshot: $snapshot/$f"
done
[[ -f "$snapshot/tokenizer.json" || -f "$snapshot/spiece.model" || -f "$snapshot/vocab.json" ]] \
    || fail "snapshot without tokenizer: $snapshot"
shopt -s nullglob globstar
weights=("$snapshot"/*.safetensors "$snapshot"/pytorch_model*.bin)
(( ${#weights[@]} > 0 )) || fail "snapshot without weights: $snapshot"
for entry in "$snapshot"/**/*; do
    [[ -e "$entry" ]] || fail "dangling snapshot file: $entry"
done
"$PY" - "$snapshot" <<'PY'
import json
import sys
from pathlib import Path
snapshot = Path(sys.argv[1])
for index in snapshot.glob("*.index.json"):
    for name in set(json.load(open(index))["weight_map"].values()):
        assert (snapshot / name).is_file(), f"missing weight shard: {snapshot / name}"
PY
echo "HF snapshot ok: $snapshot"
if (( CHECK )); then
    "$PY" - "$DATASET" "$ANN" "$IMAGES" "$IDS" "$SEMART" "$A28_CSV" "$A28_ZIP" "$A28_TRANS" "$A28_READ" <<'PY'
import sys
from pathlib import Path
import numpy as np
from scripts.extract_features import adapter_artelingo_full, adapter_artelingo28, adapter_semart
dataset, ann, images, ids, semart, csv, zip_path, trans, read = sys.argv[1:]
if dataset == "artelingo_full":
    spec = adapter_artelingo_full(ann, images, np.load(ids, allow_pickle=False))
    paths = [Path(images) / r["image"] for r in spec.images]
elif dataset == "artelingo28":
    spec = adapter_artelingo28(read, csv, trans, zip_path)
    paths = []  # adapter checks every archive member
else:
    spec = adapter_semart(semart)
    paths = [Path(semart) / "Images" / r["image_file"] for r in spec.images]
missing = [p for p in paths if not p.is_file()]
assert not missing, f"{len(missing)} missing images; first: {missing[:5]}"
print(f"inputs ok: {len(spec.images)} images, {len(spec.texts)} texts")
PY
    EXTRA+=(--limit 4)
fi
exec "$PY" scripts/extract_features.py "${ARGS[@]}" "${EXTRA[@]}"
