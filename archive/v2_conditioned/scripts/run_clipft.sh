#!/usr/bin/env bash
# Lightweight CLIP fine-tuning comparator (docs/superpowers/specs/2026-10-07-clip-lightweight-ft-design.md) on a DAS6 node.
# Launch from the CoSiR worktree:  cluster launch --node node404 -- bash scripts/run_clipft.sh <LP|LB|LoRA> <lr> [--check-only|--smoke]
# The job's cwd is its code worktree; results land in outputs/clipft/<variant>_lr<lr>[_smoke] (`cluster pull --tag` fetches them).
# `--check-only` verifies the inputs and exits without training. The 7.4 GB image cache is SHA-256 verified by default (about a minute); CLIPFT_SKIP_SHA=1 skips it (local smoke tests only).
# Data reach the node with scripts/das6_sync_clipft.py. Node paths below can be overridden for a local check;
# CLIPFT_ALLOW_NO_GPU=1 skips the CUDA check (local check only).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export COSIR_ARTELINGO_FEATURES="${COSIR_ARTELINGO_FEATURES:-/local/wding/pre_extract/artelingo/features}"
export COSIR_ARTELINGO_ANNOTATIONS="${COSIR_ARTELINGO_ANNOTATIONS:-/local/wding/Dataset/artelingo/artelingo_train.json}"
export CLIPFT_IMAGE_CACHE="${CLIPFT_IMAGE_CACHE:-/local/wding/Dataset/pre_extract/artelingo_clip224}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/scratch/wding/cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

MODEL=openai/clip-vit-base-patch32
EPOCHS=10

fail() { echo "run_clipft: $*" >&2; exit 2; }

(( $# >= 2 )) || fail "usage: run_clipft.sh <LP|LB|LoRA> <lr> [--check-only|--smoke]"
VARIANT="$1"; LR="$2"; MODE="${3:-}"
case "$VARIANT" in LP|LB|LoRA) ;; *) fail "variant must be LP, LB or LoRA, got '$VARIANT'" ;; esac
[[ "$LR" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)([eE]-?[0-9]+)?$ ]] || fail "lr must be a positive number such as 1e-3, got '$LR'"
python -c "import sys; sys.exit(0 if float('$LR') > 0 else 1)" || fail "lr must be > 0, got '$LR'"
case "$MODE" in ""|--check-only|--smoke) ;; *) fail "unknown option '$MODE'" ;; esac

OUT="outputs/clipft/${VARIANT}_lr${LR}"
[[ "$MODE" == "--smoke" ]] && OUT="${OUT}_smoke"

echo "host $(hostname), cwd $(pwd), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi -L 2>&1 || echo "nvidia-smi not available"

[[ -f "$COSIR_ARTELINGO_FEATURES/metadata.json" ]] || fail "features missing: $COSIR_ARTELINGO_FEATURES/metadata.json"
[[ -f "$COSIR_ARTELINGO_ANNOTATIONS" ]] || fail "annotations missing: $COSIR_ARTELINGO_ANNOTATIONS"
for f in images_uint8.npy paintings.json cache_record.json; do
    [[ -f "$CLIPFT_IMAGE_CACHE/$f" ]] || fail "image cache incomplete, missing: $CLIPFT_IMAGE_CACHE/$f"
done
# Sizes against cache_record.json (shape/dtype/length), then the SHA-256 of both files unless CLIPFT_SKIP_SHA=1.
python - "$CLIPFT_IMAGE_CACHE" "${CLIPFT_SKIP_SHA:-0}" <<'PY' || fail "image cache does not match cache_record.json"
import hashlib, json, os, sys
d, verify = sys.argv[1], sys.argv[2] != "1"
rec = json.load(open(f"{d}/cache_record.json"))
n, shape = rec["n_images"], rec["shape"]
size = os.path.getsize(f"{d}/images_uint8.npy")
payload = n * 224 * 224 * 3
if not (payload < size <= payload + 4096) or shape != [n, 224, 224, 3] or rec["dtype"] != "uint8":
    sys.exit(f"images_uint8.npy is {size} bytes, record says {n} images {shape} {rec['dtype']} ({payload} payload bytes)")
paintings = json.load(open(f"{d}/paintings.json"))["paintings"]
if len(paintings) != n:
    sys.exit(f"paintings.json has {len(paintings)} paintings, record says {n}")
if verify:
    for name, sha in rec["sha256"].items():
        h = hashlib.sha256()
        with open(f"{d}/{name}", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            sys.exit(f"SHA-256 mismatch: {name}")
    print("image cache SHA-256 ok")
print(f"image cache ok: {n} images, {size} bytes" + ("" if verify else " (sizes only; SHA skipped by CLIPFT_SKIP_SHA=1)"))
PY

repo="$HF_HUB_CACHE/models--${MODEL//\//--}"
[[ -f "$repo/refs/main" ]] || fail "model not in the hub cache: $repo/refs/main"
snapshot="$repo/snapshots/$(cat "$repo/refs/main")"
[[ -f "$snapshot/config.json" ]] || fail "incomplete snapshot (config.json): $snapshot"
[[ -e "$snapshot/model.safetensors" || -e "$snapshot/pytorch_model.bin" ]] || fail "incomplete snapshot (no weights): $snapshot"
broken=0
for entry in "$snapshot"/*; do [[ -e "$entry" ]] || { echo "dangling: $entry" >&2; broken=$((broken + 1)); }; done
(( broken == 0 )) || fail "$broken snapshot file(s) point at missing blobs in $snapshot"
echo "model snapshot ok: $snapshot ($(ls "$snapshot" | wc -l) files)"

python -c "import torch, peft, transformers; print('torch', torch.__version__, 'peft', peft.__version__, 'transformers', transformers.__version__)" \
    || fail "this env cannot import torch, peft and transformers"
if [[ "${CLIPFT_ALLOW_NO_GPU:-0}" == "1" ]]; then
    echo "GPU check skipped (CLIPFT_ALLOW_NO_GPU=1)"
else
    python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('cuda', torch.cuda.get_device_name(0))" \
        || fail "no visible CUDA device"
fi
echo "inputs ok"
[[ "$MODE" == "--check-only" ]] && exit 0

SMOKE=()
[[ "$MODE" == "--smoke" ]] && SMOKE=(--smoke)
exec python src/test/20261124_clip_lightweight_ft/ft_train.py --variant "$VARIANT" --lr "$LR" --epochs "$EPOCHS" --out "$OUT" ${SMOKE[@]+"${SMOKE[@]}"}
