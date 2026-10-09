#!/usr/bin/env bash
# Round 6 describe-then-score verbaliser (src/test/20261125_artelingo_held_test/DECISION_RULE.md section 7 item 1) on a
# DAS6 node, or on the local GPU under its lock. Launch from the CoSiR checkout:
#   cluster launch --node node401 -- bash scripts/run_r6_verbalise.sh <job> --wordings W1,W2,W3,W4 [--start i --stop j]
# <job> is a job-input folder written by r6_gpu_inputs.py, under $R6_JOB_ROOT (default /local/wding/r6_jobs), shipped
# to the node by scripts/das6_sync_r6.py. Outputs go to outputs/r6_verbalise/<job> (R6_OUT overrides; give each shard
# run from one checkout its own R6_OUT) and are fetched with `cluster pull --tag`.
# `--check-only` checks the inputs, loads the model and runs two episodes; `--check-only --no-model` stops before the
# model (no GPU). The job reads no label, aspect name or candidate order and prints no metric.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export COSIR_WIKIART_DIR="${COSIR_WIKIART_DIR:-/local/wding/Dataset/wikiart_proj/wikiart}"
export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-/var/scratch/wding/cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
R6_JOB_ROOT="${R6_JOB_ROOT:-/local/wding/r6_jobs}"
PY="${R6_PYTHON:-python}"
F=src/test/20261125_artelingo_held_test

fail() { echo "run_r6_verbalise: $*" >&2; exit 2; }

(( $# >= 1 )) || fail "usage: run_r6_verbalise.sh <job> --wordings W1[,W2...] [--start i] [--stop j] [--check-only [--no-model]]"
JOB="$1"
shift
[[ "$JOB" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "bad job name: $JOB"
JOB_DIR="$R6_JOB_ROOT/$JOB"
OUT="${R6_OUT:-outputs/r6_verbalise/$JOB}"

echo "host $(hostname), cwd $(pwd), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
for f in rows_manifest.npz verbalise_input.npz images.txt; do
    [[ -f "$JOB_DIR/$f" ]] || fail "job input missing: $JOB_DIR/$f"
done

"$PY" -c "import transformers; from transformers import Qwen3VLForConditionalGeneration; print('transformers', transformers.__version__)" \
    || fail "this env's transformers has no Qwen3VLForConditionalGeneration"
ids=$("$PY" -c "import json; m = json.load(open('$F/dts_settings.json'))['model']; print(m['id'], m['snapshot'])") \
    || fail "cannot read $F/dts_settings.json"
MODEL="${ids% *}"
SNAPSHOT="${ids#* }"
snapshot="$HF_HUB_CACHE/models--${MODEL//\//--}/snapshots/$SNAPSHOT"
[[ -f "$snapshot/config.json" && -f "$snapshot/model.safetensors.index.json" ]] || fail "pinned snapshot missing or incomplete: $snapshot"
broken=0
for entry in "$snapshot"/*; do [[ -e "$entry" ]] || { echo "dangling: $entry" >&2; broken=$((broken + 1)); }; done
(( broken == 0 )) || fail "$broken snapshot file(s) point at missing blobs in $snapshot"
echo "model snapshot ok: $snapshot ($(ls "$snapshot" | wc -l) files)"

total=0 missing=0
while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    total=$((total + 1))
    if [[ ! -f "$COSIR_WIKIART_DIR/$rel" ]]; then
        missing=$((missing + 1))
        (( missing <= 5 )) && echo "missing image: $COSIR_WIKIART_DIR/$rel" >&2
    fi
done < "$JOB_DIR/images.txt"
echo "images: $total listed, $missing missing under $COSIR_WIKIART_DIR"
(( total > 0 && missing == 0 )) || fail "$missing of $total job images missing under $COSIR_WIKIART_DIR"
echo "inputs ok"

exec "$PY" "$F/r6_gpu_verbalise.py" --job-dir "$JOB_DIR" --out "$OUT" --image-root "$COSIR_WIKIART_DIR" "$@"
