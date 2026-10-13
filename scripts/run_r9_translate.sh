#!/usr/bin/env bash
# Offline NLLB translation on a DAS6 node; --check-only validates mapping and shards without loading weights.
#   bash scripts/run_r9_translate.sh --shard 0 --nshards 6 [--check-only]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-/var/scratch/wding/cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY="${R9_PYTHON:-python}"
CSV="${R9_A28_CSV:-/var/scratch/wding/Dataset/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv}"
SHARD= NSHARDS= CHECK=0
ARGS=()
fail() { echo "run_r9_translate: $*" >&2; exit 2; }
while (( $# )); do
    case "$1" in
        --shard) (( $# >= 2 )) || fail "--shard needs a value"; SHARD="$2"; shift 2 ;;
        --nshards) (( $# >= 2 )) || fail "--nshards needs a value"; NSHARDS="$2"; shift 2 ;;
        --check-only) CHECK=1; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
[[ "$SHARD" =~ ^(0|[1-9][0-9]*)$ && "$NSHARDS" =~ ^[1-9][0-9]*$ ]] || fail "--shard i --nshards n required"
(( SHARD < NSHARDS )) || fail "shard must be less than nshards"
[[ -f "$CSV" ]] || fail "CSV missing: $CSV"
echo "host $(hostname), cwd $(pwd), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
OUT="${R9_OUT_ROOT:-/local/wding/res/r9}/a28_translate/shard_$SHARD"
if (( CHECK )); then ARGS+=(--dry-run); fi
exec "$PY" scripts/translate_artelingo28.py --csv "$CSV" --out "$OUT" \
    --shard "$SHARD" --nshards "$NSHARDS" "${ARGS[@]}"
