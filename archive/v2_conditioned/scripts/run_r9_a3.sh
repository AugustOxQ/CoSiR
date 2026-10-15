#!/usr/bin/env bash
# Round 9 recipe II: A3 (seed 42) on one backbone's recipe parts, on a DAS6 node (or any GPU host), with the identity
# gates of rule section 3.2 item 2. Launch from the CoSiR checkout on the node:
#   cluster launch --node node406 -- bash scripts/run_r9_a3.sh <backbone> <L1|L2|C1|main> [--check-only] [extra args]
# <backbone> is clip_b32, qwen3vl_emb_2b or siglip2_so400m. C1 is the CLIP cluster replay (rule 3.2 item 2).
# Extra args go to r9_recipe_a3.py (--inputs DIR, --compare-to CKPT). --check-only runs `--dry-run` (config gate and
# inputs; no training, no output files).
# Output: ${R9_OUT_ROOT:-/local/wding/res/r9}/a3_<backbone>_<run>/ (A3_seed42.pt, history_A3_seed42.json, picked.json,
# identity_gates.json and the .record.json of each); fetch with `cluster pull --tag`.
#
# Inputs that must be on the node (nothing here ships them; check-only fails with the name of the first missing one):
#   - the CoSiR checkout at the commit to run, with its ignored files that rounds 1 to 6 left in the main checkout:
#     src/test/20261013_stage_d_selection/cache/prepare.npz, src/test/20261014_stage_d_final/cache/held_codes.npz,
#     src/test/20261101_aspect_factor_gonogo/{results/history_A3_seed42.json,checkpoints/A3_seed42.pt} (the pinned A3),
#     src/test/20261016_factor_learning_grid/results/history_C0_seed42.json, and the earlier rounds' folders that
#     r9_common imports (the whole src/test tree of the main checkout);
#   - CLIP's feature shard store (labels, sample IDs and, for CLIP, the features): $COSIR_ARTELINGO_FEATURES, default
#     /var/scratch/wding/Dataset/artelingo/features (scripts/das6_sync_r9.py does not ship it yet);
#   - ArtELingo annotations: $COSIR_ARTELINGO_ANNOTATIONS, default /var/scratch/wding/Dataset/artelingo/artelingo_train.json;
#   - the backbone's parts: CLIP: the pinned files (partitions, graph, bank_AIC and build_record.json in
#     src/test/20261031_pseudo_partitions/results/, graph also in src/test/20261016_factor_learning_grid/cache/), or
#     --inputs DIR; other backbones: src/test/20261202_r9_scope/results/recipe/<backbone>/{partitions.npz,graph.npz,
#     bank_AIC.npz,build_record.json} (made locally by r9_recipe_groups.py; the graph needs the local GPU);
#   - other backbones: the feature store /data/SSD2/pre_extract/artelingo_full/<backbone>/{img.npy,txt.npy,index.json}
#     (a fixed path in r9_backbone.py; link it on the node).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export COSIR_ARTELINGO_FEATURES="${COSIR_ARTELINGO_FEATURES:-/var/scratch/wding/Dataset/artelingo/features}"
export COSIR_ARTELINGO_ANNOTATIONS="${COSIR_ARTELINGO_ANNOTATIONS:-/var/scratch/wding/Dataset/artelingo/artelingo_train.json}"
PY="${R9_PYTHON:-python}"
F=src/test/20261202_r9_scope
fail() { echo "run_r9_a3: $*" >&2; exit 2; }

(( $# >= 2 )) || fail "usage: run_r9_a3.sh <backbone> <L1|L2|C1|main> [--check-only] [extra args]"
BACKBONE="$1" RUN="$2"
shift 2
case "$BACKBONE" in clip_b32|qwen3vl_emb_2b|siglip2_so400m) ;; *) fail "unknown backbone: $BACKBONE" ;; esac
case "$RUN" in L1|L2|C1|main) ;; *) fail "unknown run: $RUN" ;; esac
CHECK=0
EXTRA=()
for a in "$@"; do
    case "$a" in
        --check-only) CHECK=1 ;;
        --backbone|--backbone=*|--run|--run=*|--out|--out=*|--dry-run) fail "set by the wrapper, not an extra arg: $a" ;;
        *) EXTRA+=("$a") ;;
    esac
done
OUT="${R9_OUT_ROOT:-/local/wding/res/r9}/a3_${BACKBONE}_${RUN}"

echo "host $(hostname), cwd $(pwd), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
for f in "$F/r9_recipe_a3.py" "$F/r9_common.py" \
         src/test/20261101_aspect_factor_gonogo/results/history_A3_seed42.json \
         src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt \
         src/test/20261013_stage_d_selection/cache/prepare.npz src/test/20261014_stage_d_final/cache/held_codes.npz \
         "$COSIR_ARTELINGO_ANNOTATIONS"; do
    [[ -f "$f" ]] || fail "input missing: $f"
done
[[ -d "$COSIR_ARTELINGO_FEATURES" ]] || fail "CLIP feature store missing: $COSIR_ARTELINGO_FEATURES"
if [[ "$BACKBONE" != clip_b32 ]]; then
    for f in img.npy txt.npy index.json; do
        [[ -f "/data/SSD2/pre_extract/artelingo_full/$BACKBONE/$f" ]] || fail "feature store missing: /data/SSD2/pre_extract/artelingo_full/$BACKBONE/$f"
    done
    for f in partitions.npz graph.npz bank_AIC.npz build_record.json; do
        [[ " ${EXTRA[*]:-} " == *" --inputs "* ]] || [[ -f "$F/results/recipe/$BACKBONE/$f" ]] || fail "recipe part missing: $F/results/recipe/$BACKBONE/$f"
    done
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if (( CHECK )); then
    exec "$PY" "$F/r9_recipe_a3.py" --backbone "$BACKBONE" --run "$RUN" --dry-run ${EXTRA[@]+"${EXTRA[@]}"}
fi
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)" || fail "no GPU visible: A3 trains on the GPU"
exec "$PY" "$F/r9_recipe_a3.py" --backbone "$BACKBONE" --run "$RUN" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
