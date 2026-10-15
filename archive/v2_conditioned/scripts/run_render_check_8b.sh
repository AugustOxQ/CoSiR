#!/usr/bin/env bash
# Render-equivalence check (processor only, no model weights, no GPU) for the 8B probe on a DAS6 node.
# Launch from the CoSiR worktree:  cluster launch --node node404 -- bash scripts/run_render_check_8b.sh
# Same env as scripts/run_mllm_probe_8b.sh. Output: outputs/render_check_node.json
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export COSIR_ARTELINGO_FEATURES="${COSIR_ARTELINGO_FEATURES:-/local/wding/pre_extract/artelingo/features}"
export COSIR_ARTELINGO_ANNOTATIONS="${COSIR_ARTELINGO_ANNOTATIONS:-/local/wding/Dataset/artelingo/artelingo_train.json}"
export COSIR_WIKIART_GENRE_DIR="${COSIR_WIKIART_GENRE_DIR:-/local/wding/Dataset/wikiart_genre}"
export COSIR_WIKIART_DIR="${COSIR_WIKIART_DIR:-/local/wding/Dataset/wikiart_proj/wikiart}"
export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-/var/scratch/wding/cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

mkdir -p outputs
exec python src/test/20261106_mllm_probe_8b/render_check.py --out outputs/render_check_node.json
