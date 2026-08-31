#! /bin/bash
set -euo pipefail

# Run CLIP baseline fine-tuning for multiple modes on a given dataset.
# Usage:
#   bash scripts/run_baseline.sh                          # impressions, all 3 modes, seed 42
#   bash scripts/run_baseline.sh --dataset coco           # coco, all 3 modes
#   bash scripts/run_baseline.sh --modes linear           # single mode
#   bash scripts/run_baseline.sh --seeds "42 123 456"     # multi-seed (sequential)
#   bash scripts/run_baseline.sh --epochs 10 --ratio 0.1  # quick test run

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoSiR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET="impressions"
MODES="last_blocks linear lora"
SEEDS="42"
EPOCHS=30
BATCH_SIZE=256
LR=1e-5
RATIO=1.0
EVAL_INTERVAL=5
NUM_WORKERS=4
WANDB_PROJECT="cosir_baseline"
WANDB_ENTITY="augustoxq"
WANDB_MODE="online"
OUTPUT_DIR="res/baseline"
EXTRA_ARGS=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)       DATASET="$2";       shift 2 ;;
    --modes)         MODES="$2";         shift 2 ;;
    --seeds)         SEEDS="$2";         shift 2 ;;
    --epochs)        EPOCHS="$2";        shift 2 ;;
    --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
    --lr)            LR="$2";            shift 2 ;;
    --ratio)         RATIO="$2";         shift 2 ;;
    --eval_interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --num_workers)   NUM_WORKERS="$2";   shift 2 ;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_entity)  WANDB_ENTITY="$2";  shift 2 ;;
    --wandb_mode)    WANDB_MODE="$2";    shift 2 ;;
    --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
    --no_wandb)      WANDB_MODE="disabled"; shift ;;
    *)               EXTRA_ARGS="$EXTRA_ARGS $1 $2"; shift 2 ;;
  esac
done

# ── Run ───────────────────────────────────────────────────────────────────────
for MODE in $MODES; do
  for SEED in $SEEDS; do
    echo ""
    echo "=========================================="
    echo "  dataset=${DATASET}  mode=${MODE}  seed=${SEED}"
    echo "=========================================="

    python src/baseline/train_baseline.py \
      --dataset        "$DATASET" \
      --mode           "$MODE" \
      --epochs         "$EPOCHS" \
      --batch_size     "$BATCH_SIZE" \
      --lr             "$LR" \
      --ratio          "$RATIO" \
      --seed           "$SEED" \
      --eval_interval  "$EVAL_INTERVAL" \
      --num_workers    "$NUM_WORKERS" \
      --wandb_project  "$WANDB_PROJECT" \
      --wandb_entity   "$WANDB_ENTITY" \
      --wandb_mode     "$WANDB_MODE" \
      --output_dir     "$OUTPUT_DIR" \
      $EXTRA_ARGS
  done
done
