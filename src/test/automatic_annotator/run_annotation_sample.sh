#!/usr/bin/env bash
# Run the full 5k×5k annotation.
# Supports resume: safe to kill and restart — resumes from last completed image row.
#
# LOCAL usage:
#   bash run_annotation.sh local
#
# CLUSTER usage (after sshing in):
#   bash run_annotation.sh cluster
#
# CUSTOM usage (override any arg):
#   bash run_annotation.sh local --batch_size 100 --n_samples 500

MODE=${1:-local}
shift || true   # allow extra args to pass through

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoSiR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$MODE" = "cluster" ]; then
    ANNOTATION_PATH="/var/scratch/wding/Dataset/redcaps_plus/redcaps_test.json"
    IMAGE_ROOT="/var/scratch/wding/Dataset"
    OUTPUT_PATH="/local/wding/annotations/redcaps_50_10"
elif [ "$MODE" = "local" ]; then
    ANNOTATION_PATH="/data/PDD/redcaps/redcaps_plus/redcaps_test.json"
    IMAGE_ROOT="/data/PDD"
    OUTPUT_PATH="/project/CoSiR/res/annotations/redcaps_50_10"
else
    echo "Unknown mode: $MODE. Use 'local' or 'cluster'."
    exit 1
fi

echo "============================================"
echo " Qwen Automatic Annotator — 50×10"
echo "============================================"
echo " Mode:            $MODE"
echo " Annotation file: $ANNOTATION_PATH"
echo " Image root:      $IMAGE_ROOT"
echo " Output:          $OUTPUT_PATH"
echo "============================================"
echo ""

# The vLLM server must already be running (see launch_vllm.sh).
# We wait for it to be ready before starting.
echo "Waiting for vLLM server on port 8000 ..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server is ready."
echo ""

mkdir -p "$OUTPUT_PATH"

nohup python "$SCRIPT_DIR/qwenannotator.py" \
    --annotation_path "$ANNOTATION_PATH" \
    --image_root "$IMAGE_ROOT" \
    --output_path "$OUTPUT_PATH" \
    --n_samples 50 \
    --batch_size 50 \
    --port 8000 \
    "$@" \
    > "$OUTPUT_PATH/annotation_run.log" 2>&1 &

ANNOTATOR_PID=$!
echo "Annotator started (PID $ANNOTATOR_PID)"
echo "Log: $OUTPUT_PATH/annotation_run.log"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_PATH/annotation_run.log"
echo ""
echo "To stop:  kill $ANNOTATOR_PID"
