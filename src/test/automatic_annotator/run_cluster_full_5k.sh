#!/usr/bin/env bash
# Full 5k×5k annotation run with Qwen3.6-35B-A3B-FP8 on the cluster (A6000 48GB).
# Resume-safe: if killed, re-run the same command and it picks up from the last checkpoint.
#
# Usage (after sshing into the cluster):
#   bash /path/to/run_cluster_full_5k.sh [batch_size]
#
#   batch_size defaults to 50. Set based on the test results from run_cluster_test_35b.sh.
#   Example: bash run_cluster_full_5k.sh 100
#
# Results saved to: /local/wding/res/20260421_qwen35b_5k5k/

set -e

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATION_PATH="/var/scratch/wding/Dataset/redcaps_plus/redcaps_test.json"
IMAGE_ROOT="/var/scratch/wding/Dataset"
OUTPUT_PATH="/local/wding/res/20260421_qwen35b_5k5k"
MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000
BATCH_SIZE=${1:-50}   # override with: bash run_cluster_full_5k.sh 100

mkdir -p "$OUTPUT_PATH"

# ── Conda ─────────────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoSiR

# ── Start vLLM server (only if not already running) ───────────────────────────
if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "vLLM server already running on port $PORT — reusing it."
else
    echo "Starting vLLM server: $MODEL ..."
    nohup vllm serve "$MODEL" \
        --port $PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        > "$OUTPUT_PATH/vllm_server.log" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID (saved to $OUTPUT_PATH/vllm.pid)"
    echo $VLLM_PID > "$OUTPUT_PATH/vllm.pid"

    echo "Waiting for model to load (can take ~3-5 min) ..."
    WAIT=0
    until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 10; WAIT=$((WAIT + 10))
        echo "  ${WAIT}s ..."
        if [ $WAIT -ge 600 ]; then
            echo "ERROR: Server failed to start. See $OUTPUT_PATH/vllm_server.log"
            exit 1
        fi
    done
    echo "Server ready after ${WAIT}s."
fi
echo ""

# ── Launch annotator detached under nohup ─────────────────────────────────────
# Runs in background; safe to close the SSH session.
# Monitor with: tail -f $OUTPUT_PATH/annotation_run.log
LOG="$OUTPUT_PATH/annotation_run.log"

echo "============================================"
echo " Full 5k×5k annotation"
echo " Model:      $MODEL"
echo " Batch size: $BATCH_SIZE captions/call"
echo " Output:     $OUTPUT_PATH"
echo " Log:        $LOG"
echo "============================================"
echo ""

nohup python "$SCRIPT_DIR/qwenannotator.py" \
    --annotation_path "$ANNOTATION_PATH" \
    --image_root      "$IMAGE_ROOT" \
    --output_path     "$OUTPUT_PATH" \
    --n_samples       5000 \
    --batch_size      "$BATCH_SIZE" \
    --port            $PORT \
    --model_name      "$MODEL" \
    >> "$LOG" 2>&1 &

ANNOTATOR_PID=$!
echo $ANNOTATOR_PID > "$OUTPUT_PATH/annotator.pid"

echo "Annotator started (PID $ANNOTATOR_PID)"
echo ""
echo "You can now close this SSH session safely."
echo ""
echo "Monitor:  tail -f $LOG"
echo "Progress: grep 'Progress:' $LOG | tail -5"
echo "Stop:     kill \$(cat $OUTPUT_PATH/annotator.pid)"
