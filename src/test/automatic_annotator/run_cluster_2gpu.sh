#!/usr/bin/env bash
# Self-contained annotation on a cluster node with TWO A6000 (48GB each).
#
# Serves Qwen3.6-27B — a DENSE vision-language model (not MoE) — split across
# both GPUs with tensor-parallel-size 2 (~27GB/GPU at bf16, lots of KV headroom),
# then launches the annotator. Dense + bf16 avoids the MoE-VRAM and FP8-on-Ampere
# issues that stalled the single-GPU 35B-FP8 run.
#
# Resume-safe (re-run to continue from the last checkpoint). Runs under nohup, so
# it survives closing the SSH session.
#
# Usage (after sshing into a 2-GPU node):
#   bash run_cluster_2gpu.sh                                          # 5000 samples, batch 50
#   bash run_cluster_2gpu.sh --n_samples 1000 --batch_size 100
#   bash run_cluster_2gpu.sh --output_path /local/wding/annotations/redcaps_27b_1k --n_samples 1000
#
# Any extra args (--n_samples / --batch_size / --output_path / ...) pass through
# to qwenannotator.py; the LAST value wins over the defaults below.

set -e

# ── Config ─────────────────────────────────────────────────────────────────────
ANNOTATION_PATH="/var/scratch/wding/Dataset/redcaps_plus/redcaps_test.json"
IMAGE_ROOT="/var/scratch/wding/Dataset/redcaps_test"
OUTPUT_PATH="/local/wding/res/20260623_qwen27b_2gpu"   # override with --output_path
MODEL="Qwen/Qwen3.6-27B"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000

# ── HuggingFace cache ───────────────────────────────────────────────────────────
# The default $HOME often has no space on clusters. Point HF_HOME at a filesystem
# that has space AND persists across jobs, so the ~54GB model isn't re-downloaded
# every run. Defaults next to the dataset (persistent); override by exporting
# HF_HOME first. Avoid node-local /local and the per-run output dir (may be wiped).
export HF_HOME="${HF_HOME:-/var/scratch/wding/hf_cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"

# ── Pull --output_path out of the passthrough args so the vLLM log, annotator ──
# log, and output dir all agree. Everything else passes through to the annotator.
PASS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --output_path)    OUTPUT_PATH="$2"; shift 2 ;;
        --output_path=*)  OUTPUT_PATH="${1#*=}"; shift ;;
        *)                PASS_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$OUTPUT_PATH"

# ── Env ─────────────────────────────────────────────────────────────────────────
# Activate your environment BEFORE running this script (cluster uses anaconda):
#   conda activate <your-env>      # e.g. annot
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "WARNING: no conda env active — activate one first." >&2
fi

# ── GPUs ───────────────────────────────────────────────────────────────────────
# Pin to two GPUs. Override before calling, e.g. CUDA_VISIBLE_DEVICES=2,3 bash ...
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
echo "Using GPUs: $CUDA_VISIBLE_DEVICES  (tensor-parallel-size 2)"
echo ""

# ── Start vLLM (TP=2), reuse if already running ─────────────────────────────────
if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "vLLM server already running on port $PORT — reusing it."
else
    echo "Starting vLLM: $MODEL on 2× A6000 (TP=2) ..."
    nohup vllm serve "$MODEL" \
        --port $PORT \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        > "$OUTPUT_PATH/vllm_server.log" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" > "$OUTPUT_PATH/vllm.pid"
    echo "vLLM PID: $VLLM_PID  (log: $OUTPUT_PATH/vllm_server.log)"

    echo "Waiting for model to load (27B bf16 across 2 GPUs, ~3-6 min) ..."
    WAIT=0
    until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 10; WAIT=$((WAIT + 10)); echo "  ${WAIT}s ..."
        # Bail early if the server process died instead of hanging.
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM exited during load. Last 40 lines of the log:"
            tail -40 "$OUTPUT_PATH/vllm_server.log"
            exit 1
        fi
        if [ $WAIT -ge 900 ]; then
            echo "ERROR: server not ready after 15 min. See $OUTPUT_PATH/vllm_server.log"
            exit 1
        fi
    done
    echo "Server ready after ${WAIT}s."
fi

# ── Confirm the served model id ─────────────────────────────────────────────────
LOADED=$(curl -s http://localhost:$PORT/v1/models \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
echo "Loaded model: ${LOADED:-<unknown>}"
echo ""

# ── Refuse to start a second annotator on the same output dir ───────────────────
# Multiple annotators racing on one checkpoint corrupt progress (and make deletes
# look like they "don't take" — a live process just rewrites the files).
if pgrep -u "$USER" -f "qwenannotator.py.*$OUTPUT_PATH" > /dev/null 2>&1; then
    echo "ERROR: an annotator is already running for $OUTPUT_PATH."
    echo "       Stop it first:  pkill -u $USER -f qwenannotator.py"
    exit 1
fi

# ── Launch annotator (unbuffered -> live logs, detached) ────────────────────────
LOG="$OUTPUT_PATH/annotation_run.log"
echo "============================================"
echo " 2-GPU annotation"
echo " Model:   $MODEL"
echo " Output:  $OUTPUT_PATH"
echo " Log:     $LOG"
echo " Extra:   ${PASS_ARGS[*]:-(none)}"
echo "============================================"
echo ""

nohup python -u "$SCRIPT_DIR/qwenannotator.py" \
    --annotation_path "$ANNOTATION_PATH" \
    --image_root      "$IMAGE_ROOT" \
    --output_path     "$OUTPUT_PATH" \
    --n_samples       5000 \
    --batch_size      50 \
    --port            $PORT \
    --model_name      "$MODEL" \
    "${PASS_ARGS[@]}" \
    >> "$LOG" 2>&1 &

ANNOTATOR_PID=$!
echo "$ANNOTATOR_PID" > "$OUTPUT_PATH/annotator.pid"
echo "Annotator started (PID $ANNOTATOR_PID). Safe to close the SSH session."
echo ""
echo "Monitor:  tail -f $LOG"
echo "Stop annotator: kill \$(cat $OUTPUT_PATH/annotator.pid)"
echo "Stop vLLM+GPU:  kill -9 \$(cat $OUTPUT_PATH/vllm.pid); pkill -9 -f 'VLLM::EngineCore'"
