#!/usr/bin/env bash
# Run 50×50 annotation test with Qwen3.6-35B-A3B-FP8 on the cluster (A6000 48GB).
# Tests batch_size=50 and batch_size=100 to find the faster setting.
#
# Usage (after sshing into the cluster):
#   bash /path/to/run_cluster_test_35b.sh
#
# Results saved to: /local/wding/res/20260421_qwen35b_annotation_test/

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
ANNOTATION_PATH="/var/scratch/wding/Dataset/redcaps_plus/redcaps_test.json"
IMAGE_ROOT="/var/scratch/wding/Dataset"
OUTPUT_BASE="/local/wding/res/20260421_qwen35b_annotation_test"
MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000
N_IMAGES=50

mkdir -p "$OUTPUT_BASE"

# ── Conda ────────────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoSiR

# ── Start vLLM server ────────────────────────────────────────────────────────
echo "============================================"
echo " Starting vLLM server: $MODEL"
echo " GPU: single A6000 (48GB) | max-model-len: 8192"
echo "============================================"

nohup vllm serve "$MODEL" \
    --port $PORT \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    > "$OUTPUT_BASE/vllm_server.log" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"
echo ""

# ── Wait for server ready ─────────────────────────────────────────────────────
echo "Waiting for server to be ready (model load ~3-5 min) ..."
WAIT=0
until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 10
    WAIT=$((WAIT + 10))
    echo "  ${WAIT}s elapsed..."
    if [ $WAIT -ge 600 ]; then
        echo "ERROR: Server did not start within 10 minutes. Check $OUTPUT_BASE/vllm_server.log"
        kill -9 $VLLM_PID 2>/dev/null
        pkill -9 -f "VLLM::EngineCore" 2>/dev/null
        exit 1
    fi
done
echo "Server ready after ${WAIT}s."
echo ""

# ── Confirm loaded model ──────────────────────────────────────────────────────
LOADED=$(curl -s http://localhost:$PORT/v1/models | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
echo "Loaded model: $LOADED"
echo ""

# ── Helper: run one test ──────────────────────────────────────────────────────
run_test() {
    local BSIZE=$1
    local OUT_DIR="$OUTPUT_BASE/bs${BSIZE}"
    mkdir -p "$OUT_DIR"

    echo "--------------------------------------------"
    echo " Test: batch_size=$BSIZE  →  $OUT_DIR"
    echo "--------------------------------------------"

    python "$SCRIPT_DIR/run_local_test.py" \
        --annotation_path "$ANNOTATION_PATH" \
        --image_root "$IMAGE_ROOT" \
        --batch_size "$BSIZE" \
        --port $PORT \
        --n_images $N_IMAGES \
        --model_name "$MODEL" \
        2>&1 | tee "$OUT_DIR/test_run.log"

    # Copy the saved matrix out of the cwd test_output/ to the labelled folder
    if [ -f "test_output/test_matrix.npz" ]; then
        cp test_output/test_matrix.npz "$OUT_DIR/test_matrix_${N_IMAGES}x${BSIZE}.npz"
        rm -rf test_output
    fi

    echo "  Saved → $OUT_DIR"
    echo ""
}

# ── Run tests ─────────────────────────────────────────────────────────────────
cd "$OUTPUT_BASE"

run_test 50
run_test 100

# ── Comparison summary ────────────────────────────────────────────────────────
echo "============================================"
echo " Comparison: batch_size 50 vs 100"
echo "============================================"
python3 - <<'PYEOF'
import os, re, numpy as np

base = os.environ.get("OUTPUT_BASE", "/local/wding/res/20260421_qwen35b_annotation_test")

for bs in [50, 100]:
    log = os.path.join(base, f"bs{bs}", "test_run.log")
    npz = os.path.join(base, f"bs{bs}", f"test_matrix_50x{bs}.npz")
    if not os.path.exists(log):
        print(f"bs={bs}: log not found"); continue

    with open(log) as f:
        text = f.read()

    avg_match = re.search(r"Avg time per call: ([\d.]+)s", text)
    eta_match  = re.search(r"Estimated full 5k×5k runtime: ([\S]+)", text)
    avg = float(avg_match.group(1)) if avg_match else float("nan")
    eta = eta_match.group(1) if eta_match else "?"

    gt_correct = 0
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        m = d["matrix"]
        diag = m.diagonal()
        gt_correct = int((diag == 1).sum())

    per_pair = avg / bs
    print(f"  batch_size={bs:3d} | avg/call={avg:.1f}s | per_pair={per_pair:.3f}s | "
          f"GT recall={gt_correct}/50 ({100*gt_correct/50:.0f}%) | ETA 5k×5k: {eta}")

PYEOF

export OUTPUT_BASE

# ── Shutdown server ────────────────────────────────────────────────────────────
echo ""
echo "Shutting down vLLM server ..."
# Kill the entire process group — vLLM spawns a separate EngineCore child
# that survives a simple `kill $PID` and keeps the GPU occupied.
kill -9 $VLLM_PID 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null
sleep 5
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
echo "GPU memory after shutdown: ${MEM} MiB (expect ~1)"
echo ""
echo "All results in: $OUTPUT_BASE"
