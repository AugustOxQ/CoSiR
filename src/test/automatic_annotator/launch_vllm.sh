#!/usr/bin/env bash
# Launch vLLM server for Qwen3.6-35B-A3B-FP8 on a single A6000 (48GB).
#
# FP8 weights: ~35GB — leaves ~13GB for KV cache at max-model-len 8192.
# Increase max-model-len if you see context-length errors during annotation.
#
# Usage:
#   bash launch_vllm.sh             # default port 8000
#   bash launch_vllm.sh 8001        # custom port

PORT=${1:-8000}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CoSiR

echo "Starting vLLM server on port $PORT ..."
echo "Model: Qwen/Qwen3.6-35B-A3B-FP8"
echo "GPU:   single A6000 (48GB)"
echo ""

vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --reasoning-parser qwen3 \
    --trust-remote-code
