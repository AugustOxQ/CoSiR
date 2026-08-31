#!/usr/bin/env bash
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
echo "Starting CoSiR Condition Visualizer on http://localhost:8765"
uvicorn app:app --port 8765 --host 0.0.0.0 --reload
