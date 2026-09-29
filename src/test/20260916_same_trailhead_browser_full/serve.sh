#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
cd "$DIR"
exec uvicorn server:app --host 127.0.0.1 --port 8901
