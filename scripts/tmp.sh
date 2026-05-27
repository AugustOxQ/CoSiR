#! /bin/bash

# bash scripts/run_local.sh            || echo "run_local.sh failed, continuing..."
bash scripts/run_baseline_siglip.sh  || echo "run_baseline_siglip.sh failed, continuing..."
bash scripts/run_baseline_siglip2.sh || echo "run_baseline_siglip2.sh failed, continuing..."
