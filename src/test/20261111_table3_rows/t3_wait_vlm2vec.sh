#!/bin/bash
# Usage: run with Bash run_in_background from the main session (it polls every 60 s, up to 5 h).
# Wait for the six VLM2Vec store meta files (written last, atomically), then score the VLM2Vec row of Table 3.
T=/data/SSD2/pre_extract/artelingo/vlm2vec_v2; V=/data/SSD2/pre_extract/artelingo/vlm2vec_v2_val
for i in $(seq 1 300); do
  n=0; for a in emotion style genre; do [ -f $T/$a/meta.json ] && n=$((n+1)); [ -f $V/$a/meta.json ] && n=$((n+1)); done
  [ $n -eq 6 ] && break
  sleep 60
done
[ $n -eq 6 ] || { echo "timeout: $n of 6 meta files"; exit 1; }
echo "stores complete at $(TZ=Europe/Amsterdam date '+%F %H:%M')"
cd /project/CoSiR/src/test/20261111_table3_rows
res-queue cpu --short --label paper:t3_vlm2vec -- env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python t3_run.py vlm2vec > logs/vlm2vec.log 2>&1
echo "exit $?"; tail -4 logs/vlm2vec.log
