#!/bin/bash
# Waits until 19:00 Amsterdam (weekday working-hours rule: heavy, about 65 min), then extracts the Qwen store of
# affection_local under the GPU lock (waits up to 2 h for the lock). One status line at the end.
D=/project/CoSiR-wt-r11-impressions/src/test/20261029_r11_data
now=$(date +%s); start=$(TZ=Europe/Amsterdam date -d 'today 19:00' +%s)
[ "$now" -lt "$start" ] && sleep $((start - now))
echo "start $(TZ=Europe/Amsterdam date '+%F %H:%M')" > $D/queue_status.txt
cd /project/CoSiR-wt-r11-impressions
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 flock -o -w 7200 -E 75 /tmp/gpu0.lock \
  /root/miniconda3/envs/CoSiR/bin/python scripts/extract_features.py --dataset affection_local --backbone qwen3vl_emb_2b \
  > $D/affection_qwen_extract.log 2>&1
echo "exit $? at $(TZ=Europe/Amsterdam date '+%F %H:%M')" >> $D/queue_status.txt
