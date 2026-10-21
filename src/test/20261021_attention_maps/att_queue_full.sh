#!/usr/bin/env bash
# Full attention-map run: waits until 19:00 Amsterdam (weekday rule for heavy local GPU runs), then for the GPU lock.
set -u
cd /project/CoSiR
OUT=/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention
LOG=src/test/20261021_attention_maps/outputs/full_run.log
mkdir -p "$(dirname "$LOG")" "$OUT"
target=$(TZ=Europe/Amsterdam date -d 'today 19:00' +%s); now=$(date +%s)
[ "$target" -gt "$now" ] && sleep $((target - now))
echo "start $(TZ=Europe/Amsterdam date '+%F %H:%M')" >> "$LOG"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 flock -o -w 10800 -E 75 /tmp/gpu0.lock \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261021_attention_maps/att_extract.py --check-full 8 --out "$OUT" \
  2>&1 | grep -v -i "warning\|Loading weights" >> "$LOG"
rc=${PIPESTATUS[0]}
echo "end $(TZ=Europe/Amsterdam date '+%F %H:%M') exit $rc; files $(ls "$OUT"/p*.npz 2>/dev/null | wc -l); size $(du -sh "$OUT" | cut -f1)" | tee -a "$LOG"
