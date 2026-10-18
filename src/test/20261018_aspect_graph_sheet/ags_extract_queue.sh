#!/usr/bin/env bash
# Controller only: style_word and genre_word training images, one at a time under the GPU lock.
cd /project/CoSiR || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
D=/project/CoSiR/src/test/20261018_aspect_graph_sheet
PY=/root/miniconda3/envs/CoSiR/bin/python
codes=""
for key in style_word genre_word; do
    flock -o -w 7200 -E 75 /tmp/gpu0.lock "$PY" -u "$D/ags_trainimg.py" --key "$key" > "$D/extract_$key.log" 2>&1
    codes="$codes $key=$?"
done
echo "AGS_EXTRACT_DONE$codes $(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee "$D/extract_status.txt"
