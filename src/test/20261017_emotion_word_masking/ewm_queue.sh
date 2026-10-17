#!/usr/bin/env bash
# Controller only. GPU extraction under the shared lock; CPU scoring runs even on failure.
cd /project/CoSiR || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
D=src/test/20261017_emotion_word_masking
PY=/root/miniconda3/envs/CoSiR/bin/python
CUDA_VISIBLE_DEVICES=0 flock -o -w 3600 -E 75 /tmp/gpu0.lock "$PY" "$D/ewm_extract.py" --device cuda > "$D/extract.log" 2>&1
x=$?
CUDA_VISIBLE_DEVICES='' "$PY" "$D/ewm_score.py" > "$D/score.log" 2>&1
s=$?
echo "EWM_DONE extract_exit=$x score_exit=$s $(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee "$D/queue_status.txt"
if [ "$x" -ne 0 ]; then exit "$x"; fi
exit "$s"
