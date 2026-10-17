#!/usr/bin/env bash
# Controller only. Keep stage 1 results even if extraction or stage 2 fails.
cd /project/CoSiR || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
D=/project/CoSiR/src/test/20261017_image_transfer_pilot
PY=/root/miniconda3/envs/CoSiR/bin/python
CUDA_VISIBLE_DEVICES=0 flock -o -w 3600 -E 75 /tmp/gpu0.lock "$PY" "$D/itp_train.py" --inputs default --device cuda > "$D/train_stage1.log" 2>&1
t1=$?
CUDA_VISIBLE_DEVICES='' "$PY" "$D/itp_score.py" > "$D/score_stage1.log" 2>&1
s1=$?
CUDA_VISIBLE_DEVICES=0 flock -o -w 3600 -E 75 /tmp/gpu0.lock "$PY" "$D/itp_extract.py" --device cuda > "$D/extract.log" 2>&1
x=$?
CUDA_VISIBLE_DEVICES=0 flock -o -w 3600 -E 75 /tmp/gpu0.lock "$PY" "$D/itp_train.py" --inputs emotion_focus --device cuda > "$D/train_stage2.log" 2>&1
t2=$?
CUDA_VISIBLE_DEVICES='' "$PY" "$D/itp_score.py" > "$D/score_stage2.log" 2>&1
s2=$?
echo "ITP_DONE train_stage1=$t1 score_stage1=$s1 extract=$x train_stage2=$t2 score_stage2=$s2 $(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee "$D/queue_status.txt"
for code in "$t1" "$s1" "$x" "$t2" "$s2"; do
    if [ "$code" -ne 0 ]; then exit "$code"; fi
done
