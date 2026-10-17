#!/usr/bin/env bash
# GPU extraction (10 prompt keys) under the shared lock, then CPU scoring; one summary line at the end.
cd /project/CoSiR || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
D=src/test/20261017_prompted_aspect_views
PY=/root/miniconda3/envs/CoSiR/bin/python
flock -o -w 3600 -E 75 /tmp/gpu0.lock $PY $D/pav_extract.py --device cuda > $D/gpu_full.log 2>&1
x=$?
if [ $x -ne 0 ]; then echo "PAV_DONE extract_exit=$x score=skipped $(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee $D/queue_status.txt; exit $x; fi
$PY $D/pav_score.py > $D/score_full.log 2>&1
s=$?
echo "PAV_DONE extract_exit=0 score_exit=$s $(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee $D/queue_status.txt
