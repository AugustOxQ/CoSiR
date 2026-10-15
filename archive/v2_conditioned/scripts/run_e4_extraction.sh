#!/usr/bin/env bash
# E4 full feature extraction, sequential. Wrap in the GPU lock and run in the background, e.g.
#   flock -n -o -E 75 /tmp/gpu0.lock bash scripts/run_e4_extraction.sh > src/test/20261103_feature_extraction/run_e4.log 2>&1
# Each step resumes from its progress.json after a crash; a finished output is refused without --overwrite.
set -u
cd /project/CoSiR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/root/miniconda3/envs/CoSiR/bin/python
X="$PY scripts/extract_features.py"
Q=qwen3vl_emb_2b
C=clip_b32
rc=0
run() { echo "=== $* $(date -Is)"; $X "$@" || { echo "FAILED: $*"; rc=1; }; }
run --dataset artelingo_full --backbone $Q
run --dataset semart --backbone $C
run --dataset semart --backbone $Q
run --dataset genecis_vg_crops --backbone $C
run --dataset genecis_vg_crops --backbone $Q
run --dataset genecis_coco --backbone $Q
run --dataset coco_train2014 --backbone $C
run --dataset coco_train2014 --backbone $Q
echo "=== all done $(date -Is) rc=$rc"
exit $rc
