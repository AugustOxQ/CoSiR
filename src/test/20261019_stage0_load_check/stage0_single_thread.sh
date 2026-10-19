#!/bin/bash
# September script 1 (commit 1b70d7f, unchanged) twice with one CPU thread (OMP/MKL/OpenBLAS = 1): identical runs?
# Reuses the tree exported by stage0_idle.sh; writes only under $S (session scratchpad).
S=/tmp/claude-0/-project-CoSiR/482cfe26-f369-448d-99da-12204934cc98/scratchpad/stage0_idle
P=src/test/20260922_percept_topic_pipeline
script=run_percept_stage1_embedding_snapshot_pilot
cd $S/tree
for r in 1 2; do
  mkdir -p $S/single_$r
  echo "single $r start $(TZ=Europe/Amsterdam date '+%H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
  env PYTHONPATH=$S/tree/$P:$S/tree PERCEPT_OUTPUT_ROOT=$S/single_$r PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/tmp \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    flock -o -w 3600 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python -u $S/tree/$P/$script.py > $S/single_$r/$script.log 2>&1
  echo "single $r exit=$? end $(TZ=Europe/Amsterdam date '+%H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
  grep -E 'held-out (emotion|genre) AMI' $S/single_$r/${script#run_}_report.md 2>/dev/null
  grep -E 'K=60 DEC epoch 010' $S/single_$r/$script.log
done
