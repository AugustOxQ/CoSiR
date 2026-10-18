#!/bin/bash
# Stage 0 lead 1: September Stage 1 snapshot script with default threads (no OMP/MKL cap).
S=/tmp/claude-0/-project-CoSiR/5dd1d425-0f6b-4b8f-bb5d-e320cf751793/scratchpad/trace_threads
T=$S/tree; P=src/test/20260922_percept_topic_pipeline
unset OMP_NUM_THREADS MKL_NUM_THREADS
cd $T
env PYTHONPATH=$T/$P:$T PERCEPT_OUTPUT_ROOT=$S/out PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/tmp \
  flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python -u $T/$P/run_percept_stage1_embedding_snapshot_pilot.py > $S/out/run.log 2>&1
rc=$?
echo "exit $rc"; grep -E 'held-out (emotion|genre) AMI' $S/out/*_report.md
