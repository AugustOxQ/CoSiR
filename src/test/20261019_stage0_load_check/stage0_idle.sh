#!/bin/bash
# Stage 0's two September scripts (commit 1b70d7f, unchanged) on an idle machine, same env as pc_stage0.py.
# Writes only under $S (session scratchpad); prints one line per script with the held-out AMIs.
S=/tmp/claude-0/-project-CoSiR/482cfe26-f369-448d-99da-12204934cc98/scratchpad/stage0_idle
P=src/test/20260922_percept_topic_pipeline
mkdir -p $S/tree $S/out $S/tmp
git -C /project/CoSiR archive 1b70d7f | tar -x -C $S/tree
cd $S/tree
for script in run_percept_stage1_embedding_snapshot_pilot run_percept_stage2_extended_seed_pilot; do
  echo "$script start $(TZ=Europe/Amsterdam date '+%H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
  env PYTHONPATH=$S/tree/$P:$S/tree PERCEPT_OUTPUT_ROOT=$S/out PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/tmp \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    flock -o -w 3600 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python -u $S/tree/$P/$script.py > $S/out/$script.log 2>&1
  echo "$script exit=$? end $(TZ=Europe/Amsterdam date '+%H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
  grep -E 'held-out (emotion|genre) AMI' $S/out/${script#run_}_report.md 2>/dev/null
done
