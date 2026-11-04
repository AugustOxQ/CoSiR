#!/usr/bin/env bash
# r18 TURTLE fits on node402 (ticket 09b): the exact commands for the controller, in order. Prints only; runs nothing.
# Use only CLUSTER=~/.claude/skills/cluster-run/cluster, from the worktree (branch) that holds the commit, always --node node402.
# The DATA_MAP pair for tpips_pylibs is already set for the TPIPS jobs (same pylibs/full copy; torch 2.11).
OM_EST_FIT=${OM_EST_FIT:-30m}      # --est per job; 412 fits over 16 jobs: refine from turtle_timing.json and the probe's seconds
SH=scripts/run_om_turtle_node.sh
PY=/root/miniconda3/envs/CoSiR/bin/python
RUN=src/test/20261103_outside_methods/om_turtle_run.py
OUTD=/data/SSD2/cosir_r18/20261103_outside_methods/outputs
R=res/r18_turtle
cat <<PLAN
CLUSTER=/root/.claude/skills/cluster-run/cluster
-- A. pack (local, CPU; 128 inputs: 8 plain, 120 c*, about 7.6 GB, written to $OUTD/turtle_pack; a rerun keeps files whose sha256 matches)
1  res-queue cpu --no-wait --label r18-turtle-pack -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY $RUN pack
2  DATA_MAP: add this pair to cluster.conf (controller edits it):
     /local/wding/r18_in/turtle_pack:$OUTD/turtle_pack
   (tpips_pylibs pair stays); the node config adopted for TPIPS needs no new key (the script's +om_in.turtle_pack_path is added on the command line):
     \$CLUSTER configs --adopt configs/dataset/om_node_cluster.yaml
-- B. sync, check, probe (commit with "cluster run" in the subject first)
3  \$CLUSTER sync --node node402 ; \$CLUSTER status --node node402
4  \$CLUSTER check --node node402 --sync-data -- bash $SH probe
5  \$CLUSTER launch --node node402 --queue --short --label r18:turtle -- bash $SH probe     (background; PROBE_ENV_OK, PROBE_FIT_OK)
   \$CLUSTER watch <tag> ; \$CLUSTER pull --tag <tag>     -> ${R}_probe/parts/turtle/mmart/qwen/plain/k20__start0.{npz,json}
   compare its loss with the local fit of mmart qwen plain K 20 start 0 (and the AMI of its labels).
-- C. the 16 jobs (one GPU each, --est $OM_EST_FIT each; same check form, once)
6  \$CLUSTER check --node node402 --sync-data -- bash $SH fit --set mmart --enc qwen --arm plain
PLAN
for s in artelingo mmart; do for e in qwen siglip2 siglip1 clip_l14; do for a in plain cstar; do
  echo "   \$CLUSTER launch --node node402 --queue --est $OM_EST_FIT --label r18:turtle -- bash $SH fit --set $s --enc $e --arm $a"
done; done; done
cat <<PLAN
   (background each; a lost job is relaunched unchanged with a new tag: fits already good in the node folder are skipped)
7  \$CLUSTER watch <tag>   per job;   \$CLUSTER pull --tag <tag>   per job (all land in $R/)
-- D. local check (412 fits against the pack manifest: input sha256, n, labels, keys), then the copy line it prints
8  env PYTHONDONTWRITEBYTECODE=1 $PY $RUN check-pulled $R
   (exit 0 and "412 of 412 fits good" -> run the printed rsync line to put parts/turtle into $OUTD/parts/; the script copies nothing)
PLAN
