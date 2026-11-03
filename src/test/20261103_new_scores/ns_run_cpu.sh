#!/bin/bash
# r17 run plan: one `res-queue cpu` job per step (spec §9), never the whole script in one job.
# Usage: bash ns_run_cpu.sh STAGE [--dry]
#   STAGE: stage0 stage1-ranges stage1-picks stage2-clu stage2-ret results c13 verdicts stage3
# Each line queues one job (res-queue cpu waits for its turn); run one stage per sitting. --dry prints the lines.
# --est values are measured where a ticket gave a cost, else guesses marked '~' (correct them from the first runs).
# The queue sets 8 threads per slot; k-means threads are set in code (D20).
set -u
STAGE=${1:?stage}
DRY=${2:-}
PY=/root/miniconda3/envs/CoSiR/bin/python
RUN="env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY /project/CoSiR/src/test/20261103_new_scores/ns_run.py"
ENCS="qwen siglip2 siglip1 clip_l14"

q() {  # q LABEL EST [--short] -- command...
  local label=$1 est=$2
  shift 2
  local short=""
  if [ "$1" = "--short" ]; then short="--short"; shift; fi
  shift   # the --
  local line="res-queue cpu $short --label r17:$label --est $est -- $*"
  if [ "$DRY" = "--dry" ]; then echo "$line"; else eval "$line" || echo "FAILED to queue r17:$label (exit $?)"; fi
}

case $STAGE in
stage0)   # gates, about 1.3 h in all; each step also rebuilds records/status.json
  q i_a_r14         3m  --short -- $RUN gate i_a_r14
  q i_a_partitions  2m  --short -- $RUN gate i_a_partitions
  q i_a_r15         6m  --short -- $RUN gate i_a_r15
  q i_b             3m  --short -- $RUN gate i_b
  q i_c_affection   1m  --short -- $RUN gate i_c_affection
  q i_c_emoset      2m  --short -- $RUN gate i_c_emoset
  q i_d_emoset      5m  --short -- $RUN gate i_d_emoset
  q i_d_class_count 6m  --short -- $RUN gate i_d_class_count
  q ii_scorer       11m --short -- $RUN gate ii_scorer
  q ii_crl          1m  --short -- $RUN gate ii_crl
  q ii_percept      1m  --short -- $RUN gate ii_percept
  q iii             2m  --short -- $RUN gate iii
  q iv_export       10m --short -- $RUN gate iv_export    # ~ guess: mapper forward passes on CPU
  q iv              2m  --short -- $RUN gate iv
  q v               2m  --short -- $RUN gate v
  q vi              18m         -- $RUN gate vi            # r11 stage 4 recomputed, about 17 min
  ;;
stage1-ranges)   # measured (07, loaded machine) 85 to 330 s per painting (set, encoder); photo ~ 5 min; about 1.2 h
  for s in artelingo mmart; do for e in $ENCS; do q ranges_${s}_$e 6m --short -- $RUN ranges --set $s --enc $e; done; done
  for e in $ENCS; do q ranges_affection_$e 5m --short -- $RUN ranges --set affection --enc $e; done
  q ranges_emoset_qwen 5m --short -- $RUN ranges --set emoset --enc qwen
  ;;
stage1-picks)    # measured (09): untold 20 to 30 min x 8, matched 3 to 4 min x 7, CRL 50 s; about 4 h
  for s in artelingo mmart; do for e in $ENCS; do q pick_untold_${s}_$e 30m -- $RUN pick --key untold/$s/$e; done; done
  for s in artelingo mmart; do for e in $ENCS; do
    if [ "$s" = artelingo ] && [ "$e" = qwen ]; then continue; fi
    q pick_matched_${s}_$e 5m --short -- $RUN pick --key matched/$s/$e
  done; done
  q pick_crl_mmart 2m --short -- $RUN pick --key crl/mmart/clip_l14
  q stage1_records 1m --short -- $RUN stage1-records
  # THEN THE CONTROLLER COMMITS records/stage1_ranges.json, records/stage1_picks.json and records/status.json
  # (rebuilt: ns_run.py status) before any stage 2 step.
  ;;
stage2-clu)      # ~ 165 s per paired_cell call (4 metrics, 2,000 draws) plus k-means; painting 35 min (clip_l14 50), photo 12; about 5 h
  for s in artelingo mmart; do for e in $ENCS; do
    est=35m; [ "$e" = clip_l14 ] && est=50m
    q cells_clu_${s}_$e $est -- $RUN cells --task clu --set $s --enc $e
  done; done
  for e in $ENCS; do q cells_clu_affection_$e 12m -- $RUN cells --task clu --set affection --enc $e; done
  q cells_clu_emoset_qwen 12m -- $RUN cells --task clu --set emoset --enc qwen
  ;;
stage2-ret)      # ~ 12 min per (set, encoder) at 10,000 draws (guess from gate (ii) 576 s); about 1.6 h
  for s in artelingo mmart; do for e in $ENCS; do q cells_ret_${s}_$e 12m -- $RUN cells --task ret --set $s --enc $e; done; done
  ;;
results)         # needs stage 2 done and records/status.json committed unchanged
  q outcomes 2m --short -- $RUN outcomes
  ;;
c13)             # C13 is separate code (run chat); this placeholder only checks records/c13.json
  q c13_check 1m --short -- $RUN c13-check
  ;;
verdicts)        # copies outcomes to records/verdicts.json only if c13.json has pass true and names the outcomes sha256
  q verdicts 1m --short -- $RUN verdicts
  ;;
stage3)          # ns_diag.py (ticket 11); the count sweep is the long one (~ estimates, replace with 11's)
  q diag_count_sweep 4h -- $RUN diag count_sweep
  q diag_per_class 30m -- $RUN diag per_class
  q diag_qwen_crl 20m -- $RUN diag qwen_crl
  q diag_told_untold 40m -- $RUN diag told_untold
  q diag_tables 10m --short -- $RUN diag tables
  ;;
*) echo "unknown stage $STAGE"; exit 2;;
esac
