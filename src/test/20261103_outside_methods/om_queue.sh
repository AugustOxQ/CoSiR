#!/bin/bash
# r18 run plan (ticket 11): one `res-queue` call per step, in order, ending with one summary line. Modelled on ns_run_cpu.sh.
# Usage: bash om_queue.sh [PHASE|all] [--run]
#   PHASE: stage0 stage1 stage2 labels diag all     (default all)
#   Without --run the calls are only printed (a dry run starts nothing). With --run each line is executed in turn
#   (res-queue waits for its turn); run one phase per sitting.
# GPU steps are their own commands: gate (ii) CUDA, TPIPS local extraction (om_tpips_local.sh), text set B (frozen; line kept
# for the record), TURTLE (node402 lines from om_turtle_node_plan.sh, which prints and starts nothing).
# --est values are guesses marked '~' unless a ticket measured them; correct them from the first runs.
# Order that matters: stage0 -> stage1 -> THE CONTROLLER COMMITS records/stage1_om_ranges.json, stage1_om_picks.json and
# status.json -> stage2 -> assemble -> C13 writes OUT/c13/c13_cells.json -> labels -> diag.
set -u
PHASE=${1:-all}
RUNIT=no
for a in "$@"; do [ "$a" = "--run" ] && RUNIT=yes; done
[ "$PHASE" = "--run" ] && PHASE=all
cd "$(dirname "$0")/../../.."
D=src/test/20261103_outside_methods
PY=/root/miniconda3/envs/CoSiR/bin/python
ENVV="env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
GPUENV="env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 LD_LIBRARY_PATH=/root/miniconda3/envs/CoSiR/lib/python3.11/site-packages/nvidia/cu13/lib"
OUTD=/data/SSD2/cosir_r18/20261103_outside_methods/outputs
RUN="$ENVV $PY $D/om_run.py"
N=0

q() {  # q cpu|gpu LABEL EST [--short] -- command...   (one res-queue call = one step)
  local kind=$1 label=$2 est=$3
  shift 3
  local short=""
  if [ "$1" = "--short" ]; then short="--short"; shift; fi
  shift   # the --
  local line="res-queue $kind $short --est $est --label r18:$label -- $*"
  N=$((N+1))
  if [ "$RUNIT" = yes ]; then eval "$line" || echo "FAILED r18:$label (exit $?)"; else echo "$line"; fi
}
note() { echo "# $*"; }
want() { [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }

if want stage0; then
  note "stage 0: gates (i) and (ii) CPU run here; (ii) CUDA, (iii), (iv) are separate steps; status reads all four files"
  q cpu stage0        5m  --short -- $RUN stage0
  q gpu gate_ii_cuda  5m  --short -- $GPUENV $PY $D/om_turtle.py gate-ii --device cuda
  q cpu gate_iii      1h        -- $ENVV $PY $D/om_gate_iii.py           # parts a, b, c; ~45 min (ticket 07); a code edit afterwards means a rerun
  note "TPIPS stores (GPU, 6 shards of ~2.5 h, one res-queue job each; the script queues them; not on weekdays 06:00 to 19:00):"
  note "  bash $D/om_tpips_local.sh 0 5          (run detached)"
  note "  after all 6, in the tpips venv: om_tpips.py merge --set artelingo --factor all --of 6 ; om_tpips.py merge --set mmart --factor all --of 6"
  note "  then gate (iv): om_tpips.py gate-iv --path local=$OUTD/stage0/probe_b1_rerun"
  if [ "$RUNIT" = yes ]; then
    echo "TPIPS: not started by this script; start om_tpips_local.sh by hand"
  else
    echo "bash $D/om_tpips_local.sh 0 5"
    echo "res-queue cpu --est 20m --label r18:tpips_merge -- env PYTHONDONTWRITEBYTECODE=1 /root/venvs/tpips/bin/python $D/om_tpips.py merge --set artelingo --factor all --of 6   # ~ guess"
    echo "res-queue cpu --est 20m --label r18:tpips_merge_mmart -- env PYTHONDONTWRITEBYTECODE=1 /root/venvs/tpips/bin/python $D/om_tpips.py merge --set mmart --factor all --of 6   # ~ guess"
    echo "res-queue cpu --est 10m --label r18:gate_iv -- env PYTHONDONTWRITEBYTECODE=1 /root/venvs/tpips/bin/python $D/om_tpips.py gate-iv --path local=$OUTD/stage0/probe_b1_rerun   # ~ guess"
  fi
  note "text set B is frozen (generated); the GPU line, only if it must be redone (it refuses to overwrite):"
  note "  res-queue gpu --est 20m --label r18:textset_b -- $GPUENV $PY $D/om_textset_b.py all"
  note "TURTLE (GPU, node402; prints its commands, starts nothing; pass OM_EST_FIT=1h):"
  OM_EST_FIT=${OM_EST_FIT:-1h} bash $D/om_turtle_node_plan.sh | sed 's/^/#   /'
  q cpu status_after_gates 2m --short -- $RUN status        # rebuild records/status.json once the gate files exist
fi

if want stage1; then
  note "stage 1 (validation only); TPIPS picks need the merged stores"
  q cpu s1_ranges      30m -- $RUN stage1 --step ranges                     # ~ guess
  for s in artelingo mmart; do
    q cpu s1_clay_$s   40m -- $RUN stage1 --step clay --set $s              # ~ guess
    q cpu s1_tpips_$s  60m -- $RUN stage1 --step tpips --set $s             # ~ guess; 4,096-wide read-out table
  done
  q cpu s1_records     2m --short -- $RUN stage1 --step records
  note "THE CONTROLLER NOW COMMITS records/stage1_om_ranges.json, stage1_om_picks.json and status.json (stage 2 refuses otherwise)"
fi

if want stage2; then
  note "stage 2: 28 units (SC, set, encoder), each its own process (k-means opens a pool)"
  for s in artelingo mmart; do
    for sc in SC1 SC2; do q cpu cells_${sc}_${s}_qwen 40m -- $RUN stage2 --sc $sc --set $s --enc qwen; done   # ~ guess
    for e in qwen clip_l14; do for sc in SC3 SC4; do q cpu cells_${sc}_${s}_$e 40m -- $RUN stage2 --sc $sc --set $s --enc $e; done; done
    for e in qwen siglip2 siglip1 clip_l14; do for sc in SC5 SC6; do q cpu cells_${sc}_${s}_$e 40m -- $RUN stage2 --sc $sc --set $s --enc $e; done; done
  done
  q cpu assemble 5m --short -- $RUN assemble
fi

if want labels; then
  note "C13 (the run chat, Codex) writes $OUTD/c13/c13_cells.json first; labels refuses without it"
  q cpu labels 5m --short -- $RUN labels
fi

if want diag; then
  note "stage 4 diagnostics (after the labels); each item its own process"
  for it in sweep adds fits clay tpips r0; do q cpu diag_$it 40m -- $ENVV $PY $D/om_diag.py $it; done    # ~ guess
fi

echo "om_queue: $N res-queue steps, phase $PHASE, $([ "$RUNIT" = yes ] && echo queued || echo printed only); the TPIPS shards (om_tpips_local.sh) and the TURTLE node jobs are separate"
