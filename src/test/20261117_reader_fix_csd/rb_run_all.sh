#!/usr/bin/env bash
# Real R-b chain on seed 42 (DECISION_RULE.md §4.2), launched by the main session. At most 3 processes at a time.
# Wave 1: heads (affect, image, caption, rand; csd already done) and banks (A1, A0, AR); wave 2: train; wave 3: eval
# A1 and A0, then AR; wave 4: summary. Stops at the first failing wave. Logs in results/rb_*.log.
set -u
cd /project/CoSiR || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
P=/root/miniconda3/envs/CoSiR/bin/python
D=src/test/20261117_reader_fix_csd
R=$D/results
stamp() { TZ=Europe/Amsterdam date '+%F %H:%M'; }

# run_wave <name> <log>=<script and args> ...   (at most 3 at a time)
run_wave() {
  local name=$1; shift
  echo "[$(stamp)] wave $name start"
  local pids=() logs=() fail=0
  for spec in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
    local log=${spec%%=*} args=${spec#*=}
    # shellcheck disable=SC2086
    $P $D/$args > "$R/rb_$log.log" 2>&1 &
    pids+=($!); logs+=("$log")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  done: ${logs[$i]}"; else echo "  FAILED: ${logs[$i]} (results/rb_${logs[$i]}.log)"; fail=1; fi
  done
  if [ $fail -ne 0 ]; then echo "[$(stamp)] wave $name FAILED"; exit 1; fi
  echo "[$(stamp)] wave $name done"
}

run_wave heads_banks "heads_affect=rb_build.py heads --grouping affect" "heads_image=rb_build.py heads --grouping image" \
  "heads_caption=rb_build.py heads --grouping caption" "heads_rand=rb_build.py heads --grouping rand" \
  "bank_A1=rb_build.py bank --config A1" "bank_A0=rb_build.py bank --config A0" "bank_AR=rb_build.py bank --config AR"
run_wave train "train_A1=rb_build.py train --config A1" "train_A0=rb_build.py train --config A0" \
  "train_AR=rb_build.py train --config AR"
run_wave eval "eval_A1=rb_eval.py --config A1" "eval_A0=rb_eval.py --config A0"
run_wave eval_AR "eval_AR=rb_eval.py --config AR"
run_wave summary "summary=rb_eval.py --summary"
echo "[$(stamp)] R-b chain complete"
