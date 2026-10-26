#!/usr/bin/env bash
# The whole round in order: stage A, stage B, stage C (rg_queue_{a,b,c}.sh). A failed gate stops only the stages behind it
# (spec §3.5): each stage checks the one before it (B: stage A ended clean; C: stage B ended clean, then B-PICK), so a stopped
# queue is noted and the chain goes on to queues that refuse themselves with the reason. Each queue can also be run alone
# (rg_queue.md). Exit 1 when any queue ended with a stop; B-NONE (no stage C) is not a stop.
queue=gpu
source "$(dirname -- "${BASH_SOURCE[0]}")/rg_queue_common.sh"
run_queue() {  # run_queue SCRIPT   foreground; a stop is recorded and the chain goes on
    bash "$here/$1" || { printf 'RG_GPU_NOTE %s ended with a stop (exit %s); its status file has the reason\n' "$1" "$?" | tee -a "$status"; failed=1; }
}
run_queue rg_queue_a.sh
run_queue rg_queue_b.sh
run_queue rg_queue_c.sh
if [[ "$dry" != 1 ]]; then grep -h "_DONE" "$outputs"/queue_{a,b,c}_status.txt 2>/dev/null | tee -a "$status"; fi
finish
