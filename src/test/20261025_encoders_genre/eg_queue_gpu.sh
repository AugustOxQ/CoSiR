#!/usr/bin/env bash
# The whole run in order: stage iii (needs no new store, spec §3.5), stores, stage i, stage ii. A queue that stops does not
# end the chain unless the rule says so: stage iii and stage i/stores are independent; stage ii checks stage i itself
# (X0/X1 stopped: it runs with K2 to K4). The CPU report queues of stage i (3, 4, 5) start in the background once stage i
# is done, one at a time (CPU lock); the held table of stage ii is built after them. Each queue can also be run alone
# (eg_queue.md). The exit code is 1 when any queue ended with a stop.
queue=gpu
source "$(dirname -- "${BASH_SOURCE[0]}")/eg_queue_common.sh"
bg=()
run_queue() {  # run_queue SCRIPT [ARG]   foreground; a stop is recorded and the chain goes on
    bash "$here/$1" "${@:2}" || { printf 'EG_GPU_NOTE %s ended with a stop (exit %s); status file of that queue has the reason\n' "$1" "$?" | tee -a "$status"; failed=1; }
}
run_queue eg_queue_stage_iii.sh
run_queue eg_queue_stores.sh
run_queue eg_queue_stage_i.sh
# the report queues run one after the other, in the background, beside stage ii
(
    for n in 3 4 5; do bash "$here/eg_queue_cpu_i.sh" "$n" > "$outputs/logs/gpu_bg_cpu_i$n.log" 2>&1; done
) &
bg+=($!); note "started in the background: eg_queue_cpu_i.sh 3, 4, 5 (one after the other)"
run_queue eg_queue_stage_ii.sh
note "GPU chain done; waiting for the background CPU queues"
for pid in "${bg[@]}"; do wait "$pid" || failed=1; done
run_queue eg_queue_stage_ii.sh held
grep -h "_DONE" "$outputs"/queue_*_status.txt | tee -a "$status"
finish
