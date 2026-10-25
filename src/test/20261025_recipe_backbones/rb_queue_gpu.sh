#!/usr/bin/env bash
# The whole GPU chain in order, stopping at the first failed queue: stage 0, stage 1 (+VP1), stage 1x beside, stores,
# stage 2, stage 3, 3a, 4, 5. The CPU queues (1x, reports of stage 3 and 4) start in the background as soon as their
# inputs exist and wait for each other on a CPU lock. Each stage queue can also be run alone (rb_queue.md).
queue=gpu
source "$(dirname -- "${BASH_SOURCE[0]}")/rb_queue_common.sh"
bg=()
run_queue() {  # run_queue SCRIPT [ARG]   foreground; a failure ends the chain
    bash "$here/$1" "${@:2}" || { printf 'RB_GPU_STOP %s exited %s\n' "$1" "$?" | tee -a "$status"; failed=1; finish; }
}
start_bg() {
    bash "$here/$1" "${@:2}" > "$outputs/logs/gpu_bg_${1%.sh}${2:-}.log" 2>&1 &
    bg+=($!); note "started in the background: $1 ${2:-}"
}
run_queue rb_queue_stage0.sh
run_queue rb_queue_stage1.sh
start_bg rb_queue_stage1x.sh
run_queue rb_queue_stores.sh
run_queue rb_queue_stage2.sh
run_queue rb_queue_stage3.sh
start_bg rb_queue_cpu.sh 3
run_queue rb_queue_stage3a.sh
run_queue rb_queue_stage4.sh
start_bg rb_queue_cpu.sh 4
run_queue rb_queue_stage5.sh
note "GPU chain done; waiting for the background CPU queues"
for pid in "${bg[@]}"; do wait "$pid" || failed=1; done
grep -h "_DONE" "$outputs"/queue_*_status.txt | tee -a "$status"
finish
