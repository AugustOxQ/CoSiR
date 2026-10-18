#!/usr/bin/env bash
# CPU follow-up queue. Workers append exit codes as soon as they finish.
set -u
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
python_bin=/root/miniconda3/envs/CoSiR/bin/python
status=queue_fu_status.txt
codes=''
failed=0
record() {
    printf '%s=%s\n' "$1" "$2" >> "$status"
}
printf 'AGS_FU_START %s\n' "$(TZ=Europe/Amsterdam date '+%F %H:%M')" >> "$status"
pids=()
names=()
for aspect in emotion style genre; do
    names+=("train_${aspect}")
    (
        "$python_bin" -u ags_fu.py --train --aspects "$aspect" > "queue_fu_train_${aspect}.log" 2>&1
        code=$?
        record "train_${aspect}" "$code"
        exit "$code"
    ) &
    pids+=("$!")
done
for index in "${!pids[@]}"; do
    wait "${pids[$index]}"
    code=$?
    codes="$codes ${names[$index]}=$code"
    if (( code != 0 )); then failed=1; fi
done
"$python_bin" -u ags_fu.py --score > queue_fu_score.log 2>&1
code=$?
record score "$code"
codes="$codes score=$code"
if (( code != 0 )); then failed=1; fi
printf 'AGS_FU_DONE%s %s\n' "$codes" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" >> "$status"
exit "$failed"
