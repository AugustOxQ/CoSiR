#!/usr/bin/env bash
# Controller entry point. Deliberately continue through failures, including to scoring.
set -u
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
python_bin=/root/miniconda3/envs/CoSiR/bin/python
"$python_bin" -u pbp_graphs.py > queue_graphs.log 2>&1
graphs_code=$?
printf 'graphs=%s\n' "$graphs_code" | tee queue_status.txt
codes="graphs=$graphs_code"
failed=$graphs_code
for variant in S_E S_P S_D S_PN; do
    "$python_bin" -u pbp_train.py --variants "$variant" --seeds 0 1 2 > "queue_train_${variant}.log" 2>&1
    code=$?
    printf 'train_%s=%s\n' "$variant" "$code" | tee -a queue_status.txt
    codes="$codes train_${variant}=$code"
    if (( code != 0 )); then failed=1; fi
done
"$python_bin" -u pbp_score.py > queue_score.log 2>&1
score_code=$?
printf 'score=%s\n' "$score_code" | tee -a queue_status.txt
codes="$codes score=$score_code"
if (( score_code != 0 )); then failed=1; fi
printf 'PBP_DONE %s %s\n' "$codes" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee -a queue_status.txt
if (( failed != 0 )); then exit 1; fi
