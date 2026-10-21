#!/usr/bin/env bash
stage=2
source "$(dirname -- "${BASH_SOURCE[0]}")/ab_queue_common.sh"
analysis_args=()
score_args=()
if [[ ${AB_SMOKE:-0} == 1 ]]; then
    analysis_args=(--smoke)
    score_args=(--resamples 20 --aspects emotion)
fi
step views cpu ab_analysis.py --part views --workers 4 --outputs "$outputs" "${analysis_args[@]}"
has_rows=0
for metadata in "$outputs"/train/*/ab_row.json; do
    [[ -f $metadata ]] || continue
    row=${metadata%/ab_row.json}
    row=${row##*/}
    if [[ $row != R0 && $row != R2 ]]; then has_rows=1; break; fi
done
if (( has_rows )); then
    step score cpu ab_score.py --workers 8 --outputs "$outputs" "${score_args[@]}"
    step rows cpu ab_analysis.py --part rows --workers 4 --outputs "$outputs" "${analysis_args[@]}"
fi
finish
