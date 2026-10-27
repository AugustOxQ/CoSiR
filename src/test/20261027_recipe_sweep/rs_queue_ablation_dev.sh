#!/usr/bin/env bash
# Development ablation (spec §3.2 AB, D22, rule §2.5, §3.5; GPU then CPU), by day: each of the eight removals of F at seeds 1 to
# 3 on the frozen sample, then `rs_ablation dev --final F`, then `rs_vp ablation --final F` (the point values, rule §4.5).
# F = `rs_queue_names.py final` (c* of stage3/verdict.json under PICK, else G; refused when no stage 3 file and no SWEEP-NONE or
# TOP-NONE explains it). F's runs are stage 3's (or 12-27's G seeds 1 to 3): nothing to train for F.
# Gates checked first: V1a, V3a (`rs_ablation` refuses without gates/v3a.json whatever stages 1 and 2 gave; checked here first) and,
# when a stage 3 file exists, its re-derivation V3c. A graph removal (content_correction, instructed_graphs) whose sample-scale
# graph check failed (gates/ablation_graphs.json) is skipped with a note and stops only itself; `rs_ablation dev` lists it as
# failed with the cause, and its output records `inputs.ablation_graphs_sha256`, the SHA-256 of that gate file. One arm's trainer
# call fails alone: the other arms go on; scoring starts only when every arm has a run.json. Weights of every arm are kept.
queue=ablation_dev
source "$(dirname -- "${BASH_SOURCE[0]}")/rs_queue_common.sh"
w=(--workers "$workers")
gate V1a-train "$outputs/gates/v1a_train.json" "no ablation run starts (rule §4.2 V1a)"
gate V1a-scorer "$outputs/gates/v1a_scorer.json" "no ablation run starts (rule §4.2 V1a)"
gate V3a "$outputs/gates/v3a.json" "V3a passes before the development ablation whatever stages 1 and 2 give; rs_ablation refuses without it"
if [[ "$dry" != 1 && -f "$outputs/stage3/verdict.json" ]]; then
    gate V3c "$outputs/stage3/rederive_stage3.json" "F is not stated until the stage 3 re-derivation agrees"
fi
if [[ "$dry" == 1 ]]; then
    f=F; arms=$(names arms --final-name G)
    note "dry: F and the eight removal arms come from rs_queue_names.py (final, arms); shown for F = G (for c* the arms are c*-no_<removal>)"
else
    f=$(names final --outputs "$outputs" "${smoke[@]}") || stop "F cannot be resolved (rs_queue_names.py final)"
    arms=$(names arms --outputs "$outputs" "${smoke[@]}") || stop "the arm names cannot be resolved"
    note "F = $f"
fi
seeds="1 2 3"
trained=()
while read -r removal arm; do
    [[ -n "$removal" ]] || continue
    case "$removal" in content_correction) variant=uncorrected ;; instructed_graphs) variant=default_view ;; *) variant= ;; esac
    if [[ -n "$variant" && "$dry" != 1 && "$(jget "$outputs/gates/ablation_graphs.json" "d['items'].get('$variant', {}).get('pass')")" == False ]]; then
        note "$removal: the $variant graph check failed (gates/ablation_graphs.json); this removal does not run (rule §2.5), the others go on"
        continue
    fi
    softstep "train_$removal" gpu "$here/rs_train.py" train --configs "$arm" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}" \
        || note "$removal: the trainer call for $arm failed or was refused (log $outputs/logs/${queue}_train_$removal.log); the other removals go on"
    trained+=("$arm")
done <<< "$arms"
if [[ "$dry" != 1 ]]; then
    for arm in "${trained[@]}"; do check_runs "$arm"; done
    bad=0; for arm in "${trained[@]}"; do for s in $seeds; do [[ -f "$outputs/train/$arm/s$s/run.json" ]] || bad=1; done; done
    (( bad == 0 )) || stop "an ablation arm is unfinished (trainer failure above): nothing is scored; restart skips the finished runs"
fi
step score_dev cpu "$here/rs_ablation.py" dev --final "$f" "${w[@]}" --outputs "$outputs" "${smoke[@]}" || stop "the development ablation scorer failed"
[[ "$dry" == 1 ]] || note "ablation/dev.json records inputs.ablation_graphs_sha256 = $(jget "$outputs/ablation/dev.json" "d['inputs']['ablation_graphs_sha256']")"
if [[ "${RS_SMOKE:-0}" == 1 ]]; then note "smoke: rs_vp ablation has no smoke mode; the queue ends after the scorer"; finish; fi
step v_ablation cpu "$here/rs_vp.py" ablation --final "$f" --outputs "$outputs" "${w[@]}" \
    || stop "rs_vp ablation disagrees with ablation/dev.json (rule §4.5): trace before the report"
finish
