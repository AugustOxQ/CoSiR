#!/usr/bin/env bash
# Held ablation (spec D22, rule §2.5, §3.5; GPU then CPU): each of the eight removals of F at full scale, seeds 0 to 4, held
# inference, `rs_ablation held --final F`, `rs_vh ablation --final F`. About 80 GPU minutes, then CPU scoring: START IT FROM 19:00.
# The weekday rule: it refuses to start on a weekday (Monday to Friday) before 19:00 Amsterdam time unless RS_FORCE_HOUR=1.
# Gates: V1a, V4a (both halves; the held ablation does not run when V4a fails, rule §4.4), V3a (rs_ablation refuses without it),
# and for F = c*: stage4/verdict.json names F and V4b agrees (stage4/rederive_stage4.json). For F = G the held outputs are 12-27's.
# The graph removals whose graph check failed are skipped as in the development queue (the gate file's SHA-256 goes into
# ablation/held.json as inputs.ablation_graphs_sha256). A trainer failure of one arm stops only that arm; scoring starts only
# when every arm that is to run has its outputs. Weights of every arm are kept.
queue=ablation_held
source "$(dirname -- "${BASH_SOURCE[0]}")/rs_queue_common.sh"
w=(--workers "$workers")
weekday_rule "the held ablation" 19
gate V1a-train "$outputs/gates/v1a_train.json" "no ablation run starts (rule §4.2 V1a)"
gate V1a-scorer "$outputs/gates/v1a_scorer.json" "no ablation run starts (rule §4.2 V1a)"
gate V4a-train "$outputs/gates/v4a_train.json" "the held ablation does not run (rule §4.4 V4a)"
gate V4a-scorer "$outputs/gates/v4a_scorer.json" "the held ablation does not run (rule §4.4 V4a)"
gate V3a "$outputs/gates/v3a.json" "rs_ablation refuses without it"
if [[ "$dry" == 1 ]]; then
    f=F; arms=$(names arms --final-name G)
    note "dry: F and the eight removal arms come from rs_queue_names.py (final, arms); shown for F = G (for c* the arms are c*-no_<removal>); for F = c* the gates V3c and V4b (rederive_stage3.json, rederive_stage4.json) are also required"
else
    f=$(names final --outputs "$outputs" "${smoke[@]}") || stop "F cannot be resolved (rs_queue_names.py final)"
    arms=$(names arms --outputs "$outputs" "${smoke[@]}") || stop "the arm names cannot be resolved"
    note "F = $f"
    if [[ "$f" != G ]]; then
        gate V3c "$outputs/stage3/rederive_stage3.json" "F = c* is not stated until the stage 3 re-derivation agrees"
        [[ "$(jget "$outputs/stage4/verdict.json" "d['candidate']")" == "$f" ]] || stop "stage4/verdict.json does not name F = $f: run the stage 4 queue first"
        gate V4b "$outputs/stage4/rederive_stage4.json" "F's held outcomes are not stated until V4b agrees"
    fi
fi
seeds="0 1 2 3 4"
trained=()
while read -r removal arm; do
    [[ -n "$removal" ]] || continue
    case "$removal" in content_correction) variant=uncorrected ;; instructed_graphs) variant=default_view ;; *) variant= ;; esac
    if [[ -n "$variant" && "$dry" != 1 && "$(jget "$outputs/gates/ablation_graphs.json" "d['items'].get('$variant', {}).get('pass')")" == False ]]; then
        note "$removal: the $variant graph check failed (gates/ablation_graphs.json); this removal does not run (rule §2.5), the others go on"
        continue
    fi
    if softstep "train_$removal" gpu "$here/rs_full.py" train --config "$arm" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}"; then
        softstep "held_$removal" gpu "$here/rs_full.py" held --config "$arm" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}" \
            || note "$removal: held inference of $arm failed (log $outputs/logs/${queue}_held_$removal.log); the other removals go on"
    else
        note "$removal: the full-scale training of $arm failed or was refused (log $outputs/logs/${queue}_train_$removal.log); the other removals go on"
    fi
    trained+=("$arm")
done <<< "$arms"
step score_held cpu "$here/rs_ablation.py" held --final "$f" "${w[@]}" --outputs "$outputs" "${smoke[@]}" \
    || stop "the held ablation scorer failed (an arm without held outputs and without a failure record is refused: see the log)"
[[ "$dry" == 1 ]] || note "ablation/held.json records inputs.ablation_graphs_sha256 = $(jget "$outputs/ablation/held.json" "d['inputs']['ablation_graphs_sha256']")"
if [[ "${RS_SMOKE:-0}" == 1 ]]; then note "smoke: rs_vh has no smoke mode; the queue ends after the scorer"; finish; fi
step v_ablation cpu "$here/rs_vh.py" ablation --final "$f" --outputs "$outputs" "${w[@]}" \
    || stop "rs_vh ablation disagrees with ablation/held.json (rule §4.5): trace before the report"
finish
