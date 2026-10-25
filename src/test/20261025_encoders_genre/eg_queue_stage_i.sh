#!/usr/bin/env bash
# Stage i (rule §4.1, spec §3.5; GPU and CPU): X0 gate, X1, stage 3 per candidate (K6, K7, K8, K5: student and matched arm
# train and infer; then eg_score stage 3 and eg_v2 stage 3), X2, stage 4 for the candidates with an S1 outcome other
# than S1-NO, stage 5 (ArtELingo-28, B1 and B6 at the sample scale when needed), eg_v2 stages 4 and 5.
# Gates stop what the rule says and nothing else: X0 or X1 failed: stage i stops (ii with K2 to K4 and iii go on); X2
# failed: no stage 4, stage 5 at the sample scale; V5 failed for one store: that candidate has no outcome, the others go on;
# a V2 that disagrees stops stage i (trace before any outcome is stated).
# Defer for time only with eg_run.py defer --estimate-end (recorded in part2/deferral.json).
# Stores, X0 and V5 come from eg_queue_stores.sh. Run from the merged main checkout.
queue=stage_i
source "$(dirname -- "${BASH_SOURCE[0]}")/eg_queue_common.sh"
stop() { printf 'EG_STAGE_I_STOP %s\n' "$*" | tee -a "$status"; failed=1; finish; }
step budget cpu "$run" budget --outputs "$outputs"
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_budget.log" | tee -a "$status"
regress() {  # regress x1|x2: the three halves, each skipped when its file exists (a failed one stays failed: see eg_queue.md)
    local x=$1 reg="$outputs/part2/regression"
    [[ -f "$reg/${x}_student.json" && "$dry" != 1 ]] && note "${x}_student.json exists, skipped" || \
        soft=1 step "${x}_student" gpu "$here/eg_student.py" regress "$x" --device cuda --outputs "$outputs"
    [[ -f "$reg/${x}_matched.json" && "$dry" != 1 ]] && note "${x}_matched.json exists, skipped" || \
        soft=1 step "${x}_matched" gpu "$here/eg_matched.py" regress "$x" --device cuda --outputs "$outputs"
    [[ -f "$reg/${x}_scorer.json" && "$dry" != 1 ]] && note "${x}_scorer.json exists, skipped" || \
        soft=1 step "${x}_scorer" cpu "$here/eg_score.py" --check "$x" --outputs "$outputs" --workers 8
    failed=0
}
candidate() {  # candidate SCALE K (a failed step stops this candidate only; the caller notes it)
    candidate_steps "$@"; local rc=$?; failed=0; return $rc
}
candidate_steps() {  # candidate SCALE K: student train and infer, matched train and infer (held and ArtELingo-28 outputs)
    local scale=$1 k=$2 s m kinds=()
    s=$(store_of "$k"); m=$(matched_of "$k")
    if [[ "$(jget "$outputs/stores/v5.json" "d['stores'].get('artelingo28/$s',{}).get('pass') is False")" == True ]]; then
        kinds=(--kinds held); note "$k: the ArtELingo-28 store failed V5: no ArtELingo-28 outputs, no A outcome"
    fi
    soft=1 step "train_${k}_$scale" gpu "$here/eg_student.py" train --scale "$scale" --arm "$k" --device "$device" --outputs "$outputs" "${smoke[@]}" || return 1
    soft=1 step "infer_${k}_$scale" gpu "$here/eg_student.py" infer --scale "$scale" --arm "$k" "${kinds[@]}" --outputs "$outputs" "${smoke[@]}" || return 1
    soft=1 step "matched_${m}_$scale" gpu "$here/eg_matched.py" --scale "$scale" --arm "$m" "${kinds[@]}" --device "$device" --outputs "$outputs" "${smoke[@]}" || return 1
}
# --- X0 and X1 ------------------------------------------------------------------------------------------------------
gate x0 || stop "X0 does not hold (or stores/x0.json is missing: run eg_queue_stores.sh): stage i stops; stages ii (K2 to K4) and iii go on"
regress x1
gate x1 || stop "X1 failed: no stage 3 arm trains; stage i stops; stages ii (K2 to K4) and iii go on; the cause goes in the report"
# --- stage 3 (sample scale) -----------------------------------------------------------------------------------------
cands=$(driver candidates 3) || stop "candidates 3 failed"
if [[ -z "$cands" ]]; then
    note "every stage 3 candidate already has a block or is deferred: nothing to train or score (check and V2 run again)"
else
    todo=()
    for k in $cands; do
        s=$(store_of "$k")
        if gate stage3 --cand "$k"; then
            candidate sample "$k" || note "$k failed (step line above, log in $outputs/logs): no outcome for $k; the others go on"
            todo+=("$k")
        elif [[ "$(jget "$outputs/stores/v5.json" "d['stores'].get('artelingo_full/$s',{}).get('pass') is False")" == True ]]; then
            note "$k: V5 failed for artelingo_full/$s: $k has no outcome (the scorer writes its block with the cause)"
            todo+=("$k")
        else
            note "$k: not run (V5 entry missing: run eg_queue_stores.sh); it has no block yet"
        fi
    done
    if (( ${#todo[@]} )); then
        soft=1 step verdict3 cpu "$here/eg_score.py" --stage 3 --candidates "${todo[@]}" --outputs "$outputs" --workers 8 "${smoke[@]}"
        failed=0   # exit 1 means a candidate without an outcome (reported with the cause); a crash shows in check3
    fi
fi
step check3 cpu "$run" check-verdict 3 --outputs "$outputs" || stop "check3: a candidate has no stage 3 block (see log)"
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_check3.log" | tee -a "$status"
step v2_3 cpu "$here/eg_v2.py" --stage 3 --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "V2 stage 3 stopped: see log (no candidate outcome, or disagreement; rule §4.1)"
# --- X2 and stage 4 (full scale) ------------------------------------------------------------------------------------
regress x2
if gate x2; then
    cands4=$(driver candidates 4) || stop "candidates 4 failed"
    if [[ -z "$cands4" ]]; then
        note "no candidate is left for stage 4 (S1-NO, deferred, or every block exists): nothing to train or score"
    else
        todo=()
        for k in $cands4; do
            if gate v2 --stage 3 && gate stage4 --cand "$k"; then
                candidate full "$k" || note "$k failed at full scale (step line above): no stage 4 outcome for $k; the others go on"
                todo+=("$k")
            else
                note "$k: stage 4 gate does not hold (see the gate line above): not run"
            fi
        done
        if (( ${#todo[@]} )); then
            soft=1 step verdict4 cpu "$here/eg_score.py" --stage 4 --candidates "${todo[@]}" --outputs "$outputs" --workers 8 "${smoke[@]}"
            failed=0
        fi
    fi
    if [[ -f "$outputs/part2/verdict_stage4.json" || "$dry" == 1 ]]; then
        step check4 cpu "$run" check-verdict 4 --outputs "$outputs" || stop "check4: a stage 4 candidate has no block (see log)"
        [[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_check4.log" | tee -a "$status"
        step v2_4 cpu "$here/eg_v2.py" --stage 4 --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "V2 stage 4 stopped: see log (no candidate outcome, or disagreement; rule §4.1)"
    fi
else
    note "X2 does not hold: no stage 4 arm trains; stage 5 runs at the sample scale (rule §4.1)"
fi
# --- stage 5 (ArtELingo-28) -----------------------------------------------------------------------------------------
plan=$(driver a28) || stop "a28 plan failed"
models=${plan%%|*}; rest=${plan#*|}; bscales=${rest%%|*}
score_models=$(driver candidates 5 --include-done) || stop "stage 5 candidate list failed"
if [[ -z "$score_models" ]]; then
    note "stage 5: no non-deferred stage 3 candidate: nothing to score"
    finish
fi
note "stage 5 models: $models; B1/B6 scales: $bscales; scales: ${rest#*|}"
for sc in $bscales; do
    # B1 and B6 at the full scale are 12-24's files; the sample scale is inferred here (skipped when the files exist)
    [[ "$sc" == sample ]] || continue
    for b in B1 B6; do
        soft=1 step "${b}_a28_sample" gpu "$here/eg_matched.py" --scale sample --arm "$b" --device "$device" --outputs "$outputs" "${smoke[@]}" || stop "$b ArtELingo-28 inference failed"
    done
done
failed=0
soft=1 step a28 cpu "$here/eg_score.py" --stage 5 --candidates $score_models --outputs "$outputs" --workers 8 "${smoke[@]}"
failed=0
step check5 cpu "$run" check-verdict 5 --candidates $score_models --outputs "$outputs" || stop "check5: a stage 5 model has no block (see log)"
[[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_check5.log" | tee -a "$status"
step v2_5 cpu "$here/eg_v2.py" --stage 5 --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "V2 stage 5 stopped: see log (no candidate outcome, or disagreement; rule §4.1)"
finish
