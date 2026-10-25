#!/usr/bin/env bash
# Stage iii (rule §4.3; GPU, then CPU): VG1 (both halves), the five first variants (GT-2 GT-4 GR-0.5 GR-2 GA-1) at seed 0,
# eg_score3 first, VG2 first, the two extras, eg_score3 all, VG2 all, then (when a winner gains one genre seed range) the
# winner and R6-5 on seeds 1 to 3, the parity gate, eg_score3 fresh, VG2 fresh, and eg_score3 report.
# Needs no new store. VG1 failed: no variant runs. VG2 that disagrees: stage iii stops (no outcome is stated).
# A run that fails VG3 is rerun once (eg_train3 --rerun-failed, VG4) before each scoring; a second call is never made
# for a run that was already rerun. After `all`, the weights of the variants below T_g - 0.0128 (and of failed runs) are
# appended to the pending deletions list (never deleted); fresh-seed weights stay off it.
# A finished run, score file and re-derivation is skipped. Smoke: --smoke on every runner (2 epochs, 5000 pairs, 20 draws).
queue=stage_iii
source "$(dirname -- "${BASH_SOURCE[0]}")/eg_queue_common.sh"
stage=$outputs/stage_iii
stop() { printf 'EG_STAGE_III_STOP %s\n' "$*" | tee -a "$status"; failed=1; finish; }
need_rerun() {  # need_rerun VARIANT SEED: the run failed VG3 and was not rerun yet
    [[ "$(jget "$stage/train/$1/s$2/run.json" "d['status']=='failed' and d.get('failure_kind')=='vg3' and 'rerun_of' not in d")" == True ]]
}
train_set() {  # train_set "SEEDS" VARIANT...: train, then rerun the VG3 failures once each (runs that exist are skipped)
    local seeds=$1 v s label; shift
    label="${1}_s${seeds// /}"
    step "train_$label" gpu "$here/eg_train3.py" train --variants "$@" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}" \
        || stop "training failed (log $outputs/logs/${queue}_train_$label.log)"
    for v in "$@"; do
        for s in $seeds; do
            if need_rerun "$v" "$s"; then
                note "$v s$s failed VG3: one rerun with the same seed and code (VG4)"
                step "rerun_${v}_s$s" gpu "$here/eg_train3.py" train --variants "$v" --seeds "$s" --rerun-failed --device "$device" --outputs "$outputs" "${smoke[@]}" \
                    || stop "rerun of $v s$s failed (log)"
            fi
        done
    done
}
append_pending() {  # the weights of the variants listed in score_all.json "pending_deletions" -> the pending list; no deletion
    local v dir size stamp
    stamp=$(TZ=Europe/Amsterdam date '+%F %H:%M')
    for v in $(jget "$stage/score_all.json" "' '.join(d['pending_deletions'])"); do
        dir=$stage/train/$v/s0
        [[ -d "$dir" ]] || continue
        grep -qF "$dir/" "$pending" 2>/dev/null && continue
        size=$(du -ch "$dir"/*.pt 2>/dev/null | tail -n 1 | cut -f1)
        printf '| %s | r10 build 21 run (stage iii queue) | %s/*.pt | %s | seed-0 weights of %s (best states; the exports stay) | A_genre below T_g - 0.0128, or a failed run (rule §4.3, D28); the wrap-up decides, nothing is deleted |\n' \
            "$stamp" "$dir" "${size:-?}" "$v" >> "$pending"
    done
    note "pending deletions appended to $pending (weights below T_g - 0.0128; fresh-seed weights kept)"
}
# Refuse stale scores before training or skip-if-exists decisions.
if [[ "$dry" != 1 ]]; then
    code_check=$(cpu "$here/eg_score3.py" check-code --outputs "$outputs" 2>&1) || stop "$code_check"
fi
# --- VG1 ------------------------------------------------------------------------------------------------------------
if [[ -f "$stage/vg1.json" && "$dry" != 1 ]]; then note "vg1.json exists, VG1 training half skipped"
else soft=1 step vg1_train gpu "$here/eg_train3.py" regress vg1 --device "$device" --outputs "$outputs"; failed=0; fi
if [[ -f "$stage/vg1_scorer.json" && "$dry" != 1 ]]; then note "vg1_scorer.json exists, VG1 scorer half skipped"
else soft=1 step vg1_scorer cpu "$here/eg_score3.py" check vg1 --outputs "$outputs"; failed=0; fi
gate vg1 || stop "VG1 does not hold: no variant is trained (rule §4.3); the cause goes in the report"
# --- the five first variants, seed 0 ---------------------------------------------------------------------------------
first=(GT-2 GT-4 GR-0.5 GR-2 GA-1)
train_set 0 "${first[@]}"
if [[ -f "$stage/score_first.json" && "$dry" != 1 ]]; then note "score_first.json exists, skipped"
else step score_first cpu "$run" score3 first --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "scoring the first variants failed"; fi
if [[ -f "$stage/rederive_vg2_first.json" && "$dry" != 1 ]] && gate vg2 --phase first; then note "VG2 first already agrees, skipped"
else
    step vg2_first cpu "$here/eg_vg2.py" --phase first --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "VG2 (first) disagrees: trace before any outcome is stated"
    gate vg2 --phase first || stop "VG2 (first) gate does not hold"
fi
# --- the extras ------------------------------------------------------------------------------------------------------
extras=$(jget "$stage/score_first.json" "' '.join(d['extras'])")
[[ "$dry" == 1 ]] && extras="GT-3 GR-1"   # dry run: a placeholder pair
note "extras (from score_first.json): ${extras:-none}"
[[ -z "$extras" ]] || train_set 0 $extras
if [[ -f "$stage/score_all.json" && "$dry" != 1 ]]; then note "score_all.json exists, skipped"
else step score_all cpu "$run" score3 all --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "scoring all variants failed"; fi
append_pending
if [[ -f "$stage/rederive_vg2_all.json" && "$dry" != 1 ]] && gate vg2 --phase all; then note "VG2 all already agrees, skipped"
else
    step vg2_all cpu "$here/eg_vg2.py" --phase all --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "VG2 (all) disagrees: trace before any outcome is stated"
    gate vg2 --phase all || stop "VG2 (all) gate does not hold"
fi
# --- fresh seeds, when triggered -------------------------------------------------------------------------------------
winner=$(jget "$stage/score_all.json" "d['winner']"); trigger=$(jget "$stage/score_all.json" "d['fresh_seeds']")
[[ "$dry" == 1 ]] && { winner=WINNER; trigger=True; }   # dry run: show the fresh-seed commands
note "winner: ${winner:-none}; fresh seeds triggered: ${trigger:-no}"
if [[ "$trigger" == True ]]; then
    train_set "1 2 3" "$winner" R6-5
    if [[ "${EG_SMOKE:-0}" == 1 ]]; then
        # parity has no smoke mode (it trains R6-5 at full length and compares with 12-24's K4 weights): the smoke
        # chain uses a fresh_parity.json copied into the scratch outputs (eg_queue.md)
        note "smoke: parity step skipped; stage_iii/fresh_parity.json (copied) stands"
    else
        soft=1 step parity gpu "$here/eg_train3.py" parity --device "$device" --outputs "$outputs"; failed=0
    fi
    gate parity || note "parity fails or is missing: no C outcome is stated (the scorer records the cause); trace it"
    if [[ -f "$stage/score_fresh.json" && "$dry" != 1 ]]; then note "score_fresh.json exists, skipped"
    else step score_fresh cpu "$here/eg_score3.py" fresh --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "scoring the fresh seeds failed"; fi
    if [[ -f "$stage/rederive_vg2_fresh.json" && "$dry" != 1 ]] && gate vg2 --phase fresh; then note "VG2 fresh already agrees, skipped"
    else
        step vg2_fresh cpu "$here/eg_vg2.py" --phase fresh --outputs "$outputs" --workers 8 "${smoke[@]}" || stop "VG2 (fresh) disagrees: trace before any outcome is stated"
        gate vg2 --phase fresh || stop "VG2 (fresh) gate does not hold"
    fi
fi
step report cpu "$here/eg_score3.py" report --outputs "$outputs" "${smoke[@]}" || stop "stage iii report failed"
finish
