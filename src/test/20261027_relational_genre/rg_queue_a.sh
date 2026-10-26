#!/usr/bin/env bash
# Stage A (spec §3.5, rule §4.1; GPU and CPU): VA1 (both halves) and EQ; GR-16 and GR-32; then per climb decision
# `rg_score climb`, `rg_v2 --phase rung --arm <newest>` (a disagreement stops the queue before the next rung), and the
# rung named by the last stage_a/climb.json decision (`next`; null ends the climb; the cap is GR-256); then GN-8, GS-0.02,
# GS-0.01, GQ-8; `rg_score a`; `rg_v2 --phase a`; then the loss-scale run (after stage A's outcome, spec §3.6).
# VA1 that does not hold: nothing runs in A, B or C. EQ that fails: GQ-8 still trains and is scored, but cannot win
# (rule §4.1; the scorer reads regression/eq.json). A run that fails (non-finite loss, VA3 after the trainer's one forced
# rerun) is noted and the climb goes on as the scorer decides (a failed newest rung ends it). Every runner skips finished
# work, so the queue can be started again after a stop. Pending deletions: the VA1 runs' weights, the weights in
# stage_a/verdict.json `pending_deletions`, the loss-scale run's weights (appended to the list, never deleted).
queue=a
source "$(dirname -- "${BASH_SOURCE[0]}")/rg_queue_common.sh"
a=$outputs/stage_a
climb=$a/climb.json
w=(--workers "$workers")
# --- VA1, both halves --------------------------------------------------------------------------------------------
softstep va1_train gpu "$here/rg_train.py" regress va1 --device "$device" --outputs "$outputs" "${smoke[@]}"
for d in "$outputs"/regression/va1_*_s*/; do
    [[ -d "$d" ]] && pending_add "${d%/}" "VA1 regression run weights (sample scale, switches off or GR-8)" "regression runs of the gate; the exports and va1.json are the record"
done
softstep va1_scorer cpu "$here/rg_score.py" check va1 --outputs "$outputs" "${w[@]}" "${smoke[@]}"
regression_gate VA1 "$outputs/regression/va1.json" "$outputs/regression/va1_scorer.json" \
    || stop "VA1 does not hold: nothing runs in stages A, B or C (rule §4.1); the cause goes in the report"
# --- EQ ----------------------------------------------------------------------------------------------------------
softstep eq cpu "$here/rg_train.py" eq --device cpu --outputs "$outputs" "${smoke[@]}" \
    || note "EQ fails or crashed: GQ-8 still runs, is reported with its guard, and cannot be the winner (rule §4.1)"
if [[ "$dry" != 1 && ! -f "$outputs/regression/eq.json" ]]; then stop "EQ left no regression/eq.json (the scorer needs it)"; fi
# --- GR-16 and GR-32, then the climb -------------------------------------------------------------------------------
train_arms 0 GR-16 GR-32
for i in 1 2 3 4; do   # decisions: after GR-32, then after GR-64, GR-128, GR-256 at most
    step "climb_$i" cpu "$here/rg_score.py" climb --outputs "$outputs" "${w[@]}" "${smoke[@]}" || stop "the climb decision failed"
    if [[ "$dry" == 1 ]]; then newest=NEWEST; next=NEXT; else
        newest=$(jget "$climb" "d['decisions'][-1]['newest']"); next=$(jget "$climb" "d['decisions'][-1]['next']")
        note "climb decision $i: newest $newest, reason $(jget "$climb" "d['decisions'][-1]['reason']"), next ${next}"
    fi
    step "va2_rung_$newest" cpu "$here/rg_v2.py" --phase rung --arm "$newest" --outputs "$outputs" --workers "$workers" "${smoke[@]}" \
        || stop "VA2 (rung $newest) disagrees with the scorer: settle it before the next rung runs (rule §4.1)"
    if [[ "$dry" == 1 ]]; then
        note "dry: the next rung is the one the decision names (GR-64, GR-128, GR-256; at most three more); one shown"
        train_arms 0 GR-64
        break
    fi
    [[ -z "$next" || "$next" == None ]] && break
    case "$next" in GR-64|GR-128|GR-256) ;; *) stop "climb.json names the next rung '$next': not a rung above GR-32" ;; esac
    train_arms 0 "$next"
done
if [[ "$dry" != 1 && -n "$(jget "$climb" "d['decisions'][-1]['next']" | grep -v '^None$')" ]]; then
    stop "the climb is not finished after four decisions (next $(jget "$climb" "d['decisions'][-1]['next']")); trace"
fi
# --- the other arms (GQ-8 trains after EQ ran: the scorer needs it either way) -------------------------------------
train_arms 0 GN-8 GS-0.02 GS-0.01 GQ-8
# --- the stage A outcome -------------------------------------------------------------------------------------------
step score_a cpu "$here/rg_score.py" a --outputs "$outputs" "${w[@]}" "${smoke[@]}" || stop "scoring stage A failed"
for arm in $(jget "$a/verdict.json" "' '.join(d['pending_deletions'])"); do
    pending_add "$outputs/train/$arm/s0" "seed-0 weights of $arm (stage A)" "A_genre below T_g - r_genre or a failed run (rule §4.3 Weights)"
done
[[ "$dry" == 1 ]] && note "dry: stage_a/verdict.json pending_deletions would be listed here"
step va2_a cpu "$here/rg_v2.py" --phase a --outputs "$outputs" --workers "$workers" "${smoke[@]}" \
    || stop "VA2 (stage A) disagrees with the scorer: no stage A outcome is stated; trace before stage B"
[[ "$dry" == 1 ]] || note "stage A outcome: $(jget "$a/verdict.json" "d['outcome']") (v*_A $(jget "$a/verdict.json" "d['winner']"))"
# --- the loss-scale run (spec §3.6; after stage A's outcome; failure does not stop B) --------------------------------
softstep loss_scale gpu "$here/rg_train.py" lossscale --device "$device" --outputs "$outputs" "${smoke[@]}" \
    || note "the loss-scale run failed (a diagnostic; stage B goes on); log $outputs/logs/${queue}_loss_scale.log"
pending_add "$outputs/loss_scale/run" "the loss-scale run's weights (GR-1, one epoch; no export on P)" "diagnostic run of spec §3.6; loss_scale.json is the record"
finish
