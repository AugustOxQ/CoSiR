#!/usr/bin/env bash
# Stage 0 (spec §3.5, rule §4.1; GPU then CPU): G's emotion claim at both scales. Gates first (V1a both halves, V0a: the build's
# files; the runner `rs_stage0 held` refuses without V0a and, for seed 4, without a passing gates/v1a_train.json for the trainer on
# disk; `rs_train train` does not check V1a itself, so this queue does). Then: seed 4 trains once (G at the sample scale); held
# inference of G's sample seeds 0 to 4; V0c; V0d; the verdict; V0b (`rs_vh stage0`: a disagreement states no outcome and stops
# the queue before G's analysis); then G's analysis (after the verdict file, rule §4.1). Stage 0 leads to nothing: no other stage
# waits for it. gates/v1a_train.json must not be rewritten after seed 4 trains (V0d compares the mtimes).
# Pending deletions: the V1a regression runs' weights (G's sample seed 4 weights are kept).
queue=stage0
source "$(dirname -- "${BASH_SOURCE[0]}")/rs_queue_common.sh"
w=(--workers "$workers")
gate V1a-train "$outputs/gates/v1a_train.json" "no run of stage 0 seed 4 starts (rule §4.2 V1a)"
gate V1a-scorer "$outputs/gates/v1a_scorer.json" "no run of stage 0 seed 4 starts (rule §4.2 V1a)"
gate V0a "$outputs/gates/v0a.json" "stage 0 stops (rule §4.1 V0a)"
if [[ "${RS_SMOKE:-0}" == 1 ]]; then
    # rs_stage0 has no --smoke and refuses a smoke run of seed 4 (record smoke true): a smoke ends after the training step.
    softstep train_seed4 gpu "$here/rs_train.py" train --configs G --seeds 4 --device "$device" --outputs "$outputs" "${smoke[@]}"
    note "smoke: rs_stage0 and rs_vh have no smoke mode and need the real gates; the queue ends after the training step"
    finish
fi
# --- G at the sample scale: seed 4 (once), held inference of seeds 0 to 4 -----------------------------------------------
step train_seed4 gpu "$here/rs_train.py" train --configs G --seeds 4 --device "$device" --outputs "$outputs" "${smoke[@]}" \
    || stop "seed 4's training crashed (log $outputs/logs/${queue}_train_seed4.log)"
if [[ "$dry" != 1 ]]; then
    seeds=4 check_runs G
    (( unfinished == 0 )) || stop "seed 4 left no run.json"
    [[ "$(run_status "$outputs/train/G/s4/run.json")" == ok ]] || stop "seed 4 FAILED: V0d (it is rerun once by hand, same seed and code, listed; a seed that still fails leaves the sample scale with no outcome)"
fi
pending_add "$outputs/regression/v1a_s0" "V1a regression run weights (G at seed 0, sample scale)" "regression run of the gate; gates/v1a_train.json is the record"
pending_add "$outputs/regression/v1a_s1" "V1a regression run weights (G at seed 1, sample scale)" "regression run of the gate; gates/v1a_train.json is the record"
step held_sample gpu "$here/rs_stage0.py" --outputs "$outputs" "${w[@]}" held --seeds 0 1 2 3 4 --device "$device" \
    || stop "G's sample held inference failed (V0a's GPU part or V0c rows; see the log)"
# --- the gates over the sets, the verdict, V0b ---------------------------------------------------------------------------------
softstep v0c cpu "$here/rs_stage0.py" --outputs "$outputs" "${w[@]}" check v0c \
    || note "V0c fails at a scale: that scale gets no outcome (the verdict scores the scales whose set passes); the seed is rerun once by hand (V0d)"
step v0d cpu "$here/rs_stage0.py" --outputs "$outputs" "${w[@]}" check v0d \
    || stop "V0d does not hold (seed policy): no outcome"
step verdict cpu "$here/rs_stage0.py" --outputs "$outputs" "${w[@]}" verdict || stop "the stage 0 verdict failed"
[[ "$dry" == 1 ]] || note "stage 0 outcomes: $(jget "$outputs/stage0/verdict.json" "d['outcomes']") E0=$(jget "$outputs/stage0/verdict.json" "d['E0']") (not stated until V0b agrees)"
step v0b cpu "$here/rs_vh.py" stage0 --outputs "$outputs" "${w[@]}" \
    || stop "V0b (rs_vh stage0) disagrees with the verdict: no outcome is stated; trace, G's analysis does not run"
# --- G's analysis (after the verdict file; decides nothing) ----------------------------------------------------------------------
step analysis_g cpu "$here/rs_analysis.py" --model G --outputs "$outputs" "${smoke[@]}" || stop "G's analysis failed"
finish
