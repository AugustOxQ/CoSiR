#!/usr/bin/env bash
# The local fallback of stages 1 and 2 (rule §4.2 "Fallback"; GPU and CPU): used when N1, N2 or N3 fails, stage 1 cannot start on
# the node by RS_START_BY, stage 2 has not ended there by RS_END_BY (the node plan's deadlines: the real start of stage 1 on the node,
# and that start plus 8 h; rule §4.2, exception of 2026-10-27 15:31), or the node is lost. Both stages run here from the start, with
# 12-27's local GS-0.02 seed-0 run as G0 (`rs_sweep score --frozen-g0`; frozen, V1a shows the trainer reproduces it). G0' and N3 are the
# node's; the node's results are reported only. First step: if `<outputs>/sweep/node402/` exists (the node wrote stage files before
# the fallback), it is MOVED (never deleted) to `<outputs>/aside/node402_<YYYYmmdd-HHMM>/sweep/node402/`, so that the readers that
# decide F (`rs_full.sweep_ended_without_candidates`, `rs_ablation.resolve_final`) see the local machine only; the queue refuses if
# that target exists, and prints one line for rs_node_log.md. `scores/node402/` is read by no such reader and stays. Order: G0's score; the 13 knobs in the sheet's order (their three values trained
# in one call, then scored); `rs_sweep stage1`; V1c part (a) (`rs_vp partitions`) and part (b) (`rs_vp sweep`); then, unless
# SWEEP-NONE, the grid's new points (groups of 6), `rs_sweep stage2`, V1c again. A disagreement stops the queue before the next
# stage. The k11 runs are skipped when gates/k11_sample.json failed (rule §2.2; `rs_sweep stage1` enters them as failed).
# Power (rule §4.2): the first knob's runs are timed and nvidia-smi power is sampled every 10 s into logs/fallback_power.txt; the
# queue notes mean and maximum. On a weekday between 10:00 and 19:00 it then STOPS unless RS_DAY_OK=1 (the draw was under 200 W)
# or RS_FORCE_HOUR=1; otherwise start it again from 19:00 (finished runs skip). The held ablation follows these runs.
# Pending deletions: every run of stages 1 and 2 not in TOP (TOP's and G0's are kept, rule §4.5 Weights).
queue=fallback
source "$(dirname -- "${BASH_SOURCE[0]}")/rs_queue_common.sh"
w=(--workers "$workers")
m=local
# --- the node's stage files go aside (decided: the node's results are reported only; a move, never a delete) -------------------------
node_sweep=$outputs/sweep/node402
if [[ -e "$node_sweep" ]]; then
    aside=$outputs/aside/node402_$(TZ=Europe/Amsterdam date '+%Y%m%d-%H%M')
    if [[ -e "$aside" ]]; then stop "the target $aside exists: the node's stage files are not moved (nothing was changed); look at it, then start the queue again"; fi
    if [[ "$dry" == 1 ]]; then
        note "dry: would move $node_sweep to $aside/sweep/node402 (a move; add this line to rs_node_log.md)"
    else
        mkdir -p "$aside/sweep" && mv -- "$node_sweep" "$aside/sweep/" \
            || stop "moving $node_sweep to $aside/sweep/ failed: nothing is deleted; look at both folders"
        note "rs_node_log.md entry: $(TZ=Europe/Amsterdam date '+%F %H:%M') fallback: moved $node_sweep to $aside/sweep/node402 (the node's stage files; reported only)"
    fi
fi
gate V1a-train "$outputs/gates/v1a_train.json" "no stage 1 run starts (rule §4.2 V1a)"
gate V1a-scorer "$outputs/gates/v1a_scorer.json" "no stage 1 run starts (rule §4.2 V1a)"
train_score() {   # train_score LABEL NAME...: one trainer call (seed 0, machine local) then its scores; failed runs are recorded outcomes
    local label=$1; shift
    softstep "train_$label" gpu "$here/rs_train.py" train --configs "$@" --seeds 0 --machine "$m" --device "$device" --outputs "$outputs" "${smoke[@]}" \
        || stop "the trainer call $label crashed or was refused (log $outputs/logs/${queue}_train_$label.log)"
    if [[ "$dry" != 1 ]]; then
        seeds=0 check_runs "$@"
        (( unfinished == 0 )) || stop "$label: a run has no run.json"
    fi
    step "score_$label" cpu "$here/rs_sweep.py" score --machine "$m" --runs "$@" --seed 0 --outputs "$outputs" "${smoke[@]}" \
        || stop "scoring $label failed"
}
step score_g0 cpu "$here/rs_sweep.py" score --machine "$m" --runs G --frozen-g0 --seed 0 --outputs "$outputs" "${smoke[@]}" \
    || stop "G0's score (12-27's seed-0 run) failed"
# --- stage 1: the 13 knobs ---------------------------------------------------------------------------------------------------------
knobs=$(names stage1-knobs)
first=1
while read -r knob configs; do
    if [[ "$knob" == k11 && "$dry" != 1 && "$(jget "$outputs/gates/k11_sample.json" "d['pass']")" == False ]]; then
        note "k11: the k = 20 rebuild differs from the frozen graphs (gates/k11_sample.json); k11's runs do not run (rule §2.2)"
        continue
    fi
    if (( first )); then
        poll=
        if [[ "$dry" == 1 ]]; then note "dry: nvidia-smi power.draw sampled every 10 s into $outputs/logs/fallback_power.txt during $knob"
        elif command -v nvidia-smi > /dev/null; then
            ( while :; do nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits >> "$outputs/logs/fallback_power.txt"; sleep 10; done ) &
            poll=$!
        fi
    fi
    train_score "$knob" $configs
    if (( first )); then
        first=0
        if [[ -n "${poll:-}" ]]; then
            kill "$poll" 2> /dev/null
            note "first knob ($knob) power draw: $(awk '{s+=$1; if ($1>m) m=$1; n++} END {if (n) printf "mean %.0f W, max %.0f W (%d samples; the first samples are the data load)", s/n, m, n}' "$outputs/logs/fallback_power.txt")"
        fi
        if [[ "$dry" == 1 ]]; then note "dry: on a weekday 10:00 to 19:00 the queue would stop here unless RS_DAY_OK=1 or RS_FORCE_HOUR=1 (rule §4.2: under 200 W the runs go on by day)"
        elif in_day_window && [[ "${RS_DAY_OK:-0}" != 1 && "${RS_FORCE_HOUR:-0}" != 1 ]]; then
            stop "weekday daytime: read the power note above; under 200 W start the queue again with RS_DAY_OK=1, otherwise from 19:00 (finished runs skip)"
        fi
    fi
done <<< "$knobs"
step stage1 cpu "$here/rs_sweep.py" stage1 --machine "$m" --outputs "$outputs" "${smoke[@]}" || stop "stage 1 stopped (a run is missing or its file is stale)"
[[ "$dry" == 1 ]] || note "stage 1 outcome: $(jget "$outputs/sweep/$m/stage1.json" "d['outcome']"), grid knobs $(jget "$outputs/sweep/$m/stage1.json" "d['grid_knobs']")"
if [[ "${RS_SMOKE:-0}" != 1 ]]; then
    step v1c_a_stage1 cpu "$here/rs_vp.py" partitions --machine "$m" --outputs "$outputs" --threads "$workers" || stop "V1c part (a) disagrees (stage 1): trace before stage 2"
    step v1c_b_stage1 cpu "$here/rs_vp.py" sweep --machine "$m" --outputs "$outputs" || stop "V1c part (b) disagrees (stage 1): trace before stage 2"
fi
if [[ "$dry" != 1 && "$(jget "$outputs/sweep/$m/stage1.json" "d['outcome']")" == SWEEP-NONE ]]; then
    note "SWEEP-NONE: stages 2, 3 and 4 do not run; G stays the recipe"; finish
fi
# --- stage 2: the grid's new points ------------------------------------------------------------------------------------------------------
if [[ "$dry" == 1 ]]; then points="POINT_1 POINT_2 POINT_3 POINT_4 POINT_5 POINT_6"; note "dry: the grid's new points come from sweep/local/stage1.json grid.new, in groups of 6; one group shown"
else points=$(jget "$outputs/sweep/$m/stage1.json" "' '.join(d['grid']['new'])"); fi
group=0
while read -r line; do
    [[ -n "$line" ]] || continue
    group=$((group + 1))
    train_score "grid$group" $line
done < <(printf '%s\n' $points | xargs -n 6)
step stage2 cpu "$here/rs_sweep.py" stage2 --machine "$m" --outputs "$outputs" "${smoke[@]}" || stop "stage 2 stopped (a grid point has no score, or stage 1's file is stale)"
if [[ "${RS_SMOKE:-0}" != 1 ]]; then
    step v1c_a_stage2 cpu "$here/rs_vp.py" partitions --machine "$m" --outputs "$outputs" --threads "$workers" || stop "V1c part (a) disagrees (stage 2): trace before stage 3"
    step v1c_b_stage2 cpu "$here/rs_vp.py" sweep --machine "$m" --outputs "$outputs" || stop "V1c part (b) disagrees (stage 2): trace before stage 3"
fi
if [[ "$dry" == 1 ]]; then
    note "dry: pending rows for every stage 1 and 2 run not in TOP"
else
    top=" $(jget "$outputs/sweep/$m/stage2.json" "' '.join(t['name'] for t in d['top'])") "
    note "stage 2 outcome: $(jget "$outputs/sweep/$m/stage2.json" "d['outcome']"), TOP:$top"
    for d in "$outputs"/train/*/s0; do
        n=$(basename "$(dirname "$d")")
        [[ "$top" == *" $n "* || "$n" == G || "$n" == *-no_* ]] || pending_add "$d" "seed-0 weights of $n (stages 1 and 2, local fallback)" "not in TOP (rule §4.5 Weights)"
    done
fi
finish
