#!/usr/bin/env bash
# The node plan of stages 1 and 2 (spec §3.5, D4, D19; rule §4.0, §4.2): prints (RS_DRY=1) or runs, in order, the `cluster` and
# local commands. The real mode is for the run chat; this file was only ever run DRY by its author.
#   bash rs_node_plan.sh stage1   gates n1 and n2 (fetched, pass), the 14:00 deadline, `cluster status`, `cluster sync`, three shards
#                                 of G, G-repeat and the 39 stage 1 runs (one `cluster launch` each, one GPU each), watch, pull, fetch
#                                 the scores; N3 (`run_rs_node.sh n3`); V1c part (a) on the node (`run_rs_node.sh vp`); locally
#                                 `rs_sweep stage1 --machine node402` and V1c part (b) (`rs_vp sweep --machine node402`)
#   bash rs_node_plan.sh stage2   the 22:00 deadline note, the grid's new points (sweep/node402/stage1.json grid.new) in three shards,
#                                 the same launch, watch, pull and fetch; V1c part (a) again on the node (all runs), then locally
#                                 `rs_sweep stage2 --machine node402` and V1c part (b)
# A failed step stops the plan with RS_NODE_STOP; the run chat then decides: relaunch with a NEW tag, or the local fallback
# (rs_queue_fallback.sh; rule §4.2: N1, N2 or N3 fails, stage 1 cannot start by 14:00, stage 2 not ended by 22:00, node lost).
# The launched command has no shell metacharacters (the launch accepts plain `bash scripts/run_*.sh ...`). Each node step adds an
# entry to rs_node_log.md (this plan prints the reminder; it does not write the log). Nothing is deleted anywhere.
# Env: RS_DRY=1, RS_OUTPUTS, RS_PYTHON, RS_CLUSTER (default ~/.claude/skills/cluster-run/cluster), RS_NODE (node402), RS_TAG (tag
#      prefix, default rs<MMDD>), RS_START_BY (14:00 for stage 1), RS_END_BY (22:00 for stage 2), RS_FAKE_NOW (HH:MM, tests only).
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
phase=${1:-}
case "$phase" in stage1|stage2) ;; *) echo 'usage: bash rs_node_plan.sh stage1|stage2   (RS_DRY=1 prints only)' >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1
python_bin=${RS_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}
cluster=${RS_CLUSTER:-$HOME/.claude/skills/cluster-run/cluster}
node=${RS_NODE:-node402}
outputs=${RS_OUTPUTS:-/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs}
dry=${RS_DRY:-0}
tag=${RS_TAG:-rs$(TZ=Europe/Amsterdam date '+%m%d')}
start_by=${RS_START_BY:-14:00}
end_by=${RS_END_BY:-22:00}
sync_py=$root/scripts/das6_sync_rs.py
nodes_runner='bash scripts/run_rs_node.sh'
status=/dev/null
if [[ "$dry" != 1 ]]; then mkdir -p "$outputs/logs"; status="$outputs/queue_node_${phase}_status.txt"; fi
note() { printf 'RS_NODE_NOTE %s\n' "$*" | tee -a "$status"; }
stop() { printf 'RS_NODE_STOP %s\n' "$*" | tee -a "$status"; exit 1; }
run() {   # run LABEL CMD...: print it; run it unless dry; a failure stops the plan
    local label=$1
    shift
    if [[ "$dry" == 1 ]]; then printf 'DRY %s: %s\n' "$label" "$*" | tee -a "$status"; return 0; fi
    printf 'RUN %s: %s\n' "$label" "$*" | tee -a "$status"
    "$@" > "$outputs/logs/node_${phase}_$label.log" 2>&1 || stop "$label failed (log $outputs/logs/node_${phase}_$label.log): relaunch with a new tag, or the local fallback"
}
jget() { [[ -f "$1" ]] || return 0; "$python_bin" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }
now_hhmm() { if [[ -n "${RS_FAKE_NOW:-}" ]]; then echo "$RS_FAKE_NOW"; else TZ=Europe/Amsterdam date '+%H:%M'; fi; }
gate_pass() {   # gate_pass NAME: gates/NAME.json (fetched from the node) has pass true
    if [[ "$dry" == 1 ]]; then note "dry: gates/$1.json needs pass true (N1 and N2 were run by the controller; fetch: python scripts/das6_sync_rs.py --node $node --fetch gates/$1.json --run)"; return 0; fi
    [[ "$(jget "$outputs/gates/$1.json" "d['pass']")" == True ]] || stop "gates/$1.json missing or not pass: the node route fails here, stages 1 and 2 run locally (rs_queue_fallback.sh)"
}
shard_run() {   # shard_run TAGSUFFIX NAME...: one launch of one shard (seed 0) on one GPU
    local suffix=$1; shift
    run "launch_$suffix" "$cluster" launch --node "$node" --tag "$tag-$phase-$suffix" -- $nodes_runner shard --configs "$@" --seeds 0
}
wait_pull() {   # wait_pull TAG: watch to the end (exit 0 = succeeded), then pull the job (logs and the job's results folders)
    run "watch_$1" "$cluster" watch --node "$node" "$1"
    run "pull_$1" "$cluster" pull --node "$node" --tag "$1"
}
fetch() {   # fetch REL...: copy node outputs home (the sync script refuses to overwrite a differing local file)
    run "fetch_$(echo "$1" | tr '/' '_')" "$python_bin" "$sync_py" --node "$node" --fetch "$@" --run
}
one_job() {   # one_job TAGSUFFIX MODE: a single-purpose job (n3, vp) with no arguments
    run "launch_$1" "$cluster" launch --node "$node" --tag "$tag-$phase-$1" -- $nodes_runner "$2"
    wait_pull "$tag-$phase-$1"
}
note "each node step below adds an entry to rs_node_log.md (round folder): the tag, the time, the node's GPUs, what came home"
t=$(now_hhmm)
if [[ "$phase" == stage1 ]]; then
    if [[ "$dry" == 1 ]]; then note "dry: stage 1 must start on the node by $start_by (now $t); later: the local fallback"
    elif [[ "$t" > "$start_by" ]]; then stop "stage 1 cannot start on the node by $start_by (now $t): stages 1 and 2 run locally (rs_queue_fallback.sh)"; fi
    gate_pass n1
    gate_pass n2
    run status "$cluster" status --node "$node"
    run sync "$cluster" sync --node "$node"
    shards=$("$python_bin" "$here/rs_queue_names.py" shards 3)
else
    if [[ "$dry" == 1 ]]; then note "dry: stage 2 must end by $end_by (now $t); after that: the local fallback from the start, the node results reported only"; fi
    gate_pass n3
    run status "$cluster" status --node "$node"
    if [[ "$dry" == 1 ]]; then
        note "dry: the grid's new points are read from $outputs/sweep/$node/stage1.json (grid.new, up to 72) and dealt to 3 shards; the first two names shown are placeholders"
        shards=$'0 POINT_1 POINT_4\n1 POINT_2 POINT_5\n2 POINT_3 POINT_6'
    else
        [[ "$(jget "$outputs/sweep/$node/stage1.json" "d['outcome']")" == SWEEP-NONE ]] && stop "stage 1 gave SWEEP-NONE: stages 2, 3 and 4 do not run"
        points=$(jget "$outputs/sweep/$node/stage1.json" "' '.join(d['grid']['new'])")
        [[ -n "$points" ]] || stop "no new grid points in sweep/$node/stage1.json: run rs_sweep stage1 first"
        shards=$("$python_bin" "$here/rs_queue_names.py" shards 3 $points)
    fi
fi
tags=()
while read -r i names_; do
    [[ -n "$names_" ]] || continue
    suffix=$(printf 'abc' | cut -c$((i + 1)))
    shard_run "$suffix" $names_
    tags+=("$tag-$phase-$suffix")
done <<< "$shards"
for t_ in "${tags[@]}"; do wait_pull "$t_"; done
fetch scores/$node
if [[ "$phase" == stage1 ]]; then
    one_job n3 n3
    fetch gates/n3.json
    if [[ "$dry" != 1 ]]; then [[ "$(jget "$outputs/gates/n3.json" "d['pass']")" == True ]] || stop "N3 fails (G0 and G0' differ by more than half a seed range, or a run failed V1b): stages 1 and 2 run locally (rs_queue_fallback.sh)"; fi
    note "N3 read: gates/n3.json (a pass is required; its parity readings are reported, no gate)"
fi
one_job vp vp
fetch scores/$node
if [[ "$dry" != 1 ]]; then [[ "$(jget "$outputs/scores/$node/rederive_partitions.json" "d['pass']")" == True ]] || stop "V1c part (a) on the node disagrees: trace and fix before $phase is stated"; fi
run "$phase" "$python_bin" "$here/rs_sweep.py" "$phase" --machine "$node" --outputs "$outputs"
run "v1c_b_$phase" "$python_bin" "$here/rs_vp.py" sweep --machine "$node" --outputs "$outputs"
if [[ "$phase" == stage1 ]]; then
    [[ "$dry" == 1 ]] || note "stage 1 outcome: $(jget "$outputs/sweep/$node/stage1.json" "d['outcome']"); SWEEP-NONE ends the route (G stays); otherwise run: bash rs_node_plan.sh stage2"
else
    [[ "$dry" == 1 ]] || note "stage 2 outcome: $(jget "$outputs/sweep/$node/stage2.json" "d['outcome']"), then the stage 3 queue (RS_SWEEP_MACHINE=$node)"
    note "pending list: the node's outputs stay on the node; add one row with their size (cluster du --node $node) to /project/CoSiR/.scratch/pending_deletions.md at the wrap-up; G0 and TOP's weights and exports are copied home before 2026-10-30"
fi
printf 'RS_NODE_DONE %s %s\n' "$phase" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee -a "$status"
