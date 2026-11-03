#!/usr/bin/env bash
# r19 past-picks build: node jobs on node402 (a copy of run_bs_node.sh with r19's modes). Launched through the cluster CLI,
# one GPU per job (use ~/.claude/skills/cluster-run/cluster; the `cluster` on PATH is Graphviz's):
#   cluster launch --queue --node node402 --est 10m --label r19:probe -- bash scripts/run_pp_node.sh probe aux_art aux_aff art_clip_b32 aff_clip_b32 m5
#   cluster launch --queue --node node402 --est 20m --label r19:smoke -- bash scripts/run_pp_node.sh smoke
#   cluster launch --queue --node node402 --est 25m --label r19:ii    -- bash scripts/run_pp_node.sh regress-ii
#   cluster launch --queue --node node402 --est 90m --label r19:cstar_clip_b32 -- bash scripts/run_pp_node.sh part-a --encoder clip_b32
#   cluster launch --queue --node node402 --est 40m --label r19:photo_clip_b32 -- bash scripts/run_pp_node.sh photo --encoder clip_b32
#   cluster launch --queue --node node402 --est 10m --label r19:m5 -- bash scripts/run_pp_node.sh m5
# Modes (python entry points run in the job's chroot):
#   probe GROUPS     every bound input reads, packages import (also pp_node and pp_matched), the GPU shows; then stops
#   smoke            part-a and photo for clip_b32, seed 0, 2 epochs, into r19_pp_smoke (needs clip_b32's Affection store)
#   regress-ii       gate (ii) training half: CLIP L/14 c* seed 0 and photo seed 0 (the comparison runs locally after the pull)
#   part-a --encoder E   X-c*, seeds 0 to 4 (dinov2_large | clip_b32 | clip_b16)
#   photo --encoder E    X photo, seeds 0 to 4, from E's X-c* emotion weights (the part-a job of E must have finished)
#   m5               DINOv2 M5 at full scale, seeds 0 to 4 (pp_matched.py m5 --scale full), CUBLAS_WORKSPACE_CONFIG=:4096:8
# Inputs: DATA_MAP puts each input at its node path (`pp_node_inputs.py overrides`); the chroot binds it at its local path.
# Results: node /local/wding/res/r19_pp (probe: r19_pp_probe, smoke: r19_pp_smoke), bound at the local OUT folder; pull with
#   cluster pull --tag <tag>   into res/r19_pp (a symlink to the real OUT folder, as r14 did).
# Dry: CLUSTER_DRY_RUN=1 prints the `python main_cosir.py ...` line the cluster check resolves; --plan (last) prints the plan.
set -euo pipefail
usage='usage: bash scripts/run_pp_node.sh smoke|regress-ii|m5 | part-a|photo --encoder dinov2_large|clip_b32|clip_b16 | probe GROUP... [--plan]'
mode=${1:-}
case "$mode" in smoke|regress-ii|m5|part-a|photo|probe) shift ;; *) echo "$usage" >&2; exit 2 ;; esac
encoder= groups=() plan=0 field=
for arg in "$@"; do
    case "$arg" in
        --encoder) field=encoder ;;
        --plan) plan=1; field= ;;
        --*) echo "unknown option $arg; $usage" >&2; exit 2 ;;
        *) case "$field" in
               encoder) [[ "$arg" =~ ^(dinov2_large|clip_b32|clip_b16)$ ]] || { echo "bad encoder: $arg" >&2; exit 2; }; encoder=$arg; field= ;;
               *) if [[ "$mode" == probe && "$arg" =~ ^[a-z0-9_]+$ ]]; then groups+=("$arg")
                  else echo "stray argument $arg; $usage" >&2; exit 2; fi ;;
           esac ;;
    esac
done
[[ "$mode" == part-a || "$mode" == photo ]] && { [[ -n "$encoder" ]] || { echo "--encoder is required for $mode" >&2; exit 2; }; }
[[ "$mode" == probe ]] && { [[ ${#groups[@]} -gt 0 ]] || { echo 'probe needs one or more groups' >&2; exit 2; }; }
[[ "$mode" != part-a && "$mode" != photo && -n "$encoder" ]] && { echo '--encoder goes with part-a and photo only' >&2; exit 2; }
CODE=$(pwd -P)
JOB=$(dirname -- "$CODE")
ROUND=src/test/20261103_past_picks
HELPER=$ROUND/pp_node_inputs.py
[[ -f "$CODE/$HELPER" ]] || { echo 'run from the job code directory (inputs helper missing)' >&2; exit 1; }
PY3=/usr/bin/python3; [[ -x "$PY3" ]] || PY3=python3
helper() { "$PY3" "$CODE/$HELPER" "$@"; }
args=(); [[ -n "$encoder" ]] && args+=(--encoder "$encoder")
mapfile -t gl < <(helper groups-for "$mode" "${args[@]}" "${groups[@]}")
read -r -a GROUPS_NEEDED <<< "${gl[0]:-}"
[[ ${#GROUPS_NEEDED[@]} -gt 0 ]] || { echo "no input groups resolved for mode $mode" >&2; exit 1; }
BASE=/data/SSD2/cosir_r19/20261103_past_picks
case "$mode" in
    smoke) NODE_RES=/local/wding/res/r19_pp_smoke; LOCAL_OUT=$BASE/smoke_outputs ;;
    probe) NODE_RES=/local/wding/res/r19_pp_probe; LOCAL_OUT=$BASE/probe_outputs ;;
    *) NODE_RES=/local/wding/res/r19_pp; LOCAL_OUT=$BASE/outputs ;;   # pp_paths.DEFAULT_OUT
esac
if [[ -n "${CLUSTER_DRY_RUN:-}" ]]; then
    # The line `cluster check` composes with Hydra: every +bs_in.*_path value is an input it must find or sync on the node.
    echo "python main_cosir.py dataset=bs_node_cluster ++experiment.results_dir=$NODE_RES $(helper overrides "${GROUPS_NEEDED[@]}")"
    exit 0
fi
mapfile -t binds < <(helper binds "${GROUPS_NEEDED[@]}")
[[ ${#binds[@]} -gt 0 ]] || { echo "no binds resolved for mode $mode" >&2; exit 1; }
if [[ "$plan" == 1 ]]; then
    echo "PLAN $mode: code=$CODE; groups=${GROUPS_NEEDED[*]}; outputs=$NODE_RES -> $LOCAL_OUT; job-local root/tmp/home/cache under $JOB"
    echo "PLAN encoder=${encoder:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (needs a single GPU index)"
    echo "bind $CODE -> /project/CoSiR"
    echo "bind $NODE_RES -> $LOCAL_OUT"
    for pair in "${binds[@]}"; do echo "bind ${pair%%$'\t'*} -> ${pair##*$'\t'}"; done
    echo 'No writes or node contact.'
    exit 0
fi
[[ "$CODE" == /local/wding/jobs/*/code && "$JOB" == /local/wding/jobs/* ]] || {
    echo "REFUSING: a real run needs /local/wding/jobs/<tag>/code as cwd: $CODE" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
    echo "REFUSING: CUDA_VISIBLE_DEVICES must be one GPU index from cluster launch, got '${CUDA_VISIBLE_DEVICES:-unset}'" >&2; exit 1;
}
mkdir -p "$NODE_RES"
for pair in "${binds[@]}"; do
    source=${pair%%$'\t'*}
    [[ -e "$source" ]] || { echo "Missing bind source: $source (cluster check --sync-data -- bash scripts/run_pp_node.sh $mode ...)" >&2; exit 1; }
done
mkdir -p "$JOB/pp-tmp" "$JOB/pp-home" "$JOB/pp-cache" "$JOB/pp-mpl"
R=$(mktemp -d -p "$JOB" pp-root.XXXXXX)
export NODE_RES LOCAL_OUT CODE JOB R
export TMPDIR="$JOB/pp-tmp" MPLCONFIGDIR="$JOB/pp-mpl" HOME="$JOB/pp-home" XDG_CACHE_HOME="$JOB/pp-cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PP_PYTHON=/var/scratch/wding/conda-envs/CoSiR/bin/python
export PP_ENCODER="$encoder" PP_GROUPS="${GROUPS_NEEDED[*]}"
# CUDA_VISIBLE_DEVICES is deliberately inherited from the allocation.
unshare --user --map-root-user --mount bash -s -- "$mode" "${binds[@]}" <<'NS'
set -euo pipefail
mode=$1; shift
mount --make-rprivate /
mount -t tmpfs tmpfs "$R"
for d in bin sbin lib lib64 lib32 usr etc var home local opt dev proc sys run srv cm; do
    [[ -e "/$d" ]] || continue
    if [[ -L "/$d" ]]; then ln -s "$(readlink "/$d")" "$R/$d"
    else mkdir -p "$R/$d"; mount --rbind "/$d" "$R/$d"; fi
done
bind_path() {
    local source=$1 target=$2
    [[ -e "$source" ]] || { echo "Missing bind source: $source" >&2; exit 1; }
    if [[ -d "$source" ]]; then mkdir -p "$R$target"
    else mkdir -p "$(dirname "$R$target")"; [[ -e "$R$target" ]] || touch "$R$target"; fi
    mount --bind "$source" "$R$target"
}
bind_path "$JOB/pp-tmp" /tmp
bind_path "$CODE" /project/CoSiR
bind_path "$NODE_RES" "$LOCAL_OUT"
for pair in "$@"; do bind_path "${pair%%$'\t'*}" "${pair##*$'\t'}"; done
chroot "$R" /bin/bash -s -- "$mode" <<'INNER'
set -euo pipefail
cd /project/CoSiR
round=src/test/20261103_past_picks
echo "PP_NODE $1 host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) groups=$PP_GROUPS"
case "$1" in
probe)
    PP_PROBE_OUT="$LOCAL_OUT" "$PP_PYTHON" "$round/pp_node_inputs.py" probe $PP_GROUPS
    # the node env never imported these modules (their imports pull r14, r15, 10-25 and 10-26 code)
    PP_OUTPUTS="$LOCAL_OUT" "$PP_PYTHON" -c "import sys; sys.path.insert(0, '$round'); import pp_node, pp_matched; print('PP_NODE imports ok')"
    ;;
smoke)
    export PP_OUTPUTS="$LOCAL_OUT" PP_SMOKE=1 BS_NODE_SMOKE=1
    "$PP_PYTHON" "$round/pp_node.py" part-a --encoder clip_b32 --seeds 0 --device cuda
    "$PP_PYTHON" "$round/pp_node.py" photo --encoder clip_b32 --seeds 0 --device cuda
    "$PP_PYTHON" -c "import sys; sys.path.insert(0, '$round'); import pp_matched; print('PP_NODE import pp_matched ok')"
    ;;
regress-ii)
    PP_OUTPUTS="$LOCAL_OUT" "$PP_PYTHON" "$round/pp_node.py" regress-ii --no-compare --device cuda
    ;;
part-a|photo)
    PP_OUTPUTS="$LOCAL_OUT" "$PP_PYTHON" "$round/pp_node.py" "$1" --encoder "$PP_ENCODER" --device cuda
    ;;
m5)
    PP_OUTPUTS="$LOCAL_OUT" CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PP_PYTHON" "$round/pp_matched.py" m5 --scale full --seeds 0 1 2 3 4 --device cuda
    ;;
esac
echo "PP_NODE $1 done"
INNER
NS
