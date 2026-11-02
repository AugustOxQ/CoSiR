#!/usr/bin/env bash
# r14 backbones sweep: node jobs on node402 (modelled on run_rs_node.sh). Launched through the cluster CLI, one GPU per job:
#   cluster launch --node node402 -- bash scripts/run_bs_node.sh probe aux_art aux_aff art_siglip2 aff_siglip2
#   cluster launch --node node402 -- bash scripts/run_bs_node.sh smoke
#   cluster launch --node node402 -- bash scripts/run_bs_node.sh viii
#   cluster launch --node node402 -- bash scripts/run_bs_node.sh prep
#   cluster launch --node node402 -- bash scripts/run_bs_node.sh part-a --encoder siglip2     (also siglip1, clip_l14)
#   cluster launch --node node402 -- bash scripts/run_bs_node.sh sweep --kind photo knob [--reset-claims]
# All modes except probe call src/test/20261102_backbones_sweep/bs_node.py <mode> ... --device cuda in the job's chroot.
# `probe GROUPS` (groups of bs_node_inputs.json) checks every bound input reads, the packages and the GPU, then stops.
# Inputs: DATA_MAP puts each input at its node path (`bs_node_inputs.py overrides`); the chroot binds it at its local path.
# Results: node /local/wding/res/r14_bs (probe: r14_bs_probe, smoke: r14_bs_smoke), bound at the local OUT folder.
# Dry: CLUSTER_DRY_RUN=1 prints the `python main_cosir.py ...` line the cluster check resolves; --plan (last) prints the plan.
set -euo pipefail
usage='usage: bash scripts/run_bs_node.sh smoke|viii|prep | part-a --encoder siglip2|siglip1|clip_l14 | sweep --kind photo|knob... [--reset-claims] | probe GROUP... [--plan]'
mode=${1:-}
case "$mode" in smoke|viii|prep|part-a|sweep|probe) shift ;; *) echo "$usage" >&2; exit 2 ;; esac
encoder= kinds=() groups=() reset=0 plan=0 field=
for arg in "$@"; do
    case "$arg" in
        --encoder) field=encoder ;;
        --kind) field=kind ;;
        --reset-claims) reset=1; field= ;;
        --plan) plan=1; field= ;;
        --*) echo "unknown option $arg; $usage" >&2; exit 2 ;;
        *) case "$field" in
               encoder) [[ "$arg" =~ ^(siglip2|siglip1|clip_l14)$ ]] || { echo "bad encoder: $arg" >&2; exit 2; }; encoder=$arg; field= ;;
               kind) [[ "$arg" =~ ^(photo|knob)$ ]] || { echo "bad kind: $arg" >&2; exit 2; }; kinds+=("$arg") ;;
               *) if [[ "$mode" == probe && "$arg" =~ ^[a-z0-9_]+$ ]]; then groups+=("$arg")
                  else echo "stray argument $arg; $usage" >&2; exit 2; fi ;;
           esac ;;
    esac
done
[[ "$mode" == part-a ]] && { [[ -n "$encoder" ]] || { echo '--encoder is required for part-a' >&2; exit 2; }; }
[[ "$mode" == sweep ]] && { [[ ${#kinds[@]} -gt 0 ]] || { echo '--kind is required for sweep' >&2; exit 2; }; }
[[ "$mode" == probe ]] && { [[ ${#groups[@]} -gt 0 ]] || { echo 'probe needs one or more groups' >&2; exit 2; }; }
[[ "$mode" != part-a && -n "$encoder" ]] && { echo '--encoder goes with part-a only' >&2; exit 2; }
[[ "$mode" != sweep && ( ${#kinds[@]} -gt 0 || "$reset" == 1 ) ]] && { echo '--kind and --reset-claims go with sweep only' >&2; exit 2; }
CODE=$(pwd -P)
JOB=$(dirname -- "$CODE")
HELPER=src/test/20261102_backbones_sweep/bs_node_inputs.py
[[ -f "$CODE/$HELPER" ]] || { echo 'run from the job code directory (inputs helper missing)' >&2; exit 1; }
PY3=/usr/bin/python3; [[ -x "$PY3" ]] || PY3=python3
helper() { "$PY3" "$CODE/$HELPER" "$@"; }
args=(); [[ -n "$encoder" ]] && args+=(--encoder "$encoder"); [[ ${#kinds[@]} -gt 0 ]] && args+=(--kind "${kinds[@]}")
mapfile -t gl < <(helper groups-for "$mode" "${args[@]}" "${groups[@]}")
read -r -a GROUPS_NEEDED <<< "${gl[0]:-}"
[[ ${#GROUPS_NEEDED[@]} -gt 0 ]] || { echo "no input groups resolved for mode $mode" >&2; exit 1; }
case "$mode" in
    smoke) NODE_RES=/local/wding/res/r14_bs_smoke; LOCAL_OUT=/data/SSD2/cosir_r14/20261102_backbones_sweep/smoke_outputs ;;
    probe) NODE_RES=/local/wding/res/r14_bs_probe; LOCAL_OUT=/data/SSD2/cosir_r14/20261102_backbones_sweep/probe_outputs ;;
    *) NODE_RES=/local/wding/res/r14_bs; LOCAL_OUT=/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs ;;   # bs_paths.DEFAULT_OUT
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
    echo "PLAN encoder=${encoder:-} kinds=${kinds[*]:-} reset_claims=$reset CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (needs a single GPU index)"
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
    [[ -e "$source" ]] || { echo "Missing bind source: $source (cluster check --sync-data -- bash scripts/run_bs_node.sh $mode ...)" >&2; exit 1; }
done
mkdir -p "$JOB/bs-tmp" "$JOB/bs-home" "$JOB/bs-cache" "$JOB/bs-mpl"
R=$(mktemp -d -p "$JOB" bs-root.XXXXXX)
export NODE_RES LOCAL_OUT CODE JOB R
export TMPDIR="$JOB/bs-tmp" MPLCONFIGDIR="$JOB/bs-mpl" HOME="$JOB/bs-home" XDG_CACHE_HOME="$JOB/bs-cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BS_PYTHON=/var/scratch/wding/conda-envs/CoSiR/bin/python
export BS_ENCODER="$encoder" BS_KINDS="${kinds[*]:-}" BS_RESET="$reset" BS_GROUPS="${GROUPS_NEEDED[*]}"
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
bind_path "$JOB/bs-tmp" /tmp
bind_path "$CODE" /project/CoSiR
bind_path "$NODE_RES" "$LOCAL_OUT"
for pair in "$@"; do bind_path "${pair%%$'\t'*}" "${pair##*$'\t'}"; done
chroot "$R" /bin/bash -s -- "$mode" <<'INNER'
set -euo pipefail
cd /project/CoSiR
round=src/test/20261102_backbones_sweep
echo "BS_NODE $1 host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) groups=$BS_GROUPS"
case "$1" in
probe)
    BS_PROBE_OUT="$LOCAL_OUT" "$BS_PYTHON" "$round/bs_node_inputs.py" probe $BS_GROUPS
    ;;
smoke)
    BS_OUTPUTS="$LOCAL_OUT" "$BS_PYTHON" "$round/bs_node.py" smoke --device cuda
    ;;
viii|prep)
    BS_OUTPUTS="$LOCAL_OUT" "$BS_PYTHON" "$round/bs_node.py" "$1" --device cuda
    ;;
part-a)
    BS_OUTPUTS="$LOCAL_OUT" "$BS_PYTHON" "$round/bs_node.py" part-a --encoder "$BS_ENCODER" --device cuda
    ;;
sweep)
    extra=(); [[ "$BS_RESET" == 1 ]] && extra=(--reset-claims)
    BS_OUTPUTS="$LOCAL_OUT" "$BS_PYTHON" "$round/bs_node.py" sweep --kind $BS_KINDS "${extra[@]}" --device cuda
    ;;
esac
echo "BS_NODE $1 done"
INNER
NS
