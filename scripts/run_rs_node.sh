#!/usr/bin/env bash
# The recipe sweep's node route (modelled on run_ct2_node.sh). Launched through the cluster CLI, one GPU per job:
#   cluster launch --node node402 -- bash scripts/run_rs_node.sh check                       N1
#   cluster launch --node node402 -- bash scripts/run_rs_node.sh smoke                       N2
#   cluster launch --node node402 -- bash scripts/run_rs_node.sh shard --configs G G-repeat --seeds 0
#   cluster launch --node node402 -- bash scripts/run_rs_node.sh n3                          N3 (rs_sweep n3, on the node's CPU)
# shard: for each config in order, rs_train (finished runs skip) then rs_sweep score on the node's CPU, per seed.
# A config name that fails to parse or a run that raises stops the shard (a failed run is a recorded outcome, not an error).
# Dry: --plan as the last argument prints the plan and touches nothing (no node contact, no mounts).
set -euo pipefail
usage='usage: bash scripts/run_rs_node.sh check|smoke|n3|shard [--configs NAME...] [--seeds N...] [--plan]'
mode=${1:-}
case "$mode" in check|smoke|n3|shard) shift ;; *) echo "$usage" >&2; exit 2 ;; esac
configs=() seeds=() plan=0 field=
for arg in "$@"; do
    case "$arg" in
        --configs) field=configs ;;
        --seeds) field=seeds ;;
        --plan) plan=1 ;;
        --*) echo "unknown option $arg; $usage" >&2; exit 2 ;;
        *) case "$field" in
               configs) [[ "$arg" =~ ^[A-Za-z0-9_.=+-]+$ ]] || { echo "bad config name: $arg" >&2; exit 2; }; configs+=("$arg") ;;
               seeds) [[ "$arg" =~ ^[0-4]$ ]] || { echo "bad seed: $arg" >&2; exit 2; }; seeds+=("$arg") ;;
               *) echo "stray argument $arg; $usage" >&2; exit 2 ;;
           esac ;;
    esac
done
if [[ "$mode" == shard ]]; then
    [[ ${#configs[@]} -gt 0 ]] || { echo '--configs is required for shard' >&2; exit 2; }
    [[ ${#seeds[@]} -gt 0 ]] || seeds=(0)
elif [[ ${#configs[@]} -gt 0 || ${#seeds[@]} -gt 0 ]]; then
    echo "--configs and --seeds go with shard only" >&2; exit 2
fi
NR=/local/wding/r10_rs
OUT=/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs      # rs_paths.OUTPUTS: the node folder is bound here
CODE=$(pwd -P)
JOB=$(dirname -- "$CODE")
manifest="$CODE/scripts/rs_node_inputs.json"
[[ -f "$manifest" ]] || { echo 'run from the job code directory (manifest missing)' >&2; exit 1; }
# One line per bind: SOURCE<TAB>TARGET. Every input is bound at its local path; the items under outputs/ are inside the
# node outputs folder, which is bound as a whole.
bind_list() {
    /usr/bin/python3 - "$manifest" "$NR" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1])):
    if not item.get('outputs'):
        print(sys.argv[2] + '/' + item['remote'] + '\t' + item['local'])
PY
}
if [[ "$plan" == 1 ]]; then
    echo "PLAN $mode: code=$CODE; node inputs=$NR; outputs=$NR/outputs -> $OUT; job-local root/tmp/home/cache under $JOB"
    echo "PLAN configs=${configs[*]:-} seeds=${seeds[*]:-}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (shard, check and smoke need a single GPU index)"
    echo "bind $CODE -> /project/CoSiR"
    echo "bind $NR/outputs -> $OUT"
    bind_list | while IFS=$'\t' read -r source target; do echo "bind $source -> $target"; done
    case "$mode" in
        check) echo 'run: python scripts/rs_node_check.py n1' ;;
        smoke) echo 'run: flock $JOB/gpu0.lock python scripts/rs_node_check.py n2' ;;
        n3) echo 'run: python rs_sweep.py n3 --machine node402' ;;
        shard) for c in "${configs[@]}"; do
                   echo "run: rs_train.py train --configs $c --seeds ${seeds[*]} --machine node402 --device cuda; then per seed S: rs_sweep.py score --machine node402 --runs $c --seed S"
               done ;;
    esac
    echo 'No writes or node contact.'
    exit 0
fi
[[ "$CODE" == /local/wding/jobs/*/code && "$JOB" == /local/wding/jobs/* ]] || {
    echo "REFUSING: a real run needs /local/wding/jobs/<tag>/code as cwd: $CODE" >&2; exit 1;
}
if [[ "$mode" != n3 ]]; then
    # One GPU per job from the cluster allocation: launch sets CUDA_VISIBLE_DEVICES to the slot's device.
    [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
        echo "REFUSING: CUDA_VISIBLE_DEVICES must be one GPU index from cluster launch, got '${CUDA_VISIBLE_DEVICES:-unset}'" >&2; exit 1;
    }
fi
[[ -f "$NR/inputs_manifest.json" ]] || { echo "inputs not synced: $NR/inputs_manifest.json missing (das6_sync_rs.py --inputs --run)" >&2; exit 1; }
mkdir -p "$NR/outputs"
mapfile -t binds < <(bind_list)
for pair in "${binds[@]}"; do
    source=${pair%%$'\t'*}
    [[ -e "$source" ]] || { echo "Missing bind source: $source (run das6_sync_rs.py --inputs --run)" >&2; exit 1; }
done
mkdir -p "$JOB/rs-tmp" "$JOB/rs-home" "$JOB/rs-cache" "$JOB/rs-mpl"
R=$(mktemp -d -p "$JOB" rs-root.XXXXXX)
export NR OUT CODE JOB R
export TMPDIR="$JOB/rs-tmp" MPLCONFIGDIR="$JOB/rs-mpl" HOME="$JOB/rs-home" XDG_CACHE_HOME="$JOB/rs-cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RS_PYTHON=/root/miniconda3/envs/CoSiR/bin/python
export RS_CONFIGS="${configs[*]:-}" RS_SEEDS="${seeds[*]:-}" RS_GPU_LOCK="$JOB/gpu0.lock"
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
bind_path "$JOB/rs-tmp" /tmp
bind_path "$CODE" /project/CoSiR
bind_path "$NR/outputs" "$OUT"
for pair in "$@"; do bind_path "${pair%%$'\t'*}" "${pair##*$'\t'}"; done
chroot "$R" /bin/bash -s -- "$mode" <<'INNER'
set -euo pipefail
cd /project/CoSiR
round=src/test/20261027_recipe_sweep
echo "RS_NODE $1 host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} commit=$(git rev-parse HEAD)"
case "$1" in
check)
    "$RS_PYTHON" scripts/rs_node_check.py n1
    ;;
smoke)
    flock -o -w 14400 "$RS_GPU_LOCK" "$RS_PYTHON" scripts/rs_node_check.py n2
    ;;
n3)
    "$RS_PYTHON" "$round/rs_sweep.py" n3 --machine node402
    ;;
shard)
    for config in $RS_CONFIGS; do
        "$RS_PYTHON" "$round/rs_train.py" train --configs "$config" --seeds $RS_SEEDS --machine node402 --device cuda
        for seed in $RS_SEEDS; do
            "$RS_PYTHON" "$round/rs_sweep.py" score --machine node402 --runs "$config" --seed "$seed"
        done
    done
    echo "RS_NODE shard done: configs=$RS_CONFIGS seeds=$RS_SEEDS"
    ;;
esac
INNER
NS
