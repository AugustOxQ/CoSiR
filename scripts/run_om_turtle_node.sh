#!/usr/bin/env bash
# r18 TURTLE fits on node402 (ticket 09b; modelled on run_om_tpips_node.sh). One GPU per job:
#   cluster launch --node node402 -- bash scripts/run_om_turtle_node.sh probe
#   cluster launch --node node402 -- bash scripts/run_om_turtle_node.sh fit --set artelingo|mmart --enc qwen|siglip2|siglip1|clip_l14 --arm plain|cstar
# fit    om_turtle_run.py fit --pack on one GPU: the 412 fits are 16 jobs (set x encoder x arm). Inputs and keys come from
#        the pack (om_turtle_run.py pack, local), not from the older rounds' stores; each fit's input sha256 is asserted
#        against the pack manifest. A rerun skips fits already good, so a lost job is relaunched with the same arguments.
# probe  env (torch 2.11, CUDA, torch.special.entr on the GPU, om_data and om_paths import), then ONE full fit:
#        mmart qwen plain K 20 start 0 (6000 outer steps), to compare loss and AMI with the local fit.
# Python: the node's CoSiR env interpreter; all numeric packages come from tpips_pylibs/full on PYTHONPATH (torch 2.11, the
#         version gate (ii) passed on). nvrtc builtins: its nvidia/cu13/lib is PREPENDED to LD_LIBRARY_PATH.
# Inputs: the pack at node /local/wding/r18_in/turtle_pack (DATA_MAP), bound at its local path outputs/turtle_pack;
#         tpips_pylibs/full at /local/wding/r18_in/tpips_pylibs/full (DATA_MAP pair of the TPIPS jobs).
# Results: node /local/wding/res/r18_turtle (probe: r18_turtle_probe) bound as OM_OUTPUTS; fits land in parts/turtle/...,
#          so a pull gives the tree of local OUT/parts/ (check with om_turtle_run.py check-pulled).
# Dry: CLUSTER_DRY_RUN=1 prints the `python main_cosir.py ...` line the cluster check resolves; --plan (last) prints the plan.
set -euo pipefail
usage='usage: bash scripts/run_om_turtle_node.sh probe | fit --set S --enc E --arm plain|cstar [--plan]'
mode=${1:-}
case "$mode" in probe|fit) shift ;; *) echo "$usage" >&2; exit 2 ;; esac
set_= enc= arm= plan=0 field=
for arg in "$@"; do
    case "$arg" in
        --set) field=set ;;
        --enc) field=enc ;;
        --arm) field=arm ;;
        --plan) plan=1; field= ;;
        --*) echo "unknown option $arg; $usage" >&2; exit 2 ;;
        *) case "$field" in
               set) [[ "$arg" =~ ^(artelingo|mmart)$ ]] || { echo "bad --set: $arg" >&2; exit 2; }; set_=$arg ;;
               enc) [[ "$arg" =~ ^(qwen|siglip2|siglip1|clip_l14)$ ]] || { echo "bad --enc: $arg" >&2; exit 2; }; enc=$arg ;;
               arm) [[ "$arg" =~ ^(plain|cstar)$ ]] || { echo "bad --arm: $arg" >&2; exit 2; }; arm=$arg ;;
               *) echo "stray argument $arg; $usage" >&2; exit 2 ;;
           esac; field= ;;
    esac
done
if [[ "$mode" == fit ]]; then
    [[ -n "$set_" && -n "$enc" && -n "$arm" ]] || { echo '--set, --enc and --arm are required for fit' >&2; exit 2; }
else
    [[ -z "$set_$enc$arm" ]] || { echo '--set, --enc and --arm go with fit only' >&2; exit 2; }
fi
CODE=$(pwd -P)
JOB=$(dirname -- "$CODE")
ROUND=src/test/20261103_outside_methods
[[ -f "$CODE/$ROUND/om_turtle_run.py" ]] || { echo 'run from the job code directory (om_turtle_run.py missing)' >&2; exit 1; }
NODE_PACK=/local/wding/r18_in/turtle_pack
LOCAL_PACK=/data/SSD2/cosir_r18/20261103_outside_methods/outputs/turtle_pack
NODE_LIBS=/local/wding/r18_in/tpips_pylibs/full
LOCAL_LIBS=/data/SSD2/cosir_r18/20261103_outside_methods/node_inputs/tpips_pylibs/full
LOCAL_OUT=/data/SSD2/cosir_r18/20261103_outside_methods/node_out     # OM_OUTPUTS inside the job (not om_paths.DEFAULT_OUT)
case "$mode" in
    probe) NODE_RES=/local/wding/res/r18_turtle_probe ;;
    *) NODE_RES=/local/wding/res/r18_turtle ;;
esac
TURTLE_PY=${TURTLE_PY:-/var/scratch/wding/conda-envs/CoSiR/bin/python}
if [[ -n "${CLUSTER_DRY_RUN:-}" ]]; then
    # The line `cluster check` composes with Hydra: each +om_in.* value is an input it must find or sync on the node.
    echo "python main_cosir.py dataset=om_node_cluster ++experiment.results_dir=$NODE_RES +om_in.turtle_pack_path=$NODE_PACK +om_in.tpips_pylibs_path=$NODE_LIBS"
    exit 0
fi
if [[ "$plan" == 1 ]]; then
    echo "PLAN $mode: code=$CODE; set=${set_:-} enc=${enc:-} arm=${arm:-}; outputs=$NODE_RES -> $LOCAL_OUT; job-local root/tmp/home/cache under $JOB"
    echo "PLAN python=$TURTLE_PY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (needs a single GPU index)"
    echo "PLAN LD_LIBRARY_PATH=$LOCAL_LIBS/nvidia/cu13/lib:<inherited>"
    echo "bind $CODE -> /project/CoSiR"
    echo "bind $NODE_RES -> $LOCAL_OUT"
    echo "bind $NODE_PACK -> $LOCAL_PACK"
    echo "bind $NODE_LIBS -> $LOCAL_LIBS (PYTHONPATH)"
    echo 'No writes or node contact.'
    exit 0
fi
[[ "$CODE" == /local/wding/jobs/*/code && "$JOB" == /local/wding/jobs/* ]] || {
    echo "REFUSING: a real run needs /local/wding/jobs/<tag>/code as cwd: $CODE" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || {
    echo "REFUSING: CUDA_VISIBLE_DEVICES must be one GPU index from cluster launch, got '${CUDA_VISIBLE_DEVICES:-unset}'" >&2; exit 1;
}
[[ -x "$TURTLE_PY" ]] || { echo "Missing python $TURTLE_PY (this script installs nothing)" >&2; exit 1; }
[[ -f "$NODE_PACK/manifest.json" ]] || { echo "Missing bind source: $NODE_PACK (cluster check --sync-data -- bash scripts/run_om_turtle_node.sh $mode ...)" >&2; exit 1; }
[[ -d "$NODE_LIBS/torch" ]] || { echo "Missing bind source: $NODE_LIBS (cluster check --sync-data -- bash scripts/run_om_turtle_node.sh $mode ...)" >&2; exit 1; }
mkdir -p "$NODE_RES"
mkdir -p "$JOB/om-tmp" "$JOB/om-home" "$JOB/om-cache" "$JOB/om-mpl"
R=$(mktemp -d -p "$JOB" om-root.XXXXXX)
export NODE_RES LOCAL_OUT NODE_PACK LOCAL_PACK NODE_LIBS LOCAL_LIBS CODE JOB R TURTLE_PY ROUND
export TMPDIR="$JOB/om-tmp" MPLCONFIGDIR="$JOB/om-mpl" HOME="$JOB/om-home" XDG_CACHE_HOME="$JOB/om-cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH="$LOCAL_LIBS"
export LD_LIBRARY_PATH="$LOCAL_LIBS/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"    # PREPENDED: keeps the node's driver path; torch's run-time kernels need libnvrtc-builtins
export WANDB_MODE=disabled
export OM_OUTPUTS="$LOCAL_OUT" OM_SET="$set_" OM_ENC="$enc" OM_ARM="$arm"
# CUDA_VISIBLE_DEVICES is deliberately inherited from the allocation.
unshare --user --map-root-user --mount bash -s -- "$mode" <<'NS'
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
    mkdir -p "$R$target"
    mount --bind "$source" "$R$target"
}
bind_path "$JOB/om-tmp" /tmp
bind_path "$CODE" /project/CoSiR
bind_path "$NODE_RES" "$LOCAL_OUT"
bind_path "$NODE_PACK" "$LOCAL_PACK"
bind_path "$NODE_LIBS" "$LOCAL_LIBS"
chroot "$R" /bin/bash -s -- "$mode" <<'INNER'
set -euo pipefail
cd /project/CoSiR
cd "$ROUND"
echo "OM_NODE $1 host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
case "$1" in
probe)
    "$TURTLE_PY" - <<'PY'
import os, sys
import torch
print('torch', torch.__version__, 'from', torch.__file__)
problems = []
if not torch.__version__.startswith('2.11'): problems.append(f'torch is {torch.__version__}, not 2.11')
if not torch.__file__.startswith(os.environ['LOCAL_LIBS']): problems.append(f'torch not from the input folder: {torch.__file__}')
print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
if not torch.cuda.is_available(): problems.append('no CUDA')
else:
    try:
        y = torch.special.entr(torch.rand(8, device='cuda'))
        torch.cuda.synchronize()
        print('torch.special.entr on the GPU ok', float(y.sum()) > 0)
    except Exception as e: problems.append(f'torch.special.entr on the GPU: {e!r}')
print('LD_LIBRARY_PATH starts with', os.environ['LD_LIBRARY_PATH'].split(':')[0])
try:
    import om_data, om_paths, om_turtle_run
    print('imports ok: om_data, om_paths, om_turtle_run; pack', os.environ['LOCAL_PACK'])
except Exception as e: problems.append(f'import of om_data / om_paths / om_turtle_run: {e!r}')
print('PROBLEMS' if problems else 'PROBE_ENV_OK', *problems, sep='\n  ')
sys.exit(1 if problems else 0)
PY
    # one full fit (6000 outer steps): the loss and AMI are compared with the local fit
    "$TURTLE_PY" om_turtle_run.py fit --pack "$LOCAL_PACK" --set mmart --enc qwen --arm plain --only-k 20 --only-start 0 --device cuda
    echo PROBE_FIT_OK
    ;;
fit)
    "$TURTLE_PY" om_turtle_run.py fit --pack "$LOCAL_PACK" --set "$OM_SET" --enc "$OM_ENC" --arm "$OM_ARM" --device cuda
    ;;
esac
echo "OM_NODE $1 done"
INNER
NS
