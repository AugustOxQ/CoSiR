#!/usr/bin/env bash
# r18 TPIPS-Embed-Qwen3VL-8B image vectors on node402 (ticket 04; modelled on run_bs_node.sh). One GPU per job:
#   cluster launch --node node402 -- bash scripts/run_om_tpips_node.sh probe
#   cluster launch --node node402 -- bash scripts/run_om_tpips_node.sh gate-probe [--batch N]
#   cluster launch --node node402 -- bash scripts/run_om_tpips_node.sh extract --shard I --of 3 [--batch N] [--set artelingo|mmart]
# probe       env (transformers >= 5.0, peft >= 0.19, tpips 0.1.2, CUDA), both HF snapshots, 3 images through the mapped
#             root, one image embedded; then stops.
# gate-probe  the 48 probe images, 3 factors, one shard, merged: the store gate (iv) compares with the local batch-1 run.
#             Use the same --batch as the shard jobs (gate (iv) pairs host and batch).
# extract     om_tpips.py extract for shard I of N on the sets (default: artelingo, then mmart); a rerun skips finished chunks.
# Python: TPIPS_PY (default the node's CoSiR env). tpips 0.1.2 and peft 0.19.1 come from the input folder tpips_pylibs
#         (pure Python, copied from the local venv) on PYTHONPATH; nothing is installed at run time.
# Hub: a job-local HF_HUB_CACHE with refs/main pinned to the om_paths hashes, over the real hub (see pin_ref).
# Inputs: manifests (om_tpips.py prepare) at node /local/wding/r18_in/tpips_inputs and tpips_pylibs at
#         /local/wding/r18_in/tpips_pylibs (both DATA_MAP), bound at their local paths;
#         images through --image-root-map (local /data/PDD/wikiart_proj/wikiart -> node /var/scratch/wding/Dataset/...);
#         weights from the node HF hub /var/scratch/wding/cache/hub (offline, pinned snapshots).
# Results: node /local/wding/res/r18_tpips (probe: r18_tpips_probe), bound at the local node_out folder; stores under tpips_store/.
# Dry: CLUSTER_DRY_RUN=1 prints the `python main_cosir.py ...` line the cluster check resolves; --plan (last) prints the plan.
set -euo pipefail
usage='usage: bash scripts/run_om_tpips_node.sh probe | gate-probe [--batch N] | extract --shard I --of N [--batch N] [--set artelingo|mmart] [--plan]'
mode=${1:-}
case "$mode" in probe|gate-probe|extract) shift ;; *) echo "$usage" >&2; exit 2 ;; esac
shard= of= batch=1 sets= plan=0 field=
for arg in "$@"; do
    case "$arg" in
        --shard) field=shard ;;
        --of) field=of ;;
        --batch) field=batch ;;
        --set) field=set ;;
        --plan) plan=1; field= ;;
        --*) echo "unknown option $arg; $usage" >&2; exit 2 ;;
        *) case "$field" in
               shard) [[ "$arg" =~ ^[0-9]+$ ]] || { echo "bad shard: $arg" >&2; exit 2; }; shard=$arg ;;
               of) [[ "$arg" =~ ^[1-9][0-9]*$ ]] || { echo "bad --of: $arg" >&2; exit 2; }; of=$arg ;;
               batch) [[ "$arg" =~ ^[1-9][0-9]*$ ]] || { echo "bad --batch: $arg" >&2; exit 2; }; batch=$arg ;;
               set) [[ "$arg" =~ ^(artelingo|mmart)$ ]] || { echo "bad --set: $arg" >&2; exit 2; }; sets=$arg ;;
               *) echo "stray argument $arg; $usage" >&2; exit 2 ;;
           esac; field= ;;
    esac
done
if [[ "$mode" == extract ]]; then
    [[ -n "$shard" && -n "$of" ]] || { echo '--shard and --of are required for extract' >&2; exit 2; }
    (( shard < of )) || { echo "--shard $shard must be below --of $of" >&2; exit 2; }
else
    [[ -z "$shard$of$sets" ]] || { echo '--shard, --of and --set go with extract only' >&2; exit 2; }
fi
[[ "$mode" != probe || "$batch" == 1 ]] || { echo '--batch goes with extract and gate-probe' >&2; exit 2; }
[[ -n "$sets" ]] || sets='artelingo mmart'
CODE=$(pwd -P)
JOB=$(dirname -- "$CODE")
ROUND=src/test/20261103_outside_methods
[[ -f "$CODE/$ROUND/om_tpips.py" ]] || { echo 'run from the job code directory (om_tpips.py missing)' >&2; exit 1; }
# One bound input: the manifests. NODE_IN is where DATA_MAP puts them, LOCAL_IN the path the code reads (om_paths.OUT / tpips_inputs).
NODE_IN=/local/wding/r18_in/tpips_inputs
LOCAL_IN=/data/SSD2/cosir_r18/20261103_outside_methods/outputs/tpips_inputs
NODE_LIBS=/local/wding/r18_in/tpips_pylibs
LOCAL_LIBS=/data/SSD2/cosir_r18/20261103_outside_methods/node_inputs/tpips_pylibs
LOCAL_OUT=/data/SSD2/cosir_r18/20261103_outside_methods/node_out     # OM_OUTPUTS inside the job (not om_paths.DEFAULT_OUT)
case "$mode" in
    probe) NODE_RES=/local/wding/res/r18_tpips_probe ;;
    *) NODE_RES=/local/wding/res/r18_tpips ;;
esac
TPIPS_PY=${TPIPS_PY:-/var/scratch/wding/conda-envs/CoSiR/bin/python}
HUB=/var/scratch/wding/cache/hub
IMG_MAP='/data/PDD/wikiart_proj/wikiart=/var/scratch/wding/Dataset/wikiart_proj/wikiart'
if [[ -n "${CLUSTER_DRY_RUN:-}" ]]; then
    # The line `cluster check` composes with Hydra: the +om_in.* value is an input it must find or sync on the node.
    echo "python main_cosir.py dataset=om_node_cluster ++experiment.results_dir=$NODE_RES +om_in.tpips_inputs_path=$NODE_IN +om_in.tpips_pylibs_path=$NODE_LIBS"
    exit 0
fi
if [[ "$plan" == 1 ]]; then
    echo "PLAN $mode: code=$CODE; shard=${shard:-} of=${of:-} batch=$batch sets=$sets; outputs=$NODE_RES -> $LOCAL_OUT; job-local root/tmp/home/cache under $JOB"
    echo "PLAN python=$TPIPS_PY hub=$HUB image-root-map=$IMG_MAP CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (needs a single GPU index)"
    echo "bind $CODE -> /project/CoSiR"
    echo "bind $NODE_RES -> $LOCAL_OUT"
    echo "bind $NODE_IN -> $LOCAL_IN"
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
[[ -x "$TPIPS_PY" ]] || { echo "Missing python $TPIPS_PY (this script installs nothing)" >&2; exit 1; }
[[ -d "$NODE_IN" ]] || { echo "Missing bind source: $NODE_IN (cluster check --sync-data -- bash scripts/run_om_tpips_node.sh $mode ...)" >&2; exit 1; }
[[ -d "$NODE_LIBS/tpips" ]] || { echo "Missing bind source: $NODE_LIBS (cluster check --sync-data -- bash scripts/run_om_tpips_node.sh $mode ...)" >&2; exit 1; }
mkdir -p "$NODE_RES"
mkdir -p "$JOB/om-tmp" "$JOB/om-home" "$JOB/om-cache" "$JOB/om-mpl"
R=$(mktemp -d -p "$JOB" om-root.XXXXXX)
export NODE_RES LOCAL_OUT NODE_IN LOCAL_IN NODE_LIBS LOCAL_LIBS CODE JOB R TPIPS_PY HUB IMG_MAP ROUND
export TMPDIR="$JOB/om-tmp" MPLCONFIGDIR="$JOB/om-mpl" HOME="$JOB/om-home" XDG_CACHE_HOME="$JOB/om-cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH="$LOCAL_LIBS"
# hf-fetch --revision writes no refs/main, and tpips loads the backbone by repo name: offline that name needs the ref.
# A job-local hub links the real snapshots and pins refs/main to the two om_paths hashes; the probe checks them.
HUB_JOB="$JOB/om-hub"
pin_ref() {
    mkdir -p "$HUB_JOB/$1/refs"
    ln -sfn "$HUB/$1/snapshots" "$HUB_JOB/$1/snapshots"
    printf %s "$2" > "$HUB_JOB/$1/refs/main"
}
pin_ref models--sywang--TPIPS-Embed-Qwen3VL-8B 86a1493c60121e656a1c075bc7aed88f25fcd4d4   # om_paths.TPIPS_SNAPSHOT
pin_ref models--Qwen--Qwen3-VL-Embedding-8B 2c4565515e0f265c6511776e7193b22c0968ddc7      # om_paths.TPIPS_BACKBONE_SNAPSHOT
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE="$HUB_JOB"
export OM_OUTPUTS="$LOCAL_OUT" OM_BATCH="$batch" OM_SETS="$sets" OM_SHARD="$shard" OM_OF="$of"
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
bind_path "$NODE_IN" "$LOCAL_IN"
bind_path "$NODE_LIBS" "$LOCAL_LIBS"
chroot "$R" /bin/bash -s -- "$mode" <<'INNER'
set -euo pipefail
cd /project/CoSiR
T="$TPIPS_PY $ROUND/om_tpips.py"
MAN() { echo "$LOCAL_IN/$1.json"; }
TPIPS_SNAP="$HUB/models--sywang--TPIPS-Embed-Qwen3VL-8B/snapshots/86a1493c60121e656a1c075bc7aed88f25fcd4d4"   # om_paths.TPIPS_SNAPSHOT
echo "OM_NODE $1 host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) batch=$OM_BATCH"
case "$1" in
probe)
    "$TPIPS_PY" - <<'PY'
import importlib.metadata as md, json, os, sys
from pathlib import Path
from PIL import Image
def ver(p): return tuple(int(x) for x in md.version(p).split('.')[:2] if x.isdigit())
problems = []
v = {p: md.version(p) for p in ('torch', 'transformers', 'peft', 'tpips')}
print('versions', v)
if ver('transformers') < (5, 0): problems.append('transformers < 5.0')
if ver('peft') < (0, 19): problems.append('peft < 0.19')
if v['tpips'] != '0.1.2': problems.append('tpips != 0.1.2')
import torch, tpips                                  # noqa: F401  (importable)
print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
if not torch.cuda.is_available(): problems.append('no CUDA')
sys.path.insert(0, os.environ['ROUND'])
import om_paths as P
hub = Path(os.environ['HUB'])
for repo, snap in ((P.TPIPS_REPO, P.TPIPS_SNAPSHOT), (P.TPIPS_BACKBONE_REPO, P.TPIPS_BACKBONE_SNAPSHOT)):
    d = hub / ('models--' + repo.replace('/', '--')) / 'snapshots' / snap
    files = [f for f in d.rglob('*') if f.is_file()] if d.is_dir() else []   # is_file() follows the blob link; layouts differ per repo
    ok = len(files) > 0 and any(f.stat().st_size > 1e9 for f in files)
    gb = sum(f.stat().st_size for f in files) / 1e9 if ok else 0
    print('snapshot', repo, snap[:8], 'ok' if ok else 'MISSING', f'{gb:.1f} GB in {len(files)} files')
    if not ok: problems.append(f'snapshot {repo}@{snap[:8]} missing or incomplete in {hub}')
    ref = Path(os.environ['HF_HUB_CACHE']) / ('models--' + repo.replace('/', '--')) / 'refs' / 'main'
    if ref.read_text() != snap: problems.append(f'job hub ref of {repo} is {ref.read_text()}, not {snap}')
old, new = os.environ['IMG_MAP'].split('=', 1)
man = json.loads(Path(os.environ['LOCAL_IN'], 'probe.json').read_text())
for p in man['image'][:3]:
    assert p.startswith(old), p
    q = new + p[len(old):]
    try:
        with Image.open(q) as im: im.convert('RGB').load(); print('image ok', q, im.size)
    except Exception as e: problems.append(f'image {q}: {e!r}')
for s in ('artelingo', 'mmart'):                     # every image the shard jobs will open, not only the three above
    paths = json.loads(Path(os.environ['LOCAL_IN'], s + '.json').read_text())['image']
    gone = [p for p in paths if not (p.startswith(old) and os.path.exists(new + p[len(old):]))]
    print('images', s, len(paths), 'missing', len(gone), *gone[:3])
    if gone: problems.append(f'{s}: {len(gone)} of {len(paths)} images missing on the node')
print('PROBLEMS' if problems else 'PROBE_ENV_OK', *problems, sep='\n  ')
sys.exit(1 if problems else 0)
PY
    # one image through the model (shard 0 of 1, first row), in a store apart from every real one
    $T extract --set probe --factor all --shard 0 --of 1 --batch 1 --device cuda --limit 1 \
        --manifest "$(MAN probe)" --store-root "$OM_OUTPUTS/tpips_probe_check" --image-root-map "$IMG_MAP" --snapshot-dir "$TPIPS_SNAP"
    echo PROBE_EMBED_OK
    ;;
gate-probe)
    $T extract --set probe --factor all --shard 0 --of 1 --batch "$OM_BATCH" --device cuda \
        --manifest "$(MAN probe)" --store-root "$OM_OUTPUTS/tpips_store/probe" --image-root-map "$IMG_MAP" --snapshot-dir "$TPIPS_SNAP"
    $T merge --set probe --factor all --of 1 --manifest "$(MAN probe)" --store-root "$OM_OUTPUTS/tpips_store/probe"
    ;;
extract)
    for s in $OM_SETS; do
        $T extract --set "$s" --factor all --shard "$OM_SHARD" --of "$OM_OF" --batch "$OM_BATCH" --device cuda \
            --manifest "$(MAN "$s")" --store-root "$OM_OUTPUTS/tpips_store/$s" --image-root-map "$IMG_MAP" --snapshot-dir "$TPIPS_SNAP"
    done
    ;;
esac
echo "OM_NODE $1 done"
INNER
NS
