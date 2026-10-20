#!/usr/bin/env bash
# Launch through cluster: bash scripts/run_ct2_node.sh check|smoke|run --run
# Without --run this prints a local plan and never creates directories or mounts.
set -euo pipefail
mode=${1:-}
case "$mode" in check|smoke|run) ;; *) echo 'usage: bash scripts/run_ct2_node.sh check|smoke|run [--run]' >&2; exit 2;; esac
act=${2:-}
[[ $# -le 2 && ( -z "$act" || "$act" == --run ) ]] || { echo 'unexpected arguments' >&2; exit 2; }
NR=/local/wding/r10_ct2
CODE=$(pwd -P)
JOB=$(dirname -- "$CODE")
manifest="$CODE/scripts/ct2_node_inputs.json"
[[ -f "$manifest" ]] || { echo 'run from the job code directory (manifest missing)' >&2; exit 1; }
if [[ "$act" != --run ]]; then
    echo "PLAN $mode: code=$CODE; node inputs=$NR; job-local root/tmp/home/cache/lock under job dir"
    /usr/bin/python3 - "$manifest" "$NR" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1])):
    print('bind', item.get('source', sys.argv[2]+'/'+item.get('remote','')), '->',item['bind'],
          '('+item['certainty']+')')
PY
    echo "bind $NR/env/lib/python3.11/site-packages/nvidia/cu13/lib -> legacy env lib/python3.10/site-packages/nvidia/cu13/lib"
    echo 'No writes or node contact. Add --run only in the controller-created node job.'
    exit 0
fi
if [[ "$mode" == run && "${CUDA_VISIBLE_DEVICES:-}" != 0 ]]; then
    echo 'REFUSING: the frozen queue pins GPU 0; launch when slot 0 is free (CUDA_VISIBLE_DEVICES must be exactly 0)' >&2
    exit 1
fi
[[ "$CODE" == /local/wding/jobs/*/code && "$JOB" == /local/wding/jobs/* ]] || {
    echo "REFUSING: --run requires /local/wding/jobs/<tag>/code as cwd: $CODE" >&2; exit 1;
}
# Resolve all sources before creating a namespace. Only our private /local root is writable storage.
mapfile -t binds < <(/usr/bin/python3 - "$manifest" "$NR" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1])):
    print(item.get('source',sys.argv[2]+'/'+item.get('remote',''))+'\t'+item['bind']+'\t'+item['certainty'])
PY
)
# feature_extract.py retains a Python 3.10 CUDA-library hint, while the copied
# local env is Python 3.11. Alias only that library directory, preserving code.
binds+=("$NR/env/lib/python3.11/site-packages/nvidia/cu13/lib"$'\t'"/root/miniconda3/envs/CoSiR/lib/python3.10/site-packages/nvidia/cu13/lib"$'\t'certain)
for pair in "${binds[@]}"; do
    source=${pair%%$'\t'*}
    [[ -e "$source" || "${pair##*$'\t'}" != possible ]] || continue
    [[ -e "$source" ]] || { echo "Missing bind source: $source (run das6_sync_ct2.py --inputs or controller data sync)" >&2; exit 1; }
done
mkdir -p "$JOB/ct2-tmp" "$JOB/ct2-home" "$JOB/ct2-cache" "$JOB/ct2-mpl"
R=$(mktemp -d -p "$JOB" ct2-root.XXXXXX)
export NR CODE JOB R
export CT_GPU_LOCK="$JOB/gpu0.lock" TMPDIR="$JOB/ct2-tmp" MPLCONFIGDIR="$JOB/ct2-mpl"
export HOME="$JOB/ct2-home" XDG_CACHE_HOME="$JOB/ct2-cache"
export HF_HOME=/data/SSD2/HF_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CT_PYTHON=/root/miniconda3/envs/CoSiR/bin/python
unset CT2_OUTPUTS COSIR_WIKIART_GENRE_DIR
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
    local source=$1 target=$2 certainty=${3:-certain}
    if [[ ! -e "$source" && "$certainty" == possible ]]; then
        echo "skip possible bind: $target (source absent)"
        return 0
    fi
    [[ -e "$source" ]] || { echo "Missing bind source: $source" >&2; exit 1; }
    if [[ -d "$source" ]]; then mkdir -p "$R$target"
    else mkdir -p "$(dirname "$R$target")"; [[ -e "$R$target" ]] || touch "$R$target"; fi
    mount --bind "$source" "$R$target"
}
bind_path "$JOB/ct2-tmp" /tmp
bind_path /local/wding/Dataset/pre_extract /data/SSD2/pre_extract
bind_path /var/scratch/wding/Dataset/artelingo /data/PDD/artelingo
bind_path "$CODE" /project/CoSiR
bind_path "$NR/buddy" /project/CoSiR-buddy_prototype_conditioning
# Pre-extract is a host bind. Create extra targets in a tmpfs overlay parent,
# not in the controller's shared Dataset folder.
# Reconstruct that parent to keep every newly created mount point job-local.
mount -t tmpfs tmpfs "$R/data/SSD2/pre_extract"
for d in artelingo_full artelingo; do bind_path "/local/wding/Dataset/pre_extract/$d" "/data/SSD2/pre_extract/$d"; done
for pair in "$@"; do
    source=${pair%%$'\t'*}; remainder=${pair#*$'\t'}
    target=${remainder%%$'\t'*}; certainty=${remainder##*$'\t'}
    bind_path "$source" "$target" "$certainty"
done
chroot "$R" /bin/bash -s -- "$mode" <<'INNER'
set -euo pipefail
cd /project/CoSiR
round=src/test/20261020_claim_test
case "$1" in
check)
    "$CT_PYTHON" - <<'PY'
import hashlib,importlib,sys
print('python',sys.version)
for name in ('torch','numpy','sklearn','scipy','igraph','leidenalg'):
    m=importlib.import_module(name)
    print(name,m.__version__)
import torch
print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))
h=hashlib.sha256()
with open('/data/PDD/artelingo/artelingo_train.json','rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
print('artelingo_train.json sha256',h.hexdigest())
PY
    "$CT_PYTHON" scripts/ct2_node_check.py
    ;;
smoke)
    flock -o -w 14400 "$CT_GPU_LOCK" "$CT_PYTHON" "$round/ct2_smoke.py" "$JOB/ct2-smoke" --device cuda
    ;;
run)
    # Preserve exit status while collecting reports even after a failed queue.
    set +e
    bash "$round/ct2_queue_gpu.sh" && bash "$round/ct2_queue_cpu.sh"
    code=$?
    for queue in gpu cpu; do
        status="$round/outputs/stage2/queue_${queue}_status.txt"
        if [[ -f "$status" ]]; then cat "$status"; else echo "status absent: $status"; fi
    done
    mkdir -p "$round/outputs/stage2/reports"
    for name in stage2a.json verdict_stage2.json diag_stage2.json diag_stage2.md; do
        [[ ! -f "$round/$name" ]] || cp -p "$round/$name" "$round/outputs/stage2/reports/$name"
    done
    exit "$code"
    ;;
esac
INNER
NS
