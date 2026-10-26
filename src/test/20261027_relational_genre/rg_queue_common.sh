#!/usr/bin/env bash
# Sourced by every rg_queue_*.sh (copy of eg_queue_common.sh with this round's paths). Set `queue` before sourcing.
# Env: RG_OUTPUTS (outputs folder; default the round's, on /data/SSD2), RG_SMOKE=1 (scratch outputs under /tmp/claude-0/
#      only), RG_PYTHON, RG_LOCK_WAIT (default 14400 s), RG_GPU_LOCK (default /tmp/gpu0.lock),
#      RG_DEVICE (cuda; "cpu" only for a plumbing test), RG_WORKERS (scorer workers, default 8),
#      RG_DRY=1 (print every command in order, run none, write nothing), RG_PENDING (pending deletions list).
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin=${RG_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}
lock_wait=${RG_LOCK_WAIT:-14400}
lock_file=${RG_GPU_LOCK:-/tmp/gpu0.lock}
device=${RG_DEVICE:-cuda}
workers=${RG_WORKERS:-8}
dry=${RG_DRY:-0}
outputs=${RG_OUTPUTS:-/data/SSD2/cosir_r10/20261027_relational_genre/outputs}
export RG_OUTPUTS="$outputs"
smoke=()
pending=${RG_PENDING:-/project/CoSiR/.scratch/pending_deletions.md}
if [[ "${RG_SMOKE:-0}" == 1 ]]; then
    smoke=(--smoke)
    case "$(realpath -m "$outputs")" in
        /tmp/claude-0/*) ;;
        *) echo "RG_${queue^^}_STOP smoke needs RG_OUTPUTS under /tmp/claude-0/ (got $outputs)"; exit 2 ;;
    esac
    pending=${RG_PENDING:-$outputs/pending_deletions.md}
fi
status="$outputs/queue_${queue}_status.txt"
if [[ "$dry" == 1 ]]; then
    status=/dev/null     # a dry run writes nothing (its DONE lines must never pass for a real one)
else
    mkdir -p "$outputs/logs"
fi
printf 'RG_%s_START %s smoke=%s\n' "${queue^^}" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" "${RG_SMOKE:-0}" | tee -a "$status"
failed=0
soft=0
step() {   # step LABEL CMD...   one status line; a failed step is recorded, the caller decides to stop
    local label=$1 code start=$SECONDS
    shift
    if [[ "$dry" == 1 ]]; then
        if [[ "$1" == gpu ]]; then
            printf 'DRY %s: flock -o -w %s %s env CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u %s\n' "$label" "$lock_wait" "$lock_file" "${*:2}" | tee -a "$status"
        else
            printf 'DRY %s: python -u %s\n' "$label" "${*:2}" | tee -a "$status"
        fi
        return 0
    fi
    "$@" > "$outputs/logs/${queue}_${label}.log" 2>&1
    code=$?
    printf '%s=%s seconds=%s\n' "$label" "$code" "$((SECONDS - start))" | tee -a "$status"
    if (( code != 0 )); then
        failed=1
        [[ "${soft:-0}" == 1 ]] || printf 'RG_%s_STOP %s failed (exit %s); log %s/logs/%s_%s.log\n' "${queue^^}" "$label" "$code" "$outputs" "$queue" "$label" | tee -a "$status"
    fi
    return "$code"
}
softstep() {   # a step whose failure the caller handles; leaves `failed` as it was
    local before=$failed code
    soft=1 step "$@"
    code=$?
    soft=0
    failed=$before
    return "$code"
}
gpu() {    # GPU step under the lock (the queue waits; never -n)
    local vis=0
    [[ "$device" == cpu ]] && vis=''
    flock -o -w "$lock_wait" "$lock_file" env CUDA_VISIBLE_DEVICES="$vis" "$python_bin" -u "$@"
}
cpu() {
    env CUDA_VISIBLE_DEVICES='' "$python_bin" -u "$@"
}
finish() {
    printf 'RG_%s_DONE failed=%s %s\n' "${queue^^}" "$failed" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee -a "$status"
    exit "$failed"
}
note() { printf 'RG_%s_NOTE %s\n' "${queue^^}" "$*" | tee -a "$status"; }
stop() { printf 'RG_%s_STOP %s\n' "${queue^^}" "$*" | tee -a "$status"; failed=1; finish; }
jget() {   # jget FILE PYEXPR  (d = the loaded JSON) -> value; empty when the file is missing
    [[ -f "$1" ]] || return 0
    "$python_bin" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"
}
is_true() { [[ "$(jget "$1" "$2")" == True ]]; }
last_done() {   # last_done QUEUE -> the last RG_<QUEUE>_DONE line of that queue's status file ("" when none)
    local file="$outputs/queue_$1_status.txt"
    [[ -f "$file" ]] && grep "^RG_${1^^}_DONE " "$file" | tail -n 1
    return 0
}
queue_ok() {   # queue_ok QUEUE: its last run ended without a stop
    [[ "$(last_done "$1")" == "RG_${1^^}_DONE failed=0 "* ]]
}
regression_gate() {  # regression_gate NAME FILE...: every file has pass true. A smoke cannot pass a regression (2 epochs): it notes and goes on
    local name=$1 f; shift
    if [[ "$dry" == 1 ]]; then note "dry: gate $name needs pass true in: $*"; return 0; fi
    for f in "$@"; do
        if ! is_true "$f" "d['pass']"; then
            if [[ "${RG_SMOKE:-0}" == 1 ]]; then note "smoke: $name does not hold in $f (expected at 2 epochs); the chain goes on"; return 0; fi
            return 1
        fi
    done
    note "gate $name holds"
}
run_status() {   # run_status RUN_JSON -> ok | failed | missing
    local s
    s=$(jget "$1" "d['status']")
    echo "${s:-missing}"
}
train_arms() {  # train_arms "SEEDS" ARM...: one trainer call. A run that FAILED (non-finite loss, or VA3 after its one rerun, both done
                # by the trainer) is recorded in its run.json and noted, the queue goes on (rule "Failed runs"); a run without a
                # run.json means the trainer crashed: stop.
    local seeds=$1 arm s label st bad=0 code; shift
    label="$(IFS=_; echo "$*")_s${seeds// /}"
    softstep "train_$label" gpu "$here/rg_train.py" train --arms "$@" --seeds $seeds --device "$device" --outputs "$outputs" "${smoke[@]}"
    code=$?
    if [[ "$dry" == 1 || $code == 0 ]]; then return 0; fi
    for arm in "$@"; do for s in $seeds; do
        st=$(run_status "$outputs/train/$arm/s$s/run.json")
        case "$st" in
            failed) note "$arm s$s FAILED ($(jget "$outputs/train/$arm/s$s/run.json" "d.get('cause')")): no A_genre, or not confirmed; the scorer reads run.json; reported" ;;
            ok) ;;
            *) bad=1; note "$arm s$s has no finished run.json" ;;
        esac
    done; done
    (( bad == 0 )) || stop "training $label crashed (log $outputs/logs/${queue}_train_$label.log)"
}
pending_add() {  # pending_add DIR WHAT WHY: the *.pt files under DIR -> one row on the pending list (once); nothing is deleted
    local dir=$1 what=$2 why=$3 size stamp
    if [[ "$dry" == 1 ]]; then note "dry: pending row for $dir ($what)"; return 0; fi
    [[ -n "$(find "$dir" -name '*.pt' -print -quit 2>/dev/null)" ]] || return 0
    grep -qF "$dir/" "$pending" 2>/dev/null && return 0
    size=$(find "$dir" -name '*.pt' -print0 | du -ch --files0-from=- 2>/dev/null | tail -n 1 | cut -f1)
    stamp=$(TZ=Europe/Amsterdam date '+%F %H:%M')
    printf '| %s | %s | %s/ (*.pt) | %s | %s | %s; the wrap-up decides, nothing is deleted |\n' \
        "$stamp" "r10 build 25 run (relational genre queue)" "$dir" "${size:-?}" "$what" "$why" >> "$pending"
    note "pending deletion row added: $dir"
}
