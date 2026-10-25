#!/usr/bin/env bash
# Sourced by every eg_queue_*.sh (style of rb_queue_common.sh). Set `queue` before sourcing.
# Env: EG_OUTPUTS (outputs folder; default the round's, on /data/SSD2), EG_SMOKE=1 (scratch outputs under /tmp/claude-0/
#      only), EG_PYTHON, EG_LOCK_WAIT (default 14400 s), EG_GPU_LOCK (default /tmp/gpu0.lock),
#      EG_DEVICE (cuda; "cpu" only for a plumbing test),
#      EG_DRY=1 (print the commands, run none), EG_PENDING (pending deletions list).
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin=${EG_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}
lock_wait=${EG_LOCK_WAIT:-14400}
lock_file=${EG_GPU_LOCK:-/tmp/gpu0.lock}
device=${EG_DEVICE:-cuda}
dry=${EG_DRY:-0}
outputs=${EG_OUTPUTS:-/data/SSD2/cosir_r10/20261025_encoders_genre/outputs}
export EG_OUTPUTS="$outputs"
root=$(cd -- "$here/../../.." && pwd)
pre=/data/SSD2/pre_extract
smoke=()
pending=${EG_PENDING:-/project/CoSiR/.scratch/pending_deletions.md}
if [[ "${EG_SMOKE:-0}" == 1 ]]; then
    smoke=(--smoke)
    case "$(realpath -m "$outputs")" in
        /tmp/claude-0/*) ;;
        *) echo "EG_${queue^^}_STOP smoke needs EG_OUTPUTS under /tmp/claude-0/ (got $outputs)"; exit 2 ;;
    esac
    pending=${EG_PENDING:-$outputs/pending_deletions.md}
fi
mkdir -p "$outputs/logs"
status="$outputs/queue_${queue}_status.txt"
printf 'EG_%s_START %s smoke=%s\n' "${queue^^}" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" "${EG_SMOKE:-0}" | tee -a "$status"
failed=0
step() {   # step LABEL CMD...   one status line; a failed step is recorded, the caller decides to stop
    local label=$1 code start=$SECONDS
    shift
    if [[ "$dry" == 1 ]]; then
        printf 'DRY %s: %s\n' "$label" "$*" | tee -a "$status"
        return 0
    fi
    "$@" > "$outputs/logs/${queue}_${label}.log" 2>&1
    code=$?
    printf '%s=%s seconds=%s\n' "$label" "$code" "$((SECONDS - start))" | tee -a "$status"
    if (( code != 0 )); then
        failed=1
        [[ "${soft:-0}" == 1 ]] || printf 'EG_%s_STOP %s failed (exit %s); log %s/logs/%s_%s.log\n' "${queue^^}" "$label" "$code" "$outputs" "$queue" "$label" | tee -a "$status"
    fi
    return "$code"
}
gpu() {    # GPU step under the lock
    local vis=0
    [[ "$device" == cpu ]] && vis=''
    flock -o -w "$lock_wait" "$lock_file" env CUDA_VISIBLE_DEVICES="$vis" "$python_bin" -u "$@"
}
cpu() {
    env CUDA_VISIBLE_DEVICES='' "$python_bin" -u "$@"
}
run="$here/eg_run.py"
# eg_run.py takes no --smoke (except score3 and lift-score); it only reads files and decides.
gate() {   # gate NAME [--cand K | --stage N | --phase P]: a failed gate returns 1; its line goes to the status file
    local label="gate_$*"
    label=${label//--/}; label=${label// /_}
    local before=$failed code
    soft=1 step "$label" cpu "$run" gate "$@" --outputs "$outputs"
    code=$?
    [[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_$label.log" | tee -a "$status"
    failed=$before   # a gate that does not hold stops what depends on it; the caller decides (and sets failed)
    return "$code"
}
driver() {  # driver ARGS... -> stdout of eg_run.py (read-only lists, including dry runs)
    cpu "$run" "$@" --outputs "$outputs"
}
finish() {
    printf 'EG_%s_DONE failed=%s %s\n' "${queue^^}" "$failed" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee -a "$status"
    exit "$failed"
}
note() { printf 'EG_%s_NOTE %s\n' "${queue^^}" "$*" | tee -a "$status"; }
# --- helpers -------------------------------------------------------------------------------------------------------
store_of() { case "$1" in K6) echo clip_b16;; K7) echo clip_l14;; K8) echo siglip_so400m;; K5) echo dinov2_large;; esac; }
matched_of() { case "$1" in K6) echo M6;; K7) echo M7;; K8) echo M8;; K5) echo M5;; esac; }
all_cands() {   # the candidates minus recorded deferrals, including candidates already scored
    driver candidates 3 --include-done
}
stage_i_stopped() {   # only an existing FAILED record, never an absent gate
    local file
    for file in "$outputs/stores/x0.json" "$outputs"/part2/regression/x1_{student,matched,scorer}.json; do
        [[ "$(jget "$file" "d.get('pass') is False")" == True ]] && return 0
    done
    return 1
}
jget() {   # jget FILE PYEXPR  (d = the loaded JSON) -> value; empty when the file is missing
    [[ -f "$1" ]] || return 0
    "$python_bin" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"
}
