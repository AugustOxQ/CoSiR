#!/usr/bin/env bash
# Sourced by every rb_queue_*.sh (style of ct2_queue_common.sh). Set `queue` before sourcing.
# Env: RB_OUTPUTS (outputs folder; default the main checkout's), RB_SMOKE=1 (scratch outputs under /tmp/claude-0/ only),
#      RB_PYTHON, RB_LOCK_WAIT (default 14400 s), RB_GPU_LOCK (default /tmp/gpu0.lock), RB_DEFER ("K5": candidates
#      deferred for time), RB_DEVICE (cuda; "cpu" only for CPU smoke tests of the plumbing), RB_DRY=1 (print the
#      commands, run none).
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin=${RB_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}
lock_wait=${RB_LOCK_WAIT:-14400}
lock_file=${RB_GPU_LOCK:-/tmp/gpu0.lock}
device=${RB_DEVICE:-cuda}
dry=${RB_DRY:-0}
outputs=${RB_OUTPUTS:-/project/CoSiR/src/test/20261025_recipe_backbones/outputs}
export RB_OUTPUTS="$outputs"
smoke=()
if [[ "${RB_SMOKE:-0}" == 1 ]]; then
    smoke=(--smoke)
    case "$(realpath -m "$outputs")" in
        /tmp/claude-0/*) ;;
        *) echo "RB_${queue^^}_STOP smoke needs RB_OUTPUTS under /tmp/claude-0/ (got $outputs)"; exit 2 ;;
    esac
fi
mkdir -p "$outputs/logs"
status="$outputs/queue_${queue}_status.txt"
printf 'RB_%s_START %s smoke=%s\n' "${queue^^}" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" "${RB_SMOKE:-0}" | tee -a "$status"
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
        [[ "${soft:-0}" == 1 ]] || printf 'RB_%s_STOP %s failed (exit %s); log %s/logs/%s_%s.log\n' "${queue^^}" "$label" "$code" "$outputs" "$queue" "$label" | tee -a "$status"
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
run="$here/rb_run.py"
gate() {   # gate NAME   a stop on a failed gate, with its line
    step "gate_$1" cpu "$run" gate "$1" --outputs "$outputs" "${smoke[@]}"
    local code=$?
    [[ "$dry" == 1 ]] || tail -n 1 "$outputs/logs/${queue}_gate_$1.log" | tee -a "$status"
    return "$code"
}
driver() {  # driver ARGS... -> stdout of rb_run.py (dry: a fixed full list)
    if [[ "$dry" == 1 && ( "$1" == candidates || "$1" == triggers ) ]]; then echo "K2 K3 K4 K5"; return 0; fi
    cpu "$run" "$@" --outputs "$outputs" "${smoke[@]}"
}
finish() {
    printf 'RB_%s_DONE failed=%s %s\n' "${queue^^}" "$failed" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee -a "$status"
    exit "$failed"
}
note() { printf 'RB_%s_NOTE %s\n' "${queue^^}" "$*" | tee -a "$status"; }
# --- part 2 helpers ------------------------------------------------------------------------------------------------
enc_of() { case "$1" in K2) echo siglip2_so400m;; K3) echo clip_b32;; K4) echo qwen3vl_emb_2b;; K5) echo dinov2_large;; esac; }
matched_of() { case "$1" in K2) echo M2;; K3) echo M3;; K5) echo M5;; *) echo "";; esac; }   # K4's matched arm is B1 (CT's)
candidate() {  # candidate SCALE K: student (pick recipe), its inference, its matched arm; held and ArtELingo-28 outputs
    local scale=$1 k=$2 m
    step "train_${k}_$scale" gpu "$here/rb_student.py" train --scale "$scale" --encoder "$(enc_of "$k")" --pick --device "$device" --outputs "$outputs" "${smoke[@]}" || return 1
    step "infer_${k}_$scale" gpu "$here/rb_student.py" infer --scale "$scale" --arm "$k" --outputs "$outputs" "${smoke[@]}" || return 1
    m=$(matched_of "$k")
    if [[ -n "$m" ]]; then
        step "matched_${m}_$scale" gpu "$here/rb_matched.py" --scale "$scale" --arm "$m" --device "$device" --outputs "$outputs" "${smoke[@]}" || return 1
    fi
}
