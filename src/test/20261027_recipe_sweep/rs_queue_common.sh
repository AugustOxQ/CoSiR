#!/usr/bin/env bash
# Sourced by every rs_queue_*.sh (the structure of 12-27's rg_queue_common.sh, RS_ prefix, this round's paths). Set `queue`
# before sourcing.
# Env: RS_OUTPUTS (outputs folder; default the round's, on /data/SSD2), RS_SMOKE=1 (scratch outputs under /tmp/claude-0/ only),
#      RS_PYTHON, RS_LOCK_WAIT (default 14400 s), RS_GPU_LOCK (default /tmp/gpu0.lock), RS_DEVICE (cuda; "cpu" only for a
#      plumbing test), RS_WORKERS (scorer workers, default 8), RS_DRY=1 (print every command in order, run none, write nothing),
#      RS_PENDING (pending deletions list), RS_SWEEP_MACHINE (local or node402: the machine whose stage 1 and 2 files stage 3
#      reads; default: local when sweep/local/stage2.json exists, else node402), RS_FORCE_HOUR=1 (skip the weekday rule),
#      RS_DAY_OK=1 (fallback only: the first local run drew under 200 W, so the rest may run on weekday daytime),
#      RS_FAKE_TIME="<ISO weekday 1-7> <hour 0-23>" (tests of the weekday rule only).
set -u
set -o pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin=${RS_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}
lock_wait=${RS_LOCK_WAIT:-14400}
lock_file=${RS_GPU_LOCK:-/tmp/gpu0.lock}
device=${RS_DEVICE:-cuda}
workers=${RS_WORKERS:-8}
dry=${RS_DRY:-0}
outputs=${RS_OUTPUTS:-/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs}
export RS_OUTPUTS="$outputs"
smoke=()
pending=${RS_PENDING:-/project/CoSiR/.scratch/pending_deletions.md}
if [[ "${RS_SMOKE:-0}" == 1 ]]; then
    smoke=(--smoke)
    case "$(realpath -m "$outputs")" in
        /tmp/claude-0/*) ;;
        *) echo "RS_${queue^^}_STOP smoke needs RS_OUTPUTS under /tmp/claude-0/ (got $outputs)"; exit 2 ;;
    esac
    pending=${RS_PENDING:-$outputs/pending_deletions.md}
fi
status="$outputs/queue_${queue}_status.txt"
if [[ "$dry" == 1 ]]; then
    status=/dev/null     # a dry run writes nothing (its DONE lines must never pass for a real one)
else
    mkdir -p "$outputs/logs"
fi
printf 'RS_%s_START %s smoke=%s\n' "${queue^^}" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" "${RS_SMOKE:-0}" | tee -a "$status"
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
        [[ "${soft:-0}" == 1 ]] || printf 'RS_%s_STOP %s failed (exit %s); log %s/logs/%s_%s.log\n' "${queue^^}" "$label" "$code" "$outputs" "$queue" "$label" | tee -a "$status"
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
    printf 'RS_%s_DONE failed=%s %s\n' "${queue^^}" "$failed" "$(TZ=Europe/Amsterdam date '+%F %H:%M')" | tee -a "$status"
    exit "$failed"
}
note() { printf 'RS_%s_NOTE %s\n' "${queue^^}" "$*" | tee -a "$status"; }
stop() { printf 'RS_%s_STOP %s\n' "${queue^^}" "$*" | tee -a "$status"; failed=1; finish; }
jget() {   # jget FILE PYEXPR  (d = the loaded JSON) -> value; empty when the file is missing
    [[ -f "$1" ]] || return 0
    "$python_bin" -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"
}
is_true() { [[ "$(jget "$1" "$2")" == True ]]; }
names() {   # names SUBCOMMAND...: rs_queue_names.py (F, the removal arms, the knobs, the shards)
    "$python_bin" "$here/rs_queue_names.py" "$@"
}
gate() {   # gate NAME FILE WHY: the gate file has pass true, else stop. A dry run notes it; a smoke notes and goes on (its scratch
           # outputs hold no real gates; the runners still refuse what they refuse)
    local name=$1 file=$2 why=$3
    if [[ "$dry" == 1 ]]; then note "dry: gate $name needs pass true in $file"; return 0; fi
    if is_true "$file" "d['pass']"; then note "gate $name holds"; return 0; fi
    if [[ "${RS_SMOKE:-0}" == 1 ]]; then note "smoke: gate $name does not hold in $file (scratch outputs); the chain goes on"; return 0; fi
    stop "gate $name does not hold ($file missing or pass not true): $why"
}
run_status() {   # run_status RUN_JSON -> ok | failed | missing
    local s
    s=$(jget "$1" "d['status']")
    echo "${s:-missing}"
}
check_runs() {   # check_runs TRAIN_ROOT_KIND LABEL NAME... : after a trainer call, note failed runs; echo the unfinished count in $unfinished
    # kind sample: runs under $outputs/train/<name>/s<seed>/run.json for the seeds in $seeds; the trainer records a failed run
    # (non-finite loss, V1b after its one rerun) in run.json: noted, the queue goes on. No run.json: the trainer crashed or refused.
    local name s st
    unfinished=0
    for name in "$@"; do for s in $seeds; do
        st=$(run_status "$outputs/train/$name/s$s/run.json")
        case "$st" in
            ok) ;;
            failed) note "$name s$s FAILED ($(jget "$outputs/train/$name/s$s/run.json" "d.get('cause')")): a recorded outcome, no effect on the other runs" ;;
            *) unfinished=$((unfinished + 1)); note "$name s$s has no finished run.json" ;;
        esac
    done; done
}
hour_now() {   # -> "<ISO weekday> <hour>" in Amsterdam time
    if [[ -n "${RS_FAKE_TIME:-}" ]]; then echo "$RS_FAKE_TIME"; else TZ=Europe/Amsterdam date '+%u %-H'; fi
}
weekday_rule() {   # weekday_rule LABEL EARLIEST_HOUR: refuses on Monday to Friday before EARLIEST_HOUR (19 for the held ablation) unless RS_FORCE_HOUR=1.
                   # The fallback uses the day window 10 to 19 only (weekday_day_window).
    local label=$1 from=$2 dow hh
    read -r dow hh <<< "$(hour_now)"
    if (( dow <= 5 && hh < from )) && [[ "${RS_FORCE_HOUR:-0}" != 1 ]]; then
        if [[ "$dry" == 1 ]]; then note "dry: the weekday rule would refuse $label now (weekday $dow, hour $hh, before $from:00); RS_FORCE_HOUR=1 skips it"; return 0; fi
        stop "$label: the weekday rule (a weekday before $from:00 Amsterdam time; weekday $dow, hour $hh): start it from $from:00 or set RS_FORCE_HOUR=1"
    fi
    note "$label: the weekday rule holds (weekday $dow, hour $hh)"
}
in_day_window() {   # true on Monday to Friday from 10:00 to before 19:00 (the loud hours)
    local dow hh
    read -r dow hh <<< "$(hour_now)"
    (( dow <= 5 && hh >= 10 && hh < 19 ))
}
sweep_machine() {   # the machine of stages 1 and 2 for stage 3 and later: RS_SWEEP_MACHINE, else local when its stage 2 file exists, else node402
    if [[ -n "${RS_SWEEP_MACHINE:-}" ]]; then echo "$RS_SWEEP_MACHINE"
    elif [[ -f "$outputs/sweep/local/stage2.json" || -f "$outputs/sweep/local/stage1.json" ]]; then echo local
    else echo node402; fi
}
pending_add() {  # pending_add DIR WHAT WHY: the *.pt files under DIR -> one row on the pending list (once); nothing is deleted
    local dir=$1 what=$2 why=$3 size stamp
    if [[ "$dry" == 1 ]]; then note "dry: pending row for $dir ($what)"; return 0; fi
    [[ -n "$(find "$dir" -name '*.pt' -print -quit 2>/dev/null)" ]] || return 0
    grep -qF "$dir/" "$pending" 2>/dev/null && return 0
    size=$(find "$dir" -name '*.pt' -print0 | du -ch --files0-from=- 2>/dev/null | tail -n 1 | cut -f1)
    stamp=$(TZ=Europe/Amsterdam date '+%F %H:%M')
    printf '| %s | %s | %s/ (*.pt) | %s | %s | %s; the wrap-up decides, nothing is deleted |\n' \
        "$stamp" "r10 recipe sweep run (rs queue)" "$dir" "${size:-?}" "$what" "$why" >> "$pending"
    note "pending deletion row added: $dir"
}
