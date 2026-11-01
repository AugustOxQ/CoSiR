#!/usr/bin/env bash
# r13 run chat's one queue (spec 2026-10-31-photo-finetune-design.md 3.5 and 9; model: r12's ac_queue.sh).
#   bash pf_queue.sh full [--dry-run]
#   bash pf_queue.sh smoke <scratch_dir> [--smoke-n N] [--dry-run]       (N = train keys, default 2000)
# Order: extraction (images, texts), stage 0a (its views part checks the views), stage 1, commit the sheet and the
# near-copy record, stage 2, commit the stage 2 record, stage 3, C13 (pf_rederive derive, compare), diag.
# Every step: log OUT/logs/queue_<step>.log, status line in OUT/queue_status.txt (time, step, exit code or skipped),
# seconds in OUT/queue_seconds.txt (pf_run.py keeps its own OUT/logs/<stage>.log and OUT/step_seconds.txt).
# A nonzero exit stops the queue. A step whose sealed output exists is recorded as `skipped`.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
PYTHON=/root/miniconda3/envs/CoSiR/bin/python
DEFAULT_OUT=/data/SSD2/cosir_r13/20261031_photo_finetune/outputs
SMOKE_N=2000
SMOKE_DRAWS=(--draws-clu 200 --draws-ret 500)
VIEWS=/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts   # pf_paths.VIEWS
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
GPU=(flock -o -w 3600 -E 75 /tmp/gpu0.lock)
MODE=${1:-}
DRY_RUN=0
SCRATCH=

usage() {
    printf 'Usage: %s full [--dry-run] | smoke <scratch_dir> [--smoke-n N] [--dry-run]\n' "${0##*/}" >&2
    exit 2
}

case "$MODE" in
    full) shift; OUT=$DEFAULT_OUT ;;
    smoke)
        shift
        [[ $# -ge 1 ]] || usage
        SCRATCH=$1
        shift
        OUT=$(realpath -m -- "$SCRATCH")
        ;;
    *) usage ;;
esac
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --smoke-n)
            [[ $MODE == smoke && $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || usage
            SMOKE_N=$2; shift 2 ;;
        *) usage ;;
    esac
done

unset AC_SMOKE AC_OUTPUTS      # pf_paths asserts both are unset
if [[ $MODE == smoke ]]; then
    [[ $OUT != "$(realpath -m -- "$DEFAULT_OUT")" ]] || {
        printf 'smoke scratch directory must differ from the default OUTPUTS: %s\n' "$DEFAULT_OUT" >&2
        exit 2
    }
    export PF_SMOKE=$SMOKE_N PF_OUTPUTS=$OUT
    RECORDS=$OUT/records
    TRAIN_IMG_VIEW=$OUT/smoke_views/emotion_focus_train
    TXT_VIEW=$OUT/smoke_views/emotion_focus_txt
else
    unset PF_SMOKE PF_OUTPUTS
    RECORDS=$SCRIPT_DIR
    TRAIN_IMG_VIEW=$VIEWS/emotion_focus_train
    TXT_VIEW=$VIEWS/emotion_focus_txt
fi
NEARCOPY_FILE=$RECORDS/stage0a_nearcopy.json
SHEET_FILE=$RECORDS/stage1_sheet.json
STAGE2_FILE=$RECORDS/stage2_records.json

TOPLEVEL=/project/CoSiR
if [[ $DRY_RUN -eq 0 ]]; then
    cd -- "$REPO" || exit 2
    if [[ $MODE == full ]]; then
        TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null) || {
            printf 'full mode requires the main checkout at /project/CoSiR\n' >&2
            exit 2
        }
        BRANCH=$(git -C "$TOPLEVEL" branch --show-current 2>/dev/null) || exit 2
        [[ $TOPLEVEL == /project/CoSiR && $BRANCH == main ]] || {
            printf 'full mode requires checkout /project/CoSiR on branch main\n' >&2
            exit 2
        }
        if [[ -n $(git -C "$TOPLEVEL" status --porcelain --untracked-files=no -- "$SCRIPT_DIR") ]]; then
            printf 'full mode requires no tracked changes in %s\n' "$SCRIPT_DIR" >&2
            exit 2
        fi
    fi
    mkdir -p -- "$OUT/logs" || exit 2
fi

N_RUN=0
N_SKIP=0
SEALS=()

now() { TZ=Europe/Amsterdam date '+%F %H:%M'; }

quote_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

record_status() {
    [[ $DRY_RUN -eq 1 ]] && return 0
    printf '%s\t%s\t%s\n' "$(now)" "$1" "$2" >>"$OUT/queue_status.txt"
}

# A seal is sealed when it exists and, for a .json gate file, holds no failed item ("ok": false).
sealed() {
    [[ $# -gt 0 ]] || return 1
    local f
    for f in "$@"; do
        [[ -f $f ]] || return 1
        if [[ $f == *.json ]] && grep -aq '"ok": false' "$f"; then return 1; fi
    done
    return 0
}

skip_step() {   # skip_step <step> <reason>: a step that does not apply or is already done
    N_SKIP=$((N_SKIP + 1))
    if [[ $DRY_RUN -eq 1 ]]; then printf '  # %s skipped (%s)\n' "$1" "$2"; else record_status "$1" skipped; fi
}

# run_step <step> <cmd...>; set SEALS=(files) before the call to make the step skippable.
run_step() {
    local step=$1
    shift
    local seals=("${SEALS[@]}")
    SEALS=()
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '# %s (%d sealed outputs checked)\n' "$step" "${#seals[@]}"
        quote_command "$@"
        N_RUN=$((N_RUN + 1))
        return 0
    fi
    if sealed "${seals[@]}"; then
        skip_step "$step" 'sealed output exists'
        return 0
    fi
    local logfile=$OUT/logs/queue_$step.log t0=$SECONDS
    "$@" >"$logfile" 2>&1
    local code=$?
    printf '%s\t%s\n' "$step" "$((SECONDS - t0))" >>"$OUT/queue_seconds.txt"
    record_status "$step" "$code"
    if [[ $code -ne 0 ]]; then
        printf 'Queue failed at %s (exit %s); log: %s\n' "$step" "$code" "$logfile"
        exit "$code"
    fi
    N_RUN=$((N_RUN + 1))
}

# commit_records <step> <message> <file...>: full mode, one scoped commit of the files not yet in git (add -f: the repo
# ignores *.json), then push. A rerun: a committed file must be unchanged (a changed one is a failure).
commit_records() {
    local step=$1 msg=$2
    shift 2
    if [[ $MODE == smoke ]]; then
        skip_step "$step" 'smoke: the runner writes record copies under the scratch folder'
        return 0
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '# %s\n' "$step"
        printf '  git -C %q add -f -- %s\n' "$TOPLEVEL" "$(printf '%q ' "$@")"
        printf '  git -C %q commit -m %q -- %s\n' "$TOPLEVEL" "$msg" "$(printf '%q ' "$@")"
        printf '  git -C %q push\n' "$TOPLEVEL"
        N_RUN=$((N_RUN + 1))
        return 0
    fi
    local log=$OUT/logs/queue_$step.log code f todo=()
    for f in "$@"; do
        if git -C "$TOPLEVEL" ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
            if ! git -C "$TOPLEVEL" diff --quiet HEAD -- "$f"; then
                record_status "$step" 1
                printf 'Queue failed at %s: the stage changed the committed %s\n' "$step" "$f"
                exit 1
            fi
        elif [[ -f $f ]]; then
            todo+=("$f")
        else
            record_status "$step" 1
            printf 'Queue failed at %s: %s does not exist\n' "$step" "$f"
            exit 1
        fi
    done
    if [[ ${#todo[@]} -eq 0 ]]; then
        skip_step "$step" 'committed and unchanged'
        return 0
    fi
    git -C "$TOPLEVEL" add -f -- "${todo[@]}" >"$log" 2>&1
    code=$?
    if [[ $code -eq 0 ]]; then
        git -C "$TOPLEVEL" commit -m "$msg" -- "${todo[@]}" >>"$log" 2>&1   # only these paths, whatever else is staged
        code=$?
    fi
    record_status "$step" "$code"
    if [[ $code -ne 0 ]]; then
        printf 'Queue failed at %s (exit %s); log: %s\n' "$step" "$code" "$log"
        exit "$code"
    fi
    N_RUN=$((N_RUN + 1))
    git -C "$TOPLEVEL" push >"$OUT/logs/queue_push_$step.log" 2>&1
    code=$?
    if [[ $code -ne 0 ]]; then
        record_status "push_$step" "$code"
        printf 'Push failed (exit %s); continuing; log: %s\n' "$code" "$OUT/logs/queue_push_$step.log"
    else
        record_status "push_$step" 0
    fi
}

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'DRY RUN (%s), outputs: %s\n' "$MODE" "$OUT"
    if [[ $MODE == smoke ]]; then
        printf 'env: PF_SMOKE=%s PF_OUTPUTS=%s\n' "$PF_SMOKE" "$PF_OUTPUTS"
    else
        printf 'env: PF_SMOKE and PF_OUTPUTS unset\n'
    fi
else
    record_status start 0
fi

PFRUN=("$PYTHON" "$SCRIPT_DIR/pf_run.py")
PFEXTRACT=("$PYTHON" "$SCRIPT_DIR/pf_extract.py")
PFREDERIVE=("$PYTHON" "$SCRIPT_DIR/pf_rederive.py")
EXTRACT_FLAGS=() S0A_FLAGS=() S1_FLAGS=() S2_FLAGS=() S3_FLAGS=() DERIVE_FLAGS=()
if [[ $MODE == smoke ]]; then
    EXTRACT_FLAGS=(--smoke)
    S0A_FLAGS=("${SMOKE_DRAWS[@]}")
    S1_FLAGS=(--rows 1,8 --max-epochs 2)
    S2_FLAGS=(--max-epochs 2)
    S3_FLAGS=("${SMOKE_DRAWS[@]}")
    DERIVE_FLAGS=("${SMOKE_DRAWS[@]}")
fi

# ---- extraction (GPU). The extractor refuses an existing target, so a sealed view is skipped. -----------------------
SEALS=("$TRAIN_IMG_VIEW/meta.json")
run_step extract_images "${GPU[@]}" "${PFEXTRACT[@]}" images "${EXTRACT_FLAGS[@]}"
SEALS=("$TXT_VIEW/meta.json")
run_step extract_texts "${GPU[@]}" "${PFEXTRACT[@]}" texts "${EXTRACT_FLAGS[@]}"

# ---- stage 0a (GPU; its `views` part runs the view check), then stages 1 to 3 --------------------------------------
SEALS=("$OUT/stage0a/gates.json")
run_step stage0a "${GPU[@]}" "${PFRUN[@]}" 0a "${S0A_FLAGS[@]}"

SEALS=("$SHEET_FILE")
run_step stage1 "${GPU[@]}" "${PFRUN[@]}" 1 "${S1_FLAGS[@]}"
if [[ $MODE == smoke ]]; then
    skip_step commit_sheet_nearcopy 'smoke: no commits'
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '# check_records\n  test -f %q -a -f %q\n' "$SHEET_FILE" "$NEARCOPY_FILE"
    elif [[ ! -f $SHEET_FILE || ! -f $NEARCOPY_FILE ]]; then
        record_status check_records 1
        printf 'Queue failed at check_records: missing %s or %s\n' "$SHEET_FILE" "$NEARCOPY_FILE"
        exit 1
    else
        record_status check_records 0
    fi
else
    commit_records commit_sheet_nearcopy \
        'r13 run: stage 1 sheet and near-copy record, committed before any test number' \
        "$SHEET_FILE" "$NEARCOPY_FILE"
fi

SEALS=("$STAGE2_FILE")
run_step stage2 "${GPU[@]}" "${PFRUN[@]}" 2 "${S2_FLAGS[@]}"
commit_records commit_stage2 'r13 run: stage 2 record, committed before any test number' "$STAGE2_FILE"

# Stage 3 and C13 are CPU only: pf_export.infer builds the model on CPU and loads checkpoints with map_location cpu.
SEALS=("$OUT/stage3/verdict.json")
run_step stage3 "${PFRUN[@]}" 3 "${S3_FLAGS[@]}"
SEALS=("$OUT/rederive/theirs.json")
run_step c13_derive "${PFREDERIVE[@]}" derive --workers 8 "${DERIVE_FLAGS[@]}"
SEALS=("$OUT/rederive/rederive.json")
run_step c13_compare "${PFREDERIVE[@]}" compare
SEALS=("$OUT/diag/diag.json")
run_step diag "${PFRUN[@]}" diag

printf 'All steps done (%s): %d run, %d skipped.\n' "$MODE" "$N_RUN" "$N_SKIP"
