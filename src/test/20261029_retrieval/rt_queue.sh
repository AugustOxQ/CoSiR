#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON=/root/miniconda3/envs/CoSiR/bin/python
# Smoke rows: each parity half (SMOKE_N / 2 rows) must exceed the largest whitening size, 1,024, or whiten1024
# keeps an arbitrary basis of the fit half's null space and C13 cannot compare it (fix-c13, round log).
SMOKE_N=2100
INPUTS="${RT_OUTPUTS:-/data/SSD2/cosir_r11/20261029_retrieval/outputs}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
MODE=${1:-}
DRY_RUN=0
SCRATCH=

usage() {
    printf 'Usage: %s full [--dry-run] | smoke <scratch_dir> [--dry-run]\n' "${0##*/}" >&2
    exit 2
}

case "$MODE" in
    full)
        shift
        OUT=$INPUTS
        ;;
    smoke)
        shift
        [[ $# -ge 1 ]] || usage
        SCRATCH=$1
        shift
        OUT=$SCRATCH
        ;;
    *) usage ;;
esac
if [[ ${1:-} == --dry-run ]]; then
    DRY_RUN=1
    shift
fi
[[ $# -eq 0 ]] || usage

if [[ $DRY_RUN -eq 0 ]]; then
    cd -- "$SCRIPT_DIR" || exit 2
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
    else
        mkdir -p -- "$SCRATCH" || exit 2
        SCRATCH_REAL=$(cd -- "$SCRATCH" && pwd -P) || exit 2
        INPUTS_REAL=$(cd -- "$INPUTS" 2>/dev/null && pwd -P) || INPUTS_REAL=$(realpath -m -- "$INPUTS")
        [[ $SCRATCH_REAL != "$INPUTS_REAL" ]] || {
            printf 'smoke scratch directory must differ from INPUTS: %s\n' "$INPUTS_REAL" >&2
            exit 2
        }
        OUT=$SCRATCH_REAL
    fi
    mkdir -p -- "$OUT/logs" || exit 2
else
    TOPLEVEL=/project/CoSiR
fi

quote_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_step() {
    local step=$1 logfile=$2
    shift 2
    if [[ $DRY_RUN -eq 1 ]]; then
        quote_command "$@"
        return 0
    fi
    "$@" >"$logfile" 2>&1
    local code=$?
    printf '%s\t%s\t%s\n' "$(TZ=Europe/Amsterdam date '+%F %H:%M')" "$step" "$code" >>"$OUT/queue_status.txt"
    if [[ $code -ne 0 ]]; then
        printf 'Queue failed at %s (exit %s); log: %s\n' "$step" "$code" "$logfile"
        exit "$code"
    fi
}

record_status() {
    [[ $DRY_RUN -eq 1 ]] && return 0
    printf '%s\t%s\t%s\n' "$(TZ=Europe/Amsterdam date '+%F %H:%M')" "$1" "$2" >>"$OUT/queue_status.txt"
}

python_command() {
    local script=$1
    shift
    if [[ $MODE == smoke ]]; then
        quote_command "$PYTHON" "$script" "$@" --smoke "$SMOKE_N" --outputs "$OUT"
    else
        quote_command "$PYTHON" "$script" "$@" --outputs "$OUT"
    fi
}

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'DRY RUN (%s), outputs: %s\n' "$MODE" "$OUT"
else
    printf '%s\tstart\t0\n' "$(TZ=Europe/Amsterdam date '+%F %H:%M')" >>"$OUT/queue_status.txt"
fi

STAGE0=("$PYTHON" rt_stage0.py 0a --outputs "$OUT")
if [[ $MODE == smoke ]]; then STAGE0+=(--smoke "$SMOKE_N"); fi
run_step stage0a "$OUT/logs/stage0a.log" "${STAGE0[@]}"

if [[ $MODE == full ]]; then
    RANGE_FILE=$SCRIPT_DIR/stage0a_ranges.json
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  git -C %q add %q\n' "$TOPLEVEL" "$RANGE_FILE"
        printf '  git -C %q commit -m %q -- %q\n' "$TOPLEVEL" 'r11 run: stage 0a seed ranges, committed before any held retrieval' "$RANGE_FILE"
        printf '  git -C %q push\n' "$TOPLEVEL"
    elif git -C "$TOPLEVEL" ls-files --error-unmatch -- "$RANGE_FILE" >/dev/null 2>&1; then
        # A rerun: the committed r_a stays fixed; 0a writes only rerun-stable keys, so it must reproduce the file byte for byte.
        if git -C "$TOPLEVEL" diff --quiet HEAD -- "$RANGE_FILE"; then
            record_status commit_ranges skipped
        else
            record_status commit_ranges 1
            printf 'Queue failed at commit_ranges: 0a changed the committed %s\n' "$RANGE_FILE"
            exit 1
        fi
    else
        git -C "$TOPLEVEL" add -- "$RANGE_FILE" >"$OUT/logs/commit_ranges.log" 2>&1
        code=$?
        if [[ $code -eq 0 ]]; then
            # Only this path: anything else staged in the checkout stays out of the commit.
            git -C "$TOPLEVEL" commit -m 'r11 run: stage 0a seed ranges, committed before any held retrieval' \
                -- "$RANGE_FILE" >>"$OUT/logs/commit_ranges.log" 2>&1
            code=$?
        fi
        record_status commit_ranges "$code"
        if [[ $code -ne 0 ]]; then
            printf 'Queue failed at commit_ranges (exit %s); log: %s\n' "$code" "$OUT/logs/commit_ranges.log"
            exit "$code"
        fi
        git -C "$TOPLEVEL" push >"$OUT/logs/push_ranges.log" 2>&1
        code=$?
        if [[ $code -ne 0 ]]; then
            record_status push_failed "$code"
            printf 'Push failed (exit %s); continuing; log: %s\n' "$code" "$OUT/logs/push_ranges.log"
        else
            record_status push_ranges 0
        fi
    fi
fi

STAGE0=("$PYTHON" rt_stage0.py 0b --outputs "$OUT")
if [[ $MODE == smoke ]]; then STAGE0+=(--smoke "$SMOKE_N"); fi
run_step stage0b "$OUT/logs/stage0b.log" "${STAGE0[@]}"

check_export_files() {
    local step=$1
    shift
    local missing=0 path
    for path in "$@"; do
        [[ -f $path ]] || { printf '%s\n' "$path"; missing=1; }
    done
    if [[ $missing -ne 0 ]]; then
        printf 'Required derived input files for %s are missing; populate the full inputs before smoke mode.\n' "$step" >&2
        exit 2
    fi
}

if [[ $MODE == smoke ]]; then
    if [[ $DRY_RUN -eq 0 ]]; then
        check_export_files derived_exports \
            "$INPUTS"/percept_val_full/P3__s0.npz "$INPUTS"/percept_val_full/P3__s1.npz \
            "$INPUTS"/percept_val_full/P3__s2.npz "$INPUTS"/percept_val_full/P3__s3.npz "$INPUTS"/percept_val_full/P3__s4.npz \
            "$INPUTS"/crl/lists.json "$INPUTS"/crl/emotion_text.json "$INPUTS"/crl/style_text.json "$INPUTS"/crl/genre_text.json
    else
        printf '  # percept_val, crl_lists, crl_texts skipped (smoke uses existing derived inputs)\n'
    fi
    record_status percept_val skipped
    record_status crl_lists skipped
    record_status crl_texts skipped
else
    PERCEPT_FILES=("$INPUTS"/percept_val_full/P3__s0.npz "$INPUTS"/percept_val_full/P3__s1.npz "$INPUTS"/percept_val_full/P3__s2.npz "$INPUTS"/percept_val_full/P3__s3.npz "$INPUTS"/percept_val_full/P3__s4.npz)
    if [[ -f ${PERCEPT_FILES[0]} && -f ${PERCEPT_FILES[1]} && -f ${PERCEPT_FILES[2]} && -f ${PERCEPT_FILES[3]} && -f ${PERCEPT_FILES[4]} ]]; then
        record_status percept_val skipped
        [[ $DRY_RUN -eq 1 ]] && printf '  # percept_val skipped (all outputs exist)\n'
    else
        run_step percept_val "$OUT/logs/percept_val.log" flock -o -w 3600 -E 75 /tmp/gpu0.lock "$PYTHON" rt_percept_val.py
    fi
    if [[ -f $INPUTS/crl/lists.json ]]; then
        record_status crl_lists skipped
        [[ $DRY_RUN -eq 1 ]] && printf '  # crl_lists skipped (output exists)\n'
    else
        run_step crl_lists "$OUT/logs/crl_lists.log" flock -o -w 3600 -E 75 /tmp/gpu0.lock "$PYTHON" rt_crl.py lists --outputs "$OUT"
    fi
    if [[ -f $INPUTS/crl/emotion_text.json && -f $INPUTS/crl/style_text.json && -f $INPUTS/crl/genre_text.json ]]; then
        record_status crl_texts skipped
        [[ $DRY_RUN -eq 1 ]] && printf '  # crl_texts skipped (all outputs exist)\n'
    else
        run_step crl_texts "$OUT/logs/crl_texts.log" flock -o -w 3600 -E 75 /tmp/gpu0.lock "$PYTHON" rt_crl.py texts --outputs "$OUT"
    fi
fi

# C13 confirms stage 4's picks (rt_vh.py stage4) before stage 5 reads them.
STEPS=(stage1 stage2 stage3 stage4 vh_stage4 stage5 diag vh_stage0a vh_stage1 vh_stage2 vh_stage3 vh_stage5)
for step in "${STEPS[@]}"; do
    if [[ $step == vh_* ]]; then
        VH=("$PYTHON" rt_vh.py "${step#vh_}" --outputs "$OUT")
        if [[ $MODE == smoke ]]; then VH+=(--smoke "$SMOKE_N"); fi
        run_step "$step" "$OUT/logs/$step.log" "${VH[@]}"
        continue
    fi
    stage=$step
    RUN=("$PYTHON" rt_run.py "$stage" --outputs "$OUT")
    if [[ $MODE == smoke ]]; then RUN+=(--smoke "$SMOKE_N"); fi
    run_step "$stage" "$OUT/logs/$stage.log" "${RUN[@]}"
done

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'All steps done.\n'
else
    printf 'All steps done.\n'
fi
