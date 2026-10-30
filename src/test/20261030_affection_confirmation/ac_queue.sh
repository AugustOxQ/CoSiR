#!/usr/bin/env bash
# r12 run chat's one queue (spec 2026-10-30-affection-confirmation-design.md 3.5 and 9; model: r11's rt_queue.sh).
#   bash ac_queue.sh full [--dry-run]
#   bash ac_queue.sh smoke <scratch_dir> [--dry-run]
# Every step: log OUT/logs/<step>.log, status line in OUT/queue_status.txt (time, step, exit code), seconds in
# OUT/step_seconds.txt; a nonzero exit stops the queue. A step whose sealed output exists is recorded as `skipped`.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
PYTHON=/root/miniconda3/envs/CoSiR/bin/python
DEFAULT_OUT=/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs
# Smoke rows per split: each parity half must exceed the largest whitening size, 1,024 (r11 found it; ac_vh's default).
SMOKE_N=2100
SMOKE_ARGS=(--draws-clu 200 --draws-ret 500)
AFF=/data/SSD2/pre_extract/affection_local
STORE_FI=$AFF/qwen3vl_emb_2b_fi
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
GPU=(flock -o -w 3600 -E 75 /tmp/gpu0.lock)
MODE=${1:-}
DRY_RUN=0
SCRATCH=

usage() {
    printf 'Usage: %s full [--dry-run] | smoke <scratch_dir> [--dry-run]\n' "${0##*/}" >&2
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
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || usage

if [[ $MODE == smoke ]]; then
    [[ $OUT != "$(realpath -m -- "$DEFAULT_OUT")" ]] || {
        printf 'smoke scratch directory must differ from the default OUTPUTS: %s\n' "$DEFAULT_OUT" >&2
        exit 2
    }
    export AC_SMOKE=$SMOKE_N AC_OUTPUTS=$OUT
else
    unset AC_SMOKE AC_OUTPUTS
fi

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
    local logfile=$OUT/logs/$step.log t0=$SECONDS
    "$@" >"$logfile" 2>&1
    local code=$?
    printf '%s\t%s\n' "$step" "$((SECONDS - t0))" >>"$OUT/step_seconds.txt"
    record_status "$step" "$code"
    if [[ $code -ne 0 ]]; then
        printf 'Queue failed at %s (exit %s); log: %s\n' "$step" "$code" "$logfile"
        exit "$code"
    fi
    N_RUN=$((N_RUN + 1))
}

# commit_record <step> <file> <message>: commit only that file (add -f: the repo ignores *.json), then push.
commit_record() {
    local step=$1 file=$2 msg=$3
    if [[ $MODE == smoke ]]; then
        skip_step "$step" 'smoke: the runner writes record copies under the scratch folder'
        return 0
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '# %s\n' "$step"
        printf '  git -C %q add -f -- %q\n' "$TOPLEVEL" "$file"
        printf '  git -C %q commit -m %q -- %q\n' "$TOPLEVEL" "$msg" "$file"
        printf '  git -C %q push\n' "$TOPLEVEL"
        return 0
    fi
    local log=$OUT/logs/$step.log code
    if git -C "$TOPLEVEL" ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
        # A rerun: the committed record stays fixed; the stage must reproduce it byte for byte.
        if git -C "$TOPLEVEL" diff --quiet HEAD -- "$file"; then
            skip_step "$step" 'committed and unchanged'
            return 0
        fi
        record_status "$step" 1
        printf 'Queue failed at %s: the stage changed the committed %s\n' "$step" "$file"
        exit 1
    fi
    git -C "$TOPLEVEL" add -f -- "$file" >"$log" 2>&1
    code=$?
    if [[ $code -eq 0 ]]; then
        git -C "$TOPLEVEL" commit -m "$msg" -- "$file" >>"$log" 2>&1   # only this path, whatever else is staged
        code=$?
    fi
    record_status "$step" "$code"
    if [[ $code -ne 0 ]]; then
        printf 'Queue failed at %s (exit %s); log: %s\n' "$step" "$code" "$log"
        exit "$code"
    fi
    N_RUN=$((N_RUN + 1))
    git -C "$TOPLEVEL" push >"$OUT/logs/push_$step.log" 2>&1
    code=$?
    if [[ $code -ne 0 ]]; then
        record_status "push_$step" "$code"
        printf 'Push failed (exit %s); continuing; log: %s\n' "$code" "$OUT/logs/push_$step.log"
    else
        record_status "push_$step" 0
    fi
}

if [[ $DRY_RUN -eq 1 ]]; then
    printf 'DRY RUN (%s), outputs: %s\n' "$MODE" "$OUT"
else
    record_status start 0
fi

S0A=$OUT/stage0a
ACRUN=("$PYTHON" "$SCRIPT_DIR/ac_run.py")
ACVH=("$PYTHON" "$SCRIPT_DIR/ac_vh.py")
STAGE_FLAGS=()   # ac_run.py and ac_vh.py
VH_FLAGS=()
EXPORT_SMOKE=()
if [[ $MODE == smoke ]]; then
    STAGE_FLAGS=(--smoke "$SMOKE_N" "${SMOKE_ARGS[@]}")
    VH_FLAGS=("${STAGE_FLAGS[@]}" --views-root "$OUT/smoke_views")
    EXPORT_SMOKE=(--smoke "$SMOKE_N")
fi

# ---- 0a(i): FI-only extraction, merge, swap --------------------------------------------------------------------------
if [[ $MODE == smoke ]]; then
    # --limit 64 writes under <scratch>/pre_extract/_smoke/ (never the shared feature folders); it only exercises
    # the GPU path.
    run_step fi_extract "${GPU[@]}" "$PYTHON" "$REPO/scripts/extract_features.py" \
        --dataset affection_fi --backbone qwen3vl_emb_2b --limit 64 --out-root "$OUT/pre_extract"
    skip_step merge_swap 'smoke: needs FI in the store'
else
    # The FI folder is needed only until the merge is sealed; a folder that is already swapped away may be gone.
    SEALS=("$STORE_FI/index.json")
    if [[ $DRY_RUN -eq 0 ]] && sealed "$S0A/store.json"; then SEALS=("$S0A/store.json"); fi
    run_step fi_extract "${GPU[@]}" "$PYTHON" "$REPO/scripts/extract_features.py" \
        --dataset affection_fi --backbone qwen3vl_emb_2b
    SEALS=("$S0A/store.json")
    run_step merge_swap "$PYTHON" "$SCRIPT_DIR/ac_store.py" merge --fi "$STORE_FI" \
        --out "$AFF/qwen3vl_emb_2b_merged" --swap
fi

# ---- 0a(ii): extractor checks; (b) needs the merged store ------------------------------------------------------------
SEALS=("$S0A/extractor_artelingo.json")
run_step parity_a "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_extract.py" parity a
if [[ $MODE == smoke ]]; then
    skip_step parity_b 'smoke: needs FI in the store'
else
    SEALS=("$S0A/extractor_default_affection.json")
    run_step parity_b "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_extract.py" parity b
fi

# ---- the three views (validation and test keys) ----------------------------------------------------------------------
if [[ $MODE == smoke ]]; then
    VIEWS_DIR=$OUT/smoke_views/qwen3vl_emb_2b_aspect_prompts   # the folder names of ac_paths.VIEWS and CLIP_VIEW
    CLIP_DIR=$OUT/smoke_views/clip_l14_valtest
    KEYS_FILE=$OUT/smoke_keys.json
    SEALS=("$KEYS_FILE")
    run_step smoke_keys "$PYTHON" -c '
import json, sys
sys.path.insert(0, sys.argv[1])
import ac_data
n = int(sys.argv[2])
json.dump(ac_data.keys("val")[:n] + ac_data.keys("test")[:n], open(sys.argv[3], "w"))
' "$SCRIPT_DIR" "$SMOKE_N" "$KEYS_FILE"
    KEYS_ARGS=(--keys "$KEYS_FILE")
else
    VIEWS_DIR=$AFF/qwen3vl_emb_2b_aspect_prompts
    CLIP_DIR=$AFF/clip_l14_valtest
    KEYS_ARGS=(--keys valtest)
fi
SEALS=("$VIEWS_DIR/emotion_focus/meta.json")
run_step view_emotion_focus "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_extract.py" extract --view emotion_focus \
    "${KEYS_ARGS[@]}" --out "$VIEWS_DIR/emotion_focus"
SEALS=("$VIEWS_DIR/emotion_focus_photo/meta.json")
run_step view_emotion_focus_photo "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_extract.py" extract --view emotion_focus_photo \
    "${KEYS_ARGS[@]}" --out "$VIEWS_DIR/emotion_focus_photo"
SEALS=("$CLIP_DIR/meta.json")
run_step view_clip_l14 "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_extract.py" extract --view clip_l14 \
    "${KEYS_ARGS[@]}" --out "$CLIP_DIR"

# ---- exports ---------------------------------------------------------------------------------------------------------
PERCEPT_SEALS=()
CSTAR_SEALS=()
for split in val test; do
    for s in 0 1 2 3 4; do
        PERCEPT_SEALS+=("$OUT/exports/percept/$split/P3__s$s.npz")
        for a in emotion style genre; do CSTAR_SEALS+=("$OUT/exports/cstar/$split/${a}__s$s.npz"); done
    done
done
SEALS=("${PERCEPT_SEALS[@]}")
run_step export_percept "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_export.py" percept --device cuda "${EXPORT_SMOKE[@]}"
SEALS=("$S0A/percept_loader.json")   # full-scale ArtELingo held check: it has no --smoke option
run_step export_percept_check "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_export.py" percept-check --device cuda
SEALS=("${CSTAR_SEALS[@]}")
run_step export_cstar "$PYTHON" "$SCRIPT_DIR/ac_export.py" cstar "${EXPORT_SMOKE[@]}"
SEALS=("$S0A/cstar_reinfer.json")
run_step export_reinfer "$PYTHON" "$SCRIPT_DIR/ac_export.py" reinfer "${EXPORT_SMOKE[@]}"
SEALS=("$OUT/exports/cstar/impressions/emotion__s0.npz")
run_step export_impressions "$PYTHON" "$SCRIPT_DIR/ac_export.py" impressions "${EXPORT_SMOKE[@]}"

# ---- CRL texts and check (GPU), 0a(vi) CLIP check ---------------------------------------------------------------------
CRL_SEALS=()
for v in clip_painting_emotion clip_photo_emotion qwen_painting_emotion qwen_photo_emotion \
         qwen_painting_style qwen_painting_genre; do
    CRL_SEALS+=("$OUT/crl/$v.npy" "$OUT/crl/$v.json")
done
SEALS=("${CRL_SEALS[@]}")
run_step crl_texts "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_crl.py" texts
SEALS=("$S0A/crl.json")
run_step crl_check "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_crl.py" check
SEALS=("$S0A/clip_l14_artelingo.json")
run_step parity_c "${GPU[@]}" "$PYTHON" "$SCRIPT_DIR/ac_extract.py" parity c

# ---- stages ----------------------------------------------------------------------------------------------------------
SEALS=("$S0A/gates.json")
run_step stage0a "${ACRUN[@]}" --stage 0a "${STAGE_FLAGS[@]}"

SEALS=("$OUT/stage0b/ranges.json")
run_step stage0b "${ACRUN[@]}" --stage 0b "${STAGE_FLAGS[@]}"
SEALS=("$OUT/vh/0b.json")
run_step vh_0b "${ACVH[@]}" --stage 0b --outputs "$OUT" "${VH_FLAGS[@]}"
commit_record commit_ranges "$SCRIPT_DIR/stage0b_ranges.json" \
    'r12 run: stage 0b seed ranges, committed before any Affection test number'

SEALS=("$OUT/stage1/picks.json")
run_step stage1 "${ACRUN[@]}" --stage 1 "${STAGE_FLAGS[@]}"
SEALS=("$OUT/vh/1.json")
run_step vh_1 "${ACVH[@]}" --stage 1 --outputs "$OUT" "${VH_FLAGS[@]}"
commit_record commit_picks "$SCRIPT_DIR/stage1_picks.json" \
    'r12 run: stage 1 picks, committed before any Affection test number'

SEALS=("$OUT/stage2/verdict.json")
run_step stage2 "${ACRUN[@]}" --stage 2 "${STAGE_FLAGS[@]}"
SEALS=("$OUT/vh/2.json")
run_step vh_2 "${ACVH[@]}" --stage 2 --outputs "$OUT" "${VH_FLAGS[@]}"
SEALS=("$OUT/stage3/verdict.json")
run_step stage3 "${ACRUN[@]}" --stage 3 "${STAGE_FLAGS[@]}"
SEALS=("$OUT/vh/3.json")
run_step vh_3 "${ACVH[@]}" --stage 3 --outputs "$OUT" "${VH_FLAGS[@]}"
SEALS=("$OUT/diag/diag.json")
run_step diag "${ACRUN[@]}" --stage diag "${STAGE_FLAGS[@]}"
SEALS=("$OUT/stage4/impressions.png")
run_step stage4 "${ACRUN[@]}" --stage 4 "${STAGE_FLAGS[@]}"

printf 'All steps done (%s): %d run, %d skipped.\n' "$MODE" "$N_RUN" "$N_SKIP"
