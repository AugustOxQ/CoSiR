# Shared by the r21_queue_stage*.sh scripts (source it; not run alone). Ticket 04.
# Paths come from this file's own location, so the scripts run from any checkout (the run chat uses main).
set -euo pipefail
R21_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R21_PY=/root/miniconda3/envs/CoSiR/bin/python
R21_ENV=(env PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8)
R21_PICKS=/data/SSD2/cosir_r21/20261111_dec_heads/scores/stage1_picks.json
R21_DRY=0
R21_STEPS=0

r21_args() {   # --dry-run, --picks FILE
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) R21_DRY=1 ;;
      --picks) shift; R21_PICKS="$1" ;;
      *) echo "usage: $0 [--dry-run] [--picks FILE]" >&2; exit 2 ;;
    esac
    shift
  done
}

# r21_gpu STAGE EST CMD...   a GPU job: waits in the queue, stops the script when it fails
r21_gpu() { local stage="$1" est="$2"; shift 2; r21_step gpu "$stage" "$est" "$@"; }
# r21_cpu STAGE EST CMD...   a CPU job (EST may be "-" for the queue's default)
r21_cpu() { local stage="$1" est="$2"; shift 2; r21_step cpu "$stage" "$est" "$@"; }

r21_step() {
  local kind="$1" stage="$2" est="$3"; shift 3
  local estarg=()
  [ "$est" = "-" ] || estarg=(--est "$est")
  local cmd=(res-queue "$kind" "${estarg[@]}" --label "r21:$stage" -- "${R21_ENV[@]}" "$@")
  R21_STEPS=$((R21_STEPS + 1))
  if [ "$R21_DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
}

R21_RUN=("$R21_PY" "$R21_HERE/r21_run.py")
R21_SCORE=("$R21_PY" "$R21_HERE/r21_score.py")
R21_TABLE=("$R21_PY" "$R21_HERE/r21_table.py")

# r21_pick KEY...  prints a pick from the picks file: stage1a, or stage1b <form>
r21_pick() {
  "$R21_PY" - "$R21_PICKS" "$@" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); level = sys.argv[2]
sel = p[level] if level == 'stage1a' else p[level][sys.argv[3]]
assert sel, f'{sys.argv[1]}: no pick for {sys.argv[2:]}'
print(sel['pick'])
PY
}

r21_done() { echo "r21 $1: $R21_STEPS steps $([ "$R21_DRY" = 1 ] && echo listed '(dry run, nothing run)' || echo finished OK)"; }
