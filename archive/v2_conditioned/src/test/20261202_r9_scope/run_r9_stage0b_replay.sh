#!/usr/bin/env bash
# Stage 0b replay queue (DECISION_RULE.md section 3.2; ticket 12): the CLIP recipe, re-made from scratch by the recipe
# scripts' own command lines, in chain order, into results/replay/clip_b32/ of the checkout that runs it. Each step
# reads the replay's earlier outputs, never CLIP's pinned upstream files. Then run_r9_stage0b.py compares.
#
#   cd <integration checkout>
#   nohup bash src/test/20261202_r9_scope/run_r9_stage0b_replay.sh > src/test/20261202_r9_scope/results/stage0b_replay.log 2>&1 &
#
# Steps (the two GPU steps run under one `flock -n` on /tmp/gpu0.lock, taken once before 07 and released after 08; exit
# 75 = another session holds the GPU, nothing is run or retried; the rest is CPU, CUDA_VISIBLE_DEVICES empty):
#   1 07  r9_recipe_groups.py   GPU  partitions, content graph (GPU, fp16), bank AIC, heads         ~12 min
#   2 08  r9_recipe_a3.py       GPU  A3 run L1 (--inputs = the replay), identity gates, state_equal   ~12 min
#   3 09  r9_recipe_reader.py   CPU  heads, banks, half-readers, tau (--upstream = the replay)       ~14 min
#   4 10  r9_recipe_baselines.py CPU RCA and PM lambdas, B/B0/B1 picks (--partitions-dir, --a3-dir = the replay) ~10 min
#   5 11  run_r9_stage0b.py --make-cells   CPU  cells_seed42.json through ticket 11's functions, replay routed in   ~5 min
# It refuses an existing output folder and prints one summary line at the end.
#
# Environment: OUT (default results/replay/clip_b32 beside this script), A3_RUN (L1; L2 makes the second chain of case 2:
# set OUT to another folder and L1_CKPT to the first chain's a3_L1/A3_seed42.pt, which 08 then compares with).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${R9_PYTHON:-/root/miniconda3/envs/CoSiR/bin/python}"
OUT="${OUT:-$HERE/results/replay/clip_b32}"
GPU_LOCK="${GPU_LOCK:-/tmp/gpu0.lock}"
A3_RUN="${A3_RUN:-L1}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1

fail() { echo "run_r9_stage0b_replay: $*" >&2; exit 2; }

[[ "$A3_RUN" == L1 || "$A3_RUN" == L2 ]] || fail "A3_RUN must be L1 or L2, not $A3_RUN"
[[ ! -e "$OUT" ]] || fail "$OUT exists; a replay folder is never reused (remove it yourself, or set OUT)"
A3_ARGS=()
if [[ "$A3_RUN" == L2 ]]; then
    [[ -n "${L1_CKPT:-}" && -f "$L1_CKPT" ]] || fail "A3_RUN=L2 needs L1_CKPT=<the first chain's a3_L1/A3_seed42.pt>"
    A3_ARGS=(--compare-to "$L1_CKPT")
fi
for f in r9_recipe_groups.py r9_recipe_a3.py r9_recipe_reader.py r9_recipe_baselines.py run_r9_stage0b.py; do
    [[ -f "$HERE/$f" ]] || fail "missing $HERE/$f"
done

T0=$(date +%s)
echo "host $(hostname), checkout $(cd "$HERE/../../.." && pwd), commit $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "output $OUT, A3 run $A3_RUN, started $(TZ=Europe/Amsterdam date '+%F %H:%M')"
echo "GPU compute apps now: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"

step() { echo "== $1 [$(( $(date +%s) - T0 )) s] $(TZ=Europe/Amsterdam date '+%H:%M')"; }

# The GPU lock is taken once for both GPU steps (07 and 08), so no other session can take the GPU between them and make
# 08 exit 75 on a half-made $OUT. fd 9 holds the lock (the same flock(2) lock as `flock -n -o /tmp/gpu0.lock cmd`); the
# children run with fd 9 closed, and it is released before the CPU steps.
exec 9>"$GPU_LOCK"
if ! flock -n 9; then
    echo "run_r9_stage0b_replay: the GPU lock $GPU_LOCK is held by another session" >&2
    exit 75
fi

step "1/5 07 groups, graph, bank, heads"
"$PY" "$HERE/r9_recipe_groups.py" --backbone clip_b32 --out "$OUT" 9>&-

step "2/5 08 A3 $A3_RUN"
"$PY" "$HERE/r9_recipe_a3.py" --backbone clip_b32 --run "$A3_RUN" --inputs "$OUT" --out "$OUT/a3_$A3_RUN" ${A3_ARGS[@]+"${A3_ARGS[@]}"} 9>&-
exec 9>&-                                            # the GPU is free again

export CUDA_VISIBLE_DEVICES=
step "3/5 09 reader and tau"
"$PY" "$HERE/r9_recipe_reader.py" --backbone clip_b32 --stage all --out "$OUT" --upstream "$OUT"

step "4/5 10 RCA, PM lambdas, picks"
"$PY" "$HERE/r9_recipe_baselines.py" --backbone clip_b32 --out "$OUT" --partitions-dir "$OUT" --a3-dir "$OUT/a3_$A3_RUN"

step "5/5 11 cells"
"$PY" "$HERE/run_r9_stage0b.py" --make-cells --replay "$OUT" --a3-run "$A3_RUN"

N=$(find "$OUT" -type f | wc -l)
echo "replay done: $OUT ($N files, A3 run $A3_RUN, $(( $(date +%s) - T0 )) s); next: run_r9_stage0b.py --replay $OUT"
