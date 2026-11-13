# Sourced by r29_run_stage{0,1,2}.sh (plumbing: no verdict code). Each step is its own process through res-queue.
#   --dry-run            print every command (and every check) without running or enforcing anything
#   R29_FORCE=1          do not skip steps whose output record already exists
# Never passes --allow-uncommitted, --smoke or R29_TEXT_SMOKE; never takes /tmp/gpu0.lock (res-queue does).
RD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$RD" || exit 1
for v in R29_TEXT_SMOKE R29_TEXT_OUT R29_OUTPUTS; do
  [[ -z ${!v:-} ]] || { echo "refuse: $v is set (smoke only)"; exit 2; }
done
OUT=/data/SSD2/cosir_r29/20261112_siglip_only
PYBIN=/root/miniconda3/envs/CoSiR/bin/python
PYC=(env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/data/SSD2/HF_home HF_HUB_OFFLINE=1
     TRANSFORMERS_OFFLINE=1 "$PYBIN")
DRY=0
ARGS=()
for a in "$@"; do if [[ $a == --dry-run ]]; then DRY=1; else ARGS+=("$a"); fi; done
FAILS=()
SKIPPED=()
amsterdam() { TZ=Europe/Amsterdam date '+%F %H:%M'; }

# qrun <gpu|cpu> <short|EST> <label> -- cmd...   (no failure bookkeeping; returns the exit status)
qrun() {
  local kind=$1 est=$2 label=$3; shift 4
  local rq=(res-queue "$kind")
  if [[ $kind == gpu && $est == short ]]; then rq+=(--short); else rq+=(--est "$est"); fi
  rq+=(--label "r29:$label" --)
  if ((DRY)); then printf '[dry-run] '; printf '%q ' "${rq[@]}" "$@"; printf '\n'; return 0; fi
  "${rq[@]}" "$@"
}
# q: qrun, and a failed step goes on the failure list
q() { local label=$3; qrun "$@" || { FAILS+=("$label"); return 1; }; }
# have <file>: true (and noted) when the step's output record exists
have() {
  [[ -e $1 && -z ${R29_FORCE:-} ]] || return 1
  echo "skip (exists): $1"; SKIPPED+=("$1"); return 0
}
jget() { "$PYBIN" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
committed() {   # in git HEAD and unchanged (staged or not)
  git -C "$RD" ls-files --error-unmatch -- "$1" >/dev/null 2>&1 && git -C "$RD" diff --quiet HEAD -- "$1"
}
require_committed() {
  local f bad=0
  for f in "$@"; do
    if committed "$f"; then echo "committed ok: $f"
    else
      bad=1
      if ((DRY)); then echo "[dry-run] NOT committed or missing (a real run refuses): $f"; else echo "REFUSE: $f is not committed unchanged in git HEAD"; fi
    fi
  done
  if ((bad && !DRY)); then exit 2; fi
  return 0
}
# need_committed <piece> <file>...: the non-exiting check of stage 2 (§3.5: "that piece inconclusive; the others go on"). A missing
# or uncommitted record puts the piece on the failure list with the reason and returns 1; the caller skips the piece only.
# A dry run prints what a real run does and returns 0 (so the commands of the piece are still listed).
need_committed() {
  local piece=$1 f bad=() ; shift
  for f in "$@"; do
    if committed "$f"; then echo "committed ok: $f"; else bad+=("$f"); fi
  done
  ((${#bad[@]})) || return 0
  if ((DRY)); then echo "[dry-run] NOT committed or missing: ${bad[*]}; a real run skips piece $piece (failure list) and goes on"; return 0; fi
  echo "SKIP piece $piece: ${bad[*]} not committed unchanged in git HEAD (inconclusive, §3.5); the other pieces go on"
  FAILS+=("$piece: ${bad[*]} missing or not committed")
  return 1
}
# gate_status <gate name>: the status of records/gates/<name>.json, or "missing"
gate_status() { if [[ -f records/gates/$1.json ]]; then jget "records/gates/$1.json" status; else echo missing; fi; }
require_pass() {   # D3
  local s
  if [[ -f records/pass_rule.json ]]; then s=$(jget records/pass_rule.json status); else s=missing; fi
  if [[ $s == pass ]]; then echo "pass_rule.json status: pass"; return 0; fi
  if ((DRY)); then echo "[dry-run] pass_rule.json status is '$s' (a real run refuses)"; return 0; fi
  echo "REFUSE: records/pass_rule.json status is '$s', not pass (D3)"; exit 2
}
kstar() { if [[ -f records/prefix_pick.json ]]; then jget records/prefix_pick.json kstar; else echo '<kstar>'; fi; }
kform() { if [[ -f records/prefix_pick.json ]]; then jget records/prefix_pick.json kstar_form; else echo '<kstar_form>'; fi; }

# done_k <weights dir> <seed>: all three aspect heads of a seed are on disk
done_k() { local a; for a in emotion style genre; do [[ -f $1/${a}__s$2.pt ]] || return 1; done; }
# train_retry <arm> <done_fn> <done_arg> <est> <label> "<seeds>" -- cmd (with the word @SEEDS@)
# D21: a step that exits non-zero, or leaves a seed missing, reruns once for the missing seeds (same seed, same code); a
# second failure writes records/crash_<arm>.json and the caller goes on with the other arms.
train_retry() {
  local arm=$1 donefn=$2 donearg=$3 est=$4 label=$5 seedstr=$6; shift 7
  local -a tmpl=("$@") todo=() left cmd rcs=()
  local s w try rc
  for s in $seedstr; do if $donefn "$donearg" "$s" && [[ -z ${R29_FORCE:-} ]]; then echo "skip (trained): $arm s$s"; else todo+=("$s"); fi; done
  ((${#todo[@]})) || return 0
  for try in 1 2; do
    cmd=(); for w in "${tmpl[@]}"; do if [[ $w == @SEEDS@ ]]; then cmd+=("${todo[@]}"); else cmd+=("$w"); fi; done
    qrun gpu "$est" "$label" -- "${cmd[@]}"; rc=$?; rcs+=("$rc")
    ((DRY)) && return 0
    left=(); for s in "${todo[@]}"; do $donefn "$donearg" "$s" || left+=("$s"); done
    if ((rc == 0 && ${#left[@]} == 0)); then return 0; fi
    echo "train $arm: try $try failed (exit $rc, seeds missing: ${left[*]:-none})"
    ((${#left[@]})) && todo=("${left[@]}")
  done
  "$PYBIN" -I "$RD/r29_crash.py" --arm "$arm" --seed "${todo[0]}" \
    --details "D21: two failures (exit ${rcs[*]}); seeds not finished: ${todo[*]}; command: ${cmd[*]}"
  FAILS+=("train:$arm(crash record)")
  return 1
}
summary() {   # summary <stage name>
  local dry=''; ((DRY)) && dry=' (dry run)'
  echo "r29 $1 done$dry $(amsterdam); failed steps: ${FAILS[*]:-none}; skipped (output exists): ${#SKIPPED[@]}"
  ((${#FAILS[@]} == 0)) || exit 1
}
