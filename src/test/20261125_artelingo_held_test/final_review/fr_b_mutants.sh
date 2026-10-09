#!/bin/bash
# Final review B: mutation checks on a copy of F in the temp dir (MAIN pinned to /project/CoSiR, nothing else changed).
# Each mutant: one copy with one change, then the named tests; the mutant is caught if a test fails.
# usage: fr_b_mutants.sh <temp dir>
set -u
T="$1/mut"
SRC=/project/CoSiR-r6-fr/src/test/20261125_artelingo_held_test
PY=/root/miniconda3/envs/CoSiR/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=

fresh() {   # $1 mutant name -> echo the copy's F path
  local d="$T/$1/src/test/20261125_artelingo_held_test"
  mkdir -p "$d"
  cp "$SRC"/*.py "$SRC"/*.json "$SRC"/DECISION_RULE.md "$d"/ 2>/dev/null
  sed -i 's|^MAIN = main_checkout(HERE)$|MAIN = Path("/project/CoSiR")|' "$d/r6_common.py"
  grep -q '^MAIN = Path("/project/CoSiR")$' "$d/r6_common.py" || echo "PIN FAILED"
  echo "$d"
}

run() {   # $1 name, $2 file, $3 python replace (old), $4 new, $5 test file, $6 -k expr
  local d; d=$(fresh "$1")
  $PY - "$d/$2" "$3" "$4" <<'EOF'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
assert s.count(old) == 1, ("not unique", old)
open(p, "w").write(s.replace(old, new))
EOF
  (cd "$T/$1" && $PY -m pytest -q -p no:cacheprovider -x "$d/$5" -k "$6" > "$T/$1.log" 2>&1)
  local rc=$?
  echo "$1: pytest exit $rc ($( tail -1 "$T/$1.log" ))"
}

# baseline: the unmutated copy passes the same tests
d=$(fresh base)
(cd "$T/base" && $PY -m pytest -q -p no:cacheprovider "$d/test_r6_heads.py" -k "one_ulp or compare_bits" > "$T/base_heads.log" 2>&1); echo "base heads: $? ($(tail -1 $T/base_heads.log))"
(cd "$T/base" && $PY -m pytest -q -p no:cacheprovider "$d/test_r6_episodes.py" -k "value_outside" > "$T/base_eps.log" 2>&1); echo "base episodes: $? ($(tail -1 $T/base_eps.log))"
(cd "$T/base" && $PY -m pytest -q -p no:cacheprovider "$d/test_r6_score.py" -k "parity_mapping" > "$T/base_score.log" 2>&1); echo "base score: $? ($(tail -1 $T/base_score.log))"

run heads_isclose r6_heads.py "differ = got.view(np.uint32) != want.view(np.uint32)" \
    "differ = ~np.isclose(got, want, rtol=1e-6, atol=0)" test_r6_heads.py "one_ulp or compare_bits"
run eps_no_restriction r6_episodes.py "        ok_a = sorted(set(ok_a) & v_a)
        ok_b = sorted(set(ok_b) & v_b)" "        pass" test_r6_episodes.py "value_outside"
run score_parity_swap r6_score.py "return {h: parity == 1 - h for h in HALVES}" \
    "return {h: parity == h for h in HALVES}" test_r6_score.py "parity_mapping"
