"""Final review of reader-fix round 2: mutation checks. Each mutation edits one committed file of the round in place,
runs the named tests, then restores the file with `git checkout -- <file>` and checks that `git status` is clean for it.
Writes out/fr_mutate.json.

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/final_review/fr_mutate.py
"""
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
D = "src/test/20261118_reader_fix_round2"
PY = "/root/miniconda3/envs/CoSiR/bin/python"
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
       "PYTHONDONTWRITEBYTECODE": "1"}

MUTATIONS = [
    # (id, guard, file, old, new, tests that should fail ("" = the whole file))
    ("F1", "counterpart is condition-free (per-cell assertion): counterpart built from the gated, condition-dependent term",
     "r2_fusion.py", "                sc = _combine(zB, zB, G[t], lu, la)\n",
     "                sc = _combine(zB, zB, gated[t], lu, la)\n",
     ["test_r2_fusion.py::test_counterpart_is_condition_free_and_K_comes_from_B"]),
    ("F1b", "same, with the per-cell condition-free assertion also deleted (does the naive reference still catch it?)",
     "r2_fusion.py", "                sc = _combine(zB, zB, G[t], lu, la)\n",
     "                sc = _combine(zB, zB, gated[t], lu, la)\n",
     ["test_r2_fusion.py::test_counterpart_is_condition_free_and_K_comes_from_B",
      "test_r2_fusion.py::test_cell_statistics_equal_naive_reference"],
     ("                    _require_condition_free(rc_, f\"restricted counterpart (cell {cell_number(kappa, t, u, a)})\")\n",
      "")),
    ("F2", "integer tie-break: the counterpart's max-rho pick takes the LAST tied cell",
     "r2_fusion.py", "        out[half] = int(allowed[int(np.argmax(rho))])\n",
     "        out[half] = int(allowed[len(rho) - 1 - int(np.argmax(rho[::-1]))])\n",
     ["test_r2_fusion.py::test_integer_crossfit_ties_go_to_lowest_cell"]),
    ("F3", "integer criterion: the fused pick compares round 1's float means instead of integers",
     "r2_fusion.py",
     "        crit, _, _ = fused_criterion(fri, fgi, tune_mask(parity, half), ctrl[half][1], allowed)\n",
     "        _t = tune_mask(parity, half); _n = _t.sum()\n"
     "        crit = np.array([min(float(np.mean(fri[i][_t].astype(np.float64) / 4.0)) - ctrl[half][1] / 4.0 / _n,"
     " float(np.mean(fgi[i][_t].astype(np.float64) / 4.0))) for i in allowed])\n",
     ["test_r2_fusion.py::test_integer_criterion_exact_where_float_means_misorder"]),
    ("F4", "restriction: outside candidates placed below the HIGHEST score in K (min -> max); R@1 unaffected",
     "r2_fusion.py", "    low = np.where(inK, S64, np.inf).min(axis=1, keepdims=True)\n",
     "    low = np.where(inK, S64, -np.inf).max(axis=1, keepdims=True)\n",
     ["test_r2_fusion.py::test_restriction_keeps_S_inside_K_and_B_order_outside",
      "test_r2_fusion.py::test_cell_statistics_equal_naive_reference"]),
    ("F5", "D12 clause 3: gain statistic's point instead of its lower bound",
     "r2_apply_rule.py", '    r["clause3"] = bool(r["gain_ci95"][0] > 0)\n',
     '    r["clause3"] = bool(r["gain_statistic"] > 0)\n',
     ["test_r2_fusion.py::test_apply_rule_carry_tie_and_kill"]),
    ("F6", "wiring (no unit test expected): run_r2_fusion's counterpart term is the gated term itself (G_cf skipped)",
     "run_r2_fusion.py", "    G = {t: K.g_cf(gated[t]) for t in g}\n", "    G = {t: gated[t] for t in g}\n",
     ["test_r2_fusion.py"]),
    ("R1", "R2 code check: condition b is never compared",
     "r2_readers.py", "    for c in CONDITIONS:\n        a, b = np.asarray(P_own[c], dtype=np.float64), np.asarray(P_r1[c], dtype=np.float64)\n",
     "    for c in CONDITIONS[:1]:\n        a, b = np.asarray(P_own[c], dtype=np.float64), np.asarray(P_r1[c], dtype=np.float64)\n",
     ["test_r2_readers.py::test_code_check_tolerance_is_1e12_absolute"]),
    ("R2", "R3 draws: a replacement caption may come from the anchor's painting",
     "r2_readers.py", "        bad = (paint[cap] == pa) | (paint[cap] == paint[img])\n",
     "        bad = (paint[cap] == paint[img])\n",
     ["test_r2_readers.py::test_draws_respect_the_painting_constraints"]),
    ("R3", "R3 nested banks: replaced positions depend on k (order rolled by k), so banks are not nested",
     "r2_readers.py", "    mask = replaced_mask(order, k)\n    out = {}\n",
     "    mask = replaced_mask(np.roll(np.asarray(order), k, axis=2), k)\n    out = {}\n",
     ["test_r2_readers.py::test_impure_banks_follow_the_rule_and_are_nested"]),
    ("R4", "wiring (no unit test expected): run_r2_readers runs EM on condition a's rows only",
     "run_r2_readers.py", '    pi_hat, em = RR.em_prior(np.vstack([P["a"], P["b"]]), pi_train)\n',
     '    pi_hat, em = RR.em_prior(P["a"], pi_train)\n',
     ["test_r2_readers.py"]),
]


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def main():
    out = []
    for m in MUTATIONS:
        mid, guard, f, old, new, tests = m[:6]
        extra = m[6] if len(m) > 6 else None
        path = ROOT / D / f
        src = path.read_text()
        assert src.count(old) == 1, (mid, "pattern not unique")
        mutated = src.replace(old, new)
        if extra:
            assert mutated.count(extra[0]) == 1, (mid, "extra pattern not unique")
            mutated = mutated.replace(extra[0], extra[1])
        path.write_text(mutated)
        try:
            res = {}
            for t in tests:
                p = subprocess.run([PY, "-m", "pytest", f"{D}/{t}", "-q", "-p", "no:cacheprovider", "-x"], cwd=ROOT,
                                   env=ENV, capture_output=True, text=True)
                tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:]
                res[t] = {"exit": p.returncode, "failed": p.returncode != 0, "summary": tail[0] if tail else ""}
        finally:
            subprocess.run(["git", "checkout", "--", f"{D}/{f}"], cwd=ROOT, check=True)
        clean = git("status", "--porcelain", "--", f"{D}/{f}").strip() == ""
        out.append({"id": mid, "guard": guard, "file": f, "tests": res, "restored_clean": clean})
        print(mid, {t: (r["failed"], r["summary"]) for t, r in res.items()}, "clean" if clean else "NOT CLEAN", flush=True)
    (HERE / "out" / "fr_mutate.json").write_text(json.dumps(out, indent=1))
    print("git status of the round's tracked files:", repr(git("status", "--porcelain", "--", D, ":!" + D + "/final_review")))


if __name__ == "__main__":
    main()
