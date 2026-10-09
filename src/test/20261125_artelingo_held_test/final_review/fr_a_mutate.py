"""Final review A: semantic mutations (weakenings, not deletions) of the verdict-path code, each on a copy of this folder
in a temp dir (GIT_DIR points r6_common at the main checkout), with the folder's own test file run against the copy.
"caught" = the test file fails on the mutant. Also one unmutated control per test file.

    python final_review/fr_a_mutate.py <tmp dir>      -> fr_a_mutate.json beside this file, one line per mutant
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

F = Path(__file__).resolve().parents[1]
TMP = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fr_a_mutate")
OUT = Path(__file__).with_name("fr_a_mutate.json")
PY = "/root/miniconda3/envs/CoSiR/bin/python"
ST, AP, RH = "r6_stats.py", "run_r6_apply_rule.py", "run_r6_held.py"
TS, TA, TH = "test_r6_stats.py", "test_r6_apply_rule.py", "test_r6_held.py"

MUTANTS = {
    # name: (file, old, new, test file)
    "control_stats": (ST, None, None, TS),
    "control_apply": (AP, None, None, TA),
    "control_held": (RH, None, None, TH),
    "stats_int_le0_strict": (ST, "le0.append(isums[draws].sum(axis=1) <= 0)", "le0.append(isums[draws].sum(axis=1) < 0)", TS),
    "stats_holm_not_stepdown": (ST, "        earlier_pass = passes\n", "        earlier_pass = True\n", TS),
    "stats_ties_reversed": (ST, "key=lambda nm: (int(counts[nm]), names.index(nm))",
                            "key=lambda nm: (int(counts[nm]), -names.index(nm))", TS),
    "stats_holm_interval_0.05": (ST, "np.percentile(b, 100 * 0.025 / (m + 1 - k))", "np.percentile(b, 100 * 0.05 / (m + 1 - k))", TS),
    "stats_holm_interval_upper": (ST, "np.percentile(b, 100 * (1 - 0.025 / (m + 1 - k)))", "np.percentile(b, 100 * (1 - 0.025 / (m - k + 2)))", TS),
    "stats_level_m": (ST, '"level_two_sided": 1 - 0.05 / (m + 1 - k)', '"level_two_sided": 1 - 0.05 / (m - k)', TS),
    "stats_near_boundary_strict": (ST, "return abs(int(n) - boundary(k, m)) <= 1", "return abs(int(n) - boundary(k, m)) < 1", TS),
    "stats_Z_P": (ST, "Z_P = 3.532", "Z_P = 3.53", TS),
    "stats_Z_S": (ST, "Z_S = 3.083", "Z_S = 3.08", TS),
    "stats_sum_M_not_squared": (ST, "np.sum(M.astype(np.int64) ** 2)", "np.sum(M.astype(np.int64))", TS),
    "stats_P7_comparator": (ST, '"P7": ("gain", "rca", ', '"P7": ("gain", "cosine", ', TS),
    "stats_S2_comparator": (ST, '"S2": ("r1", "r1_fused", ', '"S2": ("r1", "B", ', TS),
    "stats_point_scaled_before": (ST, '"ci95": [100 * float(np.percentile(b, 2.5)), 100 * float(np.percentile(b, 97.5))]}',
                                  '"ci95": [float(np.percentile(100 * b, 2.5)), float(np.percentile(100 * b, 97.5))]}', TS),
    "apply_point_ge0": (AP, '    if c["point"] > 0:\n', '    if c["point"] >= 0:\n', TA),
    "apply_secondary_always_tested": (AP, "    if go:  # GUARD: gatekeeping\n", "    if True:  # GUARD: gatekeeping\n", TA),
    "apply_disagreements_truthy": (AP, 'if agr.get("disagreements") != []:', 'if agr.get("disagreements"):', TA),
    "apply_n_quantities_any": (AP, 'and agr["n_quantities"] >= 1', 'and agr["n_quantities"] >= 0', TA),
    "apply_no_passes_consistency": (AP, 'diffs += [f for f in ("passes", "own_count_passes") if c[f] is not e[f]]',
                                    'diffs += []', TA),
    "apply_fix1_pair_ignored": (AP, '    if (out / "held_pass_fix1.json").exists():  # GUARD: fix1-pair\n        return "_fix1"\n',
                                '    if False:\n        return "_fix1"\n', TA),
    "apply_name_P4": (AP, '"P4": "B′(A0)"', '"P4": "B0"', TA),
    "apply_name_P1": (AP, '"P1": "cosine"', '"P1": "COS"', TA),
    "apply_name_P5": (AP, '"P5": "its matched control"', '"P5": "CF"', TA),
    "apply_claim_appends_failed": (AP, 'for nm in S_CHECKS if sread[nm]["passes"])', 'for nm in S_CHECKS)', TA),
    "apply_kind_any_inconclusive": (AP, 'kind = "inconclusive" if all(', 'kind = "inconclusive" if any(', TA),
    "apply_not_reached_stop_last": (AP, 'stop = next((e["name"] for e in recomputed if not e["own_count_passes"]), None)',
                                    'stop = next((e["name"] for e in reversed(recomputed) if not e["own_count_passes"]), None)', TA),
    "apply_smoke_flag_any": (AP, '    if agr["smoke"] is not smoke:  # GUARD: smoke-flag\n', '    if False:  # GUARD: smoke-flag\n', TA),
    "held_three_attempts": (RH, "MAX_ATTEMPTS = 2 ", "MAX_ATTEMPTS = 3 ", TH),
    "held_ledger_sha_anywhere": (RH, 'row["scripts"] and row["scripts"][-1] == runner_sha', 'runner_sha in row["scripts"]', TH),
    "held_smoke_keys_of_record_only": (RH, "diff = sorted(k for k in set(mods) | set(cur) if mods.get(k) != cur.get(k))",
                                       "diff = sorted(k for k in set(mods) if mods.get(k) != cur.get(k))", TH),
    "held_crash1_ignored": (RH, "return crash if crash.is_file() else res / SMOKE_RECORDS[\"held\"]",
                            "return res / SMOKE_RECORDS[\"held\"]", TH),
    "held_fix1_record_ignored_first_read": (RH, 'if kind == "held" and (res / SMOKE_RECORDS["fix1"]).is_file():',
                                            'if False:', TH),
    "held_include_pm_true": (RH, "sc = score_bundle(env, b, picks, lambdas, include_pm=False)",
                             "sc = score_bundle(env, b, picks, lambdas, include_pm=True)", TH),
    "held_sensitivity_per_seed_N": (RH, "    n = len(seeds) * len(R.PAIRS) * int(n_per_pair)\n    _require(len(bundles) == len(seeds) and cl.shape == (n,)",
                                    "    n = len(R.PAIRS) * int(n_per_pair)\n    _require(len(bundles) == len(seeds) and cl.shape == (3 * n,)", TH),
    "held_rerun_hash_compare_skipped": (RH, "        if s in self.want:\n", "        if False:\n", TH),
    "held_after_crash_pass_check_off": (RH, "    _refuse_unless(not (res / n.pass_).exists(),",
                                        "    _refuse_unless(after_crash or not (res / n.pass_).exists(),", TH),
}


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_DIR="/project/CoSiR/.git", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8",
               MKL_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="")
    only = set(sys.argv[2:])
    res = json.loads(OUT.read_text()) if OUT.exists() and only else {}
    for name, (fname, old, new, test) in MUTANTS.items():
        if only and name not in only:
            continue
        root = TMP / name
        if root.exists():
            shutil.rmtree(root)
        dst = root / "src/test/20261125_artelingo_held_test"
        shutil.copytree(F, dst, ignore=shutil.ignore_patterns("final_review", "results", "__pycache__", "*.pyc"))
        if old is not None:
            p = dst / fname
            s = p.read_text()
            n = s.count(old)
            if n != 1:
                res[name] = {"error": f"old text found {n} times"}
                print(f"{name}: old text found {n} times", flush=True)
                continue
            p.write_text(s.replace(old, new))
        log = root / "pytest.log"
        with open(log, "w") as fh:
            code = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp",
                                   str(root / "pt"), test], cwd=dst, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=1800).returncode
        tail = [ln for ln in log.read_text().splitlines() if ln.strip()][-1:]
        failed = [ln for ln in log.read_text().splitlines() if ln.startswith("FAILED") or ln.startswith("ERROR")][:2]
        caught = code != 0
        res[name] = {"code": code, "caught": caught, "tail": tail, "first_failed": failed}
        print(f"{name}: {'caught' if caught else 'NOT caught'} ({tail[0][:90] if tail else ''})", flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
