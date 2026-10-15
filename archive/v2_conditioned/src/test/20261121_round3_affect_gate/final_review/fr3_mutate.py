"""Final review: mutation checks on scratch copies (never on the committed files).

For each mutation: a fresh copy of the round-3 folder's *.py and DECISION_RULE.md in final_review/mut/w/ (with
r3_common's ROOT pinned to the repository, so results/ of the copy stay inside the copy), round 2's r2_fusion.py
copied in as well when the mutation targets it; one guard deleted or changed; the named test files run with pytest
(basetemp inside final_review/mut/); pass/fail counts recorded. Writes out/fr3_mutate.json.

    ... python fr3_mutate.py [ids...]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

FR = Path(__file__).resolve().parent
SRC = FR.parent
ROOT = FR.parents[3]
R2 = ROOT / "src/test/20261118_reader_fix_round2"
MUT = FR / "mut"
W = MUT / "w"
PY = sys.executable
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
       "PYTHONDONTWRITEBYTECODE": "1"}

F, U, BN, WI = "test_r3_fusion.py", "test_r3_runners.py", "test_r3_bundle.py", "test_r3_wiring.py"

MUTATIONS = [
    # id, guard, file, old, new, tests
    ("M01", "AFF gate: the affect factor dropped (AFF = R1)", "r3_fusion.py",
     "(gt[c] * (np.asarray(pick[c]) == R.AFFECT).astype(np.float32))", "(gt[c] * np.ones_like(gt[c]))", [F, WI]),
    ("M02", "AFF gate opens on image picks instead of affect", "r3_fusion.py",
     "(np.asarray(pick[c]) == R.AFFECT)", "(np.asarray(pick[c]) == 1)", [F]),
    ("M03", "counterpart built from all-open gates, not the family's own gates", "r3_fusion.py",
     "G = {t: K.g_cf(gated[t]) for t in gated}",
     "G = {t: K.g_cf(K.gated_terms(zT, {c: np.ones_like(np.asarray(gates[t][c])) for c in gates[t]})) for t in gated}",
     [F, WI]),
    ("M04", "counterpart = the condition-dependent gated term", "r3_fusion.py",
     "G = {t: K.g_cf(gated[t]) for t in gated}", "G = {t: gated[t] for t in gated}", [F]),
    ("M05", "integer cross-fit: fused ties go to the highest cell (round 2's select_fused, copy)", "r2:r2_fusion.py",
     "picks[half] = int(allowed[int(np.argmax(crit))])",
     "picks[half] = int(allowed[len(crit) - 1 - int(np.argmax(crit[::-1]))])", [F]),
    ("M06", "integer cross-fit: fused criterion ignores rho_ctrl (copy of round 2)", "r2:r2_fusion.py",
     "np.minimum(rho - np.int64(rho_ctrl), gam)", "np.minimum(rho, gam)", [F]),
    ("M07", "integer cross-fit: counterpart ties go to the highest cell (copy of round 2)", "r2:r2_fusion.py",
     "out[half] = int(allowed[int(np.argmax(rho))])",
     "out[half] = int(allowed[len(rho) - 1 - int(np.argmax(rho[::-1]))])", [F]),
    ("M08", "pooled clustering: clusters made seed-specific in pooled_check", "r3_stats.py",
     "cl = np.concatenate([np.asarray(x) for x in cl_by_seed])",
     "cl = np.concatenate([np.asarray(x) + 10**9 * i for i, x in enumerate(cl_by_seed)])", [F]),
    ("M09", "pooled clustering: GO pass stores per-seed local painting ids", "run_r3_test.py",
     'arrays["cl"], arrays["pair_index"], arrays["parity"] = (np.asarray(b.cl),',
     'arrays["cl"], arrays["pair_index"], arrays["parity"] = (np.unique(np.asarray(b.cl), return_inverse=True)[1],',
     [U, WI]),
    ("M10", "GO pass: AFF's family run on R1's gates (AFF never computed)", "run_r3_test.py",
     'fa = RF.run_family(b, rd["T"], RF.gates_aff(rd["m"], rd["pick"], taus))',
     'fa = RF.run_family(b, rd["T"], RF.gates_r1(rd["m"], taus))', [U, WI]),
    ("M11", "§6.4: GO pass cross-fits R1's counterpart", "run_r3_test.py",
     'RF.gates_r1(rd["m"], taus), fused_only=True)', 'RF.gates_r1(rd["m"], taus), fused_only=False)', [U, WI]),
    ("M12", "§6.4: fused_only still selects and reports the counterpart", "r3_fusion.py",
     "if fused_only:                      # rule 6.4", "if False:                      # rule 6.4", [F]),
    ("M13", "strict inequality: pooled_check pass = lower bound >= 0", "r3_stats.py",
     'return {"point": r["point"], "ci95": r["ci95"], "pass": bool(r["ci95"][0] > 0)}',
     'return {"point": r["point"], "ci95": r["ci95"], "pass": bool(r["ci95"][0] >= 0)}', [F]),
    ("M14", "strict inequality: read_check pass = lower bound >= 0", "r3_apply_rule.py",
     'out = {"point": pt, "ci95": [lo, hi], "pass": bool(lo > 0)}\n    if lo > 0:',
     'out = {"point": pt, "ci95": [lo, hi], "pass": bool(lo >= 0)}\n    if lo >= 0:', [U]),
    ("M15", "secondary changes GO in go_checks", "r3_stats.py",
     '"go": bool(all(c["pass"] for c in checks.values())),',
     '"go": bool(all(c["pass"] for c in checks.values()) and pooled_check(*r1_minus("r1"))["pass"]),', [F]),
    ("M15b", "secondary changes GO in decide", "r3_apply_rule.py",
     'verdict = "GO" if not failed else "NO-GO"', 'verdict = "GO" if not failed and secondary["pass"] else "NO-GO"',
     [U]),
    ("M16", "hash check skips the earlier seeds", "run_r3_build.py",
     '(("new", others_new), ("earlier", earlier))', '(("new", others_new),)', [U]),
    ("M17", "hash check skips the other new seeds", "run_r3_build.py",
     '(("new", others_new), ("earlier", earlier))', '(("earlier", earlier),)', [U]),
    ("M18", "hash check: two pairs of one seed may share a SHA-256", "run_r3_build.py",
     "if len(set(vals)) != len(vals):", "if False:", [U]),
    ("M19", "verify_build_record: file SHA-256 not compared", "run_r3_build.py",
     'if not f.exists() or R3.sha_file(f) != rec["sha256"][k]:', "if not f.exists():", [U]),
    ("M20", "§5 order: item 2 not required before AFF (item 3) is computed", "run_r3_seed42.py",
     "rec.require(2)                            # no AFF number before items 1 and 2 pass", "pass", [U, BN]),
    ("M21", "descriptive pass accepts a verdict written under another rule", "run_r3_test.py",
     'if v.get("rule_sha256") != R3.RULE_SHA:\n        raise SystemExit(f"{p.name}: written under another rule; refusing")',
     'if False:\n        raise SystemExit(f"{p.name}: written under another rule; refusing")', [U]),
    ("M22", "random-share control draws condition b first", "r3_fusion.py",
     "for c in CONDITIONS:                      # CONDITIONS order", 'for c in ("b", "a"):                      # CONDITIONS order',
     [F]),
    ("M23", "counterpart gain-zero assertion removed (go_checks)", "r3_stats.py",
     'if not np.all(np.asarray(s["cf"]["gain"]) == 0):', "if False:", [F]),
    ("M24", "boundary epsilon set to 0", "r3_apply_rule.py", "BOUNDARY_EPS = 1e-12", "BOUNDARY_EPS = 0.0", [U]),
    ("M25", "rule applied without the phase-2 agreement record", "r3_apply_rule.py",
     "agreement = None if smoke else check_agreement()", "agreement = None", [U]),
    ("M26", "sensitivity formula: 9 sum m^2 + 3n instead of - 6n", "r3_stats.py",
     "9 * sum_m2 - 6 * n", "9 * sum_m2 + 3 * n", [F]),
    ("M27", "descriptive pass: cache-reproduces-GO check removed", "run_r3_test.py",
     "        if not consistent:\n", "        if False:\n", [U, WI]),
    ("M28", "pick ties go to the last grouping (reader)", "r3_fusion.py",
     '"pick": {c: pm[c][0] for c in CONDITIONS}',
     '"pick": {c: (P[c].shape[1] - 1 - np.argmax(P[c][:, ::-1], axis=1)) for c in CONDITIONS}', [F]),
    ("M29", "gate opens on m > tau instead of m >= tau", "r3_fusion.py",
     "g = K.gates(m, list(taus))",
     "g = {k: {c: (np.asarray(m[c]) > t).astype(np.float32) for c in m} for k, t in enumerate(taus)}", [F]),
    ("M30", "results never-overwrite guard disabled", "r3_common.py", "    if busy:\n", "    if False:\n", [U, F]),
    ("M31", "GO pass: episode-hash cross check of the build records not called", "run_r3_test.py",
     "    BLD.cross_check(records)\n    say(f\"build records", "    pass\n    say(f\"build records", [U, WI]),
]


def make_copy(r2):
    if W.exists():
        shutil.rmtree(W)
    W.mkdir(parents=True)
    for p in list(SRC.glob("*.py")) + [SRC / "DECISION_RULE.md"]:
        shutil.copy2(p, W / p.name)
    rc = (W / "r3_common.py").read_text()
    rc = rc.replace("ROOT = HERE.parents[2]", 'ROOT = Path("%s")' % ROOT, 1)
    if r2:
        (W / "r2copy").mkdir()
        shutil.copy2(R2 / "r2_fusion.py", W / "r2copy" / "r2_fusion.py")
        rc = rc.replace("for p in (str(ROOT), str(R1), str(R2)):", 'for p in (str(ROOT), str(R1), str(R2), str(HERE / "r2copy")):', 1)
        rc = rc.replace("if Path(F.__file__).resolve().parent != R2:", 'if Path(F.__file__).resolve().parent != HERE / "r2copy":', 1)
    (W / "r3_common.py").write_text(rc)
    assert 'ROOT = Path("' in rc


def apply(fname, old, new):
    p = (W / "r2copy" / fname.split(":", 1)[1]) if fname.startswith("r2:") else W / fname
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{fname}: target occurs {n} times")
    p.write_text(s.replace(old, new, 1))


def run_tests(tests, tag):
    res = {}
    for t in tests:
        log = MUT / f"{tag}__{t}.log"
        bt = MUT / "tmp" / f"{tag}_{t[:-3]}"
        if bt.exists():
            shutil.rmtree(bt)
        bt.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.run([PY, "-m", "pytest", str(W / t), "-q", "-x", "-p", "no:cacheprovider", "--basetemp",
                                 str(bt)], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=str(ROOT), timeout=1800).returncode
        txt = log.read_text()
        last = [ln for ln in txt.splitlines() if re.search(r"(passed|failed|error)", ln)]
        failed_tests = re.findall(r"^FAILED (\S+)", txt, re.M)
        res[t] = {"returncode": rc, "summary": last[-1] if last else "", "failed": failed_tests,
                  "seconds": round(time.time() - t0, 1)}
        if bt.exists():
            shutil.rmtree(bt)
    return res


def main(ids):
    MUT.mkdir(exist_ok=True)
    out_p = FR / "out" / "fr3_mutate.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    if "baseline" in ids or not out.get("baseline"):
        make_copy(r2=True)
        out["baseline"] = run_tests([F, U, BN, WI], "baseline")
        out_p.write_text(json.dumps(out, indent=1))
        print("baseline", {k: v["summary"] for k, v in out["baseline"].items()}, flush=True)
    for mid, guard, fname, old, new, tests in MUTATIONS:
        if ids and mid not in ids:
            continue
        make_copy(r2=fname.startswith("r2:"))
        apply(fname, old, new)
        r = run_tests(tests, mid)
        caught = any(v["returncode"] != 0 for v in r.values())
        out[mid] = {"guard": guard, "file": fname, "tests": r, "caught": caught}
        out_p.write_text(json.dumps(out, indent=1))
        print(mid, "CAUGHT" if caught else "SURVIVED", {k: v["summary"] for k, v in r.items()}, flush=True)
    if W.exists():
        shutil.rmtree(W)


if __name__ == "__main__":
    main(sys.argv[1:])
