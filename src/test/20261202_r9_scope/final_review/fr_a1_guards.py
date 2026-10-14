"""Final review A1, item 4: delete or weaken each guard of run_r9_apply_rule.py and r9_stats.py, run the in-scope tests,
restore the file at once (SHA-256 checked).

    python fr_a1_guards.py --which apply|stats --batch i --size n   -> fr_a1_guards_<which>_<i>.json
"""
import argparse
import signal
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
ROOT = F.parents[2]
PY = "/root/miniconda3/envs/CoSiR/bin/python"
ENV = {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
       "PATH": "/usr/bin:/bin", "HOME": "/root"}
T_APPLY = [str(F / "test_r9_apply_rule.py")]
T_STATS = [str(F / "test_r9_stats.py"), str(F / "test_r9_a28_stats.py"), str(F / "test_r9_apply_rule.py")]

APPLY = [  # (label, old, new); every `old` occurs exactly once
    ("G agreement-missing", "if not path.is_file():  # GUARD: agreement-missing", "if False:  # GUARD: agreement-missing"),
    ("G phase-2", 'if not (_is_int(agr.get("phase")) and agr["phase"] == 2):  # GUARD: phase-2', "if False:"),
    ("G smoke-flag", 'if agr["smoke"] is not smoke:  # GUARD: smoke-flag', "if False:"),
    ("G agreement-true", 'if agr.get("agreement") is not True:  # GUARD: agreement-true', "if False:"),
    ("G failing-items", 'if [x for x in items if not x["equal"]]:  # GUARD: failing-items', "if False:"),
    ("G failed-zero", "if not (_is_int(failed) and failed == 0):  # GUARD: failed-zero", "if False:"),
    ("G compared-count", "if not (_is_int(compared) and compared >= 1 and compared == len(items)):  # GUARD: compared-count",
     "if False:"),
    ("G read-name", 'if agr.get("read") != read:  # GUARD: read-name', "if False:"),
    ("G pass-file-name", 'if agr.get("pass_file") != pass_name:  # GUARD: pass-file-name', "if False:"),
    ("G rederived-counts", "    if not (isinstance(rc, dict) and set(rc) == set(P_CHECKS)\n"
     "            and all(_is_int(rc[nm]) and 0 <= rc[nm] <= N_BOOT for nm in P_CHECKS)):  # GUARD: rederived-counts",
     "    if False:"),
    ("G fix1-pair", 'if (out / f"held_pass_{read}_fix1.json").exists():  # GUARD: fix1-pair', "if False:"),
    ("G smoke-record", "if not path.is_file():  # GUARD: smoke-record", "if False:"),
    ("G smoke-passed", 'if rec.get("passed") is not True:  # GUARD: smoke-passed', "if False:"),
    ("G smoke-shas", "if not mods or diff:  # GUARD: smoke-shas", "if False:"),
    ("G verdict-exists", "if vpath.exists():  # GUARD: verdict-exists", "if False:"),
    ("G reserve-needs-original", 'if variant == "reserve" and not (out / f"held_verdict_{read}.json").is_file():  # GUARD: '
     "reserve-needs-original", "if False:"),
    ("G sha-binding", 'if agr["held_pass_sha256"] != pass_sha:  # GUARD: sha-binding', "if False:"),
    ("G fix1-original", 'if sfx == "_fix1" and not (out / f"held_pass_{read}.json").is_file():  # GUARD: fix1-original',
     "if False:"),
    ("G counts-equal", 'if rd[nm] != rec["checks"][nm]["n"]:  # GUARD: counts-equal', "if False:"),
    ("G smoke_guard call", "smoke_record = None if smoke else smoke_guard(out, variant)", "smoke_record = None"),
    ("u pass rule_sha256", 'if rec.get("rule_sha256") != CM.RULE_SHA256:', "if False:"),
    ("u pass mode", 'if rec.get("mode") != mode:', "if False:"),
    ("u pass seeds", 'if rec.get("seeds") != seeds:', "if False:"),
    ("u pass n_episodes", 'if rec.get("n_episodes") != 9 * (CM.N_SMOKE if smoke else CM.N_PER_PAIR):', "if False:"),
    ("u pass design", 'if (rec.get("design") == "two_level") is not (design == "two_level"):', "if False:"),
    ("u pass checks set", "if not isinstance(checks, dict) or set(checks) != set(P_CHECKS):", "if False:"),
    ("u entry level", 'ok = ok and _is_int(c.get("level")) and c["level"] == level', "ok = ok"),
    ("u holm_order schema seven", "if not (isinstance(order, list) and sorted(order) == list(P_CHECKS)):", "if False:"),
    ("u near_duplicates schema", '        if not (isinstance(nd, dict) and set(nd) == {"cos_099", "cos_098"}',
     "        if False and (isinstance(nd, dict) and set(nd) == {\"cos_099\", \"cos_098\"}"),
    ("u check_set_aside call", "        check_set_aside(sa, name)\n", "        pass\n"),
    ("u set-aside cap", "    if over:\n", "    if False:\n"),
    ("u set-aside duplicate", "if len(set(keys)) != len(keys):", "if False:"),
    ("u set-aside unknown name", "    if outside:\n", "    if False:\n"),
    ("u sensitivity x = z SE", 'if not math.isclose(c[k], want[k], rel_tol=1e-9, abs_tol=1e-12):', "if False:"),
    ("u sensitivity finite", 'if not (isinstance(c, dict) and all(_is_num(c.get(k)) and c[k] >= 0 for k in ("SE", "x", "x95"))):',
     "if False:"),
    ("u assert_consistent", "        if diffs:\n            contradiction(", "        if False:\n            contradiction("),
    ("u holm_order seven equal", 'if rec["holm_order"] != [e["name"] for e in hp]:', "if False:"),
    ("u holm_order levels equal", 'if rec["holm_order"] != {"level1": [e["name"] for e in h1], "level2": [e["name"] for e in h2]}:',
     "if False:"),
    ("u assert_rule rule sha", "if CM.sha256_bytes(raw) != CM.RULE_SHA256:", "if False:"),
    ("u assert_rule R6 sha", "if CM.sha256_file(r6) != CM.R6_RULE_SHA256:", "if False:"),
    ("u assert_rule phrases", "    if missing:\n        contradiction(", "    if False:\n        contradiction("),
    ("u assert_rule call (read)", "    assert_rule()\n    sfx = pair_suffix", "    sfx = pair_suffix"),
    ("u gatekeeping level 2", "        if l1_pass:                                       # serial gatekeeping",
     "        if True:                                       # serial gatekeeping"),
    ("u NO-GO kind any/all", 'return "inconclusive" if all(read_checks[nm]["reading_kind"] == "inconclusive" for nm in failed)',
     'return "inconclusive" if any(read_checks[nm]["reading_kind"] == "inconclusive" for nm in failed)'),
    ("u c11 point > 0", 'if c["point"] > 0:', 'if c["point"] >= 0:'),
    ("u N counts entries", 'n = N_LANGUAGES - len({s["language"] for s in sa})', "n = N_LANGUAGES - len(sa)"),
    ("u not-run pass exists", 'if ppath.exists() or (out / f"held_pass_{read}_fix1.json").exists():', "if False:"),
    ("u not-run started exists", 'if (out / f"held_started_{read}.json").exists():', "if False:"),
    ("u not-run H8", 'if read == "H8":', "if False:"),
    ("u not-run sibling", 'if sv.get("status") != NOT_RUN["switched_off"]:', "if False:"),
    ("u not-run ledger wording", "if NOT_RUN[not_run] not in cells or any(other in c for c in cells):", "if False:"),
    ("u not-run variant", "if not_run is not None and (not_run not in NOT_RUN or variant is not None):", "if False:"),
    ("u agreement smoke type", 'if type(agr.get("smoke")) is not bool:', "if False:"),
    ("u agreement sha format", 'if not (isinstance(agr.get("held_pass_sha256"), str) and SHA_RE.fullmatch(agr["held_pass_sha256"])):',
     "if False:"),
    ("u write_once link", "        os.link(tmp, path)", "        path.write_bytes(tmp.read_bytes())"),
    ("u level-2 boundary only when tested", "            out.update(read_family(checks, L2, h2, sens, 2))\n            tested_boundary(h2)",
     "            out.update(read_family(checks, L2, h2, sens, 2))"),
]

STATS = [
    ("s finite", 'if not np.isfinite(v).all():\n        raise AssertionError("a per-episode difference is not finite")',
     'if False:\n        raise AssertionError("a per-episode difference is not finite")'),
    ("s abs <= 2", "if np.abs(v).max() > MAX_ABS_DIFF:", "if False:"),
    ("s quarter", "if np.abs(4.0 * v - q).max() > QUARTER_TOL:", "if False:"),
    ("s exact sums", "if not np.array_equal(isums, exact):", "if False:"),
    ("s k < 2", 'if k < 2:\n        raise ValueError("bootstrap_draws', 'if False:\n        raise ValueError("bootstrap_draws'),
    ("s ci95 copy", 'if got != ref["ci95"] or ref["n_clusters"] != k or b.shape != (n_boot,) or float(v.mean()) != ref["point"]:',
     "if False:"),
    ("s point_ci", "if pc != C.point_ci(d, cl_c):", "if False:"),
    ("s CF gain 0", 'if not np.all(np.asarray(s[CF]["gain"], dtype=np.float64) == 0):', "if False:"),
    ("s shapes", "if np.asarray(s[key][metric]).shape != (n,):", "if False:"),
    ("s seeds", "if tuple(seeds) != want:", "if False:"),
    ("s per-seed count", 'if len(np.asarray(s["cl"])) != 3 * per_pair:', "if False:"),
    ("s extra keys", 'if set(extra) != {"episodes_sha256", "runner_sha256", "module_sha256"}:', "if False:"),
    ("s extra seeds", "if set(eps) != {str(s) for s in seeds}:", "if False:"),
    ("s rule sha", "if got != R6C.RULE_SHA256:", "if False:"),
    ("s holm names", "if len(names) != m or len(set(names)) != m or set(counts) != set(names):", "if False:"),
    ("s holm count range", "if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 0 <= n <= N_BOOT:",
     "if False:"),
    ("s mode/design", 'if mode not in (("a28", "smoke") if design == "two_level" else ("held", "regression", "smoke")):',
     "if False:"),
    ("s sens names", "if set(sig) != set(CHECKS):", "if False:"),
    ("s sens N", "if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N <= 0:", "if False:"),
    ("s sens variances", 'if not all(np.isfinite(sig[c][key]) and sig[c][key] >= 0 for key in ("sigma_a2", "sigma_eps2")):',
     "if False:"),
    ("s M_p ints", "if M.ndim != 1 or not np.issubdtype(M.dtype, np.integer) or (M <= 0).any():", "if False:"),
    ("s sum M_p = N", "if int(M.astype(np.int64).sum()) != int(N):", "if False:"),
    ("m integer form strict", "le0.append(isums[draws].sum(axis=1) <= 0)", "le0.append(isums[draws].sum(axis=1) < 0)"),
    ("m Holm factor", "own = 40 * (n + 1) * (m + 1 - k) <= 5001", "own = 40 * (n + 1) * (m + 2 - k) <= 5001"),
    ("m tie order", "order = sorted(names, key=lambda nm: (int(counts[nm]), names.index(nm)))",
     "order = sorted(names, key=lambda nm: (int(counts[nm]), -names.index(nm)))"),
    ("m Z level2", '"level2": 3.417}', '"level2": 3.532}'),
    ("m holm interval level", "return [100 * float(np.percentile(b, 100 * 0.025 / (m + 1 - k))),",
     "return [100 * float(np.percentile(b, 100 * 0.025 / (m + 2 - k))),"),
    ("m near boundary", "return abs(int(n) - boundary(k, m)) <= 1", "return abs(int(n) - boundary(k, m)) <= 2"),
    ("m L1 tie order", '("level1", ("P4", "P6"), 2, "level1")', '("level1", ("P6", "P4"), 2, "level1")'),
    ("m a28 set_aside pass-through", '"set_aside": [dict(s) for s in set_aside]', '"set_aside": []'),
    ("m a28 level field", 'entry["level"] = 1 if label == "level1" else 2', 'entry["level"] = 1'),
    ("m sens z kind", '"x": Z[kind] * se', '"x": Z_P * se'),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_one(path, label, old, new, tests):
    orig = path.read_bytes()
    text = orig.decode("utf-8")
    n = text.count(old)
    if n != 1:
        return {"label": label, "error": f"snippet occurs {n} times"}
    try:
        path.write_text(text.replace(old, new), encoding="utf-8")
        t0 = time.time()
        env = {**ENV, "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"}
        procs = [subprocess.Popen([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", t], cwd=str(ROOT), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for t in tests]
        outs = []
        try:
            for q in procs:
                outs.append((q.communicate(timeout=max(1, 90 - (time.time() - t0)))[0], q.returncode))
        finally:
            for q in procs:
                if q.poll() is None:
                    q.kill()
                    q.wait()
        p = type("R", (), {"returncode": max(rc for _, rc in outs), "stdout": "\n".join(o for o, _ in outs)})
        dt = time.time() - t0
    finally:
        path.write_bytes(orig)
    assert sha(path) == hashlib.sha256(orig).hexdigest()
    lines = p.stdout.strip().splitlines()
    failed = [ln for ln in lines if ln.startswith("FAILED") or ln.startswith("ERROR")]
    return {"label": label, "killed": p.returncode != 0, "rc": p.returncode, "seconds": round(dt, 1),
            "first_failure": failed[0][:200] if failed else None, "tail": lines[-1][:160] if lines else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("apply", "stats"), required=True)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--size", type=int, default=100)
    a = ap.parse_args()
    path = F / ("run_r9_apply_rule.py" if a.which == "apply" else "r9_stats.py")
    muts = APPLY if a.which == "apply" else STATS
    tests = T_APPLY if a.which == "apply" else T_STATS
    before = sha(path)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sel = muts[a.batch:a.batch + a.size]  # --batch is the start index
    out = []
    for m in sel:
        try:
            out.append(run_one(path, *m, tests))
        except subprocess.TimeoutExpired:
            out.append({"label": m[0], "killed": None, "error": "test run over 90 s"})
        print(out[-1]["label"], out[-1].get("killed"), out[-1].get("seconds"), flush=True)
    assert sha(path) == before
    (HERE / f"fr_a1_guards_{a.which}_{a.batch}.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(("KILLED  " if r.get("killed") else "SURVIVED" if "killed" in r else "ERROR   "), r["label"],
              r.get("seconds"), (r.get("first_failure") or r.get("error") or "")[:110])


if __name__ == "__main__":
    main()
