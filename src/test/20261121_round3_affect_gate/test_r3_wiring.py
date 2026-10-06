"""End-to-end wiring smoke test of round 3 (rule DECISION_RULE.md §10, §8 step 4) on the smoke seeds 9001, 9002, 9003.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/20261121_round3_affect_gate/test_r3_wiring.py -q \
        -p no:cacheprovider > src/test/20261121_round3_affect_gate/results/smoke/test_r3_wiring.log 2>&1

Steps, each a subprocess with stdout and stderr to a log under results/smoke/ (logs are kept):
  1. run_r3_build.py --smoke for every smoke seed without a build record (an existing smoke seed is recorded, never
     rebuilt; new ones are built with run_baselines.py --smoke, whose own log holds its scorer table by design);
  2. run_r3_test.py --phase go --smoke;
  3. r3_apply_rule.py --smoke, which reads the real results/sensitivity.json when it exists (Task 5's run); before it
     exists, a stand-in sensitivity file is written to results/smoke/ and passed with --sensitivity. A boundary stop
     (exit 3) is followed by the --boundary-reported rerun, so that path is exercised too;
  4. run_r3_test.py --phase descriptive --smoke;
  5. one wiring mutation: this file run as `--mutant-go` patches r3_fusion._terms so that AFF's gated term is passed
     where the counterpart expects G_cf, then runs the GO phase; the condition-free assertion must fire.
Assertions check existence, keys, finiteness and pass/fail only; no metric value is printed, and the logs of steps 2 to
5 must contain no decimal number (and none of the pooled values). On success the outputs of this test are deleted
(logs kept). Assertion messages carry no value (plain raises, no pytest rewriting of values).
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_common as R3  # noqa: E402

SEEDS = tuple(R3.SMOKE_SEEDS)
PY = sys.executable
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
       "PYTHONDONTWRITEBYTECODE": "1"}
MUTANT_MARK = "must be identical under both conditions (condition-free)"
GO_CHECKS = ("r1_vs_cosine", "r1_vs_rca", "r1_vs_B", "r1_vs_Bprime", "r1_vs_counterpart", "gain_statistic",
             "gain_vs_rca")
DECIMAL = re.compile(r"\d\.\d{2,}")
POOLED_KEYS = {"what", "seeds", "smoke", "n_episodes", "n_episodes_pooled", "n_clusters_pooled", "checks", "go",
               "secondary", "files_sha256", "build_record_sha256", "taus", "runtime_s", "provenance"}


def smoke_dir():
    return R3.RES / "smoke"


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(args, log):
    with open(smoke_dir() / log, "w") as f:
        return subprocess.run([PY, *map(str, args)], stdout=f, stderr=subprocess.STDOUT, env=ENV,
                              cwd=str(R3.ROOT)).returncode


def log_text(log):
    return (smoke_dir() / log).read_text()


def outputs():
    """Every file this test's runs write to results/smoke/ (logs excluded)."""
    d = smoke_dir()
    out = [d / f"{k}_seed{s}.npz" for s in SEEDS for k in ("cache", "cache_reader", "go")]
    out += [d / f"cache_seed{s}.partial.npz" for s in SEEDS]
    out += [d / f"build_seed{s}.json" for s in SEEDS]
    out += [d / n for n in ("go_pooled.json", "test_verdict.json", "test_verdict.txt", "verdict_boundary.json",
                            "descriptive.json", "descriptive.txt", "wiring_sensitivity_standin.json")]
    return out


def finite_tree(o):
    if isinstance(o, dict):
        return all(finite_tree(v) for v in o.values())
    if isinstance(o, list):
        return all(finite_tree(v) for v in o)
    if isinstance(o, float):
        return math.isfinite(o)
    return True


def check_rec(r):
    return (isinstance(r, dict) and isinstance(r.get("pass"), bool) and isinstance(r.get("point"), float)
            and len(r.get("ci95", [])) == 2 and finite_tree(r))


def write_standin():
    rec = {"what": "stand-in sensitivity for the wiring smoke test only (structure of results/sensitivity.json; not "
                   "a result)", "stand_in": True,
           "checks": {k: {"SE": 0.1, "half_width": 0.196, "x": 0.28, "seed42_half_width": 0.2}
                      for k in GO_CHECKS + ("secondary",)},
           "provenance": {"rule_sha256": R3.RULE_SHA, "smoke": True}}
    p = smoke_dir() / "wiring_sensitivity_standin.json"
    p.write_text(json.dumps(rec))
    return p


def pooled_numbers(pooled):
    vals = []
    for r in list(pooled["checks"].values()) + [pooled["secondary"]]:
        vals += [r["point"], *r["ci95"]]
    return vals


def leaked(logs, values):
    """How many of the values appear, in any usual format, in the logs; and how many decimal numbers the logs hold."""
    text = "\n".join(log_text(g) for g in logs)
    n_val = 0
    for v in values:
        for f in ("{:.2f}", "{:.3f}", "{:+.3f}", "{:.4f}", "{:.6f}", "{!r}"):
            s = f.format(v)
            if s.lstrip("+-") in ("0.00", "0.000", "0.0000", "0.000000", "0.0"):
                continue
            if s in text:
                n_val += 1
    return n_val, len(DECIMAL.findall(text))


# ---------------------------------------------------------------- the test

def test_wiring_end_to_end_with_mutation():
    sd = smoke_dir()
    sd.mkdir(parents=True, exist_ok=True)
    for p in outputs():
        if p.name.startswith("build_seed"):
            continue                       # kept between runs only while the test has not passed
        p.unlink(missing_ok=True)

    # 1. smoke seeds
    missing = [s for s in SEEDS if not (sd / f"build_seed{s}.json").exists()]
    if missing:
        rc = run([HERE / "run_r3_build.py", "--smoke", "--seeds", *missing], "wiring_build.log")
        ok(rc == 0, f"run_r3_build.py --smoke exited {rc} (see results/smoke/wiring_build.log, grep only)")
    for s in SEEDS:
        rec = json.loads((sd / f"build_seed{s}.json").read_text())
        ok(rec.get("passed") is True and rec["provenance"]["rule_sha256"] == R3.RULE_SHA,
           f"build record of smoke seed {s} did not pass")

    # 2. GO phase
    rc = run([HERE / "run_r3_test.py", "--phase", "go", "--seeds", *SEEDS, "--smoke"], "wiring_go.log")
    ok(rc == 0, f"GO phase exited {rc}")
    ok("GO_PHASE PASS" in log_text("wiring_go.log") and "Traceback" not in log_text("wiring_go.log"),
       "GO phase log lacks its PASS marker or holds a traceback")
    import run_r3_test as RT
    allowed = set(RT.go_npz_keys())
    for s in SEEDS:
        for k in ("cache", "cache_reader", "go"):
            ok((sd / f"{k}_seed{s}.npz").exists(), f"{k}_seed{s}.npz missing")
        with np.load(sd / f"go_seed{s}.npz") as z:
            ok(set(z.files) == allowed, f"go_seed{s}.npz holds other arrays than rule 6.4 allows")
            meta = json.loads(str(z["meta"][()]))
            n = int(meta["n"])
            ok(meta["seed"] == s and meta["smoke"] is True and meta["rule_sha256"] == R3.RULE_SHA,
               f"go_seed{s}.npz meta")
            for k in z.files:
                if k == "meta":
                    continue
                a = z[k]
                ok(np.isfinite(a.astype(np.float64)).all(), f"go_seed{s}.npz {k}: non-finite")
                if "__" in k or k in ("cl", "pair_index", "parity"):
                    ok(a.shape == (n,), f"go_seed{s}.npz {k}: shape")
                else:
                    ok(a.shape == (2,), f"go_seed{s}.npz {k}: shape")
            for k in ("aff_fused_cells", "aff_cf_cells", "r1_fused_cells"):
                ok(all(0 <= int(c) < 224 for c in z[k]), f"go_seed{s}.npz {k}: cell outside 0..223")
            ok(bool((z["aff_cf__gain"] == 0).all()), f"go_seed{s}: the counterpart's gain is not exactly 0")
    pooled = json.loads((sd / "go_pooled.json").read_text())
    ok(tuple(pooled["checks"]) == GO_CHECKS, "go_pooled.json: the seven checks")
    ok(all(check_rec(pooled["checks"][k]) for k in GO_CHECKS) and check_rec(pooled["secondary"]),
       "go_pooled.json: a check record is malformed or non-finite")
    ok(pooled["seeds"] == list(SEEDS) and pooled["smoke"] is True, "go_pooled.json: seeds or smoke flag")
    ok(set(pooled) == POOLED_KEYS, "go_pooled.json holds other fields than the GO pass's (a post-verdict quantity?)")
    ok(all(set(pooled["checks"][k]) <= {"point", "ci95", "pass", "n_clusters"} for k in GO_CHECKS)
       and set(pooled["secondary"]) <= {"point", "ci95", "pass", "n_clusters"}, "go_pooled.json: check record fields")
    ok(set(pooled["files_sha256"]) == {f"{k}_seed{s}.npz" for s in SEEDS for k in ("cache", "cache_reader", "go")},
       "go_pooled.json: file records")

    # 3. rule application
    real = R3.RES / "sensitivity.json"
    extra = [] if real.exists() else ["--sensitivity", write_standin()]
    rc = run([HERE / "r3_apply_rule.py", "--smoke", *extra], "wiring_apply.log")
    logs = ["wiring_go.log", "wiring_apply.log"]
    if rc == 3:
        ok((sd / "verdict_boundary.json").exists() and not (sd / "test_verdict.json").exists(),
           "boundary stop without its record, or with a verdict")
        rc = run([HERE / "r3_apply_rule.py", "--smoke", *extra, "--boundary-reported"], "wiring_apply_boundary.log")
        logs.append("wiring_apply_boundary.log")
    ok(rc == 0, f"rule application exited {rc}")
    v = json.loads((sd / "test_verdict.json").read_text())
    ok(v["rule_sha256"] == R3.RULE_SHA and v["smoke"] is True and v["verdict"] in ("GO", "NO-GO"), "verdict record")
    ok(bool(v["time_amsterdam"]) and (sd / "test_verdict.txt").exists(), "verdict time or text missing")
    ok(v["sensitivity"]["stand_in"] is (not real.exists()), "verdict read the wrong sensitivity file")
    for k in v["failed"]:
        ok(isinstance(v["checks"][k]["reading"], str) and v["checks"][k]["kind"] in ("inconclusive", "not_beaten"),
           f"failed check {k} has no rule 6.7 reading")
    ok((v["verdict"] == "GO") == (v["failed"] == []), "verdict and failed checks disagree")
    ok(isinstance(v["secondary"]["pass"], bool) and isinstance(v["secondary"]["reading"], str), "secondary check")

    # 4. descriptive phase
    rc = run([HERE / "run_r3_test.py", "--phase", "descriptive", "--seeds", *SEEDS, "--smoke"],
             "wiring_descriptive.log")
    logs.append("wiring_descriptive.log")
    ok(rc == 0, f"descriptive phase exited {rc}")
    ok("DESCRIPTIVE_PHASE PASS" in log_text("wiring_descriptive.log"), "descriptive log lacks its PASS marker")
    d = json.loads((sd / "descriptive.json").read_text())
    for k in ("item1_per_seed_and_per_pair", "item2_bar_margin_cells_frozen", "item3_R1_checks",
              "item4_gate_open_shares", "item5_pick_accuracy", "item6_redundancy", "item7_random_share_control"):
        ok(k in d, f"descriptive.json lacks {k}")
    ok(finite_tree(d), "descriptive.json holds a non-finite number")
    ok((sd / "descriptive.txt").exists(), "descriptive.txt missing")

    # no metric value in the logs of steps 2 to 4
    n_val, n_dec = leaked(logs, pooled_numbers(pooled))
    ok(n_val == 0, f"{n_val} pooled values appear in the smoke logs")
    ok(n_dec == 0, f"{n_dec} decimal numbers appear in the smoke logs")
    for g in logs:
        ok("Traceback" not in log_text(g), f"{g} holds a traceback")

    # 5. the wiring mutation must fire the condition-free assertion
    rc = run([HERE / "test_r3_wiring.py", "--mutant-go"], "wiring_mutant_go.log")
    t = log_text("wiring_mutant_go.log")
    ok("WIRING_MUTANT_APPLIED" in t, "the mutation was not applied")
    ok(rc != 0 and MUTANT_MARK in t and "GO_PHASE PASS" not in t,
       "the wiring mutation (AFF's gated term as G_cf) did not fire the condition-free assertion")
    ok(len(DECIMAL.findall(t)) == 0, "the mutant run's log holds a decimal number")
    print("WIRING_MUTATION FIRED", flush=True)

    # success: delete this test's outputs (logs kept)
    for p in outputs():
        p.unlink(missing_ok=True)
    print("WIRING_TEST PASS", flush=True)


# ---------------------------------------------------------------- the mutant driver (run only by the test above)

def _mutant_go():
    import r3_fusion as RF
    import run_r3_test as RT
    original = RF._terms

    def mutant_terms(bundle, T, gates):
        zB, gated, _ = original(bundle, T, gates)
        return zB, gated, gated              # AFF's gated term passed where the counterpart expects G_cf

    RF._terms = mutant_terms
    print("WIRING_MUTANT_APPLIED", flush=True)
    RT.main(["--phase", "go", "--seeds", *map(str, SEEDS), "--smoke"])


if __name__ == "__main__":
    if sys.argv[1:] != ["--mutant-go"]:
        raise SystemExit("run with pytest; `--mutant-go` is the test's own mutation driver")
    _mutant_go()
