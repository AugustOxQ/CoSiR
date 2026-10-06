"""Round 3: apply the rule (DECISION_RULE.md §6.5 to §6.7, §8 boundaries and agreement, §9) to the GO pass.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/r3_apply_rule.py
    ... --smoke [--sensitivity results/smoke/<stand-in>.json]     the wiring smoke test only (results/smoke/)
    ... --boundary-reported      after a boundary stop (exit 3) has been reported to the user

Reads results/go_pooled.json (its go_seed / cache files' SHA-256s checked), results/sensitivity.json (the real §6.1
projection, also in smoke mode, as rule §10 says; --sensitivity overrides the path in smoke mode only, for the wiring
test's stand-in before the real file exists) and, outside smoke mode, the re-derivation's phase-2 agreement record
rederive/out/phase2_agreement.json, which must say "all_agree": true under this rule's SHA-256 (rule §8).

Boundaries (§8): if any of the eight lower bounds (seven checks and the secondary) lies within 1e-12 of 0,
results/verdict_boundary.json is written with the bound and the threshold, and the script stops without a verdict
(exit 3); after the user has been told, --boundary-reported writes the verdict, which the strict inequality decides.

Verdict (§6.5): GO iff all seven pooled 95% lower bounds are strictly above 0. Each failed check is read by §6.7:
pooled point above 0 -> inconclusive at the detectable margin x of §6.1, with the realised pooled half-width beside it;
point at or below 0 -> "AFF did not beat <name> on fresh episodes". The secondary check (§6.6) is read the same way
with <name> = R1 and never changes GO. Writes results/test_verdict.json and .txt (rule SHA-256, Amsterdam time), never
overwritten outside smoke mode. Smoke mode prints no value.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r3_common as R3  # noqa: E402
import r3_stats as RS  # noqa: E402

AGREEMENT = R3.HERE / "rederive" / "out" / "phase2_agreement.json"
BOUNDARY_EPS = 1e-12
CHECKS8 = RS.GO_CHECKS + ("secondary",)
NAMES = {"r1_vs_cosine": "cosine", "r1_vs_rca": "RCA", "r1_vs_B": "B", "r1_vs_Bprime": "B′(A0)",
         "r1_vs_counterpart": "the matched counterpart",
         "gain_statistic": "the condition-free comparators on condition gain",
         "gain_vs_rca": "RCA on condition gain"}
LABELS = {"r1_vs_cosine": "R@1, AFF fused minus cosine", "r1_vs_rca": "R@1, AFF fused minus RCA",
          "r1_vs_B": "R@1, AFF fused minus B", "r1_vs_Bprime": "R@1, AFF fused minus B′(A0)",
          "r1_vs_counterpart": "R@1, AFF fused minus its matched counterpart",
          "gain_statistic": "gain statistic (AFF fused gain minus the counterpart's 0)",
          "gain_vs_rca": "condition gain, AFF fused minus RCA",
          "secondary": "secondary: R@1, AFF fused minus R1 fused"}
CLAIM = ("A GO shows that AFF, a reader on A0 that steers only when it picks the affect grouping, beats each comparator, "
         "pooled over the three aspect pairs, on new episodes drawn from the same 6,451 selection paintings. It does not "
         "show transfer to new paintings (the held split stays reserved for the paper test) or a margin on each aspect "
         "pair. A passed secondary check adds that AFF beats R1 on the same episodes. For the paper: the groupings were "
         "built without evaluation labels; AFF was selected on seed 42 (rule §6.10).")
DISCLOSURE = ("AFF was found among about 50 label-free variants read on seed 42, beside 15 declared oracles (with "
              "labels) and 2 controls (brainstorm §6); the median of its one-sided cluster (+0.63, range +0.42 to "
              "+0.72) is a better guide than its +0.700; the fresh seeds 49 to 51 are the protection; the frozen-cell "
              "line accompanies the test (rule §6.11).")


def say(msg):
    print(f"[r3_apply_rule] {msg}", flush=True)


# ---------------------------------------------------------------- the rule's pieces (pure)

def read_check(rec, x, name) -> dict:
    """Rule §6.7 for one check record {"point", "ci95"} (pp) with its detectable margin x (pp)."""
    pt, lo, hi = float(rec["point"]), float(rec["ci95"][0]), float(rec["ci95"][1])
    out = {"point": pt, "ci95": [lo, hi], "pass": bool(lo > 0)}
    if lo > 0:
        out.update(kind="pass", reading="95% lower bound above 0")
    elif pt > 0:
        hw = (hi - lo) / 2.0
        out.update(kind="inconclusive", x=float(x), realised_half_width=hw,
                   reading=f"inconclusive at a detectable margin of x = {float(x):.3f} pp (realised pooled half-width "
                           f"{hw:.3f} pp); not evidence that AFF fails")
    else:
        out.update(kind="not_beaten", reading=f"AFF did not beat {name} on fresh episodes")
    return out


def _consistent(rec, what):
    if bool(rec["pass"]) != bool(float(rec["ci95"][0]) > 0):
        raise SystemExit(f"go_pooled.json {what}: its pass flag disagrees with its lower bound; stop and report")


def decide(pooled, sens) -> dict:
    """Rule §6.5 to §6.7 on go_pooled's checks and the sensitivity file's x. -> verdict record."""
    checks = pooled["checks"]
    if tuple(checks) != RS.GO_CHECKS:
        raise SystemExit(f"go_pooled.json: checks {tuple(checks)} are not the rule's seven {RS.GO_CHECKS}")
    out = {}
    for k in RS.GO_CHECKS:
        _consistent(checks[k], k)
        out[k] = {**read_check(checks[k], sens["checks"][k]["x"], NAMES[k]), "label": LABELS[k]}
    failed = [k for k in RS.GO_CHECKS if not out[k]["pass"]]
    _consistent(pooled["secondary"], "secondary")
    secondary = {**read_check(pooled["secondary"], sens["checks"]["secondary"]["x"], "R1"), "label": LABELS["secondary"],
                 "note": "pre-registered secondary check (rule §6.6); it never changes GO"}
    verdict = "GO" if not failed else "NO-GO"
    if "go" in pooled and bool(pooled["go"]) != (verdict == "GO"):
        raise SystemExit("go_pooled.json: its 'go' flag disagrees with the seven checks; stop and report")
    return {"verdict": verdict, "checks": out, "failed": failed, "secondary": secondary,
            "readings": [f"{LABELS[k]}: {out[k]['reading']}" for k in failed]}


def boundary_hits(pooled) -> list:
    """Rule §8: the checks (seven and the secondary) whose lower bound lies within 1e-12 of 0."""
    hits = []
    for k in CHECKS8:
        rec = pooled["secondary"] if k == "secondary" else pooled["checks"][k]
        lo = float(rec["ci95"][0])
        if abs(lo - 0.0) <= BOUNDARY_EPS:
            hits.append({"check": k, "lower_bound": lo, "threshold": 0.0, "distance": abs(lo),
                         "point": float(rec["point"]), "upper_bound": float(rec["ci95"][1])})
    return hits


# ---------------------------------------------------------------- inputs

def load_sensitivity(override, smoke) -> tuple:
    if override is not None and not smoke:
        raise SystemExit("--sensitivity (a stand-in sensitivity file) is for the smoke wiring test only")
    p = Path(override) if override is not None else R3.RES / "sensitivity.json"
    if not p.exists():
        raise SystemExit(f"{p} is missing: run run_r3_seed42.py first (the sensitivity projection of rule §6.1)")
    rec = json.loads(p.read_text())
    prov = rec.get("provenance", {})
    if prov.get("rule_sha256") != R3.RULE_SHA:
        raise SystemExit(f"{p.name}: sensitivity file written under another rule")
    stand_in = override is not None
    if not stand_in and prov.get("smoke") is not False:
        raise SystemExit(f"{p}: the sensitivity file must be the real (non-smoke) projection of rule §6.1")
    for k in CHECKS8:
        x = rec.get("checks", {}).get(k, {}).get("x")
        if not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
            raise SystemExit(f"{p.name}: sensitivity lacks a finite detectable margin x for {k}")
    return rec, {"path": str(p), "sha256": R3.sha_file(p), "stand_in": stand_in}


def load_pooled(out, smoke) -> tuple:
    p = out / "go_pooled.json"
    if not p.exists():
        raise SystemExit(f"{p} is missing: run run_r3_test.py --phase go first")
    rec = json.loads(p.read_text())
    if rec.get("provenance", {}).get("rule_sha256") != R3.RULE_SHA:
        raise SystemExit(f"{p.name}: written under another rule")
    seeds = list(R3.SMOKE_SEEDS if smoke else R3.TEST_SEEDS)
    if rec.get("smoke") is not bool(smoke) or rec.get("seeds") != seeds:
        raise SystemExit(f"{p.name}: smoke flag or seeds differ from this mode's ({seeds})")
    for name, sha in rec.get("files_sha256", {}).items():
        f = out / name
        if not f.exists() or R3.sha_file(f) != sha:
            raise SystemExit(f"{name} is missing or its SHA-256 differs from go_pooled.json's record")
    return rec, {"path": str(p), "sha256": R3.sha_file(p)}


def check_agreement() -> dict:
    """Rule §8: the re-derivation's phase 2 agrees (all_agree true) under this rule's SHA-256."""
    p = Path(AGREEMENT)
    if not p.exists():
        raise SystemExit(f"{p} is missing: the re-derivation's phase-2 agreement record must exist before the rule is "
                         f"applied (rule §8)")
    rec = json.loads(p.read_text())
    sha = rec.get("rule_sha256", rec.get("provenance", {}).get("rule_sha256"))
    if rec.get("all_agree") is not True or sha != R3.RULE_SHA:
        raise SystemExit(f"{p.name}: the agreement record does not say all_agree true under this rule's SHA-256; "
                         f"trace the difference first (rule §8)")
    return {"path": str(p), "sha256": R3.sha_file(p)}


# ---------------------------------------------------------------- output

def _ci(r):
    return f"{r['point']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"


def verdict_text(v) -> str:
    seeds = ", ".join(str(s) for s in (R3.SMOKE_SEEDS if v["smoke"] else R3.TEST_SEEDS))
    L = [f"Rule application, round 3 (DECISION_RULE.md {R3.RULE_SHA[:12]}...), {v['time_amsterdam']}"
         f"{'  [SMOKE: not a result]' if v['smoke'] else ''}",
         f"VERDICT: {v['verdict']}",
         f"Seven GO checks, pooled over seeds {seeds} (pp, 95% painting-bootstrap intervals; pass = lower bound > 0):"]
    for k in RS.GO_CHECKS:
        c = v["checks"][k]
        L.append(f"  {c['label']:58s} {_ci(c)}  {'pass' if c['pass'] else 'FAIL'}"
                 + ("" if c["pass"] else f"  -> {c['reading']}"))
    s = v["secondary"]
    L.append(f"  {s['label']:58s} {_ci(s)}  {'pass' if s['pass'] else 'fail'}  -> {s['reading']} (never changes GO)")
    if v["failed"]:
        L.append("NO-GO readings (rule §6.7):")
        L += [f"  - {r}" for r in v["readings"]]
    if v["boundary"]:
        L.append("Boundary (rule §8), reported to the user before this verdict: " + "; ".join(
            f"{h['check']} lower bound {h['lower_bound']!r} vs 0" for h in v["boundary"]))
    sens = v["inputs"]["sensitivity"]
    L.append(f"Sensitivity: {Path(sens['path']).name}{' (stand-in, smoke only)' if sens['stand_in'] else ''}; "
             f"agreement record: {'not required (smoke)' if v['smoke'] else Path(v['inputs']['phase2_agreement']['path']).name}")
    if v["verdict"] == "GO":
        L.append("Claim licensed: " + CLAIM)
    L.append("Disclosure: " + DISCLOSURE)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sensitivity", default=None, help="stand-in sensitivity file (smoke wiring test only)")
    ap.add_argument("--boundary-reported", action="store_true",
                    help="write the verdict after a boundary stop has been reported to the user")
    args = ap.parse_args(argv)
    smoke = bool(args.smoke)
    R3.assert_rule()
    out = R3.res_dir(smoke)
    vj, vt, bj = out / "test_verdict.json", out / "test_verdict.txt", out / "verdict_boundary.json"
    R3.refuse_existing([vj, vt], smoke)
    sens, sens_info = load_sensitivity(args.sensitivity, smoke)
    pooled, pooled_info = load_pooled(out, smoke)
    agreement = None if smoke else check_agreement()
    hits = boundary_hits(pooled)
    if hits and not args.boundary_reported:
        if smoke or not bj.exists():
            R3.write_json_once(bj, {"what": "rule §8: lower bound(s) within 1e-12 of 0; report both values to the user "
                                             "before the verdict is recorded, then rerun with --boundary-reported",
                                    "hits": hits, "go_pooled": pooled_info, "smoke": smoke}, smoke)
        if smoke:
            say(f"BOUNDARY: {len(hits)} lower bound(s) within 1e-12 of 0; {bj.name} written; no verdict (exit 3)")
        else:
            for h in hits:
                say(f"BOUNDARY: {h['check']}: lower bound {h['lower_bound']!r}, threshold 0.0")
            say(f"{bj.name} written; no verdict. Report both values to the user, then rerun with --boundary-reported")
        raise SystemExit(3)
    if args.boundary_reported:
        if not hits:
            raise SystemExit("--boundary-reported given, but no lower bound lies within 1e-12 of 0 (no boundary case)")
        if not bj.exists() or [h["check"] for h in json.loads(bj.read_text())["hits"]] != [h["check"] for h in hits]:
            raise SystemExit(f"{bj.name} is missing or names other checks: run without --boundary-reported first")
    d = decide(pooled, sens)
    v = {"rule_sha256": R3.RULE_SHA, "time_amsterdam": R3.now_ams(), "smoke": smoke, "verdict": d["verdict"],
         "checks": d["checks"], "failed": d["failed"], "readings": d["readings"], "secondary": d["secondary"],
         "boundary": hits or None, "sensitivity": sens_info,
         "inputs": {"go_pooled": pooled_info, "sensitivity": sens_info, "phase2_agreement": agreement},
         "seeds": pooled["seeds"], "claim_if_go": CLAIM, "disclosure": DISCLOSURE,
         "script_sha256": R3.sha_file(Path(__file__))}
    R3.write_json_once(vj, v, smoke)
    text = verdict_text(v)
    vt.write_text(text)
    if smoke:
        say(f"{vj.name} and {vt.name} written (smoke: no value printed)")
    else:
        print(text, flush=True)
    say("APPLY_RULE PASS")
    return 0


if __name__ == "__main__":
    main()
