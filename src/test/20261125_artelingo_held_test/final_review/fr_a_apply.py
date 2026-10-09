"""Final review A, check 3 and the apply step's part of check 4: rule section 4 and 9 with my own decision function,
run against the real verdict step (run_r6_apply_rule.main, its module paths pointed at a temp folder) on a table of
crafted pass files; then every refusal of the apply step through main().

    python final_review/fr_a_apply.py <tmp dir>
Writes fr_a_apply.json beside this file; prints one summary line per part.
"""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402
import run_r6_apply_rule as A  # noqa: E402

P = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
S = ("S1", "S2")
NAME = {"P1": "cosine", "P2": "RCA", "P3": "B", "P4": "B′(A0)", "P5": "its matched control",
        "P6": "the condition-free scorers on condition gain", "P7": "RCA on condition gain", "S1": "B′(A1)", "S2": "R1"}
CLAIM = ("AFF beats COS, RCA, B, B′(A0) and its matched control on aspect R@1, and its condition gain exceeds theirs "
         "and RCA's, pooled over three aspect pairs, on paintings never used to fit, select or tune it, with Holm "
         "across the seven checks. No per-pair margin, no other dataset or backbone. Disclosures of §10.4 accompany "
         "every AFF number.")
TMP = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fr_a_apply")
OUT = Path(__file__).with_name("fr_a_apply.json")


# ---------------------------------------------------------------- my rule (from the text of section 3 and 4)
def holm(counts, names, m):
    order = sorted(names, key=lambda nm: (counts[nm], names.index(nm)))
    res, ok = {}, True
    for k, nm in enumerate(order, 1):
        own = 40 * (counts[nm] + 1) * (m + 1 - k) <= 5001
        ok = ok and own
        res[nm] = {"k": k, "passes": ok, "own": own}
    return order, res


def my_verdict(spec):
    """spec: {check: (n, point)} for P1..P7, S1, S2. -> my reading of the rule."""
    cnt = {c: spec[c][0] for c in P + S}
    order, h = holm(cnt, P, 7)
    go = all(h[c]["passes"] for c in P)
    stop = next((c for c in order if not h[c]["passes"]), None)
    out = {"verdict": "GO" if go else "NO-GO", "checks": {}, "secondary": {}}

    def c11(c):
        return "inconclusive" if spec[c][1] > 0 else f"AFF did not beat {NAME[c]} on new paintings"

    kinds = []
    for c in P:
        if h[c]["passes"]:
            out["checks"][c] = {"passes": True, "reading": "passed", "not_reached_after": None}
            continue
        r = c11(c)
        kinds.append("inconclusive" if r == "inconclusive" else "did not beat")
        nr = stop if (h[c]["own"] and c != stop) else None
        out["checks"][c] = {"passes": False, "reading": r, "not_reached_after": nr}
    out["kind"] = None if go else ("inconclusive" if all(k == "inconclusive" for k in kinds) else "did not beat")
    if go:
        so, hs = holm(cnt, S, 2)
        sstop = next((c for c in so if not hs[c]["passes"]), None)
        for c in S:
            if hs[c]["passes"]:
                out["secondary"][c] = {"passes": True, "license": f"AFF also beats {NAME[c]}", "not_reached_after": None}
            else:
                out["secondary"][c] = {"passes": False, "reading": c11(c), "license": None,
                                       "not_reached_after": sstop if (hs[c]["own"] and c != sstop) else None}
        out["secondary"]["tested"] = True
        out["claim"] = CLAIM + "".join(f" AFF also beats {NAME[c]}." for c in S if hs[c]["passes"])
    else:
        out["secondary"] = {"tested": False, **{c: {"passes": None} for c in S}}
        out["claim"] = None
    bnd = []
    for fam, names, m, hh in (("P", P, 7, h), ("S", S, 2, None)):
        if fam == "S":
            if not go:
                continue
            _, hh = holm(cnt, S, 2)
        for c in names:
            k = hh[c]["k"]
            if abs(cnt[c] - (5001 // (40 * (m + 1 - k)) - 1)) <= 1:
                bnd.append(c)
    out["boundary"] = sorted(bnd)
    return out


# ---------------------------------------------------------------- crafted files
def pass_file(spec, mode="held", seeds=None):
    """A pass file as contracts section 6 fixes it, Holm fields from my own integer Holm."""
    cnt = {c: spec[c][0] for c in P + S}
    rec = {"rule_sha256": R.RULE_SHA256, "mode": mode, "seeds": list(seeds or R.HELD_SEEDS), "n_episodes": 36864,
           "n_clusters": 9000, "checks": {}, "secondary": {}}
    for fam, names, m in (("checks", P, 7), ("secondary", S, 2)):
        order, h = holm(cnt, names, m)
        for c in names:
            n, pt = spec[c]
            k = h[c]["k"]
            e = {"quantity": c, "n": n, "point": pt, "ci95": [pt - 0.2, pt + 0.2], "holm_k": k,
                 "ci_holm": [pt - 0.27, pt + 0.27], "level_two_sided": 1 - 0.05 / (m + 1 - k)}
            if fam == "checks":
                e.update(passes=h[c]["passes"], own_count_passes=h[c]["own"])
            e["near_boundary"] = abs(n - (5001 // (40 * (m + 1 - k)) - 1)) <= 1
            rec[fam][c] = e
        if fam == "checks":
            rec["holm_order"] = order
    rec.update({"episodes_sha256": {}, "runner_sha256": "a" * 64, "module_sha256": {}, "time": "2026-10-12 10:00:00"})
    return rec


def sens_file():
    rec = {c: {"quantity": c, "sigma_a2": 4.0, "sigma_eps2": 1500.0, "SE": 0.1 + 0.01 * i,
               ("x" if c in P else "x2"): (3.532 if c in P else 3.083) * (0.1 + 0.01 * i), "x95": 2.8 * (0.1 + 0.01 * i)}
           for i, c in enumerate(P + S)}
    rec.update({"N": 36864, "n_paintings": 9000})
    return rec


def agreement(pass_bytes, pass_name="held_pass.json", smoke=False, **over):
    rec = {"phase": 2, "smoke": smoke, "all_agree": True, "held_pass_sha256": hashlib.sha256(pass_bytes).hexdigest(),
           "pass_file": pass_name, "n_quantities": 9, "disagreements": [], "time": "2026-10-12 11:00:00"}
    rec.update(over)
    return rec


def setup_dir(d, spec, smoke=False, sfx="", agr_over=None, pass_over=None, sens=True):
    d.mkdir(parents=True, exist_ok=True)
    pr = pass_file(spec, "smoke" if smoke else "held", R.SMOKE_SEEDS if smoke else R.HELD_SEEDS)
    if pass_over:
        pass_over(pr)
    raw = json.dumps(pr).encode()
    (d / f"held_pass{sfx}.json").write_bytes(raw)
    (d / f"rederive_agreement{sfx}.json").write_text(json.dumps(
        agreement(raw, f"held_pass{sfx}.json", smoke, **(agr_over or {}))))
    if sens:
        (d / ("sensitivity_held_reserve.json" if sfx == "_reserve" else "sensitivity_held.json")).write_text(
            json.dumps(sens_file()))
    return raw


def run(results, argv, smoke_dir=None):
    A.RESULTS, A.SMOKE = Path(results), Path(smoke_dir or Path(results) / "smoke")
    try:
        return A.main(argv)
    except SystemExit as e:
        return e.code


# ---------------------------------------------------------------- the decision table
def base(n=0, pt=1.0):
    return {c: (n, pt) for c in P + S}


def table():
    t = {}
    t["go_both_s_pass"] = base()
    s = base(); s["S2"] = (200, 0.3); t["go_s1_only"] = s
    s = base(); s["S1"] = (62, 0.4); s["S2"] = (30, 0.4); t["go_s_stop_at_s1_rank2"] = s          # S2 k1, S1 k2 fails
    s = base(); s["S1"] = (62, 0.4); s["S2"] = (61, 0.4); t["go_s2_first_s2_passes_s1_not"] = s
    s = base(); s["S1"] = (62, 0.4); s["S2"] = (62, 0.4); t["go_s_tie_s1_first_fails_s2_not_reached"] = s
    s = base(); s["S1"] = (2600, -0.1); s["S2"] = (2500, 0.0); t["go_s_did_not_beat"] = s
    s = base(); s["S1"] = (61, 0.4); s["S2"] = (124, 0.4); t["go_s_at_boundaries"] = s
    for c in P:                                   # each P failing alone: it has the largest count (rank 7)
        s = base(5); s[c] = (125, 0.31); t[f"{c}_fails_alone_point_pos"] = s
        s = base(5); s[c] = (2600, -0.2); t[f"{c}_fails_alone_point_neg"] = s
        s = base(5); s[c] = (2500, 0.0); t[f"{c}_fails_alone_point_zero"] = s
        s = base(5); s[c] = (124, 0.3); t[f"{c}_rank7_at_boundary_passes"] = s
    s = {c: (n, 0.5) for c, n in zip(P, (17, 18, 19, 20, 21, 22, 23))}; t["holm_stop_rank1_rest_not_reached"] = s
    s = {c: (n, 0.5) for c, n in zip(P, (5, 20, 25, 31, 41, 62, 125))}; t["holm_stop_rank2_P2"] = s
    s = {c: (n, 0.5) for c, n in zip(P, (16, 19, 24, 31, 40, 61, 124))}; t["holm_stop_rank4_P4_rest_own_pass"] = s
    s = {c: (n, 0.5) for c, n in zip(P, (16, 19, 24, 30, 40, 61, 124))}; t["go_all_at_boundaries"] = s
    s = {c: (n, 0.5) for c, n in zip(P, (17, 19, 24, 30, 40, 61, 124))}; t["stop_at_first_one_over"] = s
    s = base(5); s["P1"] = (2600, -0.3); s["P5"] = (400, 0.2); t["mixed_nogo_did_not_beat"] = s
    s = base(5); s["P2"] = (400, 0.2); s["P6"] = (300, 0.1); t["nogo_all_inconclusive"] = s
    s = base(5); s["P2"] = (400, 0.2); s["S1"] = (0, 2.0); s["S2"] = (0, 2.0); t["nogo_s_would_pass_untested"] = s
    s = {c: (12, 0.5) for c in P}; s["P7"] = (12, 0.5); t["ties_all_equal_pass"] = s
    s = {c: (30, 0.5) for c in P}; t["ties_all_equal_fail_P1_stops"] = s
    s = {c: (n, 0.4) for c, n in zip(P, (17, 18, 19, 20, 21, 2600, 23))}; s["P6"] = (2600, -0.1)
    t["stop_then_did_not_beat_later"] = s
    s = {c: (n, 0.4) for c, n in zip(P, (17, 18, 19, 20, 21, 22, 23))}; s["P5"] = (21, -0.05)
    t["not_reached_with_point_neg"] = s
    s = base(); s["S1"] = (62, 0.4); s["S2"] = (125, 0.4); t["go_s1_k1_fails_s2_fails_own"] = s
    for v in t.values():
        for c in S:
            v.setdefault(c, (0, 1.0))
    return t


def compare(v, mine):
    d = []
    for f in ("verdict", "kind", "claim"):
        if v.get(f) != mine[f]:
            d.append((f, v.get(f), mine[f]))
    for c in P:
        g, w = v["checks"][c], mine["checks"][c]
        if g["passes"] != w["passes"] or g["not_reached_after"] != w["not_reached_after"]:
            d.append((c, g["passes"], g["not_reached_after"], w))
        if not w["passes"]:
            exp = w["reading"]
            body = g["c11_reading"]
            if exp == "inconclusive":
                ok = body.startswith("inconclusive at a detectable margin of x = ") and "realised half-width" in body
            else:
                ok = body == exp
            if w["not_reached_after"]:
                ok = ok and g["reading"].startswith(f"not reached: the Holm procedure stopped at {w['not_reached_after']}")
                ok = ok and body in g["reading"]
            else:
                ok = ok and g["reading"] == body
            if not ok:
                d.append((c, "reading", g["reading"], exp))
        elif g["reading"] != "passed":
            d.append((c, "reading of a pass", g["reading"]))
    sv, sm = v["secondary"], mine["secondary"]
    if sv["tested"] != sm["tested"]:
        d.append(("secondary tested", sv["tested"], sm["tested"]))
    for c in S:
        if sv[c]["passes"] != sm[c]["passes"]:
            d.append((c, sv[c]["passes"], sm[c]["passes"]))
        if sm["tested"]:
            if sv[c].get("license") != sm[c]["license"] or sv[c]["not_reached_after"] != sm[c]["not_reached_after"]:
                d.append((c, "license/not reached", sv[c].get("license"), sv[c]["not_reached_after"], sm[c]))
            if not sm[c]["passes"]:
                exp, body = sm[c]["reading"], sv[c]["c11_reading"]
                ok = (body.startswith("inconclusive at a detectable margin of x₂ = ") if exp == "inconclusive"
                      else body == exp)
                if not ok:
                    d.append((c, "reading", body, exp))
        elif sv[c].get("reading") is not None:
            d.append((c, "read after a NO-GO"))
    if sorted(b["check"] for b in v["boundary_report"]) != mine["boundary"]:
        d.append(("boundary", sorted(b["check"] for b in v["boundary_report"]), mine["boundary"]))
    return d


def part_table():
    res = {}
    for name, spec in table().items():
        d = TMP / "table" / name
        setup_dir(d, spec)
        code = run(d, [])
        if code != 0:
            res[name] = {"code": code}
            continue
        v = json.loads((d / "held_verdict.json").read_text())
        mine = my_verdict(spec)
        diff = compare(v, mine)
        res[name] = {"code": code, "verdict": v["verdict"], "kind": v["kind"], "n_diffs": len(diff), "diffs": diff[:4]}
        # never overwritten: a second run refuses, file unchanged
        before = (d / "held_verdict.json").read_bytes()
        res[name]["second_run_code"] = run(d, [])
        res[name]["unchanged"] = (d / "held_verdict.json").read_bytes() == before
    return res


# ---------------------------------------------------------------- refusals of the apply step
def part_refusals():
    out = {}
    ok_spec = base()

    def case(name, prep, argv=(), want=4, smoke=False, sfx=""):
        d = TMP / "refuse" / name
        if d.exists():
            shutil.rmtree(d)
        res = d / "results"
        tgt = res / "smoke" if smoke else res
        setup_dir(tgt, ok_spec, smoke=smoke, sfx=sfx)
        prep(res, tgt)
        before = {p.name: p.read_bytes() for p in tgt.glob("held_verdict*.json")}
        code = run(res, list(argv))
        after = {p.name: p.read_bytes() for p in tgt.glob("held_verdict*.json")}
        written = sorted(k for k in after if before.get(k) != after[k])
        out[name] = {"code": code, "want": want, "ok": code == want and (bool(written) == (want == 0)),
                     "verdicts": written}

    def edit_agr(**kw):
        def f(res, tgt, sfx=""):
            p = tgt / f"rederive_agreement{sfx}.json"
            r = json.loads(p.read_text())
            for k, v in kw.items():
                if v is KeyError:
                    r.pop(k, None)
                else:
                    r[k] = v
            p.write_text(json.dumps(r))
        return f

    def edit_pass(fn):
        def f(res, tgt):
            p = tgt / "held_pass.json"
            r = json.loads(p.read_text())
            fn(r)
            raw = json.dumps(r).encode()
            p.write_bytes(raw)
            a = json.loads((tgt / "rederive_agreement.json").read_text())
            a["held_pass_sha256"] = hashlib.sha256(raw).hexdigest()
            (tgt / "rederive_agreement.json").write_text(json.dumps(a))
        return f

    nop = (lambda res, tgt: None)
    case("control_real_ok", nop, want=0)
    case("control_smoke_ok", nop, ["--smoke"], want=0, smoke=True)
    case("agreement_missing", lambda r, t: (t / "rederive_agreement.json").unlink())
    for k, v in (("phase", 1), ("phase", "2"), ("phase", 2.0), ("phase", True), ("phase", KeyError),
                 ("n_quantities", 0), ("n_quantities", "9"), ("n_quantities", KeyError), ("n_quantities", True),
                 ("disagreements", KeyError), ("disagreements", ["P3"]), ("disagreements", None),
                 ("disagreements", {}), ("pass_file", "results/held_pass.json"), ("pass_file", "held_pass_fix1.json"),
                 ("pass_file", KeyError), ("all_agree", False), ("all_agree", "true"), ("all_agree", 1),
                 ("smoke", True), ("smoke", "false"), ("smoke", KeyError), ("held_pass_sha256", "0" * 64),
                 ("held_pass_sha256", KeyError)):
        case(f"agr_{k}_{v if v is not KeyError else 'missing'}", edit_agr(**{k: v}))
    case("smoke_mode_refuses_real_agreement",
         lambda r, t: edit_agr(smoke=False)(r, t), ["--smoke"], smoke=True)
    case("pass_rule_sha", edit_pass(lambda r: r.update(rule_sha256="f" * 64)))
    case("pass_mode_smoke_in_real", edit_pass(lambda r: r.update(mode="smoke")))
    case("pass_mode_regression", edit_pass(lambda r: r.update(mode="regression")))
    case("pass_seeds", edit_pass(lambda r: r.update(seeds=[52, 53])))
    case("pass_flag_tampered_exit5", edit_pass(lambda r: r["checks"]["P1"].update(passes=False)), want=5)
    case("pass_order_tampered_exit5", edit_pass(lambda r: r.update(holm_order=list(reversed(r["holm_order"])))),
         want=5)
    case("pass_boundary_flag_tampered_exit5", edit_pass(lambda r: r["checks"]["P3"].update(near_boundary=True)),
         want=5)
    case("verdict_exists", lambda r, t: (t / "held_verdict.json").write_text("{}"))
    case("sensitivity_missing", lambda r, t: (t / "sensitivity_held.json").unlink())
    case("smoke_subdir_in_real_mode", nop, ["--smoke-subdir", "fix1"])
    case("reserve_without_original_verdict", nop, ["--reserve"])
    case("smoke_and_reserve", nop, ["--smoke", "--reserve"], want=2)

    def fix1_without_agreement(res, tgt):
        raw = json.dumps(pass_file(ok_spec)).encode() + b" "
        (tgt / "held_pass_fix1.json").write_bytes(raw)
    case("fix1_pass_needs_its_own_agreement", fix1_without_agreement)

    def fix1_ok(res, tgt):
        raw = json.dumps(pass_file(base(), seeds=R.HELD_SEEDS)).encode() + b"  "
        (tgt / "held_pass_fix1.json").write_bytes(raw)
        (tgt / "rederive_agreement_fix1.json").write_text(json.dumps(agreement(raw, "held_pass_fix1.json")))
    case("fix1_pair_used", fix1_ok, want=0)

    def fix1_agr_names_first(res, tgt):
        raw = json.dumps(pass_file(base())).encode() + b"  "
        (tgt / "held_pass_fix1.json").write_bytes(raw)
        (tgt / "rederive_agreement_fix1.json").write_text(json.dumps(agreement(raw, "held_pass.json")))
    case("fix1_agreement_naming_first_pass", fix1_agr_names_first)

    def reserve_ok(res, tgt):
        (tgt / "held_verdict.json").write_text("{}")
        setup_dir(tgt, base(), sfx="_reserve")
    case("reserve_ok", reserve_ok, ["--reserve"], want=0)

    def reserve_no_sens(res, tgt):
        (tgt / "held_verdict.json").write_text("{}")
        setup_dir(tgt, base(), sfx="_reserve", sens=False)
    case("reserve_needs_its_sensitivity", reserve_no_sens, ["--reserve"])

    def smoke_subdir_ok(res, tgt):
        setup_dir(res / "smoke" / "fix1", base(), smoke=True)
    d = TMP / "refuse" / "smoke_subdir_fix1"
    if d.exists():
        shutil.rmtree(d)
    smoke_subdir_ok(d / "results", None)
    code = run(d / "results", ["--smoke", "--smoke-subdir", "fix1"])
    out["smoke_subdir_fix1"] = {"code": code, "want": 0,
                                "ok": code == 0 and (d / "results/smoke/fix1/held_verdict.json").is_file()}
    # module SHAs against the latest smoke record (contracts section 8, amendment 12:20: ticket 15 adds it to
    # the apply step): a smoke record with other SHAs beside a valid real pair
    def stale_smoke_record(res, tgt):
        (tgt / "smoke_record.json").write_text(json.dumps({"passed": True, "module_sha256": {"x": "0" * 64}}))
    case("stale_smoke_record_exit4_expected", stale_smoke_record)
    return out


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    res = {"table": part_table(), "refusals": part_refusals()}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    t = res["table"]
    bad = {k: v for k, v in t.items() if v.get("code") != 0 or v.get("n_diffs") or v.get("second_run_code") != 4
           or not v.get("unchanged")}
    print(f"table: {len(t)} cases, {len(bad)} differ: {sorted(bad)[:10]}")
    r = res["refusals"]
    badr = {k: (v["code"], v["want"]) for k, v in r.items() if not v["ok"]}
    print(f"refusals: {len(r)} cases, {len(badr)} not as expected: {badr}")


if __name__ == "__main__":
    main()
