"""Apply DECISION_RULE.md section 5 items 3 to 5 (development bar, carry, kill) and the section 8 rows, mechanically, to
the stored A0 candidates of seed 42: results/cand_R{1,2,3}_A0.json. Writes results/rule_application.{json,txt} (never
overwritten; --smoke writes to results/smoke/). A candidate without a results file is reported as missing (dropped when
the rule is applied at the cutoff). R3 with k* = 4 (probs_R3_A0.json says r3_is_r1) is R1: it has R1's numbers and is
never carried. Also written, clearly labelled as diagnostics that decide nothing: the paired per-anchor differences of
item 2 (R1 minus round-1 R-c, R2 minus R1, R3 minus R1) with intervals.

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/r2_apply_rule.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_common as R  # noqa: E402

C = R.C
ORDER = ("R1", "R2", "R3")                                  # item 4 tie order
BAR, TIE = C.BAR_TARGET, 0.05                               # +0.5 pp (D12 clause 1), 0.05 pp (item 4)
CONFIG = "A0"


def find(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def row_of(rec):
    bar, gain = rec["bar"], rec["gain_statistic"]
    r = {"bar_margin": bar["r1"]["point"], "bar_ci95": bar["r1"]["ci95"], "comparator": bar["comparator"],
         "gain_statistic": gain["point"], "gain_ci95": gain["ci95"],
         "margin_r1": rec["margin"]["r1"]["point"], "fused_r1": rec["r1_means"]["fused"],
         "counterpart_r1": rec["r1_means"]["counterpart"]}
    r["clause1"] = bool(r["bar_margin"] >= BAR)
    r["clause2"] = bool(r["bar_ci95"][0] > 0)
    r["clause3"] = bool(r["gain_ci95"][0] > 0)
    r["clears"] = r["clause1"] and r["clause2"] and r["clause3"]
    return r


def load_candidate(reader, smoke):
    d = R.res_dir(smoke)
    pj, pz = d / f"cand_{reader}_{CONFIG}.json", d / f"cand_{reader}_{CONFIG}.npz"
    if not pj.exists():
        return None, None
    rec = json.loads(pj.read_text())
    if rec["provenance"]["rule_sha256"] != R.RULE_SHA:
        raise SystemExit(f"{pj} was written under another rule")
    if bool(rec["smoke"]) != bool(smoke) or rec["reader"] != reader or rec["config"] != CONFIG:
        raise SystemExit(f"{pj}: other reader, configuration or smoke flag")
    if R.sha_file(pz) != rec["npz_sha256"]:
        raise SystemExit(f"{pz} differs from its json (SHA-256)")
    return rec, np.load(pz)


def r3_is_r1(smoke):
    pj = R.res_dir(smoke) / f"probs_R3_{CONFIG}.json"
    return pj.exists() and bool(json.loads(pj.read_text()).get("r3_is_r1"))


def apply_rule(rows):
    """rows: {reader: row or None}; R3 identical to R1 is a row with 'identical_to'. -> decision dict (items 3 to 5)."""
    eligible = [n for n in ORDER if rows.get(n) and not rows[n].get("identical_to") and rows[n]["clears"]]
    if eligible:
        M = max(rows[n]["bar_margin"] for n in eligible)
        tied = [n for n in eligible if rows[n]["bar_margin"] >= M - TIE]
        carried = min(tied, key=ORDER.index)
        action = f"carry {carried} to the fresh-seed test (DECISION_RULE.md section 6)"
    else:
        M, tied, carried = None, [], None
        action = ("item 5: no candidate clears the development bar; no test is built; the seed-42 results go to the "
                  "user, who decides what follows; the A1 ablation runs on the best development candidate "
                  "(section 4.8)")
    real = [n for n in ORDER if rows.get(n) and not rows[n].get("identical_to")]
    best = carried or (max(real, key=lambda n: (rows[n]["bar_margin"], -ORDER.index(n))) if real else None)
    return {"eligible": eligible, "largest_bar_margin_in_E": M, "tied": tied, "carried": carried, "action": action,
            "ablation_candidate": best, "ablation_label": (None if best is None else
                                                          ("carried candidate" if carried else
                                                           "best development candidate, not carried"))}


def paired(x, y, cl):
    """x - y per anchor with the painting-clustered interval, for fused R@1, margin and bar margin (pp)."""
    out = {}
    for label, f in (("fused_r1", lambda z: np.asarray(z["fused__r1"], np.float64)),
                     ("margin_r1", lambda z: np.asarray(z["fused__r1"], np.float64) - np.asarray(z["cf__r1"], np.float64)),
                     ("bar_margin_r1", lambda z: np.asarray(z["bar_v"], np.float64))):
        out[label] = C.point_ci(f(x) - f(y), cl)
    return out


def diagnostics(zs, smoke):
    """Item 2's paired differences; decide nothing."""
    diag = {"note": "diagnostics of item 2: they decide nothing; paired per anchor, pp, 95% painting-bootstrap intervals"}
    cl = None
    for z in zs.values():
        if z is not None:
            cl = np.asarray(z["anchor_group"])
            break
    if cl is None:
        return diag
    if zs.get("R1") is not None:
        rc_npz = R.r1_path("results/cand_Rc_Rb_expected_A0.npz")
        R.assert_inputs(["results/cand_Rc_Rb_expected_A0.npz"])
        zc = np.load(rc_npz)
        if len(zc["fused__r1"]) != len(cl):
            diag["R1_minus_round1_Rc"] = f"not computed: {len(cl)} episodes here, round 1's R-c has {len(zc['fused__r1'])}"
        else:
            diag["R1_minus_round1_Rc"] = {**paired(zs["R1"], zc, cl), "meaning": "the top-k restriction alone"}
    for n in ("R2", "R3"):
        if zs.get(n) is not None and zs.get("R1") is not None:
            diag[f"{n}_minus_R1"] = {**paired(zs[n], zs["R1"], cl), "meaning": "the reader fix, same fusion family"}
    return diag


def ci_text(x):
    return f"{x['point']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    R.assert_rule()
    out = R.res_dir(args.smoke)
    out_json, out_txt = out / "rule_application.json", out / "rule_application.txt"
    R.refuse_existing([out_json, out_txt], args.smoke)

    recs, zs, rows, notes = {}, {}, {}, {}
    for n in ORDER:
        rec, z = load_candidate(n, args.smoke)
        recs[n], zs[n] = rec, z
        if rec is not None:
            rows[n] = {**row_of(rec), "file_sha256": R.sha_file(out / f"cand_{n}_{CONFIG}.json")}
    if recs["R3"] is None and r3_is_r1(args.smoke):
        if recs["R1"] is None:
            notes["R3"] = "k* = 4: R3 is R1, which is missing"
        else:
            rows["R3"] = {**{k: v for k, v in rows["R1"].items() if k != "file_sha256"}, "identical_to": "R1"}
            notes["R3"] = "k* = 4: R3 is R1 (rule 4.4 h); it has R1's numbers and cannot be carried"
    elif recs["R3"] is not None and r3_is_r1(args.smoke):
        raise SystemExit("R3 has both a candidate file and r3_is_r1: stop and report")
    missing = [n for n in ORDER if rows.get(n) is None]
    decision = apply_rule(rows)
    diag = diagnostics(zs, args.smoke)
    result = {"rule_sha256": R.RULE_SHA, "time_amsterdam": R.now_ams(), "smoke": bool(args.smoke), "git_head": C.git_head(),
              "script_sha256": R.sha_file(__file__), "config": CONFIG, "bar_pp": BAR, "tie_pp": TIE,
              "candidates": rows, "missing": missing, "notes": notes, "complete": not missing, **decision,
              "diagnostics_decide_nothing": diag}
    C.assert_finite_tree(C.jsonable(result))
    out_json.write_text(json.dumps(C.jsonable(result), indent=1))

    L = [f"Rule application (DECISION_RULE.md {R.RULE_SHA[:12]}), {result['time_amsterdam']}"
         f"{'  [SMOKE: not a result]' if args.smoke else ''}",
         f"{'candidate':10s} {'bar margin':>10s}  {'95% interval':>18s}  {'comparator':11s} {'gain stat':>9s}  "
         f"{'gain lower':>10s}  clauses  clears"]
    for n in ORDER:
        r = rows.get(n)
        if r is None:
            L.append(f"{n:10s} missing (no results file; dropped if the rule is applied at the cutoff)")
            continue
        L.append(f"{n:10s} {r['bar_margin']:+10.4f}  [{r['bar_ci95'][0]:+.3f}, {r['bar_ci95'][1]:+.3f}]  "
                 f"{r['comparator']:11s} {r['gain_statistic']:+9.3f}  {r['gain_ci95'][0]:+10.3f}  "
                 f"{int(r['clause1'])}{int(r['clause2'])}{int(r['clause3'])}      {r['clears']}"
                 + (f"  (= {r['identical_to']}, cannot be carried)" if r.get("identical_to") else ""))
    L += [f"missing: {missing or 'none'}", f"eligible: {decision['eligible'] or 'none'}",
          f"tied within {TIE} of the largest bar margin: {decision['tied'] or 'none'}",
          f"carried: {decision['carried']}", f"action: {decision['action']}",
          f"A1 ablation candidate: {decision['ablation_candidate']} ({decision['ablation_label']})",
          "", "Diagnostics of item 2 (decide nothing); paired per anchor, pp:"]
    for k, v in diag.items():
        if k == "note":
            continue
        if isinstance(v, str):
            L.append(f"  {k}: {v}")
        else:
            L.append(f"  {k} ({v['meaning']}): fused R@1 {ci_text(v['fused_r1'])}; margin {ci_text(v['margin_r1'])}; "
                     f"bar margin {ci_text(v['bar_margin_r1'])}")
    out_txt.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
