"""Round 4 comparators, development record, carry and GO checks (DECISION_RULE.md D8, D9, D10, section 5 items 6 to 9,
section 6.1, 6.5, 6.7). Pure numpy functions on per-anchor dicts ({"r1", "gain", ...} fraction arrays).

Intervals are round 1's common.point_ci (5,000-resample anchor-painting bootstrap, seed 42, chunk 250, percentage
points), the same helper round 3's seed-42 runner used, so AFF's seed-42 numbers reproduce exactly.
"""
import numpy as np

import r4_common as R4

RS3, C, F, RF3 = R4.RS3, R4.C, R4.RF3.F, R4.RF3
BOUNDARY_EPS = 1e-12
TIE_BAND = R4.TIE_BAND_UNITS

BASE_CHECKS = ("vs_cosine", "vs_rca", "vs_B", "vs_Bprime_A0")
AFF_CHECK = "vs_AFF"
_NAMES = {"vs_cosine": "cosine", "vs_rca": "RCA", "vs_B": "B", "vs_Bprime_A0": "B'(A0)", "vs_Bprime_A1": "B'(A1)",
          "vs_counterpart": "the matched counterpart", "vs_AFF": "AFF"}


# ---------------------------------------------------------------- comparators (D8)

def comparators(name, pB, pBp0, pBp1, cf):
    """Condition-free comparators in D8's tie order. V4 and AFF: B'(A0), counterpart, B. V2 and V24: B'(A1), B'(A0),
    counterpart, B. -> list of (label, per_anchor)."""
    if name in ("V4", "AFF", "R1", "IMGABST"):
        return [("Bprime_A0", pBp0), ("counterpart", cf), ("B", pB)]
    if name in ("V2", "V24"):
        return [("Bprime_A1", pBp1), ("Bprime_A0", pBp0), ("counterpart", cf), ("B", pB)]
    raise ValueError(f"unknown candidate {name!r}")


def bar_comparator(comps, mask=None):
    """D8: the comparator with the largest mean R@1 over the episodes in mask (all if None), full precision, ties to the
    earliest of `comps`. -> (label, per_anchor, {label: mean R@1 as a fraction})."""
    sel = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    means = [float(np.mean(np.asarray(p["r1"], dtype=np.float64)[sel])) for _, p in comps]
    best = 0
    for i in range(1, len(comps)):
        if means[i] > means[best]:
            best = i
    return comps[best][0], comps[best][1], {l: m for (l, _), m in zip(comps, means)}


# ---------------------------------------------------------------- the development record (section 5 item 6)

def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _assert_cf(cf, what="counterpart"):
    if not np.all(_f64(cf["gain"]) == 0):
        raise AssertionError(f"the {what}'s condition gain is not exactly 0")


def dev_record(name, fam, aff_fam, pB, pBp0, pBp1, cl, pair_index):
    """One candidate's seed-42 development numbers. fam / aff_fam: RF3.run_family outputs (aff_fam None gives no Delta,
    for the item 3 regression path). cl, pair_index as arrays over the episodes."""
    cl, pair_index = np.asarray(cl), np.asarray(pair_index)
    fu, cf = fam["fused"], fam["cf"]
    _assert_cf(cf)
    label, comp, _ = bar_comparator(comparators(name, pB, pBp0, pBp1, cf))
    v = _f64(fu["r1"]) - _f64(comp["r1"])
    bar = C.point_ci(v, cl)
    d3 = C.diff3(fu, cf, cl)
    gain = d3["gain"]
    c1, c2, c3 = bool(bar["point"] >= C.BAR_TARGET), bool(bar["ci95"][0] > 0), bool(gain["ci95"][0] > 0)
    rec = {"name": name,
           "fused_r1": 100 * float(np.mean(fu["r1"])), "cf_r1": 100 * float(np.mean(cf["r1"])),
           "cells": {"fpick": dict(fam["fpick"]), "cpick": dict(fam["cpick"])}, "sigma": dict(fam["sigma"]),
           "bar_comparator": label, "bar_margin": bar,
           "margin_vs_counterpart": d3["r1"], "gain_statistic": gain, "either_change": d3["either"]["point"],
           "d10": {"c1": c1, "c2": c2, "c3": c3, "clears": bool(c1 and c2 and c3)},
           "per_pair_bar_margin": {p: C.point_ci(v[pair_index == i], cl[pair_index == i])
                                   for i, p in enumerate(C.POOLED_ORDER)},
           "cell_text": {k: {int(h): RF3.describe(int(c)) for h, c in fam[p].items()}
                         for k, p in (("fused", "fpick"), ("cf", "cpick"))},
           "delta_int": None, "delta": None}
    b = []
    for lbl, val, thr in (("D10 clause 1 (bar margin point vs 0.5)", bar["point"], C.BAR_TARGET),
                          ("D10 clause 2 (bar margin lower bound vs 0)", bar["ci95"][0], 0.0),
                          ("D10 clause 3 (gain statistic lower bound vs 0)", gain["ci95"][0], 0.0)):
        if abs(val - thr) <= BOUNDARY_EPS:
            b.append(f"{name}: {lbl}: {val!r} within 1e-12 of {thr}")
    rec["boundaries"] = b
    if aff_fam is not None:
        n = len(fu["r1"])
        di = int((F.as_int4(fu["r1"], "candidate r1").astype(np.int64)
                  - F.as_int4(aff_fam["fused"]["r1"], "AFF r1").astype(np.int64)).sum())
        ci = C.point_ci(_f64(fu["r1"]) - _f64(aff_fam["fused"]["r1"]), cl)["ci95"]
        rec["delta_int"] = di
        if di == 0:
            rec["boundaries"].append(f"{name}: Delta_k is exactly 0")
        rec["delta"] = {"point": 100.0 * di / (4 * n), "ci95": ci}
    return rec


def carry(records):
    """Section 5 items 8 and 9. records: {name: dev_record}. E = candidates that clear D10 with Delta > 0 (integers);
    M = largest Delta in E; tied = members of E with M - Delta <= 24; carried = first tied in the order V4, V2, V24."""
    assert set(records) == set(R4.CANDIDATES), f"carry needs exactly {R4.CANDIDATES}, got {sorted(records)}"
    order = list(R4.CANDIDATES)
    for n in order:
        if type(records[n]["delta_int"]) is not int:
            raise AssertionError(f"{n}: Delta_k must be a Python int, got {type(records[n]['delta_int'])}")
    E = [n for n in order if records[n]["d10"]["clears"] and records[n]["delta_int"] > 0]
    bnd = [x for n in order for x in records[n].get("boundaries", [])]
    if not E:
        return {"E": [], "M": None, "tied": [], "carried": None, "boundaries": bnd}
    M = max(records[n]["delta_int"] for n in E)
    tied = [n for n in E if M - records[n]["delta_int"] <= TIE_BAND]
    bnd += [f"{n}: tie gap M - Delta_k is exactly {TIE_BAND}" for n in E if M - records[n]["delta_int"] == TIE_BAND]
    return {"E": E, "M": M, "tied": tied, "carried": tied[0], "boundaries": bnd}


# ---------------------------------------------------------------- the GO checks (section 6.5)

def _check_names(name):
    if name not in R4.CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}")
    base = list(BASE_CHECKS) + (["vs_Bprime_A1"] if R4.READS_CSD[name] else [])
    return base + ["vs_counterpart", "gain_statistic", "gain_vs_rca", AFF_CHECK]


def _diffs(name, s):
    """{check: per-episode difference array (fractions)} in the fixed order of section 6.5 for one seed's arrays."""
    if name in R4.READS_CSD and R4.READS_CSD[name] and s.get("Bp1") is None:
        raise ValueError(f"{name} reads CSD: B'(A1) is required")
    if s["aff_fused"] is s["fused"] or s["aff_fused"]["r1"] is s["fused"]["r1"]:
        raise AssertionError("AFF's fused arrays are the candidate's")
    _assert_cf(s["cf"])
    n = len(s["cl"])
    for k in ("fused", "cf", "aff_fused", "B", "Bp0", "cosine", "rca") + (("Bp1",) if R4.READS_CSD[name] else ()):
        for m in ("r1", "gain"):
            if len(s[k][m]) != n:
                raise ValueError(f"{k}.{m} and the clusters differ in length")
    r1 = lambda k: _f64(s["fused"]["r1"]) - _f64(s[k]["r1"])
    out = {"vs_cosine": r1("cosine"), "vs_rca": r1("rca"), "vs_B": r1("B"), "vs_Bprime_A0": r1("Bp0")}
    if R4.READS_CSD[name]:
        out["vs_Bprime_A1"] = r1("Bp1")
    out["vs_counterpart"] = r1("cf")
    out["gain_statistic"] = _f64(s["fused"]["gain"]) - _f64(s["cf"]["gain"])
    out["gain_vs_rca"] = _f64(s["fused"]["gain"]) - _f64(s["rca"]["gain"])
    out[AFF_CHECK] = r1("aff_fused")
    assert list(out) == _check_names(name)
    return out


def go_checks(name, per_seed):
    """Eight checks (V4) or nine (V2, V24), the AFF check last, each pooled over the seeds (RS3.pooled_check: point and
    95% interval in pp, pass = lower bound > 0). -> {"checks": {check: {point, ci95, pass}}, "go": all pass}.
    per_seed: list in seed order of dicts with cl, fused, cf, aff_fused, B, Bp0, Bp1 (None unless CSD), cosine, rca."""
    names = _check_names(name)
    per = [_diffs(name, s) for s in per_seed]
    cls = [np.asarray(s["cl"]) for s in per_seed]
    checks = {k: RS3.pooled_check([p[k] for p in per], cls) for k in names}
    return {"checks": checks, "go": bool(all(c["pass"] for c in checks.values()))}


def sensitivity_all(name, seed42):
    """Section 6.1 for every GO check: RS3.sensitivity on the seed-42 per-episode difference. seed42: one per_seed
    element. -> {check: {SE, half_width, x, seed42_half_width, ...}} in go_checks' order."""
    names = _check_names(name)
    d = _diffs(name, seed42)
    return {k: RS3.sensitivity(d[k], np.asarray(seed42["cl"])) for k in names}


# ---------------------------------------------------------------- reading a failed check (section 6.7)

def reading(check_name, point, x, half_width, candidate="the candidate"):
    """The sentence section 6.7 prescribes for one failed check."""
    if point > 0:
        return (f"inconclusive at a detectable margin of {x:.3f} pp "
                f"(realised pooled half-width {half_width:.3f} pp)")
    if check_name in _NAMES:
        what = _NAMES[check_name]
    elif check_name == "gain_statistic":
        what = "the condition-free comparators on condition gain"
    elif check_name == "gain_vs_rca":
        what = "RCA on condition gain"
    else:
        raise KeyError(check_name)
    return f"{candidate} did not beat {what} on fresh episodes"


def aff_only_failed_reading(checks, candidate):
    """If the AFF check is the only failed check: the extra reading of section 6.7; otherwise None."""
    failed = [k for k, v in checks.items() if not v["pass"]]
    if failed == [AFF_CHECK]:
        return f"{candidate} works, but no improvement over AFF was shown"
    return None
