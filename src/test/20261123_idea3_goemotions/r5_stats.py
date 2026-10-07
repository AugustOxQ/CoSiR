"""Round 5 comparators, development record, D10 and carry (DECISION_RULE.md D8, D9, D10, section 5 items 5 to 8).
Pure numpy functions on per-anchor dicts ({"r1", "gain", "other", ...} fraction arrays) and on family dicts shaped like
round 3's `run_family` output. Round 4's `bar_comparator` is reused as is; the record, the carry and the comparator
list are this round's own (round 4's are tied to V4/V2/V24 and to 24 as its own TIE_BAND).

Intervals are round 1's `common.point_ci` (5,000-resample anchor-painting bootstrap, seed 42, percentage points).
Nothing here takes a placement: the arrays it receives are already computed under the guard of r5_guard.
"""
import numpy as np

import r5_common as R5

C, RF3, RS4 = R5.C, R5.RF3, R5.RS4
F = RF3.F
BOUNDARY_EPS = 1e-12
TIE_BAND = R5.TIE_BAND_UNITS
CANDIDATES = R5.CANDIDATES
LABELS = ("Bprime_G", "Bprime_A0", "counterpart", "B")


def _f64(x):
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------- comparators (D8)

def comparators(pB, pBp0, pBpG, cf):
    """The condition-free comparators in D8's tie order: B'_G, B'(A0), the candidate's matched counterpart, B.
    -> list of (label, per_anchor)."""
    return [("Bprime_G", pBpG), ("Bprime_A0", pBp0), ("counterpart", cf), ("B", pB)]


bar_comparator = RS4.bar_comparator     # (comps, mask=None) -> (label, per_anchor, {label: mean R@1 fraction})


def _assert_cf(cf):
    if not np.all(_f64(cf["gain"]) == 0):
        raise AssertionError("the counterpart's condition gain is not exactly 0")


# ---------------------------------------------------------------- D9, D10

def delta_k(fam, aff_fam) -> int:
    """D9: sum over the episodes of 4 * (candidate fused R@1 - AFF fused R@1), from integer per-episode values
    (round 2's as_int4 asserts multiples of 0.25). A Python int."""
    a = F.as_int4(fam["fused"]["r1"], "candidate r1").astype(np.int64)
    b = F.as_int4(aff_fam["fused"]["r1"], "AFF r1").astype(np.int64)
    if a.shape != b.shape:
        raise AssertionError("candidate and AFF per-episode arrays differ in length")
    return int((a - b).sum())


def d10(record) -> dict:
    """D10 on a record's bar margin and gain statistic: clause 1 point >= 0.5, clause 2 bar lower bound > 0,
    clause 3 gain statistic lower bound > 0. Boundary flags: a value within 1e-12 of its threshold.
    -> {"clauses": {c1, c2, c3, clears}, "boundaries": [str]}."""
    name = record.get("name", "?")
    bar, gain = record["bar_margin"], record["gain_statistic"]
    c1 = bool(bar["point"] >= C.BAR_TARGET)
    c2 = bool(bar["ci95"][0] > 0)
    c3 = bool(gain["ci95"][0] > 0)
    b = []
    for lbl, val, thr in (("D10 clause 1 (bar margin point vs 0.5)", bar["point"], C.BAR_TARGET),
                          ("D10 clause 2 (bar margin lower bound vs 0)", bar["ci95"][0], 0.0),
                          ("D10 clause 3 (gain statistic lower bound vs 0)", gain["ci95"][0], 0.0)):
        if abs(val - thr) <= BOUNDARY_EPS:
            b.append(f"{name}: {lbl}: {val!r} within 1e-12 of {thr}")
    return {"clauses": {"c1": c1, "c2": c2, "c3": c3, "clears": bool(c1 and c2 and c3)}, "boundaries": b}


# ---------------------------------------------------------------- the development record (section 5 item 5)

def dev_record(name, fam, aff_fam, pB, pBp0, pBpG, pBp1, cl, pair_index):
    """One candidate's seed-42 development numbers (rule section 5 item 5). fam / aff_fam: run_family outputs
    (aff_fam None gives no Delta, for the item 4 regression path). pBp1: B'(A1)'s per-anchor dict or None (beside).
    cl, pair_index: arrays over the episodes."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}")
    cl, pair_index = np.asarray(cl), np.asarray(pair_index)
    fu, cf = fam["fused"], fam["cf"]
    _assert_cf(cf)
    label, comp, means = bar_comparator(comparators(pB, pBp0, pBpG, cf))
    v = _f64(fu["r1"]) - _f64(comp["r1"])
    bar = C.point_ci(v, cl)
    d3 = C.diff3(fu, cf, cl)
    gain = d3["gain"]
    rec = {"name": name,
           "fused_r1": 100 * float(np.mean(fu["r1"])), "cf_r1": 100 * float(np.mean(cf["r1"])),
           "cells": {"fpick": dict(fam["fpick"]), "cpick": dict(fam["cpick"])}, "sigma": dict(fam["sigma"]),
           "bar_comparator": label, "comparator_means": {k: 100 * m for k, m in means.items()},
           "bar_margin": bar, "margin_vs_counterpart": d3["r1"], "gain_statistic": gain,
           "either_change": d3["either"]["point"],
           "per_pair_bar_margin": {p: C.point_ci(v[pair_index == i], cl[pair_index == i])
                                   for i, p in enumerate(C.POOLED_ORDER)},
           "cell_text": {k: {int(h): RF3.describe(int(c)) for h, c in fam[p].items()}
                         for k, p in (("fused", "fpick"), ("cf", "cpick"))},
           "Bprime_G_minus_Bprime_A0": C.point_ci(_f64(pBpG["r1"]) - _f64(pBp0["r1"]), cl),
           "beside_Bprime_A1": None, "delta_int": None, "delta": None}
    k = d10({**rec, "name": name})
    rec["d10"], rec["boundaries"] = k["clauses"], list(k["boundaries"])
    if pBp1 is not None:
        rec["beside_Bprime_A1"] = {"mean_r1": 100 * float(np.mean(pBp1["r1"])),
                                   "candidate_minus": C.point_ci(_f64(fu["r1"]) - _f64(pBp1["r1"]), cl)}
    if aff_fam is not None:
        di = delta_k(fam, aff_fam)
        n = len(fu["r1"])
        rec["delta_int"] = di
        if di == 0:
            rec["boundaries"].append(f"{name}: Delta_k is exactly 0")
        rec["delta"] = {"point": 100.0 * di / (4 * n),
                        "ci95": C.point_ci(_f64(fu["r1"]) - _f64(aff_fam["fused"]["r1"]), cl)["ci95"]}
    return rec


# ---------------------------------------------------------------- carry (items 7 and 8)

def carry(records, dev_seed42_sha256=None):
    """Section 5 items 7 and 8. records: {name: dev_record}. E = candidates that clear all of D10 with Delta_k > 0
    (integers); M = largest Delta_k in E; tied = members of E with M - Delta_k <= 24; carried = first tied in the order
    G-T, G-TF; E empty is a kill. The returned dict is what results/carry.json holds (the runner writes it):
    E, M, tied, carried, kill, boundaries, per-candidate Delta_k and D10 clauses, the SHA-256 of
    results/dev_seed42.json (holds tau') and this rule's SHA-256."""
    assert set(records) == set(CANDIDATES), f"carry needs exactly {CANDIDATES}, got {sorted(records)}"
    order = list(CANDIDATES)
    for n in order:
        if type(records[n]["delta_int"]) is not int:
            raise AssertionError(f"{n}: Delta_k must be a Python int, got {type(records[n]['delta_int'])}")
    E = [n for n in order if records[n]["d10"]["clears"] and records[n]["delta_int"] > 0]
    bnd = [x for n in order for x in records[n].get("boundaries", [])]
    out = {"E": E, "M": None, "tied": [], "carried": None, "kill": not E,
           "candidates": {n: {"delta_int": records[n]["delta_int"], "d10": dict(records[n]["d10"])} for n in order},
           "dev_seed42_sha256": dev_seed42_sha256, "rule_sha256": R5.RULE_SHA}
    if E:
        M = max(records[n]["delta_int"] for n in E)
        tied = [n for n in E if M - records[n]["delta_int"] <= TIE_BAND]
        bnd += [f"{n}: tie gap M - Delta_k is exactly {TIE_BAND}" for n in E
                if M - records[n]["delta_int"] == TIE_BAND]
        out.update({"M": M, "tied": tied, "carried": tied[0]})
    out["boundaries"] = bnd
    return out


def console_line(c) -> str:
    """The console line of items 7 and 8."""
    if c["carried"] is None:
        return "KILL (pending the phase-1 agreement, rule §8)"
    return f"CARRY {c['carried']} (pending the phase-1 agreement, rule §8)"
