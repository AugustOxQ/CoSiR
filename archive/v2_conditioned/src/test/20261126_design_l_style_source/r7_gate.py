"""Round 7 seed-42 statistics (DECISION_RULE.md L15, L16, §5 K1, §6 S1 to S3, P1, chains, flags). Pure functions on
per-anchor arrays; no data loading, no file writes.

- L16 integer form: `sum4` (asserts multiples of 0.25 within 1e-9), `need(a, N)` = ceil(a * N / 25) in exact
  arithmetic, `point_of` = 25 * sum / N, `interval` = C9's `cluster_bootstrap` (anchor-painting clusters of the scope,
  5,000 resamples, seed 42) scaled to R@1 points.
- L15: `bar_comparator` (largest mean R@1 of B1, BX, CF[s], B0, B; ties to the earliest; decided on the integer sum of
  4 * R@1, which is the mean at full precision because every value is a multiple of 0.25); bar margin paired per
  anchor; gain statistic = s.gain - CF.gain with CF.gain = 0 asserted.
- §5 K1: `k1_zone`, `ceiling_stats` (STOP / WEAK / CLEAR).
- §6: `evaluate_scorer` (S1, S2, S3, pass), `gate_stats` (P1 per family, n, alpha, "promising", random-control flag;
  references and controls under their own key).

Array layout is the index's `seed42/arrays.npz`: per scorer `{name}__{r1,gain}` and `{name}_cf__{r1,gain}`; `B1__r1`,
`B0__r1`, `B__r1`, `BX_{X}__r1`, `AFF__r1`, `cl`, `pair_index` (0 emotion__style, 1 emotion__genre, 2 style__genre).
Scorer names `R1_{X}` and `XAFF_{X}`. Differences go in as FRACTIONS; points and intervals come out in R@1 points.
Thresholds scale with the episode count (`need`): 246, 82, 164, 50, 0, -49 at the real 12,288 / 4,096.
"""
import hashlib
import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np

QUARTER_TOL = 1e-9
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
BAR_ORDER = ("B1", "BX", "CF", "B0", "B")                 # L15 tie order
VARIANTS = tuple(f"V{i}" for i in range(1, 7))
KINDS = ("R1", "XAFF")                                    # P1 tie order: R1 before X-AFF
CANDIDATES = tuple(f"{k}_{v}" for v in VARIANTS for k in KINDS)
FAMILIES = {"csd": ("V1", "V2", "V3", "V4"), "generic": ("V5", "V6")}
ALPHA = {2: 0.0125, 1: 0.025}                             # §6 chains: n = 2 -> 0.0125 each, n = 1 -> 0.025
ZONES = ("STOP", "WEAK", "CLEAR")

# rule numbers at the real sizes, checked at import (L16)
_REAL = {(0.5, 12288): 246, (0.5, 4096): 82, (1.0, 4096): 164, (0.3, 4096): 50, (0, 4096): 0, (-0.3, 4096): -49}



# ------------------------------------------------------------------------------------------------ L16 integer form
def need(a, n):
    """ceil(a * n / 25), exact (a is taken as its decimal text, so 0.3 is 3/10)."""
    f = Fraction(str(a)) * n / 25
    return -((-f.numerator) // f.denominator)


def sum4(d):
    """Integer sum of 4 * d for per-episode differences d (fractions); asserts multiples of 0.25 within 1e-9."""
    q = 4.0 * np.asarray(d, dtype=np.float64)
    r = np.rint(q)
    assert np.all(np.abs(q - r) <= 4 * QUARTER_TOL), "a per-episode value is not a multiple of 0.25"
    return int(r.sum())


def point_of(s4, n):
    """R@1 points: 25 * sum(4d) / n."""
    return 25.0 * s4 / n


def _cluster_bootstrap():
    """`cluster_bootstrap` of src/eval/aspect_metrics.py, through `r7_common`'s loader (SHA-256 asserted, rule §10)."""
    import r7_common
    return r7_common.load("src.eval.aspect_metrics").cluster_bootstrap


def interval(d, clusters, boot=None):
    """(point, [lo, hi]) in R@1 points: C9's bootstrap of the paired differences d over the scope's clusters."""
    d = np.asarray(d, dtype=np.float64)
    assert len(d) == len(clusters), "values and clusters differ in length"
    r = (boot or _cluster_bootstrap())(d, clusters)
    return 100.0 * float(r["point"]), [100.0 * float(r["ci95"][0]), 100.0 * float(r["ci95"][1])]


# ----------------------------------------------------------------------------------------------- K1 (§5)
def k1_zone(s4, lower, n):
    """STOP if sum < need(0.5) or lower <= 0; WEAK if sum < need(1.0) (lower > 0); else CLEAR. `lower` in any unit."""
    if s4 < need(0.5, n) or not lower > 0:
        return "STOP"
    return "WEAK" if s4 < need(1.0, n) else "CLEAR"


def ceiling_stats(fused_r1, cf_r1, cl, boot=None):
    """The ceiling check on the style x genre episodes: d = fused - counterpart R@1, paired per anchor."""
    fused_r1, cf_r1, cl = np.asarray(fused_r1), np.asarray(cf_r1), np.asarray(cl)
    assert len(fused_r1) == len(cf_r1) == len(cl), "ceiling arrays differ in length"
    d = fused_r1 - cf_r1
    s4, n = sum4(d), len(d)
    pt, ci = interval(d, cl, boot)
    assert abs(pt - point_of(s4, n)) < 1e-9, "bootstrap point and 25*sum/n disagree"
    return {"sum4d": s4, "n": n, "point": point_of(s4, n), "ci95": ci, "zone": k1_zone(s4, ci[0], n)}


# ----------------------------------------------------------------------------------------------- L15 comparator
def bar_comparator(r1_by_name):
    """Name among BAR_ORDER whose R@1 array has the largest mean over the scope; ties to the earliest in the order."""
    assert set(r1_by_name) == set(BAR_ORDER), f"comparators must be {BAR_ORDER}"
    sums = {k: sum4(r1_by_name[k]) for k in BAR_ORDER}    # same scope, same n: the largest sum is the largest mean
    best = BAR_ORDER[0]
    for k in BAR_ORDER[1:]:
        if sums[k] > sums[best]:
            best = k
    return best


# -------------------------------------------------------------------------------------------- §6 S1 to S3, P1
def split_name(name):
    """'R1_V3' -> ('R1', 'V3'); 'XAFF_Z6' -> ('XAFF', 'Z6')."""
    kind, x = name.split("_", 1)
    assert kind in KINDS, f"unknown scorer kind in {name}"
    return kind, x


def _scope(arr):
    cl, pi = np.asarray(arr["cl"]), np.asarray(arr["pair_index"])
    n = len(cl)
    assert len(pi) == n and n % 3 == 0, "cl and pair_index differ in length, or n is not a multiple of 3"
    masks = [pi == p for p in range(3)]
    assert all(int(m.sum()) == n // 3 for m in masks), "pair_index must hold three equal pairs of episodes (0, 1, 2)"
    return cl, masks


def _need(arr, key):
    assert key in arr, f"missing array {key}"
    return np.asarray(arr[key], dtype=np.float64)


def evaluate_scorer(arr, name, boot=None):
    """S1 to S3 of one scorer on seed 42. Returns (record, extras): the record has the index's core keys; extras
    holds the style x genre sum of s - AFF that "promising" needs."""
    kind, x = split_name(name)
    cl, masks = _scope(arr)
    n, n_pair = len(cl), len(cl) // 3
    s_r1, s_gain = _need(arr, f"{name}__r1"), _need(arr, f"{name}__gain")
    cf_r1, cf_gain = _need(arr, f"{name}_cf__r1"), _need(arr, f"{name}_cf__gain")
    assert not np.any(cf_gain), f"{name}: the counterpart's condition gain must be 0 everywhere"
    comps = {"B1": _need(arr, "B1__r1"), "BX": _need(arr, f"BX_{x}__r1"), "CF": cf_r1,
             "B0": _need(arr, "B0__r1"), "B": _need(arr, "B__r1")}
    bar = bar_comparator(comps)
    d_bar = s_r1 - comps[bar]
    d_gain = s_gain - cf_gain
    bar4, gain4 = sum4(d_bar), sum4(d_gain)
    _, bar_ci = interval(d_bar, cl, boot)
    _, gain_ci = interval(d_gain, cl, boot)
    s1 = {"sum4d": bar4, "ci95": bar_ci, "gain_sum4d": gain4, "gain_ci95": gain_ci,
          "pass": bool(bar4 >= need(0.5, n) and bar_ci[0] > 0 and gain_ci[0] > 0)}
    sg = masks[2]
    s2_4 = sum4((s_r1 - cf_r1)[sg])
    s2 = {"sum4d": s2_4, "pass": bool(s2_4 >= need(0, n_pair))}
    aff = _need(arr, "AFF__r1")
    s3 = {}
    for p in (0, 1):
        v = sum4((s_r1 - aff)[masks[p]])
        s3[PAIRS[p]] = {"sum4d": v, "pass": bool(v >= need(-0.3, n_pair))}
    s3["pass"] = bool(s3[PAIRS[0]]["pass"] and s3[PAIRS[1]]["pass"])
    rec = {"bar_comparator": bar, "S1": s1, "S2": s2, "S3": s3, "dropped": False,
           "pass": bool(s1["pass"] and s2["pass"] and s3["pass"])}
    return rec, {"sg_vs_aff_sum4d": sum4((s_r1 - aff)[sg]), "n_pair": n_pair}


def _dropped_record():
    return {"bar_comparator": None, "S1": None, "S2": None, "S3": None, "dropped": True, "pass": False}


def _variant_key(name):
    kind, x = split_name(name)
    return (VARIANTS.index(x) if x in VARIANTS else len(VARIANTS), KINDS.index(kind))


def _pick(cands, records, family):
    """P1 for one family: passing scorers only; max pooled bar-margin sum, then max gain sum, then V order, R1 first."""
    ok = [c for c in cands if split_name(c)[1] in FAMILIES[family] and records[c]["pass"]]
    if not ok:
        return None
    return min(ok, key=lambda c: (-records[c]["S1"]["sum4d"], -records[c]["S1"]["gain_sum4d"], _variant_key(c)))


def gate_stats(arr, candidates=CANDIDATES, references=(), dropped=(), boot=None, require_controls=True):
    """The seed-42 screen. `arr`: arrays dict (module docstring). `dropped`: grouping ids ('V3', 'Z2', 'N') whose
    scorers do not pass and need no arrays. Returns the index's gate keys: candidates, references (R0, N and the
    random controls; they pick nothing), picks {csd, generic}, n, alpha {family: alpha} (picked families only),
    promising, random_control_flag {family: bool | "not evaluable"} (picked families only). "Not evaluable" only when
    the pick's Zk is dropped; a Zk that is neither dropped nor among `references` asserts (`require_controls`)."""
    dropped = set(dropped)
    assert not (set(candidates) & set(references)), "a scorer is both candidate and reference"
    recs, extras = {}, {}
    for name in list(candidates) + list(references):
        if split_name(name)[1] in dropped:
            recs[name] = _dropped_record()
        else:
            recs[name], extras[name] = evaluate_scorer(arr, name, boot)
    picks = {f: _pick(candidates, recs, f) for f in FAMILIES}
    chosen = {f: p for f, p in picks.items() if p}
    n = len(chosen)
    alpha = {f: ALPHA[n] for f in chosen}
    promising = False                                       # §6: stated only at n = 0 (spec §6: no scorer passes)
    for c in candidates:
        if recs[c]["dropped"]:
            continue
        e = extras[c]
        n_pair = e["n_pair"]
        if (n == 0 and e["sg_vs_aff_sum4d"] >= need(0.3, n_pair) and recs[c]["S2"]["sum4d"] >= 0
                and recs[c]["S1"]["sum4d"] > 0):
            promising = True
    flag = {}
    for f, p in chosen.items():
        kind, x = split_name(p)
        control = f"{kind}_Z{x[1:]}"
        if f"Z{x[1:]}" in dropped:
            flag[f] = "not evaluable"
        else:
            assert control in recs or not require_controls, f"random control {control} of the pick {p} is missing"
            flag[f] = bool(recs[control]["pass"]) if control in recs else "not evaluable"
    return {"candidates": {c: recs[c] for c in candidates}, "references": {r: recs[r] for r in references},
            "picks": picks, "n": n, "alpha": alpha, "promising": promising, "random_control_flag": flag}


for _k, _v in _REAL.items():
    assert need(*_k) == _v, f"need{_k} != {_v}"
