"""Round 8 gate, decision rule M15/M16 and sections 6 to 8. Pure array calculations.

Integer/interval/comparator helpers and the S1-S3 calculation copied from read-only
R7 r7_gate.py (SHA-256 52eb81648d514c5a76c62f94ae61313632da1d59019e3505a61c12c6ab91aa33).
Only the new SC3 gate, selection, trigger and flags differ. Array differences are
fractions; reported points and intervals are percentage points. Inject ``boot``
to run independently of r8_common; the default loads R7's checked C9 bootstrap.
Small synthetic scopes scale integer bars with R7's ``need``; file-level real
runs assert M15's 12,288 pooled anchors and 4,096 anchors per pair.
"""
from copy import deepcopy
from fractions import Fraction

import numpy as np

QUARTER_TOL = 1e-9
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
BAR_ORDER = ("B1", "BX", "CF", "B0", "B")
KINDS = ("R1", "XAFF")
MAIN = ("V100", "V67", "V33", "NB100", "NB67", "NB33", *(f"FC{i}" for i in range(1, 7)), "NBC")
SCREEN = ("AL", "DI", "SG")
TRACKS = {"main": MAIN, "screen": SCREEN}


def _cluster_bootstrap():
    import r8_common
    return r8_common.load_r7("r7_gate")._cluster_bootstrap()


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


def interval(d, clusters, boot=None):
    """(point, [lo, hi]) in R@1 points: C9's bootstrap of the paired differences d over the scope's clusters."""
    d = np.asarray(d, dtype=np.float64)
    assert len(d) == len(clusters), "values and clusters differ in length"
    r = (boot or _cluster_bootstrap())(d, clusters)
    return 100.0 * float(r["point"]), [100.0 * float(r["ci95"][0]), 100.0 * float(r["ci95"][1])]


def bar_comparator(r1_by_name):
    """Name among BAR_ORDER whose R@1 array has the largest mean over the scope; ties to the earliest in the order."""
    assert set(r1_by_name) == set(BAR_ORDER), f"comparators must be {BAR_ORDER}"
    sums = {k: sum4(r1_by_name[k]) for k in BAR_ORDER}    # same scope, same n: the largest sum is the largest mean
    best = BAR_ORDER[0]
    for k in BAR_ORDER[1:]:
        if sums[k] > sums[best]:
            best = k
    return best


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


def _r7_evaluate_scorer(arr, name, boot=None):
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


def evaluate_scorer(arr, name, boot=None):
    """SC3/SC4 and the recorded R7 S1/S2/S3 of one scorer (rule section 6.1)."""
    cl, masks = _scope(arr)
    n = len(cl)
    keys = [f"{name}__r1", f"{name}__gain", f"{name}_cf__r1", f"{name}_cf__gain",
            "AFF__r1", "B1__r1", "B0__r1", "B__r1", f"BX_{split_name(name)[1]}__r1"]
    assert cl.shape == (n,) and n > 0 and np.issubdtype(cl.dtype, np.integer)
    pi = np.asarray(arr["pair_index"])
    assert pi.shape == (n,) and np.issubdtype(pi.dtype, np.integer), "invalid pair_index shape/dtype"
    for key in keys:
        a = _need(arr, key)
        assert a.shape == (n,) and np.isfinite(a).all(), f"{key}: invalid shape or nonfinite values"
        sum4(a)
        if key.endswith("__r1"):
            assert ((a >= 0) & (a <= 1)).all(), f"{key}: R@1 outside [0, 1]"
        else:
            assert ((a >= -1) & (a <= 1)).all(), f"{key}: gain outside [-1, 1]"
    rec, _ = _r7_evaluate_scorer(arr, name, boot)
    d = _need(arr, f"{name}__r1") - _need(arr, "AFF__r1")
    s4 = sum4(d)
    pt, ci = interval(d, cl, boot)
    assert abs(pt - point_of(s4, n)) < 1e-9, "bootstrap point differs from integer point"
    assert np.isfinite(ci).all() and ci[0] <= ci[1], "invalid bootstrap interval"
    rec["SC3"] = {"sum4d": s4, "n": n, "point": point_of(s4, n), "ci95": ci,
                  "pass": bool(ci[0] > 0)}
    rec["SC4"] = deepcopy(rec["S3"])
    for p in (0, 1):
        scoped = d[masks[p]]
        _, pair_ci = interval(scoped, cl[masks[p]], boot)
        rec["SC4"][PAIRS[p]].update(n=n // 3, point=point_of(sum4(scoped), n // 3), ci95=pair_ci)
    s1, s2 = rec["S1"], rec["S2"]
    s1.update(n=n, point=point_of(s1["sum4d"], n), gain_point=point_of(s1["gain_sum4d"], n))
    sg_d = (_need(arr, f"{name}__r1") - _need(arr, f"{name}_cf__r1"))[masks[2]]
    _, sg_ci = interval(sg_d, cl[masks[2]], boot)
    s2.update(n=n // 3, point=point_of(s2["sum4d"], n // 3), ci95=sg_ci)
    rec["gain_sum4d"] = s1["gain_sum4d"]
    rec["r7_pass"] = rec["pass"]
    rec["pass"] = bool(rec["SC3"]["pass"] and rec["SC4"]["pass"])
    return rec


def scorer_names(groupings):
    return tuple(f"{kind}_{x}" for x in groupings for kind in KINDS)


def evaluate_all(arr, names, dropped=None, boot=None):
    """Evaluate named scorers; M16 drops require a nonempty recorded reason."""
    dropped = dropped or {}
    assert isinstance(dropped, dict)
    assert len(names) == len(set(names)), "duplicate scorer names"
    assert all(isinstance(reason, str) and reason.strip() for reason in dropped.values()), "drop needs a reason"
    recs = {}
    for name in names:
        x = split_name(name)[1]
        if x in dropped:
            recs[name] = {"dropped": True, "reason": dropped[x], "pass": False, "r7_pass": False,
                          "SC3": None, "SC4": None, "S1": None, "S2": None, "S3": None,
                          "gain_sum4d": None, "bar_comparator": None}
        else:
            recs[name] = evaluate_scorer(arr, name, boot)
    return recs


def select_scorer(records, track, passing_only=True):
    """Rule 6.4 order; passing_only=False supplies the descriptive H<X> target."""
    order = TRACKS[track]
    assert all(split_name(name)[1] in order for name in records), "noncandidate in selection"
    eligible = [name for name, r in records.items()
                if not r["dropped"] and (r["pass"] or not passing_only)]
    if not eligible:
        return None
    return min(eligible, key=lambda name: (
        -records[name]["SC3"]["sum4d"], -records[name]["gain_sum4d"],
        order.index(split_name(name)[1]), KINDS.index(split_name(name)[0])))


def probe_trigger(records):
    """Rule 6.3: all six NB records, no passing NB, strongest evaluated NB."""
    names = scorer_names(("NB100", "NB67", "NB33"))
    assert set(names) <= set(records), "all six NB scorers must be evaluated or dropped"
    nb = {name: records[name] for name in names}
    available = {name: r for name, r in nb.items() if not r["dropped"]}
    source = None
    reason = "an NB scorer passes"
    if not any(r["pass"] for r in nb.values()):
        if available:
            source = select_scorer(available, "main", passing_only=False)
            reason = "none of the six NB scorers passes"
        else:
            # rule gap: section 6.3's literal "none passes" would trigger even
            # when all six NB scorers drop. Keep triggered=False/no NBC run in
            # that case, preserving ticket 05's trigger/strength contract.
            reason = "no NB scorer was evaluated; NBC does not run"
    strength = {"NB100": 1.0, "NB67": 0.67, "NB33": 0.33}
    return {"nb_records": nb, "triggered": source is not None, "source_scorer": source,
            "strength": strength[split_name(source)[1]] if source else None, "reason": reason}


def track_verdict(candidates, track, references=None, controls=None):
    """Pick, flag, and (main only) promising/H<X>/D31 inputs, from scorer records."""
    references, controls = references or {}, controls or {}
    assert not (set(candidates) & set(references) or set(candidates) & set(controls)
                or set(references) & set(controls)), "scorer has multiple roles"
    pick = select_scorer(candidates, track)
    flag = None
    if pick:
        kind, x = split_name(pick)
        control = f"{kind}_Z{x}"
        assert control in controls, f"missing random control {control} for {pick}"
        flag = "not evaluable" if controls[control]["dropped"] else bool(controls[control]["pass"])
    out = {"candidates": candidates, "references": references, "controls": controls,
           "pick": pick, "random_control_flag": flag}
    if track == "main":
        promising = pick is None and any(not r["dropped"] and all(r[s]["pass"] for s in ("S1", "S2", "S3"))
                                        for r in candidates.values())
        target = pick or select_scorer(candidates, track, passing_only=False)
        head = None
        # rule gap: section 6.6 defines no H<X> target when all main candidates
        # are dropped; record null rather than attempting to rank unevaluated scorers.
        if target:
            kind, x = split_name(target)
            head = {"grouping": x, "scorer_kind": kind, "scorer": f"{kind}_H{x}", "source_scorer": target}
        out.update(promising=bool(promising), head_target=head, main_line_fails_at_gate=pick is None)
    return out


def gate_stats(arr, dropped=None, include_nbc=False, boot=None):
    """Full two-track gate, with separate roles; no grouping is silently omitted."""
    dropped = dropped or {}
    main = MAIN if include_nbc else MAIN[:-1]
    refs = ("R0own", "NRK", "TP")
    # A dropped candidate has no control (M13), unless the control's own drop was explicitly recorded.
    zs = tuple(f"Z{x}" for x in (*main, *SCREEN) if x not in dropped or f"Z{x}" in dropped)
    recs = evaluate_all(arr, scorer_names((*main, *SCREEN, *refs, *zs)), dropped, boot)
    out = {}
    for track, xs in (("main", main), ("screen", SCREEN)):
        candidates = {n: recs[n] for n in scorer_names(xs)}
        references = {n: recs[n] for n in scorer_names(refs if track == "main" else ("R0own",))}
        controls = {n: recs[n] for x in xs for n in scorer_names((f"Z{x}",)) if n in recs}
        out[track] = track_verdict(candidates, track, references, controls)
    out["trigger"] = probe_trigger(out["main"]["candidates"])
    out["fallback"] = {"main_has_pick": out["main"]["pick"] is not None,
                       "main_line_fails_at_gate": out["main"]["pick"] is None,
                       "screen_has_pick": out["screen"]["pick"] is not None,
                       "screen_read_if_main_fails": out["screen"]["pick"] is not None,
                       "screen_eligible_at_gate": out["main"]["pick"] is None and out["screen"]["pick"] is not None,
                       "main_failure_after_held": "held read without Tier 1 (D31); determined under HELD_RULE.md",
                       "held_rule_required": True, "held_start_cutoff": "2026-10-23 12:00 Europe/Amsterdam"}
    return out


def merge_batches(batches):
    """Union batches over the same anchors; repeated arrays must agree exactly."""
    assert batches, "no seed-42 batches"
    merged = {}
    for batch in batches:
        assert {"cl", "pair_index", "parity", "AFF__r1", "B__r1", "B0__r1", "B1__r1"} <= set(batch), "missing baseline/alignment arrays"
        cl, _ = _scope(batch)
        parity = np.asarray(batch["parity"])
        assert parity.shape == cl.shape and np.issubdtype(parity.dtype, np.integer)
        assert np.isin(parity, [0, 1]).all(), "parity must be 0 or 1"
        for key, value in batch.items():
            value = np.asarray(value)
            if key in merged:
                assert value.dtype == merged[key].dtype and np.array_equal(value, merged[key]), f"batches disagree on {key}"
            else:
                merged[key] = value
    return merged
