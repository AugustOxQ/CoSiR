"""Round 5 re-derivation: comparisons, the bar comparator, the development record, D10, Delta_k, the carry and the
detectable margin x (round 3 §6.1), written from the rules' text. Uses only the allowed `cluster_bootstrap`."""
import math

import numpy as np

from rd5_core import as_int4

BOUNDARY_EPS = 1e-12
TIE_BAND = 24                       # round 5 §5 item 7 (integer literal)
CANDIDATE_ORDER = ("G-T", "G-TF")   # tie order of the carry
K_X = 2.80


def point_ci(values, clusters, cluster_bootstrap) -> dict:
    r = cluster_bootstrap(np.asarray(values, np.float64), clusters)
    return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}


def mean_pp(values) -> float:
    return 100 * float(np.asarray(values, np.float64).mean())


def either(pa) -> np.ndarray:
    return np.asarray(pa["r1"]) + np.asarray(pa["other"])


def bar_comparator(comps) -> tuple:
    """comps: ordered list of (name, per-anchor dict). Largest mean R@1 at full precision; ties to the earliest."""
    best, best_mean = None, None
    means = {}
    for name, pa in comps:
        m = float(np.asarray(pa["r1"], np.float64).mean())
        means[name] = 100 * m
        if best is None or m > best_mean:
            best, best_mean = name, m
    return best, means


def dev_record(fused, cf, comps, aff_fused, bp1, clusters, pair_index, cluster_bootstrap, pairs) -> dict:
    """The development numbers of one fused reader (round 5 §5 item 5 and D8 to D10).

    comps: ordered (name, per-anchor) condition-free comparators, the counterpart among them as "counterpart"."""
    for name, pa in comps:
        if not (np.asarray(pa["gain"]) == 0).all():
            raise AssertionError(f"{name} is not condition-free (gain != 0)")
    bar, means = bar_comparator(comps)
    bar_pa = dict(comps)[bar]
    out = {"fused_r1": mean_pp(fused["r1"]), "cf_r1": mean_pp(cf["r1"]), "comparator_means": means,
           "bar_comparator": bar}
    out["bar_margin"] = point_ci(fused["r1"] - bar_pa["r1"], clusters, cluster_bootstrap)
    out["margin_vs_cf"] = point_ci(fused["r1"] - cf["r1"], clusters, cluster_bootstrap)
    out["gain_statistic"] = point_ci(fused["gain"] - cf["gain"], clusters, cluster_bootstrap)
    out["either_change_vs_cf"] = point_ci(either(fused) - either(cf), clusters, cluster_bootstrap)["point"]
    out["per_pair_bar_margin"] = {name: mean_pp((fused["r1"] - bar_pa["r1"])[pair_index == i])
                                  for i, name in enumerate(pairs)}
    if aff_fused is not None:
        delta = int(as_int4(fused["r1"]).sum() - as_int4(aff_fused["r1"]).sum())
        n = len(fused["r1"])
        out["delta_vs_aff"] = {"delta_int": delta, "point": 100 * delta / (4 * n),
                               "ci95": point_ci(fused["r1"] - aff_fused["r1"], clusters, cluster_bootstrap)["ci95"]}
    if bp1 is not None:
        out["beside_Bp_A1"] = {"Bp_A1_r1": mean_pp(bp1["r1"]),
                               "minus_Bp_A1": point_ci(fused["r1"] - bp1["r1"], clusters, cluster_bootstrap)}
    out["D10"] = d10(out)
    return out


def d10(rec) -> dict:
    c1 = rec["bar_margin"]["point"] >= 0.5
    c2 = rec["bar_margin"]["ci95"][0] > 0
    c3 = rec["gain_statistic"]["ci95"][0] > 0
    flags = {"c1_within_1e-12": abs(rec["bar_margin"]["point"] - 0.5) <= BOUNDARY_EPS,
             "c2_within_1e-12": abs(rec["bar_margin"]["ci95"][0]) <= BOUNDARY_EPS,
             "c3_within_1e-12": abs(rec["gain_statistic"]["ci95"][0]) <= BOUNDARY_EPS}
    return {"c1_bar_point_ge_0.5": bool(c1), "c2_bar_lower_gt_0": bool(c2), "c3_gain_lower_gt_0": bool(c3),
            "clears": bool(c1 and c2 and c3), "boundary": flags}


def carry(records: dict) -> dict:
    """round 5 §5 items 7 and 8. records: {name: dev record with D10 and delta_vs_aff}."""
    E = [k for k in CANDIDATE_ORDER if k in records and records[k]["D10"]["clears"]
         and records[k]["delta_vs_aff"]["delta_int"] > 0]
    out = {"E": E, "deltas": {k: records[k]["delta_vs_aff"]["delta_int"] for k in records},
           "boundary": {"delta_zero": [k for k in records if records[k]["delta_vs_aff"]["delta_int"] == 0]}}
    if not E:
        out.update(M=None, tied=[], carried=None, decision="KILL")
        out["boundary"]["gap_24"] = []
        return out
    M = max(records[k]["delta_vs_aff"]["delta_int"] for k in E)
    tied = [k for k in E if M - records[k]["delta_vs_aff"]["delta_int"] <= TIE_BAND]
    carried = next(k for k in CANDIDATE_ORDER if k in tied)
    out.update(M=M, tied=tied, carried=carried, decision="CARRY")
    out["boundary"]["gap_24"] = [k for k in E if M - records[k]["delta_vs_aff"]["delta_int"] == TIE_BAND]
    return out


def sensitivity(diff, clusters) -> dict:
    """round 3 §6.1: one-way decomposition of a seed-42 per-episode difference by anchor painting, projected to the
    pooled SE over three seeds; values in the units of `diff` (multiply by 100 for points)."""
    d = np.asarray(diff, np.float64)
    _, idx = np.unique(np.asarray(clusters), return_inverse=True)
    P = int(idx.max()) + 1
    n = len(d)
    m = np.bincount(idx, minlength=P).astype(np.float64)
    sums = np.bincount(idx, weights=d, minlength=P)
    means = sums / m
    grand = d.mean()
    ssw = float(((d - means[idx]) ** 2).sum())
    ssb = float((m * (means - grand) ** 2).sum())
    msw = ssw / (n - P)
    msb = ssb / (P - 1)
    sum_m2 = float((m ** 2).sum())
    n0 = (n - sum_m2 / n) / (P - 1)
    sa2 = max(0.0, (msb - msw) / n0)
    se2 = (sa2 * (9 * sum_m2 - 6 * n) + msw * 3 * n) / (3 * n) ** 2
    se = math.sqrt(se2)
    return {"sigma_eps2": msw, "sigma_a2": sa2, "n0": n0, "n": n, "P": P, "sum_m2": sum_m2, "SE": se,
            "half_width_1.96SE": 1.96 * se, "x_2.80SE": K_X * se}
