"""Round 3 pooled statistics (DECISION_RULE.md section 2, 6.1, 6.5 to 6.6, 7 item 2). Pure numpy functions; Task 3's
runner builds the inputs.

Pooling: per-anchor values of the seeds are concatenated in the order given (49, 50, 51); clusters are the anchor
paintings, so a painting that anchors episodes on several seeds is one cluster. Intervals: 5,000-resample painting
bootstrap, seed 42, chunk 250 (src.eval.aspect_metrics.cluster_bootstrap), in percentage points (round 1's
common.point_ci). pass = lower bound strictly above 0.
"""
import numpy as np

import r3_common as R

C = R.C
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

Z_HALF = 1.96            # rule 6.1: projected half-width = 1.96 * SE
K_DETECT = 2.80          # rule 6.1: detectable margin x = 2.80 * SE

GO_CHECKS = ("r1_vs_cosine", "r1_vs_rca", "r1_vs_B", "r1_vs_Bprime", "r1_vs_counterpart", "gain_statistic",
             "gain_vs_rca")


def pooled_check(arrays_by_seed, cl_by_seed) -> dict:
    """arrays_by_seed: sequence of (E_s,) per-anchor value arrays (seed order); cl_by_seed: sequence of the painting
    ids of the same episodes (the same painting on several seeds is one cluster). -> {"point", "ci95", "pass"} in
    percentage points; pass = lower bound > 0 (strict)."""
    if len(arrays_by_seed) != len(cl_by_seed):
        raise ValueError("one cluster array per seed")
    for v, c in zip(arrays_by_seed, cl_by_seed):
        if len(v) != len(c):
            raise ValueError("values and clusters differ in length")
    v = np.concatenate([np.asarray(x, dtype=np.float64) for x in arrays_by_seed])
    cl = np.concatenate([np.asarray(x) for x in cl_by_seed])
    r = C.point_ci(v, cl)
    return {"point": r["point"], "ci95": r["ci95"], "pass": bool(r["ci95"][0] > 0)}


def _by_seed(per_seed, fn):
    return [fn(s) for s in per_seed], [s["cl"] for s in per_seed]


def go_checks(per_seed) -> dict:
    """The seven GO checks (rule 6.5) and the secondary check (6.6), pooled over the seeds.

    per_seed: list in seed order of dicts with
      "cl"      (E,) anchor-painting ids;
      "aff"     per-anchor dict of AFF's fused reader ("r1", "gain" arrays, fractions as per_anchor returns them);
      "cf"      per-anchor dict of AFF's matched counterpart (its "gain" must be exactly 0);
      "r1"      per-anchor dict of R1's fused reader (secondary check only);
      "cosine", "rca", "B", "Bp"   per-anchor dicts of the external and condition-free comparators (B' = B'(A0)).
    -> {"checks": {name: {"point", "ci95", "pass"}} for the seven GO_CHECKS, "go": all seven pass,
        "secondary": {"point", "ci95", "pass"} (R@1, AFF fused minus R1 fused; never changes "go")}.
    """
    for s in per_seed:
        if not np.all(np.asarray(s["cf"]["gain"]) == 0):
            raise AssertionError("the matched counterpart's condition gain is not exactly 0")

    def r1_minus(name):
        return _by_seed(per_seed, lambda s: np.asarray(s["aff"]["r1"], np.float64) - np.asarray(s[name]["r1"], np.float64))

    checks = {
        "r1_vs_cosine": pooled_check(*r1_minus("cosine")),
        "r1_vs_rca": pooled_check(*r1_minus("rca")),
        "r1_vs_B": pooled_check(*r1_minus("B")),
        "r1_vs_Bprime": pooled_check(*r1_minus("Bp")),
        "r1_vs_counterpart": pooled_check(*r1_minus("cf")),
        "gain_statistic": pooled_check(*_by_seed(
            per_seed, lambda s: np.asarray(s["aff"]["gain"], np.float64) - np.asarray(s["cf"]["gain"], np.float64))),
        "gain_vs_rca": pooled_check(*_by_seed(
            per_seed, lambda s: np.asarray(s["aff"]["gain"], np.float64) - np.asarray(s["rca"]["gain"], np.float64))),
    }
    assert tuple(checks) == GO_CHECKS
    return {"checks": checks, "go": bool(all(c["pass"] for c in checks.values())),
            "secondary": pooled_check(*r1_minus("r1"))}


def bar_info_pooled(per_seed) -> tuple:
    """Rule 7 item 2: AFF's bar margin on the pooled episodes, the comparator chosen once over the pooled mean R@1
    (common.bar_comparator, ties B', counterpart, B). per_seed dicts as in go_checks plus "pair_index" (E,), with
    "Bp" and "B", "cf", "aff". -> (per-anchor margin vector, common.bar_info's record)."""
    cat = {k: {m: np.concatenate([np.asarray(s[k][m]) for s in per_seed]) for m in per_seed[0][k]}
           for k in ("aff", "cf", "Bp", "B")}
    cl = np.concatenate([np.asarray(s["cl"]) for s in per_seed])
    pi = np.concatenate([np.asarray(s["pair_index"]) for s in per_seed])
    return C.bar_info(cat["aff"], cat["cf"], cat["Bp"], cat["B"], cl, pi)


def sensitivity(diff, cl) -> dict:
    """Rule 6.1 on one seed's per-episode paired difference `diff` (per-anchor FRACTIONS, as pooled_check takes them;
    scaled by 100 inside), clustered by anchor painting `cl`. SE, half_width, x, seed42_half_width, sigma_a2 and
    sigma_e2 are in PERCENTAGE POINTS (sigma_*2 in pp squared).

    One-way decomposition: sigma_e2 = within-painting mean square; sigma_a2 = max(0, (between MS - sigma_e2) / n0),
    n0 = (n - sum_p m_p^2 / n) / (P - 1). Projected pooled SE over three seeds:
    SE^2 = (sigma_a2 * (9 sum_p m_p^2 - 6 n) + sigma_e2 * 3 n) / (3 n)^2.
    -> {"SE", "half_width" (1.96 SE), "x" (2.80 SE), "seed42_half_width" (half the width of this difference's own 95%
    painting-bootstrap interval), "sigma_a2", "sigma_e2", "n0", "n", "P"}."""
    d = 100.0 * np.asarray(diff, dtype=np.float64)
    _, idx = np.unique(np.asarray(cl), return_inverse=True)
    n, P = len(d), int(idx.max()) + 1
    if P < 2 or n <= P:
        raise ValueError("need at least two paintings and more episodes than paintings")
    m = np.bincount(idx, minlength=P).astype(np.float64)
    mean_p = np.bincount(idx, weights=d, minlength=P) / m
    grand = d.mean()
    ssb = float(np.sum(m * (mean_p - grand) ** 2))
    ssw = float(np.sum((d - mean_p[idx]) ** 2))
    msb, msw = ssb / (P - 1), ssw / (n - P)
    sum_m2 = float(np.sum(m ** 2))
    n0 = (n - sum_m2 / n) / (P - 1)
    sig_e2 = msw
    sig_a2 = max(0.0, (msb - sig_e2) / n0)
    se2 = (sig_a2 * (9 * sum_m2 - 6 * n) + sig_e2 * 3 * n) / (3 * n) ** 2
    se = float(np.sqrt(se2))
    ci = cluster_bootstrap(d, cl)["ci95"]   # pp, as d is
    return {"SE": se, "half_width": Z_HALF * se, "x": K_DETECT * se,
            "seed42_half_width": 0.5 * (ci[1] - ci[0]), "sigma_a2": float(sig_a2), "sigma_e2": float(sig_e2),
            "n0": float(n0), "n": int(n), "P": P}
