"""Round 7, ticket 15: independent re-derivation of the verdict statistics (DECISION_RULE.md of this folder, section 9
"Re-derivation (C13)"; items L15, L16, section 5 K1, section 6 S1 to S3 and P1, section 7.5 F1 to F3, section 11).

Written from the rule's words and the index's "Verdict JSON core keys" alone. It never opens or imports the pipeline's
r7_gate, r7_chain, r7_fusion or run_r7 files, nor round 3's or round 6's statistics modules. It takes from the
library only `cluster_bootstrap` (C9 intervals of K1 and S1; also the copy check of the F1 to F3 loop) and, through
`r7_common`, the paths, the rule SHA, the write-once writer and the module loader.

Inputs (per-anchor arrays, read by key):
  ceiling  seed42/arrays_ceiling.npz    cl, pair_index, told_T__r1, told_T_cf__r1
  gate     seed42/arrays.npz            cl, pair_index, AFF/B/B0/B1 and per scorer `{name}__r1`, `{name}__gain`,
                                        `{name}_cf__r1`, `{name}_cf__gain`, `BX_{X}__r1`
  test     seed{55,56,57}/arrays.npz    the pick's arrays and its CF, B, B0, B1, BX_{X}, AFF (r1 only)
           gate_verdict.json            picks and alpha only
`--overlay` merges the phase files of ticket 14 (`rederive/phaseK.npz`, `phaseA_seed42.npz`, `phaseB_seed{s}.npz`)
over the pipeline's files; a key that both hold must be equal.

Conventions. Scenario scopes: pair_index 0 = emotion__style, 1 = emotion__genre, 2 = style__genre. Scorer names are
the arrays' names: `R1_{X}` and `XAFF_{X}` for X in V1..V6, R0, N, Z1..Z6. A grouping whose scorer arrays are absent
is dropped (L7, L11). Integers: d is a multiple of 0.25, Sum4d = sum of 4d (L16); a point in R@1 points is
25 * Sum4d / N; an interval is 100 times `cluster_bootstrap`'s. Output: `rederive/stats_{stage}.json`.
"""
import argparse
import json
import sys
from fractions import Fraction
from math import ceil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import r7_common as R  # noqa: E402

AM = R.load("src.eval.aspect_metrics")

GROUPS = ["V1", "V2", "V3", "V4", "V5", "V6"]
REFS = ["R0", "N"] + [f"Z{k}" for k in range(1, 7)]
KINDS = ["R1", "XAFF"]
FAMILY = {"csd": ["V1", "V2", "V3", "V4"], "generic": ["V5", "V6"]}
PAIRS = {"emotion__style": 0, "emotion__genre": 1, "style__genre": 2}
SG = PAIRS["style__genre"]
CHECKS = ["B", "B0", "B1", "BX", "CF"]       # F1's order, which breaks ties in Holm
N_BOOT, CHUNK, SEED, DENOM = 5000, 250, 42, 5001


# ------------------------------------------------------------------ L16

def q4(d):
    """4d as int64; asserts d is a multiple of 0.25 (R3 D8.5) within 1e-9."""
    x = 4.0 * np.asarray(d, dtype=np.float64)
    q = np.rint(x)
    assert np.max(np.abs(x - q), initial=0.0) <= 1e-9, "a per-episode difference is not a multiple of 0.25"
    return q.astype(np.int64)


def sum4(d) -> int:
    return int(q4(d).sum())


def need(a, n) -> int:
    """ceil(a * n / 25), exact (a is a string or a Fraction)."""
    return ceil(Fraction(str(a)) * n / 25)


def point(s4: int, n: int) -> float:
    return 25 * s4 / n


def ci_points(values, cl):
    """C9's 95% interval in R@1 points."""
    lo, hi = AM.cluster_bootstrap(values, cl)["ci95"]
    return [100 * lo, 100 * hi]


# ------------------------------------------------------------------ K1

def ceiling_scope(A, per_pair=None):
    """Told T fused and counterpart R@1 and the painting ids on the style x genre episodes."""
    sel = np.asarray(A["pair_index"]) == SG
    if per_pair is not None:
        assert int(sel.sum()) == per_pair, f"style x genre episodes: {int(sel.sum())}, not {per_pair}"
    for k in ("told_T__r1", "told_T_cf__r1", "cl"):
        assert len(A[k]) == len(sel), f"{k}: {len(A[k])} rows, pair_index has {len(sel)}"
    return A["told_T__r1"][sel], A["told_T_cf__r1"][sel], A["cl"][sel]


def ceiling_stats(fused, cf, cl):
    """K1 (section 5): STOP if Sum4d < 82 or the lower bound is at or below 0; WEAK if 82 <= Sum4d < 164 and the lower
    bound is above 0; CLEAR if Sum4d >= 164 and the lower bound is above 0 (82 and 164 at n = 4,096, from L16)."""
    d = np.asarray(fused, dtype=np.float64) - np.asarray(cf, dtype=np.float64)
    n = len(d)
    s4 = sum4(d)
    ci = ci_points(d, cl)
    if s4 < need("0.5", n) or ci[0] <= 0:
        zone = "STOP"
    elif s4 < need("1.0", n):
        zone = "WEAK"
    else:
        zone = "CLEAR"
    return {"sum4d": s4, "n": n, "point": point(s4, n), "ci95": ci, "zone": zone}


# ------------------------------------------------------------------ S1 to S3, P1

def parse(name):
    kind, grouping = name.split("_", 1)
    assert kind in KINDS and grouping in GROUPS + REFS, f"unknown scorer {name!r}"
    return kind, grouping


def norm_name(name):
    """'R1[V1]', 'X-AFF[V3]', 'XAFF_V3' -> the array name."""
    name = name.strip()
    if "[" in name:
        kind, rest = name.split("[", 1)
        kind = "XAFF" if kind.replace("-", "").upper() == "XAFF" else kind
        return f"{kind}_{rest.rstrip(']')}"
    return name


def present(A, name):
    return f"{name}__r1" in A


def scorer_stats(A, name):
    """S1, S2, S3 of one scorer (section 6) with the bar comparator of L15; also the quantities P1 and 'promising' use."""
    _, grouping = parse(name)
    cl, pair = A["cl"], A["pair_index"]
    n = len(cl)
    s = A[f"{name}__r1"]
    cf = A[f"{name}_cf__r1"]
    comps = [("B1", A["B1__r1"]), ("BX", A[f"BX_{grouping}__r1"]), ("CF", cf), ("B0", A["B0__r1"]), ("B", A["B__r1"])]
    totals = [int(q4(c).sum()) for _, c in comps]            # the mean R@1 up to the constant 4 / n: exact in integers
    bar_name, bar = comps[int(np.argmax(totals))]            # argmax returns the first maximum: ties to the earliest
    d_bar = s - bar
    cf_gain = A[f"{name}_cf__gain"]
    assert not np.any(cf_gain), f"{name}: the counterpart's gain is not 0 on every episode"
    g = A[f"{name}__gain"] - cf_gain
    s1_sum, g_sum = sum4(d_bar), sum4(g)
    s1_ci, g_ci = ci_points(d_bar, cl), ci_points(g, cl)
    s1 = {"sum4d": s1_sum, "ci95": s1_ci, "gain_sum4d": g_sum, "gain_ci95": g_ci,
          "pass": bool(s1_sum >= need("0.5", n) and s1_ci[0] > 0 and g_ci[0] > 0)}
    m_sg = pair == SG
    s2_sum = sum4((s - cf)[m_sg])
    s2 = {"sum4d": s2_sum, "pass": bool(s2_sum >= 0)}
    s3 = {}
    for pname in ("emotion__style", "emotion__genre"):
        m = pair == PAIRS[pname]
        t = sum4((s - A["AFF__r1"])[m])
        s3[pname] = {"sum4d": t, "pass": bool(t >= need("-0.3", int(m.sum())))}
    s3["pass"] = all(s3[p]["pass"] for p in ("emotion__style", "emotion__genre"))
    ok = s1["pass"] and s2["pass"] and s3["pass"]
    out = {"bar_comparator": bar_name, "S1": s1, "S2": s2, "S3": s3, "pass": bool(ok), "dropped": False}
    aff_sg = sum4((s - A["AFF__r1"])[m_sg])
    return out, {"aff_sg": aff_sg, "n_sg": int(m_sg.sum())}


DROPPED = {"bar_comparator": None, "S1": None, "S2": None, "S3": None, "dropped": True, "pass": False}


def gate_block(A, groupings):
    """{scorer: record} for the given groupings; also the extra quantities by scorer."""
    out, extra = {}, {}
    for g in groupings:
        have = [present(A, f"{k}_{g}") for k in KINDS]
        assert have[0] == have[1], f"grouping {g}: only one of its two scorers has arrays"
        for k in KINDS:
            name = f"{k}_{g}"
            if not have[0]:
                out[name] = dict(DROPPED)
                continue
            out[name], extra[name] = scorer_stats(A, name)
    return out, extra


def gate_stats(A, per_pair=None):
    """S1 to S3 and P1 for all twelve candidates, the references and controls, n, alpha, promising, flags."""
    n = len(A["cl"])
    assert n % 3 == 0 and (per_pair is None or n == 3 * per_pair), f"{n} episodes"
    counts = [int((A["pair_index"] == p).sum()) for p in range(3)]
    assert counts[0] == counts[1] == counts[2] == n // 3, f"pair sizes {counts}"
    if present(A, "BX_R0") and "B1__r1" in A:
        assert np.array_equal(A["B1__r1"], A["BX_R0__r1"]), "B1 differs from BX_R0"
    cands, cx = gate_block(A, GROUPS)
    refs, _ = gate_block(A, REFS)
    picks, alpha, flags = {}, {}, {}
    for fam, members in FAMILY.items():
        rows = []
        for g in members:
            for k in KINDS:
                name = f"{k}_{g}"
                if cands[name]["pass"]:
                    rows.append((-cands[name]["S1"]["sum4d"], -cands[name]["S1"]["gain_sum4d"],
                                 GROUPS.index(g), KINDS.index(k), name))
        picks[fam] = min(rows)[-1] if rows else None
    n_chains = sum(v is not None for v in picks.values())
    for fam, name in picks.items():
        if name is None:
            continue
        alpha[fam] = 0.0125 if n_chains == 2 else 0.025
        kind, g = parse(name)
        ctl = refs[f"{kind}_Z{GROUPS.index(g) + 1}"]
        flags[fam] = "not evaluable" if ctl["dropped"] else bool(ctl["pass"])
    promising = n_chains == 0 and any(
        (not cands[nm]["dropped"]) and cx[nm]["aff_sg"] >= need("0.3", cx[nm]["n_sg"])
        and cands[nm]["S2"]["sum4d"] >= 0 and cands[nm]["S1"]["sum4d"] > 0 for nm in cands)
    return {"candidates": cands, "references": refs, "picks": picks, "n": n_chains, "alpha": alpha,
            "promising": bool(promising), "random_control_flag": flags}


# ------------------------------------------------------------------ the bootstrap loop (section 7.5; R6 section 3)

def bootstrap_checks(values, cl, n_boot=N_BOOT, seed=SEED, chunk=CHUNK):
    """One draw set for every check in `values` {name: per-episode differences}: clusters = the sorted unique ids of
    `cl`; default_rng(42); chunks of 250 draws of size (rows, k) from rng.integers(0, k). Per check: the resample
    means b (float) and n_j = the number of resamples whose integer 4 * sum of cluster sums is at or below 0."""
    _, idx = np.unique(np.asarray(cl), return_inverse=True)
    k = int(idx.max()) + 1
    assert k >= 2, "fewer than two clusters"
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    sums, isums = {}, {}
    for nm, v in values.items():
        v = np.asarray(v, dtype=np.float64)
        assert len(v) == len(idx), f"{nm}: {len(v)} values for {len(idx)} episodes"
        sums[nm] = np.bincount(idx, weights=v, minlength=k)
        isums[nm] = np.rint(np.bincount(idx, weights=q4(v).astype(np.float64), minlength=k)).astype(np.int64)
    b = {nm: [] for nm in values}
    n_le0 = {nm: 0 for nm in values}
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, chunk):
        draws = rng.integers(0, k, size=(min(chunk, n_boot - start), k))
        den = counts[draws].sum(axis=1)
        for nm in values:
            b[nm].append(sums[nm][draws].sum(axis=1) / den)
            n_le0[nm] += int((isums[nm][draws].sum(axis=1) <= 0).sum())
    return {nm: {"b": np.concatenate(b[nm]), "n_j": n_le0[nm]} for nm in values}


def pct(b, lo):
    """Interval in points between the lo-th and (100 - lo)-th percentiles of the resample means."""
    return [100 * float(np.percentile(b, lo)), 100 * float(np.percentile(b, 100 - lo))]


# ------------------------------------------------------------------ Holm and single checks

def c_of(alpha) -> int:
    c = round(1 / alpha)
    assert c in (80, 40) and abs(c * alpha - 1) < 1e-12, f"alpha {alpha} is not 0.0125 or 0.025"
    return c


def single_pass(n_j, c) -> bool:
    return bool(c * (n_j + 1) <= DENOM)


def holm(ns, c):
    """Order by n_j ascending, ties in the dict's order; the k-th (1 to 5) passes if every earlier one passes and
    c * (n_(k) + 1) * (6 - k) <= 5,001. Returns (order, {name: passes})."""
    names = list(ns)
    order = sorted(names, key=lambda nm: (ns[nm], names.index(nm)))
    passes, alive = {}, True
    for k, nm in enumerate(order, start=1):
        alive = alive and bool(c * (ns[nm] + 1) * (len(names) + 1 - k) <= DENOM)
        passes[nm] = alive
    return order, {nm: passes[nm] for nm in names}


# ------------------------------------------------------------------ F1 to F3

def check_numbers(d, res):
    s4 = sum4(d)
    return {"n_j": res["n_j"], "sum4d": s4, "point": point(s4, len(d)), "ci95": pct(res["b"], 2.5)}


def copy_check(d, cl, ci95, label):
    """The loop's 95% interval equals cluster_bootstrap's (as R6 section 3 asserts of its copy)."""
    want = AM.cluster_bootstrap(d, cl)["ci95"]
    assert [ci95[0] / 100, ci95[1] / 100] == want or np.allclose(
        [ci95[0] / 100, ci95[1] / 100], want, rtol=0, atol=1e-12), f"{label}: loop differs from cluster_bootstrap"


def chain_stats(arrays, pick, alpha, verify_copy=True):
    """F1 to F3 of one chain over the test seeds' arrays (a list, in seed order 55, 56, 57)."""
    kind, grouping = parse(pick)
    c = c_of(alpha)
    cat = lambda key: np.concatenate([np.asarray(a[key], dtype=np.float64) for a in arrays])  # noqa: E731
    cl = np.concatenate([np.asarray(a["cl"]) for a in arrays])
    pair = np.concatenate([np.asarray(a["pair_index"]) for a in arrays])
    s = cat(f"{pick}__r1")
    other = {"B": cat("B__r1"), "B0": cat("B0__r1"), "B1": cat("B1__r1"), "BX": cat(f"BX_{grouping}__r1"),
             "CF": cat(f"{pick}_cf__r1")}
    diffs = {nm: s - other[nm] for nm in CHECKS}
    diffs["AFF"] = s - cat("AFF__r1")
    res = bootstrap_checks(diffs, cl)
    order, passes = holm({nm: res[nm]["n_j"] for nm in CHECKS}, c)
    checks = {}
    for nm in CHECKS:
        k = order.index(nm) + 1
        row = check_numbers(diffs[nm], res[nm])
        row["ci_level"] = pct(res[nm]["b"], 100 * alpha / (6 - k))
        checks[nm] = row
        if verify_copy:
            copy_check(diffs[nm], cl, row["ci95"], nm)
    f1 = {"checks": checks, "holm_order": order, "passes": passes, "pass": bool(all(passes.values()))}
    # F2, tested only if F1 passes
    r2 = check_numbers(diffs["AFF"], res["AFF"])
    if verify_copy:
        copy_check(diffs["AFF"], cl, r2["ci95"], "AFF")
    if f1["pass"]:
        r2["ci_level"] = pct(res["AFF"]["b"], 100 * alpha)
        r2["pass"] = single_pass(res["AFF"]["n_j"], c)
        f2 = {k: r2[k] for k in ("n_j", "sum4d", "point", "ci95", "ci_level", "pass")}
    else:
        f2 = {"status": "not tested", "point": r2["point"], "ci95": r2["ci95"]}
    # F3 over the pooled style x genre episodes, its own draws, tested only if F2 passes
    sel = pair == SG
    d3 = (s - other["B1"])[sel]
    res3 = bootstrap_checks({"F3": d3}, cl[sel])["F3"]
    r3 = check_numbers(d3, res3)
    if verify_copy:
        copy_check(d3, cl[sel], r3["ci95"], "F3")
    if f1["pass"] and f2["pass"]:
        r3["ci_level"] = pct(res3["b"], 100 * alpha)
        r3["pass"] = single_pass(res3["n_j"], c)
        f3 = {k: r3[k] for k in ("n_j", "sum4d", "point", "ci95", "ci_level", "pass")}
    else:
        f3 = {"status": "not tested", "point": r3["point"], "ci95": r3["ci95"]}
    return {"pick": pick, "alpha": alpha, "F1": f1, "F2": f2, "F3": f3}


def test_stats(arrays, picks, alphas, verify_copy=True):
    """Chains for the picked families; FV is GO if F1 passes in at least one chain."""
    chains = {}
    for fam, pick in picks.items():
        if pick is None:
            continue
        assert fam in alphas, f"no alpha for family {fam}"
        chains[fam] = chain_stats(arrays, pick, alphas[fam], verify_copy)
    assert chains, "no pick: there is no chain to test"
    assert set(alphas) == set(chains), f"alpha for {sorted(alphas)}, chains for {sorted(chains)}"
    go = any(ch["F1"]["pass"] for ch in chains.values())
    return {"chains": chains, "FV": "GO" if go else "NO-GO"}


# ------------------------------------------------------------------ files and stages

def load_arrays(path, overlay=None):
    """The npz at `path` as a dict; keys of `overlay` (ticket 14's file) are added, and equal where both have them."""
    with np.load(path) as z:
        A = {k: z[k] for k in z.files}
    if overlay is not None:
        with np.load(overlay) as z:
            for k in z.files:
                if k in A:
                    assert np.array_equal(A[k], z[k]), f"{k}: {overlay} differs from {path}"
                else:
                    A[k] = z[k]
    return A


def _arrays(results, rel, overlay_rel, overlay):
    ov = Path(results) / overlay_rel if overlay else None
    return load_arrays(Path(results) / rel, ov)


def run_stage(stage, results, dev, tests, per_pair=None, overlay=False):
    results = Path(results)
    if stage == "ceiling":
        A = _arrays(results, f"seed{dev}/arrays_ceiling.npz", "rederive/phaseK.npz", overlay)
        rec = ceiling_stats(*ceiling_scope(A, per_pair))
        if per_pair is not None:
            assert rec["n"] == per_pair
    elif stage == "gate":
        A = _arrays(results, f"seed{dev}/arrays.npz", f"rederive/phaseA_seed{dev}.npz", overlay)
        rec = gate_stats(A, per_pair)
    elif stage == "test":
        gv = json.loads((results / "gate_verdict.json").read_text())
        picks, alphas = gv["picks"], gv["alpha"]
        picks = {f: (None if p is None else norm_name(p)) for f, p in picks.items()}
        fams = [f for f, p in picks.items() if p is not None]
        assert set(alphas) == set(fams), f"gate_verdict alpha {sorted(alphas)} vs picks {fams}"
        assert all(abs(alphas[f] - (0.0125 if len(fams) == 2 else 0.025)) < 1e-12 for f in fams), "alpha vs n"
        arrays = []
        for s in tests:
            A = _arrays(results, f"seed{s}/arrays.npz", f"rederive/phaseB_seed{s}.npz", overlay)
            assert per_pair is None or len(A["cl"]) == 3 * per_pair, f"seed {s}: {len(A['cl'])} episodes"
            arrays.append(A)
        rec = test_stats(arrays, picks, alphas)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    path = results / "rederive" / f"stats_{stage}.json"
    R.write_json_once(path, rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["ceiling", "gate", "test"])
    ap.add_argument("--chain-smoke", action="store_true", help="smoke seeds 9001 to 9003 under results/smoke/chain")
    ap.add_argument("--overlay", action="store_true", help="merge ticket 14's phase files over the pipeline's arrays")
    ap.add_argument("--results", help="results folder (default: r7_common's RESULTS, or the chain smoke's)")
    args = ap.parse_args(argv)
    R.assert_rule()
    roles = R.roles("chain" if args.chain_smoke else "real")
    results = Path(args.results) if args.results else roles.root
    for s in (roles.dev,) + tuple(roles.tests):
        R.check_seed(s, roles.smoke)
    per_pair = None if roles.smoke else 4096
    rec = run_stage(args.stage, results, roles.dev, roles.tests, per_pair, args.overlay)
    head = {"ceiling": lambda r: f"sum4d {r['sum4d']} n {r['n']} zone {r['zone']}",
            "gate": lambda r: f"n {r['n']} picks {r['picks']} alpha {r['alpha']} promising {r['promising']}",
            "test": lambda r: f"FV {r['FV']} chains {sorted(r['chains'])}"}[args.stage](rec)
    print(f"rd7_stats {args.stage}: {head}; wrote {results / 'rederive' / f'stats_{args.stage}.json'}")


if __name__ == "__main__":
    main()
