"""Final review A, check 1 and 2 (rule section 3 and section 8 item 2), with code written from the rule's text.

Part 1: crafted count vectors through r6_stats.holm against my own integer Holm (and against the float p-value form),
        every one-count boundary and tie pattern of both families.
Part 2: synthetic pooled held runs of the real shapes (3 seeds x 12,288 episodes, ~10,000 anchor clusters,
        quarter-valued differences) through r6_stats.pass_record (mode "held"); every field compared with my own
        draws, integer counts, Holm, intervals. Planted shifts put the counts near the Holm boundaries.
Part 3: section 8.2's SE, x, x95, x2 from my own one-way split and formula, against r6_stats / run_r6_held.

Writes a JSON summary next to this file (fr_a_stats.json, gitignored). Prints one line per part.
"""
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402  first
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402  (the rule's reference for the ci95)

OUT = Path(__file__).with_suffix(".json")
NB, CH = 5000, 250


# ------------------------------------------------------------------ my own code, from the rule's text
def my_draws(v, cl):
    """Rule section 3: clusters by numpy.unique(.., return_inverse=True); rng default_rng(42); chunks of 250 rows
    rng.integers(0, k, size=(250, k)); b_r = sum of cluster sums / sum of cluster counts. Also the integer count."""
    v = np.asarray(v, np.float64)
    _, idx = np.unique(np.asarray(cl), return_inverse=True)
    k = idx.max() + 1
    sums = np.bincount(idx, weights=v, minlength=k)
    cnt = np.bincount(idx, minlength=k).astype(np.float64)
    q = np.rint(4 * v).astype(np.int64)
    assert np.all(q == 4 * v)
    isum = np.zeros(k, np.int64)
    np.add.at(isum, idx, q)
    rng = np.random.default_rng(42)
    b, n_int, n_float = [], 0, 0
    for _ in range(NB // CH):
        d = rng.integers(0, k, size=(CH, k))
        bb = sums[d].sum(axis=1) / cnt[d].sum(axis=1)
        b.append(bb)
        n_int += int((isum[d].sum(axis=1) <= 0).sum())
        n_float += int((bb <= 0).sum())
    return np.concatenate(b), n_int, n_float, int(k)


def my_holm(counts, names, m):
    order = sorted(names, key=lambda nm: (counts[nm], names.index(nm)))
    out, ok = {}, True
    for k, nm in enumerate(order, 1):
        own = 40 * (counts[nm] + 1) * (m + 1 - k) <= 5001
        ok = ok and own
        out[nm] = {"k": k, "passes": ok, "own": own}
    return order, out


def float_holm(counts, names, m):
    """The p-value form p = (n+1)/5001 <= 0.025/(m+1-k), as a cross-check of the integer form."""
    order = sorted(names, key=lambda nm: (counts[nm], names.index(nm)))
    out, ok = {}, True
    for k, nm in enumerate(order, 1):
        own = (counts[nm] + 1) / 5001 <= 0.025 / (m + 1 - k)
        ok = ok and own
        out[nm] = ok
    return out


# ------------------------------------------------------------------ part 1: count vectors
def part1():
    rng = np.random.default_rng(7)
    fams = {"P": (ST.CHECKS, 7), "S": (ST.SECONDARY, 2)}
    nvec, bad, float_dis = 0, [], 0
    for fam, (names, m) in fams.items():
        names = tuple(names)
        bnd = [5001 // (40 * (m + 1 - k)) - 1 for k in range(1, m + 1)]
        assert bnd == list(ST.BOUNDARIES[fam]), (bnd, ST.BOUNDARIES[fam])
        vecs = []
        # every check at n*_k + delta for its rank, deltas in {-1, 0, +1}, rank assignment by permutation
        perms = list(itertools.permutations(range(m))) if m <= 3 else [tuple(rng.permutation(m)) for _ in range(60)]
        perms.append(tuple(range(m)))
        for perm in perms:
            for deltas in (itertools.product((-1, 0, 1), repeat=m) if m <= 2 else
                           [tuple(rng.integers(-1, 2, m)) for _ in range(200)] + [(0,) * m, (1,) * m, (-1,) * m]
                           + [tuple(1 if i == j else 0 for i in range(m)) for j in range(m)]):
                # rank r gets n*_{r+1} + delta: assign it to name perm[r]
                c = {names[perm[r]]: max(0, bnd[r] + int(deltas[r])) for r in range(m)}
                vecs.append(c)
        # ties: equal counts in random patterns, and all equal
        for _ in range(3000):
            pool = rng.integers(0, 140, size=rng.integers(1, m + 1))
            vecs.append({nm: int(rng.choice(pool)) for nm in names})
        for v in range(0, 200):
            vecs.append({nm: v for nm in names})
        # random wide counts
        for _ in range(3000):
            vecs.append({nm: int(rng.integers(0, 5001)) if rng.random() < 0.2 else int(rng.integers(0, 150))
                         for nm in names})
        for c in vecs:
            nvec += 1
            order, mine = my_holm(c, names, m)
            got = ST.holm(c, names, m)
            got_order = [e["name"] for e in got]
            fl = float_holm(c, names, m)
            float_dis += sum(fl[nm] != mine[nm]["passes"] for nm in names)
            ok = got_order == order and all(
                e["k"] == mine[e["name"]]["k"] and e["passes"] == mine[e["name"]]["passes"]
                and e["own_count_passes"] == mine[e["name"]]["own"] and e["n"] == c[e["name"]]
                and e["level_two_sided"] == 1 - 0.05 / (m + 1 - e["k"])
                and ST.near_boundary(e["n"], e["k"], m) == (abs(e["n"] - bnd[e["k"] - 1]) <= 1)
                for e in got)
            if not ok:
                bad.append((fam, c))
    return {"n_vectors": nvec, "n_disagree": len(bad), "first_bad": bad[:3], "float_vs_int_disagreements": float_dis}


# ------------------------------------------------------------------ part 2: pooled synthetic runs, real shapes
def make_clusters(rng, n_seeds=3, per_seed=3 * 4096, n_paint=10_000):
    w = rng.pareto(1.5, n_paint) + 1.0
    w /= w.sum()
    ids = rng.choice(np.arange(5_000_000, 5_000_000 + 4 * n_paint, 4), size=n_paint, replace=False)
    return [ids[rng.choice(n_paint, size=per_seed, p=w)] for _ in range(n_seeds)]


def quarter(rng, n, lo=-4, hi=4):
    return rng.integers(lo, hi + 1, n) / 4.0


def counts_matrix(cl_all):
    """C[r, j] = how often cluster j is drawn in resample r (the same draws as the rule's loop)."""
    _, idx = np.unique(cl_all, return_inverse=True)
    k = idx.max() + 1
    rng = np.random.default_rng(42)
    C = np.zeros((NB, k), np.int16)
    for c in range(NB // CH):
        d = rng.integers(0, k, size=(CH, k))
        for i in range(CH):
            C[c * CH + i] = np.bincount(d[i], minlength=k)
    return C.astype(np.float64), idx, k


def plant(d0, idx, k, C, target, rng):
    """Add +0.25 to episodes (fixed random order) until the integer count n = #{r: C[r]·isum <= 0} reaches target."""
    order = rng.permutation(len(d0))
    def n_of(j):
        d = d0.copy()
        d[order[:j]] += 0.25
        d = np.clip(d, -2, 2)
        isum = np.bincount(idx, weights=np.rint(4 * d), minlength=k).astype(np.int64)
        return int((C @ isum.astype(np.float64) <= 0).sum()), d
    lo, hi = 0, len(d0)
    n_lo, _ = n_of(lo)
    if n_lo <= target:
        return n_of(lo)[1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        n_mid, _ = n_of(mid)
        if n_mid > target:
            lo = mid
        else:
            hi = mid
    return n_of(hi)[1]


def per_seed_dicts(cl_seeds, diffs, rng):
    """score_seed-like dicts whose AFF minus comparator differences are exactly `diffs` (check -> pooled array)."""
    n = sum(len(c) for c in cl_seeds)
    a_r1 = quarter(rng, n, 0, 4)
    a_gain = diffs["P6"]                             # CF gain is 0, so AFF gain - 0 = P6's difference
    comp = {"cosine": ("P1", "r1"), "rca": ("P2", "r1"), "B": ("P3", "r1"), "B0": ("P4", "r1"),
            "aff_cf": ("P5", "r1"), "B1": ("S1", "r1"), "r1_fused": ("S2", "r1")}
    full = {"aff_fused": {"r1": a_r1, "gain": a_gain}}
    for key, (chk, _) in comp.items():
        full[key] = {"r1": a_r1 - diffs[chk], "gain": quarter(rng, n)}
    full["aff_cf"]["gain"] = np.zeros(n)
    full["rca"]["gain"] = a_gain - diffs["P7"]
    out, s0 = [], 0
    for c in cl_seeds:
        sl = slice(s0, s0 + len(c))
        out.append({"cl": c, **{key: {m: np.asarray(v[m][sl], np.float64) for m in ("r1", "gain")}
                                for key, v in full.items()}})
        s0 += len(c)
    return out


def my_record(diffs, cl):
    mine = {}
    for c in ST.CHECKS + ST.SECONDARY:
        b, n_int, n_float, k = my_draws(diffs[c], cl)
        ref = cluster_bootstrap(diffs[c], cl)
        assert [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] == ref["ci95"]
        mine[c] = {"b": b, "n": n_int, "n_float": n_float, "k_clusters": k,
                   "point": 100 * float(np.mean(diffs[c])),
                   "ci95": [100 * float(np.percentile(b, 2.5)), 100 * float(np.percentile(b, 97.5))]}
    out = {}
    for fam, names, m in (("P", ST.CHECKS, 7), ("S", ST.SECONDARY, 2)):
        order, h = my_holm({c: mine[c]["n"] for c in names}, tuple(names), m)
        for c in names:
            k = h[c]["k"]
            lv = 0.025 / (m + 1 - k)
            out[c] = {"n": mine[c]["n"], "n_float": mine[c]["n_float"], "point": mine[c]["point"],
                      "ci95": mine[c]["ci95"], "holm_k": k,
                      "ci_holm": [100 * float(np.percentile(mine[c]["b"], 100 * 0.025 / (m + 1 - k))),
                                  100 * float(np.percentile(mine[c]["b"], 100 * (1 - 0.025 / (m + 1 - k))))],
                      "level_two_sided": 1 - 0.05 / (m + 1 - k),
                      "near_boundary": abs(mine[c]["n"] - (5001 // (40 * (m + 1 - k)) - 1)) <= 1,
                      "passes": h[c]["passes"], "own": h[c]["own"], "lv": lv}
        out[f"order_{fam}"] = order
    out["n_clusters"] = mine["P1"]["k_clusters"]
    return out


def compare_record(rec, mine, n_ep):
    diffs = []
    if rec["n_episodes"] != n_ep or rec["n_clusters"] != mine["n_clusters"]:
        diffs.append(("sizes", rec["n_episodes"], rec["n_clusters"]))
    if rec["holm_order"] != mine["order_P"]:
        diffs.append(("holm_order", rec["holm_order"], mine["order_P"]))
    for fam, names in (("checks", ST.CHECKS), ("secondary", ST.SECONDARY)):
        for c in names:
            g, w = rec[fam][c], mine[c]
            for f in ("n", "point", "ci95", "holm_k", "ci_holm", "level_two_sided", "near_boundary"):
                if g[f] != w[f]:
                    diffs.append((c, f, g[f], w[f]))
            if fam == "checks":
                if g["passes"] != w["passes"] or g["own_count_passes"] != w["own"]:
                    diffs.append((c, "passes", g["passes"], w["passes"]))
            elif "passes" in g:
                diffs.append((c, "secondary has a pass field"))
    return diffs


def extra_for(seeds):
    return {"episodes_sha256": {s: {p: f"{(s * 7 + i):064x}" for i, p in enumerate(R.PAIR_NAMES)} for s in seeds},
            "runner_sha256": "c" * 64, "module_sha256": {"x.py": "d" * 64}}


SCENARIOS = {
    # check -> target integer count (None: plain noise around a positive shift picked at random)
    "go_at_boundaries": {"P3": 16, "P1": 19, "P2": 24, "P4": 30, "P5": 40, "P6": 61, "P7": 124, "S1": 61, "S2": 124},
    "first_rank_one_over": {"P3": 17, "P1": 19, "P2": 24, "P4": 30, "P5": 40, "P6": 61, "P7": 124, "S1": 62, "S2": 0},
    "middle_fail": {"P1": 5, "P2": 10, "P3": 20, "P4": 31, "P5": 35, "P6": 50, "P7": 100, "S1": 125, "S2": 30},
    "ties_and_large": {"P1": 12, "P2": 12, "P3": 12, "P4": 300, "P5": 2600, "P6": 0, "P7": 0, "S1": 3, "S2": 3},
}


def part2(rng):
    cl_seeds = make_clusters(rng)
    cl = np.concatenate(cl_seeds)
    C, idx, k = counts_matrix(cl)
    res = {}
    for name, targets in SCENARIOS.items():
        diffs = {}
        for c in ST.CHECKS + ST.SECONDARY:
            d0 = quarter(rng, len(cl), -3, 3)
            diffs[c] = plant(d0, idx, k, C, targets[c], rng)
        per_seed = per_seed_dicts(cl_seeds, diffs, rng)
        rec = ST.pass_record(per_seed, "held", list(R.HELD_SEEDS), extra_for(R.HELD_SEEDS))
        mine = my_record(diffs, cl)
        dif = compare_record(rec, mine, len(cl))
        res[name] = {"n_clusters": mine["n_clusters"],
                     "counts_mine": {c: mine[c]["n"] for c in ST.CHECKS + ST.SECONDARY},
                     "counts_code": {**{c: rec["checks"][c]["n"] for c in ST.CHECKS},
                                     **{c: rec["secondary"][c]["n"] for c in ST.SECONDARY}},
                     "float_vs_int": {c: [mine[c]["n_float"], mine[c]["n"]] for c in ST.CHECKS + ST.SECONDARY
                                      if mine[c]["n_float"] != mine[c]["n"]},
                     "passes": {c: rec["checks"][c]["passes"] for c in ST.CHECKS},
                     "holm_order": rec["holm_order"], "n_field_diffs": len(dif), "field_diffs": dif[:5]}
        print(f"part2 {name}: {len(dif)} field differences; counts mine==code "
              f"{res[name]['counts_mine'] == res[name]['counts_code']}", flush=True)
    return res, cl_seeds


# ------------------------------------------------------------------ part 3: section 8.2
def my_sigma(diff, cl):
    """R3 rule section 6.1's one-way split (pp): sigma_e2 = within MS; sigma_a2 = max(0, (between MS - within MS)/n0)."""
    d = 100.0 * np.asarray(diff, np.float64)
    _, idx = np.unique(cl, return_inverse=True)
    P, n = idx.max() + 1, len(d)
    m = np.bincount(idx, minlength=P).astype(float)
    mu = np.bincount(idx, weights=d, minlength=P) / m
    msb = np.sum(m * (mu - d.mean()) ** 2) / (P - 1)
    msw = np.sum((d - mu[idx]) ** 2) / (n - P)
    n0 = (n - np.sum(m ** 2) / n) / (P - 1)
    return {"sigma_a2": max(0.0, (msb - msw) / n0), "sigma_eps2": msw}


def part3(rng, cl_seeds):
    out = {"z": {"P": float(norm.ppf(1 - 0.025 / 7) + norm.ppf(0.8)), "S": float(norm.ppf(1 - 0.025 / 2)
                                                                                    + norm.ppf(0.8))}}
    # sigma parts on a seed-42-shaped array (12,288 episodes, ~3,000 paintings)
    cl42 = make_clusters(rng, 1, 12288, 3000)[0]
    d42 = quarter(rng, len(cl42), -2, 2) + 0.25 * (rng.random(3000)[np.unique(cl42, return_inverse=True)[1]] < 0.3)
    mine_s, code_s = my_sigma(d42, cl42), ST.sigma_split(d42, cl42)
    out["sigma_rel_diff"] = {k: abs(mine_s[k] - code_s[k]) / max(1e-300, abs(mine_s[k])) for k in mine_s}
    # SE, x, x95, x2 from pooled held anchors, via run_r6_held.sensitivity_held (the function that writes the file)
    sigma = {c: {"sigma_a2": 3.0 + i * 0.7, "sigma_eps2": 1400.0 + 13 * i} for i, c in
             enumerate(ST.CHECKS + ST.SECONDARY)}
    bundles = [SimpleNamespace(cl=c) for c in cl_seeds]
    got = RH.sensitivity_held(bundles, list(R.HELD_SEEDS), sigma, R.N_PER_PAIR)
    cl = np.concatenate(cl_seeds)
    _, M = np.unique(cl, return_counts=True)
    N = 36864
    worst = 0.0
    for c in ST.CHECKS + ST.SECONDARY:
        se = np.sqrt((sigma[c]["sigma_a2"] * np.sum(M.astype(float) ** 2) + sigma[c]["sigma_eps2"] * N) / N ** 2)
        want = {"SE": se, "x95": 2.80 * se, ("x" if c in ST.CHECKS else "x2"): (3.532 if c in ST.CHECKS else 3.083) * se}
        for kk, v in want.items():
            worst = max(worst, abs(got[c][kk] - v) / v)
        if set(got[c]) != {"quantity", "sigma_a2", "sigma_eps2", "SE", "x95", "x" if c in ST.CHECKS else "x2"}:
            out.setdefault("key_problems", []).append((c, sorted(got[c])))
    out.update({"N": got["N"], "n_paintings": got["n_paintings"], "n_paintings_mine": int(M.size),
                "worst_rel_diff_SE_x": worst})
    return out


def main():
    rng = np.random.default_rng(20261009)
    summary = {"part1": part1()}
    print(f"part1: {summary['part1']['n_vectors']} count vectors, {summary['part1']['n_disagree']} disagree, "
          f"float-vs-int {summary['part1']['float_vs_int_disagreements']}", flush=True)
    summary["part2"], cl_seeds = part2(rng)
    summary["part3"] = part3(rng, cl_seeds)
    p3 = summary["part3"]
    print(f"part3: z P {p3['z']['P']:.4f} S {p3['z']['S']:.4f}; sigma rel diff {max(p3['sigma_rel_diff'].values()):.1e};"
          f" SE/x worst rel diff {p3['worst_rel_diff_SE_x']:.1e}; N {p3['N']}; paintings {p3['n_paintings']} vs "
          f"{p3['n_paintings_mine']}; key problems {p3.get('key_problems')}")
    OUT.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
