"""Final review A1: my own reference for rule §4 (r9) and R6 rule §3, written from the rule text, and the data maker.

Nothing here imports r9_stats or run_r9_apply_rule; the scripts that compare import both.
"""
from fractions import Fraction
from statistics import NormalDist

import numpy as np

N_BOOT, SEED, CHUNK = 5000, 42, 250
CHECKS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
L1, L2 = ("P4", "P6"), ("P1", "P2", "P3", "P5", "P7")
PAIRS = ("emotion__style", "emotion__genre", "style__genre")


# ---------------------------------------------------------------- draws (R6 rule §3 "Draws")
class Draws:
    """Resample multiplicities W[r, c] (times cluster c is drawn in resample r): numpy.unique(..., return_inverse),
    default_rng(42), chunks of 250 rows of rng.integers(0, k, size=(rows, k))."""

    def __init__(self, cl):
        labels, inv = np.unique(np.asarray(cl), return_inverse=True)
        self.inv, self.k = inv.reshape(-1), len(labels)
        k = self.k
        rng = np.random.default_rng(SEED)
        W = np.empty((N_BOOT, k), np.int64)
        r = 0
        while r < N_BOOT:
            rows = min(CHUNK, N_BOOT - r)
            d = rng.integers(0, k, size=(rows, k))
            W[r:r + rows] = np.bincount((d + k * np.arange(rows)[:, None]).ravel(), minlength=rows * k).reshape(rows, k)
            r += rows
        self.W = W
        self.WT = np.ascontiguousarray(W.T)
        self.m = np.bincount(self.inv, minlength=k)

    def stats(self, q):
        """q: per-episode difference in QUARTER units (int). -> n (integer form), point, ci95, b (fractions)."""
        q = np.asarray(q, np.int64)
        isums = np.bincount(self.inv, weights=q, minlength=self.k).astype(np.int64)
        tot = self.W @ isums
        n = int((tot <= 0).sum())
        b = (self.W @ (isums / 4.0)) / (self.W @ self.m.astype(np.float64))
        return {"n": n, "n_float": int((b <= 0).sum()), "point": 100 * float(np.mean(q / 4.0)),
                "ci95": [100 * float(np.percentile(b, 2.5)), 100 * float(np.percentile(b, 97.5))], "b": b}


def holm_ci(b, k, m):
    a = 0.025 / (m + 1 - k)
    return [100 * float(np.percentile(b, 100 * a)), 100 * float(np.percentile(b, 100 * (1 - a)))]


# ---------------------------------------------------------------- Holm (R6 rule §3 "Holm", r9 rule §4.1, §4.2)
def own_int(n, k, m):
    return 40 * (n + 1) * (m + 1 - k) <= 5001


def own_frac(n, k, m):
    return Fraction(n + 1, 5001) <= Fraction(1, 40) / (m + 1 - k)


def own_float(n, k, m):
    return (n + 1) / 5001 <= 0.025 / (m + 1 - k)


def nstar(k, m):
    n = -1
    while own_frac(n + 1, k, m):
        n += 1
    return n


def my_holm(counts, names):
    m = len(names)
    order = sorted(names, key=lambda c: (counts[c], names.index(c)))
    out, ok = {}, True
    for k, c in enumerate(order, 1):
        own = own_frac(counts[c], k, m)
        ok = ok and own
        out[c] = {"holm_k": k, "passes": ok, "own_count_passes": own, "n_star": nstar(k, m),
                  "near_boundary": abs(counts[c] - nstar(k, m)) <= 1, "level_two_sided": 1 - 0.05 / (m + 1 - k)}
    return order, out


def z_table():
    nd = NormalDist()
    z80 = nd.inv_cdf(0.8)
    return {"seven": nd.inv_cdf(1 - 0.025 / 7) + z80, "level1": nd.inv_cdf(1 - 0.025 / 2) + z80,
            "level2": nd.inv_cdf(1 - 0.025 / 5) + z80, "x95": nd.inv_cdf(0.975) + z80}


def variance_split(d_points, cl):
    """R3 rule §6.1 one-way split (in the units given)."""
    d = np.asarray(d_points, np.float64)
    _, inv = np.unique(cl, return_inverse=True)
    P, n = inv.max() + 1, len(d)
    m = np.bincount(inv, minlength=P).astype(float)
    mu = np.bincount(inv, weights=d, minlength=P) / m
    msb = np.sum(m * (mu - d.mean()) ** 2) / (P - 1)
    msw = np.sum((d - mu[inv]) ** 2) / (n - P)
    n0 = (n - np.sum(m ** 2) / n) / (P - 1)
    return {"sigma_a2": max(0.0, (msb - msw) / n0), "sigma_eps2": msw}


# ---------------------------------------------------------------- data with planted counts
def clusters(n_paint, per_seed, seeds=3, seed=0):
    """Pareto-sized anchor clusters with non-contiguous painting IDs; one array per seed."""
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(10 ** 6, size=n_paint, replace=False))
    w = rng.pareto(1.1, n_paint) + 1.0
    w /= w.sum()
    return [ids[rng.choice(n_paint, size=per_seed, p=w)] for _ in range(seeds)]


def plant(dr, target, rng, tries=40):
    """Per-episode quarters q (int in [-4, 4]) whose integer-form count is exactly `target`; None if not found."""
    E = len(dr.inv)
    for _ in range(tries):
        u = rng.normal(0, 1.2, dr.k)
        q = np.clip(np.rint(u[dr.inv] + rng.normal(0, 1.8, E)), -4, 4).astype(np.int64)
        isums = np.bincount(dr.inv, weights=q, minlength=dr.k).astype(np.int64)
        T = dr.W @ isums
        cnt = int((T <= 0).sum())
        step = 1 if cnt > target else -1
        order = rng.permutation(E)
        done = cnt == target
        for _pass in range(9):
            if done or (step == 1 and cnt <= target) or (step == -1 and cnt >= target):
                break
            for e in order:
                if (step == 1 and q[e] < 4) or (step == -1 and q[e] > -4):
                    q[e] += step
                    T += step * dr.WT[dr.inv[e]]
                    cnt = int((T <= 0).sum())
                    if (step == 1 and cnt <= target) or (step == -1 and cnt >= target):
                        break
        if cnt == target:
            return q
    return None


def per_seed_dicts(cls, qs, rng):
    """score_seed-shaped dicts (fractions) whose P1..P7 differences are qs[c] / 4, split back into seeds."""
    E = sum(len(c) for c in cls)
    a = rng.integers(0, 5, E) / 4.0
    d = {c: np.asarray(qs[c], np.int64) / 4.0 for c in CHECKS}
    full = {"aff_fused": {"r1": a, "gain": d["P6"]},
            "cosine": {"r1": a - d["P1"], "gain": np.zeros(E)},
            "rca": {"r1": a - d["P2"], "gain": d["P6"] - d["P7"]},
            "B": {"r1": a - d["P3"], "gain": np.zeros(E)},
            "B0": {"r1": a - d["P4"], "gain": np.zeros(E)},
            "aff_cf": {"r1": a - d["P5"], "gain": np.zeros(E)}}
    out, s = [], 0
    for c in cls:
        e = s + len(c)
        out.append({"cl": c, **{key: {m: v[s:e].copy() for m, v in arms.items()} for key, arms in full.items()}})
        s = e
    return out


def extra_for(seeds):
    h = lambda i: f"{i:064x}"  # noqa: E731
    return {"episodes_sha256": {s: {p: h(1000 * s + j + 1) for j, p in enumerate(PAIRS)} for s in seeds},
            "runner_sha256": "a" * 64, "module_sha256": {"run_r9_held.py": "b" * 64}}
