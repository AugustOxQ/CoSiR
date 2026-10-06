"""Final review of reader-fix round 2: this review's own implementation of the round-2 fusion family
(DECISION_RULE.md sections 4.5, 4.6, D10 to D12), written from the rule text. Only src/eval is imported
(_zdict for the D6 z-score, cluster_bootstrap for the intervals). No round-2 module is imported.

Hits are computed without building the restricted score: with the top-k set K taken from B, candidate j is strictly
first under the restriction iff j is in K and S(j) exceeds S(i) for every other i in K (every candidate outside K
scores at least 1 below the lowest in K). For k_top = 13 this is the plain strict first place.
"""
from pathlib import Path

import numpy as np
import torch

from src.eval.aspect_metrics import cluster_bootstrap
from src.eval.aspect_nested import _zdict

NU = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NA = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
KT = (13, 5, 3, 2)
DIRS = ("i2t", "t2i")
CONDS = ("a", "b")
SUMS = sorted({u + a for u in NU for a in NA})
assert len(SUMS) == 30
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def cellno(kappa, t, u, a):
    return ((kappa * 4 + t) * 7 + u) * 8 + a


def decode(i):
    return i // 224, (i // 56) % 4, (i // 8) % 7, i % 8


def load_cache():
    z = np.load(OUT / "fr_cache.npz")
    return {k: z[k] for k in z.files}


def B_of(cache):
    return {c: {d: cache[f"B__{c}__{d}"] for d in DIRS} for c in CONDS}


def positions(B):
    """pos[d][n, j] = position of candidate j in numpy.argsort(-b, kind='stable') of B's row (condition a == b)."""
    out = {}
    for d in DIRS:
        assert np.array_equal(B["a"][d], B["b"][d])
        o = np.argsort(-B["a"][d], axis=1, kind="stable")
        pos = np.empty_like(o)
        rows = np.arange(len(o))[:, None]
        pos[rows, o] = np.arange(13)[None, :]
        out[d] = pos
    return out


def first(S, col, inK):
    """1 where candidate `col` is strictly first among the candidates of K (inK bool mask)."""
    S = np.asarray(S)
    Sm = np.where(inK, S, -np.inf)
    Sm[:, col] = -np.inf
    return (inK[:, col] & (S[:, col] > Sm.max(axis=1))).astype(np.int64)


def int_stats(sc, pos, k):
    """(4*R@1, 4*gain, 4*other) per episode as int64 of a score dict {cond: {dir: (E, 13)}} restricted to B's top k."""
    r = np.zeros(len(pos["i2t"]), np.int64)
    o = np.zeros_like(r)
    for d in DIRS:
        inK = pos[d] < k
        r += first(sc["a"][d], 0, inK) + first(sc["b"][d], 1, inK)
        o += first(sc["b"][d], 0, inK) + first(sc["a"][d], 1, inK)
    return r, r - o, o


def comb(zB, X, lu, la):
    """aspect_nested._combine's arithmetic in float32 torch: z(B) [+ lu z(B)] [+ la X], weight-0 terms left out."""
    out = {}
    for c in CONDS:
        out[c] = {}
        for d in DIRS:
            s = zB[c][d]
            if lu > 0:
                s = s + lu * zB[c][d]
            if la > 0:
                s = s + la * X[c][d]
            out[c][d] = s.numpy().astype(np.float32)
    return out


def taus_of(margins):
    m = np.concatenate([margins["a"], margins["b"]]).astype(np.float64)
    assert len(m) == 24_576
    return [float(x) for x in np.percentile(m, [0, 25, 50, 75])]


def terms(P, cache, parts):
    """T^c = sum_h P^c(h) s_h (float64 sum, float32 result), picks (first max), top-two margins."""
    T, picks, margins = {}, {}, {}
    for c in CONDS:
        Pc = np.asarray(P[c], np.float64)
        T[c] = {}
        for d in DIRS:
            acc = np.zeros(cache[f"s__affect__{d}"].shape, np.float64)
            for j, h in enumerate(parts):
                acc += Pc[:, j:j + 1] * cache[f"s__{h}__{d}"].astype(np.float64)
            T[c][d] = acc.astype(np.float32)
        picks[c] = Pc.argmax(axis=1)
        srt = np.sort(Pc, axis=1)
        margins[c] = srt[:, -1] - srt[:, -2]
    return T, picks, margins


def family(cache, T, margins, taus, n_kappa=4):
    """All cells: fri, fgi (fused 4R@1, 4gain), cri (counterpart 4R@1), int64 (cells, E). Also the gated z-terms."""
    B = B_of(cache)
    pos = positions(B)
    zB, zT = _zdict(B), _zdict(T)
    E = len(cache["cl"])
    n = n_kappa * 224
    fri, fgi, cri = (np.zeros((n, E), np.int8) for _ in range(3))
    gated, G = {}, {}
    for t, tau in enumerate(taus):
        g = {c: torch.as_tensor((np.asarray(margins[c], np.float64) >= tau).astype(np.float32)) for c in CONDS}
        gated[t] = {c: {d: g[c][:, None] * zT[c][d] for d in DIRS} for c in CONDS}
        avg = {d: (0.5 * (gated[t]["a"][d].numpy().astype(np.float64) + gated[t]["b"][d].numpy().astype(np.float64)))
               .astype(np.float32) for d in DIRS}
        G[t] = {c: {d: torch.as_tensor(avg[d].copy()) for d in DIRS} for c in CONDS}
        for u, lu in enumerate(NU):
            for a, la in enumerate(NA):
                sf = comb(zB, gated[t], lu, la)
                sc = comb(zB, G[t], lu, la)
                for d in DIRS:
                    assert np.array_equal(sc["a"][d], sc["b"][d])
                for kap in range(n_kappa):
                    i = cellno(kap, t, u, a)
                    r, gn, _ = int_stats(sf, pos, KT[kap])
                    fri[i], fgi[i] = r, gn
                    rc_, gc_, _ = int_stats(sc, pos, KT[kap])
                    assert not gc_.any(), "counterpart is not condition-free"
                    cri[i] = rc_
    return {"fri": fri, "fgi": fgi, "cri": cri, "gated": gated, "G": G, "zB": zB, "pos": pos, "B": B}


def control(cache, zB):
    parity = cache["parity"]
    pos = positions(B_of(cache))
    rho = []
    for s in SUMS:
        r, _, _ = int_stats(comb(zB, zB, s, 0.0), pos, 13)
        rho.append(r)
    out = {}
    for h in (0, 1):
        tune = parity == h
        vals = [int(r[tune].sum()) for r in rho]
        best = max(vals)
        out[h] = (SUMS[vals.index(best)], best)          # first (smallest sigma) with the largest rho
    return out


def pick(fam, ctrl, parity, allowed=None):
    n = len(fam["fri"])
    allowed = list(range(n)) if allowed is None else sorted(allowed)
    fp, cp, crit_out = {}, {}, {}
    for h in (0, 1):
        tune = parity == h
        rho = fam["fri"][allowed][:, tune].astype(np.int64).sum(1)
        gam = fam["fgi"][allowed][:, tune].astype(np.int64).sum(1)
        crit = np.minimum(rho - ctrl[h][1], gam)
        best = crit.max()
        fp[h] = allowed[int(np.flatnonzero(crit == best)[0])]
        crho = fam["cri"][allowed][:, tune].astype(np.int64).sum(1)
        cp[h] = allowed[int(np.flatnonzero(crho == crho.max())[0])]
        crit_out[h] = {"fused_crit": int(best), "fused_ties": int((crit == best).sum()),
                       "fused_rho": int(rho[allowed.index(fp[h])]), "fused_gamma": int(gam[allowed.index(fp[h])]),
                       "cf_rho": int(crho.max()), "cf_ties": int((crho == crho.max()).sum())}
    return fp, cp, crit_out


def restricted_order_key(S, pos, k):
    """A float64 score with S' 's order (rule 4.5 item 6) for the swap metric: inside K S itself, outside K below every
    inside value and in B's order."""
    S = np.asarray(S, np.float64)
    inK = pos < k
    if k >= 13:
        return S
    low = np.where(inK, S, np.inf).min(axis=1, keepdims=True)
    return np.where(inK, S, low - 1.0 - (pos - k))


def assembled_scores(fam, terms_by_t, picks, parity):
    """Scores of each half's cell applied to the other half (float64), with the restriction applied."""
    zB, pos = fam["zB"], fam["pos"]
    out = {c: {d: np.empty(zB[c][d].shape, np.float64) for d in DIRS} for c in CONDS}
    for h in (0, 1):
        app = parity != h
        kap, t, u, a = decode(picks[h])
        s = comb(zB, terms_by_t[t], NU[u], NA[a])
        for c in CONDS:
            for d in DIRS:
                out[c][d][app] = restricted_order_key(s[c][d], pos[d], KT[kap])[app]
    return out


def per_episode(sc):
    """r1, gain, other, swap, strict per episode (as src.eval.aspect_metrics.per_anchor defines them), own code."""
    acc = {m: np.zeros(len(sc["a"]["i2t"])) for m in ("r1", "gain", "other", "swap", "strict")}
    all13 = np.ones(sc["a"]["i2t"].shape, bool)
    for d in DIRS:
        sa, sb = np.asarray(sc["a"][d], np.float64), np.asarray(sc["b"][d], np.float64)
        aa, bb = first(sa, 0, all13).astype(float), first(sb, 1, all13).astype(float)
        ab, ba = first(sb, 0, all13).astype(float), first(sa, 1, all13).astype(float)
        r1 = 0.5 * (aa + bb)
        oth = 0.5 * (ab + ba)
        acc["r1"] += 0.5 * r1
        acc["other"] += 0.5 * oth
        acc["gain"] += 0.5 * (r1 - oth)
        acc["swap"] += 0.5 * ((sa[:, 0] > sa[:, 1]) & (sb[:, 1] > sb[:, 0]))
        acc["strict"] += 0.5 * (aa * bb)
    return acc


def from_cells(M, picks, parity):
    """Per-episode metric (pp/100) of the assembled cross-fit from a per-cell integer matrix (4x metric)."""
    v = np.empty(M.shape[1], np.float64)
    for h in (0, 1):
        app = parity != h
        v[app] = M[picks[h]][app] / 4.0
    return v


def ci(values, cl):
    r = cluster_bootstrap(np.asarray(values, np.float64), cl)
    return {"point": 100 * r["point"], "ci95": [100 * x for x in r["ci95"]]}


def comparator(cache, cf_r1, cfg="A0"):
    means = {"B_prime": float(np.mean(cache[f"pBp_{cfg}__r1"].astype(np.float64))),
             "counterpart": float(np.mean(np.asarray(cf_r1, np.float64))),
             "B": float(np.mean(cache["pB__r1"].astype(np.float64)))}
    arrays = {"B_prime": cache[f"pBp_{cfg}__r1"], "counterpart": cf_r1, "B": cache["pB__r1"]}
    best = "B_prime"
    for k in ("counterpart", "B"):
        if means[k] > means[best]:
            best = k
    return best, np.asarray(arrays[best], np.float64), {k: 100 * v for k, v in means.items()}


def decision(cache, f_r1, f_gain, c_r1, c_gain, cfg="A0", with_ci=True):
    cl = cache["cl"]
    name, comp, means = comparator(cache, c_r1, cfg)
    bar_v = np.asarray(f_r1, np.float64) - comp
    out = {"fused": 100 * float(np.mean(f_r1)), "cf": 100 * float(np.mean(c_r1)), "comparator": name,
           "comparator_means": means, "bar_v": bar_v}
    if with_ci:
        out["bar"] = ci(bar_v, cl)
        out["gain_stat"] = ci(np.asarray(f_gain, np.float64) - np.asarray(c_gain, np.float64), cl)
        out["margin"] = ci(np.asarray(f_r1, np.float64) - np.asarray(c_r1, np.float64), cl)
        out["clauses"] = [out["bar"]["point"] >= 0.5, out["bar"]["ci95"][0] > 0, out["gain_stat"]["ci95"][0] > 0]
    return out
