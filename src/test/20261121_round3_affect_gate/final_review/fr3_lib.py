"""Final review of round 3: own implementation of the rule's computations (DECISION_RULE.md D3 to D14, §2, §6.1,
§6.5 to §6.7, §7), written from the rule text. Imports only the components the review brief allows: zscore_rows
(D8) and cluster_bootstrap (§2). Nothing here imports a round-3 module or round 1's / round 2's fusion code.
"""
import sys
from pathlib import Path

import numpy as np
import torch

FR = Path(__file__).resolve().parent
R3DIR = FR.parent
ROOT = FR.parents[3]
OUT = FR / "out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402  (allowed)
from src.model.aspect_rule import zscore_rows  # noqa: E402  (allowed)

CONDS = ("a", "b")
DIRS = ("i2t", "t2i")
A0 = ("affect", "image", "caption")
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
TAUS = (3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211)   # rule D6
LAM_U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LAM_A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SIGMAS = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 3, 4, 4.25, 4.5, 5, 6, 8, 8.25, 8.5, 9, 10, 12, 16, 16.25,
          16.5, 17, 18, 20, 24, 32)                                                                   # rule D8.5
N_CELLS = 4 * 7 * 8


def cell_of(t, u, a):
    return (t * 7 + u) * 8 + a


def decode(cell):
    t, rest = divmod(int(cell), 56)
    u, a = divmod(rest, 8)
    return t, LAM_U[u], LAM_A[a]


# ---------------------------------------------------------------- scores and metrics

def z(x):
    """D8.1: per-row z-score over the 13 candidates (zscore_rows, float32)."""
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy()


def first(S, col):
    """bool (n,): candidate `col` strictly above every other candidate (ties miss)."""
    S = np.asarray(S)
    t = S[:, col:col + 1]
    others = np.delete(S, col, axis=1)
    return (others < t).all(axis=1) & np.isfinite(S).all(axis=1)


def counts(scores):
    """scores {c: {d: (n,13)}} -> (rho, other) int arrays: target hits and other-aspect hits over the 4 rankings.
    Condition a: target column 0, other column 1; condition b: target column 1, other column 0."""
    rho = 0
    oth = 0
    for d in DIRS:
        rho = rho + first(scores["a"][d], 0).astype(np.int64) + first(scores["b"][d], 1).astype(np.int64)
        oth = oth + first(scores["a"][d], 1).astype(np.int64) + first(scores["b"][d], 0).astype(np.int64)
    return rho, oth


def metrics(scores):
    """Per-episode r1, gain, other, swap, strict (fractions), my own definitions of §2."""
    rho, oth = counts(scores)
    swap = 0.0
    strict = 0.0
    for d in DIRS:
        sa, sb = np.asarray(scores["a"][d], np.float64), np.asarray(scores["b"][d], np.float64)
        swap = swap + ((sa[:, 0] > sa[:, 1]) & (sb[:, 1] > sb[:, 0])).astype(np.float64)
        strict = strict + (first(sa, 0) & first(sb, 1)).astype(np.float64)
    return {"r1": rho / 4.0, "gain": (rho - oth) / 4.0, "other": oth / 4.0, "swap": swap / 2.0, "strict": strict / 2.0}


def ci(values, clusters):
    """Point and 95% painting-bootstrap interval in percentage points (rule §2)."""
    r = cluster_bootstrap(np.asarray(values, np.float64), np.asarray(clusters))
    return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}


# ---------------------------------------------------------------- reader (D5) and gates (D6)

def features(post, ep):
    """D5: the 18 features per condition, from the posteriors (float32 agreements, as D3), in A0 order: S, C, Delta,
    sd(S pairs, ddof 1), sd(C pairs, ddof 1), match share of the support pairs."""
    out = {}
    for c in CONDS:
        if c == "a":
            si, st, ci_, ct = ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt
        else:
            si, st, ci_, ct = ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt
        cols = []
        for h in A0:
            pi, pt = post[h]["img"], post[h]["txt"]
            sup = np.einsum("nsc,nsc->ns", pi[si], pt[st])
            con = np.einsum("nsc,nsc->ns", pi[ci_], pt[ct])
            S, Cc = sup.mean(axis=1), con.mean(axis=1)
            match = (pi[si].argmax(-1) == pt[st].argmax(-1)).mean(axis=1)
            cols += [S, Cc, S - Cc, sup.astype(np.float64).std(axis=1, ddof=1),
                     con.astype(np.float64).std(axis=1, ddof=1), match]
        out[c] = np.stack([np.asarray(x, np.float64) for x in cols], axis=1)
    return out


def grouping_scores(post, ep):
    """D4: s_h(q, k) = p_h(q) . p_h(k), each item with its own modality's head. {d: (n, 3, 13) float32}."""
    out = {}
    for d in DIRS:
        qm, km = ("img", "txt") if d == "i2t" else ("txt", "img")
        out[d] = np.stack([np.einsum("nc,nkc->nk", post[h][qm][ep.anchor], post[h][km][ep.candidates])
                           for h in A0], axis=1)
    return out


def read(pk, F, stack):
    """D5: P^c (mean of the two half-readers), T^c, pick (np.argmax), margin (largest minus second largest)."""
    P, T, pick, m = {}, {}, {}, {}
    for c in CONDS:
        probs = [h["model"].predict_proba(h["scaler"].transform(F[c])) for h in pk["halves"]]
        P[c] = np.mean(np.stack([np.asarray(p, np.float64) for p in probs]), axis=0)
        pick[c] = np.argmax(P[c], axis=1).astype(np.int64)
        srt = np.sort(P[c], axis=1)
        m[c] = srt[:, -1] - srt[:, -2]
        T[c] = {d: np.einsum("nh,nhk->nk", P[c], stack[d].astype(np.float64)).astype(np.float32) for d in DIRS}
    return P, T, pick, m


def gates(m, pick, kind, taus=TAUS):
    """D6: list over tau index of {c: (n,) float32}; kind 'R1' or 'AFF' (AFF also needs pick == 0 = affect)."""
    out = []
    for t in taus:
        g = {}
        for c in CONDS:
            open_ = np.asarray(m[c], np.float64) >= t
            if kind == "AFF":
                open_ = open_ & (np.asarray(pick[c]) == 0)
            g[c] = open_.astype(np.float32)
        out.append(g)
    return out


def random_gates(g_r1, share, gen_seed, n):
    """§7 item 7: keep^c = 1[u < share_c], one generator, condition a drawn first."""
    gen = np.random.default_rng(gen_seed)
    keep = {}
    for c in CONDS:
        keep[c] = (gen.random(n) < share[c]).astype(np.float32)
    return [{c: g[c] * keep[c] for c in CONDS} for g in g_r1]


# ---------------------------------------------------------------- the 224-cell family (D8, D9)

def combine(zB, term, lu, la):
    """D8.3 in _combine's order: zB, + lu*zB if lu > 0, + la*term if la > 0, float32."""
    s = zB
    if lu > 0:
        s = s + np.float32(lu) * zB
    if la > 0:
        s = s + np.float32(la) * term
    return s.astype(np.float32)


class Family:
    """Integer statistics of all 224 cells for one gate set, the control, both cross-fits and the assembly."""

    def __init__(self, B, T, gate_list, parity, with_cf=True):
        self.parity = np.asarray(parity)
        n = len(self.parity)
        self.zB = {c: {d: z(B[c][d]) for d in DIRS} for c in CONDS}
        for d in DIRS:
            assert np.array_equal(self.zB["a"][d], self.zB["b"][d]), "B must be condition-free"
        self.zT = {c: {d: z(T[c][d]) for d in DIRS} for c in CONDS}
        self.gated, self.G = [], []
        for g in gate_list:
            gt = {c: {d: (g[c][:, None] * self.zT[c][d]).astype(np.float32) for d in DIRS} for c in CONDS}
            Gd = {d: (0.5 * (gt["a"][d].astype(np.float64) + gt["b"][d].astype(np.float64))).astype(np.float32)
                  for d in DIRS}
            self.gated.append(gt)
            self.G.append({c: Gd for c in CONDS})
        self.rho_f = np.zeros((N_CELLS, n), np.int64)
        self.gam_f = np.zeros((N_CELLS, n), np.int64)
        self.rho_c = np.zeros((N_CELLS, n), np.int64) if with_cf else None
        for t in range(4):
            for ui, lu in enumerate(LAM_U):
                for ai, la in enumerate(LAM_A):
                    k = cell_of(t, ui, ai)
                    S = {c: {d: combine(self.zB[c][d], self.gated[t][c][d], lu, la) for d in DIRS} for c in CONDS}
                    r, o = counts(S)
                    self.rho_f[k], self.gam_f[k] = r, r - o
                    if with_cf:
                        Sc = {c: {d: combine(self.zB[c][d], self.G[t][c][d], lu, la) for d in DIRS} for c in CONDS}
                        for d in DIRS:
                            assert np.array_equal(Sc["a"][d], Sc["b"][d]), "counterpart must be condition-free"
                        self.rho_c[k] = counts(Sc)[0]
        # nested control: sigma* = smallest sigma with the largest rho((1 + sigma) z(B)) in _combine's order
        self.ctrl = {}
        rho_s = []
        for s in SIGMAS:
            S = {c: {d: combine(self.zB[c][d], self.zB[c][d], s, 0.0) for d in DIRS} for c in CONDS}
            rho_s.append(counts(S)[0])
        rho_s = np.stack(rho_s)
        for h in (0, 1):
            tune = self.parity == h
            sums = rho_s[:, tune].sum(axis=1)
            j = int(np.argmax(sums))                        # first maximum = smallest sigma
            self.ctrl[h] = (float(SIGMAS[j]), int(sums[j]))

    def pick_fused(self):
        out, crit = {}, {}
        for h in (0, 1):
            tune = self.parity == h
            rho = self.rho_f[:, tune].sum(axis=1)
            gam = self.gam_f[:, tune].sum(axis=1)
            c = np.minimum(rho - self.ctrl[h][1], gam)
            out[h] = int(np.argmax(c))                       # ties to the lowest cell number
            crit[h] = {"rho": int(rho[out[h]]), "gamma": int(gam[out[h]]), "crit": int(c[out[h]]),
                       "n_ties": int((c == c.max()).sum())}
        return out, crit

    def pick_cf(self):
        out, crit = {}, {}
        for h in (0, 1):
            tune = self.parity == h
            rho = self.rho_c[:, tune].sum(axis=1)
            out[h] = int(np.argmax(rho))
            crit[h] = {"rho": int(rho[out[h]]), "n_ties": int((rho == rho.max()).sum())}
        return out, crit

    def scores_of(self, cell, cf=False):
        t, lu, la = decode(cell)
        term = self.G[t] if cf else self.gated[t]
        return {c: {d: combine(self.zB[c][d], term[c][d], lu, la) for d in DIRS} for c in CONDS}

    def assemble(self, picks, cf=False):
        """The cell chosen on tune half h scores the episodes of parity 1 - h. -> per-episode metrics."""
        n = len(self.parity)
        out = {c: {d: np.empty((n, 13), np.float32) for d in DIRS} for c in CONDS}
        for h in (0, 1):
            apply = self.parity != h
            S = self.scores_of(picks[h], cf)
            for c in CONDS:
                for d in DIRS:
                    out[c][d][apply] = S[c][d][apply]
        return metrics(out)


# ---------------------------------------------------------------- comparisons (D12, §6.5)

def either(pa):
    return np.asarray(pa["r1"], np.float64) + np.asarray(pa["other"], np.float64)


def bar_comparator(pBp, pcf, pB):
    """D12: largest mean R@1 of B', counterpart, B; ties to the earliest (B', counterpart, B)."""
    means = [float(np.mean(np.asarray(x["r1"], np.float64))) for x in (pBp, pcf, pB)]
    best = 0
    for i in (1, 2):
        if means[i] > means[best]:
            best = i
    return ("B_prime", "counterpart", "B")[best], (pBp, pcf, pB)[best], means


def seven(S, cl):
    """The seven GO checks and the secondary on one scope. S: dict of per-episode dicts 'fused', 'cf', 'second'
    (secondary comparator), 'cosine', 'rca', 'B', 'Bp'."""
    f = S["fused"]
    assert np.all(np.asarray(S["cf"]["gain"]) == 0)
    d = lambda k: np.asarray(f["r1"]) - np.asarray(S[k]["r1"])  # noqa: E731
    out = {"r1_vs_cosine": ci(d("cosine"), cl), "r1_vs_rca": ci(d("rca"), cl), "r1_vs_B": ci(d("B"), cl),
           "r1_vs_Bprime": ci(d("Bp"), cl), "r1_vs_counterpart": ci(d("cf"), cl),
           "gain_statistic": ci(np.asarray(f["gain"]) - np.asarray(S["cf"]["gain"]), cl),
           "gain_vs_rca": ci(np.asarray(f["gain"]) - np.asarray(S["rca"]["gain"]), cl)}
    for k in out:
        out[k]["pass"] = bool(out[k]["ci95"][0] > 0)
    out["go"] = all(out[k]["pass"] for k in list(out))
    if S.get("second") is not None:
        out["secondary"] = ci(d("second"), cl)
        out["secondary"]["pass"] = bool(out["secondary"]["ci95"][0] > 0)
    return out


def cat(dicts):
    return {m: np.concatenate([np.asarray(x[m], np.float64) for x in dicts]) for m in dicts[0]}


def sub(pa, mask):
    return {m: np.asarray(v)[mask] for m, v in pa.items()}


# ---------------------------------------------------------------- sensitivity (§6.1)

def sensitivity(diff_pp, cl):
    d = np.asarray(diff_pp, np.float64)
    _, idx = np.unique(np.asarray(cl), return_inverse=True)
    n, P = len(d), int(idx.max()) + 1
    m = np.bincount(idx, minlength=P).astype(np.float64)
    sums = np.bincount(idx, weights=d, minlength=P)
    means = sums / m
    grand = d.mean()
    ms_between = np.sum(m * (means - grand) ** 2) / (P - 1)
    ms_within = np.sum((d - means[idx]) ** 2) / (n - P)
    sm2 = np.sum(m ** 2)
    n0 = (n - sm2 / n) / (P - 1)
    se2 = max(0.0, (ms_between - ms_within) / n0) * (9 * sm2 - 6 * n) + ms_within * 3 * n
    se = np.sqrt(se2 / (3 * n) ** 2)
    r = cluster_bootstrap(d, cl)
    return {"SE": float(se), "half_width": 1.96 * float(se), "x": 2.80 * float(se),
            "seed42_half_width": 0.5 * (r["ci95"][1] - r["ci95"][0])}


# ---------------------------------------------------------------- redundancy (D7)

def redundancy(stack, B):
    out = {}
    for j, h in enumerate(A0):
        out[h] = {}
        for d in DIRS:
            x = z(stack[d][:, j]).astype(np.float64)
            y = z(B["a"][d]).astype(np.float64)
            x = x - x.mean(1, keepdims=True)
            y = y - y.mean(1, keepdims=True)
            num = (x * y).sum(1)
            den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
            ok = den > 0
            out[h][d] = float(np.mean(num[ok] / den[ok]))
    return out
