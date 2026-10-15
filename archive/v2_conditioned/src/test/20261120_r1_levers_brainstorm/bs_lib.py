"""Brainstorm harness (exploratory, seed 42, decides nothing). Pure numpy/torch on cache/bs_cache.npz.

A family is described by per-threshold terms: M[t] (condition-free, the same under both conditions) and D[t]
(antisymmetric: D^a = -D^b), so that

    fused^c   = (1 + lu) z(B) + lm * M[t] + ld * D[t]^c
    counterpart = (1 + lu) z(B) + lm * M[t]                    (the fused score with only the condition removed)

R1's own family is lm = ld = la with M = G_cf = (g^a z(T^a) + g^b z(T^b)) / 2 and D^c = (g^c z(T^c) - g^c' z(T^c')) / 2,
since g^c z(T^c) = G_cf + D^c. Cross-fits use round 2's exact integer criteria (min-margin for the fused reader against
the nested control, max-R@1 for the counterpart; ties to the lowest cell).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R2DIR = HERE.parent / "20261118_reader_fix_round2"
sys.path.insert(0, str(R2DIR))
import r2_common as R  # noqa: E402,F401  (puts round 1 and the repo root on sys.path)
import r2_fusion as F  # noqa: E402

C, K = R.C, R.K
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import NESTED_A, NESTED_U, _combine, control_sums  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

CACHE = HERE / "cache" / "bs_cache.npz"
PAIRS = C.POOLED_ORDER
A0 = ("affect", "image", "caption")
TOLD_IDX = {"emotion": 0, "style": 1, "genre": 1}           # A0 told mapping (diagnostic / declared oracle only)
ASPECTS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))


class Data:
    def __init__(self):
        z = np.load(CACHE)
        self.z = z
        self.B = {c: {d: z[f"B__{d}"] for d in DIRECTIONS} for c in CONDITIONS}
        self.zB = {c: {d: zscore_rows(torch.as_tensor(z[f"B__{d}"], dtype=torch.float32)) for d in DIRECTIONS}
                   for c in CONDITIONS}
        self.stack4 = {d: z[f"stack4__{d}"] for d in DIRECTIONS}
        self.stack = {d: self.stack4[d][:, :3] for d in DIRECTIONS}
        self.cl = z["anchor_group"]
        self.pi = z["pair_index"]
        self.parity = z["parity"]
        self.E = len(self.cl)
        self.pB = {m: z[f"pB__{m}"] for m in METRICS}
        self.pBp = {m: z[f"pBpA0__{m}"] for m in METRICS}
        self.pBp1 = {m: z[f"pBpA1__{m}"] for m in METRICS}
        self.feat = {c: z[f"feat_A1__{c}"] for c in CONDITIONS}
        self.info = F.rank_info(self.B)
        self.ctrl = F.control_choice(self.zB, self.parity)
        self.told = {c: np.array([TOLD_IDX[ASPECTS[i][j]] for i in self.pi]) for j, c in enumerate(CONDITIONS)}

    def P(self, name):
        return {c: self.z[f"P_{name}__{c}"] for c in CONDITIONS}


def zrows(x):
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32))


def term(stack, P):
    """T^c = sum_h P^c(h) s_h (float64 einsum, float32 out), as common.expected_term."""
    return {c: {d: np.einsum("nh,nhk->nk", np.asarray(P[c], np.float64), np.asarray(stack[d], np.float64))
                .astype(np.float32) for d in DIRECTIONS} for c in CONDITIONS}


def zterm(T):
    return {c: {d: zrows(T[c][d]) for d in DIRECTIONS} for c in CONDITIONS}


def margins_of(P):
    return {c: C.top_two_margin(P[c]) for c in CONDITIONS}


def thresholds(m):
    return K.thresholds(m)[0]


def gates_from(m, taus):
    return K.gates(m, taus)


def MD(zT, g):
    """M = G_cf and D^c = (g^c zT^c - g^c' zT^c') / 2 as float64 numpy, for one gate dict g = {c: (E,)}."""
    ga = np.asarray(g["a"], np.float64)[:, None]
    gb = np.asarray(g["b"], np.float64)[:, None]
    M, D = {}, {"a": {}, "b": {}}
    for d in DIRECTIONS:
        xa = ga * zT["a"][d].numpy().astype(np.float64)
        xb = gb * zT["b"][d].numpy().astype(np.float64)
        M[d] = 0.5 * (xa + xb)
        D["a"][d] = 0.5 * (xa - xb)
        D["b"][d] = -D["a"][d]
    return {c: M for c in CONDITIONS}, D


def ints(scores):
    pa = per_anchor(scores)
    return F.as_int4(pa["r1"]), F.as_int4(pa["gain"])


def score(zB64, M, D, lu, lm, ld, mask_d=None):
    out = {}
    for c in CONDITIONS:
        out[c] = {}
        for d in DIRECTIONS:
            s = (1.0 + lu) * zB64[d]
            if lm:
                s = s + lm * M[c][d]
            if ld:
                dd = D[c][d] if mask_d is None else D[c][d] * mask_d[:, None]
                s = s + ld * dd
            out[c][d] = s
    return out


class Family:
    """Generic decoupled family on the cache: cells (t, lu, lm, ld)."""

    def __init__(self, data, MDs, lus=NESTED_U, lms=NESTED_A, lds=NESTED_A, tied=False, mask_d=None):
        self.data, self.MDs = data, MDs
        self.zB64 = {d: data.zB["a"][d].numpy().astype(np.float64) for d in DIRECTIONS}
        self.mask_d = mask_d
        if tied:
            self.cells = [(t, lu, la, la) for t in range(len(MDs)) for lu in lus for la in lms]
        else:
            self.cells = [(t, lu, lm, ld) for t in range(len(MDs)) for lu in lus for lm in lms for ld in lds]
        self.cf_cells = [(t, lu, lm) for t in range(len(MDs)) for lu in lus for lm in lms]

    def stats(self):
        E = self.data.E
        fri = np.zeros((len(self.cells), E), np.int8)
        fgi = np.zeros((len(self.cells), E), np.int8)
        for i, (t, lu, lm, ld) in enumerate(self.cells):
            M, D = self.MDs[t]
            fri[i], fgi[i] = ints(score(self.zB64, M, D, lu, lm, ld, self.mask_d))
        cri = np.zeros((len(self.cf_cells), E), np.int8)
        for i, (t, lu, lm) in enumerate(self.cf_cells):
            M, D = self.MDs[t]
            cri[i] = ints(score(self.zB64, M, D, lu, lm, 0.0))[0]
        self.fri, self.fgi, self.cri = fri, fgi, cri
        return self

    def crossfit(self, allowed=None):
        par = self.data.parity
        fpick = F.select_fused(self.fri, self.fgi, self.data.ctrl, par, allowed)
        cpick = F.select_cf(self.cri, par)
        return fpick, cpick

    def assemble(self, fpick, cpick):
        par = self.data.parity
        fused = {c: {d: np.empty((self.data.E, 13)) for d in DIRECTIONS} for c in CONDITIONS}
        cf = {c: {d: np.empty((self.data.E, 13)) for d in DIRECTIONS} for c in CONDITIONS}
        for half in (0, 1):
            ap = par != half
            t, lu, lm, ld = self.cells[fpick[half]]
            M, D = self.MDs[t]
            s = score(self.zB64, M, D, lu, lm, ld, self.mask_d)
            t2, lu2, lm2 = self.cf_cells[cpick[half]]
            M2, D2 = self.MDs[t2]
            s2 = score(self.zB64, M2, D2, lu2, lm2, 0.0)
            for c in CONDITIONS:
                for d in DIRECTIONS:
                    fused[c][d][ap] = s[c][d][ap]
                    cf[c][d][ap] = s2[c][d][ap]
        return per_anchor(fused), per_anchor(cf)

    def in_sample(self):
        r = self.fri.sum(axis=1, dtype=np.int64)
        rc = self.cri.sum(axis=1, dtype=np.int64)
        i, j = int(np.argmax(r)), int(np.argmax(rc))
        return {"fused_best": 100 * r[i] / 4 / self.data.E, "fused_cell": self.cells[i],
                "cf_best": 100 * rc[j] / 4 / self.data.E, "cf_cell": self.cf_cells[j]}


def evaluate(data, pn, pc, label=""):
    """Bar margin (comparator = max of B', counterpart, B), margin, gain statistic, either, per pair."""
    bar_v, bar = C.bar_info(pn, pc, data.pBp, data.pB, data.cl, data.pi)
    d = C.diff3(pn, pc, data.cl)
    return {"label": label, "fused_r1": 100 * float(np.mean(pn["r1"])), "cf_r1": 100 * float(np.mean(pc["r1"])),
            "comparator": bar["comparator"], "bar": bar["r1"], "margin": d["r1"], "gain": d["gain"],
            "either": d["either"], "per_pair_bar": {p: v["point"] for p, v in bar["per_pair_r1"].items()},
            "per_pair_gain": {p: v["point"] for p, v in bar["per_pair_gain"].items()}}, bar_v


def fmt(r):
    b = r["bar"]
    pp = " / ".join(f"{v:+.3f}" for v in r["per_pair_bar"].values())
    return (f"{r['label']:<44s} fused {r['fused_r1']:.3f} cf {r['cf_r1']:.3f} bar {b['point']:+.3f} "
            f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] ({r['comparator']}) gain {r['gain']['point']:+.3f} "
            f"either {r['either']['point']:+.3f} | pairs e×s/e×g/s×g {pp}")


def standard_MDs(data, P, taus=None):
    """R1-style terms for a probability dict P: thresholds from P's own margins (or the given taus)."""
    T = term(data.stack, P)
    zT = zterm(T)
    m = margins_of(P)
    taus = thresholds(m) if taus is None else taus
    g = gates_from(m, taus)
    return [MD(zT, g[t]) for t in range(len(taus))], taus, zT, g, m
