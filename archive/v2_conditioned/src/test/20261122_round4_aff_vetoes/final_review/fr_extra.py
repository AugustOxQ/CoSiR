"""Final review: two extra descriptive checks on seed 42 (own code; reviewer's checks, decide nothing).

1. R1's fused minus counterpart R@1 per aspect pair and condition, at R1's chosen cells (the draft report cites
   "+0.01 and -0.26" for the emotion pairs' condition b; the brainstorm says +0.01 and -0.25).
2. How close each fused cross-fit choice was: the runner-up cell per tune half and its criterion gap; for V4 (gap 1 on
   tune half 0) the candidate's Delta_k had the runner-up been chosen (sensitivity of a descriptive statement only).
Writes out/fr_extra.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/project/CoSiR")
TST = ROOT / "src/test"
R1DIR = TST / "20261117_reader_fix_csd"
OUT = TST / "20261122_round4_aff_vetoes/final_review/out"
for p in (str(ROOT), str(R1DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402

import common as C1  # noqa: E402
import rb_build  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

f32, f64 = np.float32, np.float64
CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
A0 = ("affect", "image", "caption")
U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CELLS = [(t, u, a) for t in range(4) for u in range(7) for a in range(8)]
PAIRS = ("emotion__style", "emotion__genre", "style__genre")

bun = C1.load_bundle()
ctx, post, ep = bun.ctx, bun.post, bun.ctx.pooled
E = len(ep.anchor)
parity = np.arange(E) % 2
pidx = np.asarray(ctx.pair_index)
z = np.load(OUT / "fr_arrays.npz")
taus = json.loads((R1DIR / "results/rc_tau.json").read_text())["taus"]


def sets(c):
    return ((ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt) if c == "a"
            else (ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt))


def feats(c):
    si, st, ci, ct = sets(c)
    cols = []
    for h in A0:
        pi, pt = post[h]["img"], post[h]["txt"]
        sup = np.einsum("nsc,nsc->ns", pi[si], pt[st])
        con = np.einsum("nsc,nsc->ns", pi[ci], pt[ct])
        S, Cn = sup.mean(axis=1), con.mean(axis=1)
        match = (pi[si].argmax(axis=-1) == pt[st].argmax(axis=-1)).mean(axis=1)
        cols += [S, Cn, S - Cn, sup.astype(f64).std(axis=1, ddof=1), con.astype(f64).std(axis=1, ddof=1), match]
    return np.stack([np.asarray(x, f64) for x in cols], axis=1)


pk = rb_build.load_readers("A0", False)[0]
P = {}
for c in CONDS:
    X = feats(c)
    ps = [np.asarray(h["model"].predict_proba(h["scaler"].transform(X)), f64) for h in pk["halves"]]
    P[c] = (ps[0] + ps[1]) / 2.0
stk = {}
for d in DIRS:
    q, k = ("img", "txt") if d == "i2t" else ("txt", "img")
    stk[d] = np.stack([np.einsum("nc,nkc->nk", post[h][q][ep.anchor], post[h][k][ep.candidates]) for h in A0], axis=1)
Tm = {c: {d: np.einsum("nh,nhk->nk", P[c], stk[d].astype(f64)).astype(f32) for d in DIRS} for c in CONDS}
for c in CONDS:
    assert np.array_equal(P[c].argmax(1), z[f"pick0_{c}"])


def zs(x):
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy().astype(f32)


zB = {d: zs(bun.B["a"][d]) for d in DIRS}
zT = {c: {d: zs(Tm[c][d]) for d in DIRS} for c in CONDS}


def hits(s, col):
    tgt = s[:, col]
    oth = np.delete(s, col, axis=1)
    return ((oth < tgt[:, None]).all(axis=1) & np.isfinite(s).all(axis=1)).astype(np.int64)


def combine(base, term, lu, la):
    s = base
    if lu > 0:
        s = s + f32(lu) * base
    if la > 0:
        s = s + f32(la) * term
    return s.astype(f32)


def cell_hits(gates, cell, cf=False):
    """per-condition hit counts (sum over directions) of one cell: {c: (E,) int} for the target of c."""
    t, u, a = CELLS[cell]
    g = {c: z[f"{gates}_gate_{c}"][t] for c in CONDS}
    gated = {c: {d: (g[c][:, None] * zT[c][d]).astype(f32) for d in DIRS} for c in CONDS}
    if cf:
        G = {d: (0.5 * (gated["a"][d].astype(f64) + gated["b"][d].astype(f64))).astype(f32) for d in DIRS}
        sc = {c: {d: combine(zB[d], G[d], U[u], A[a]) for d in DIRS} for c in CONDS}
    else:
        sc = {c: {d: combine(zB[d], gated[c][d], U[u], A[a]) for d in DIRS} for c in CONDS}
    return {"a": sum(hits(sc["a"][d], 0) for d in DIRS), "b": sum(hits(sc["b"][d], 1) for d in DIRS)}


def assembled_cond(gates, picks, cf=False):
    h = {p: cell_hits(gates, picks[p], cf) for p in (0, 1)}
    out = {}
    for c in CONDS:
        out[c] = np.where(parity == 1, h[0][c], h[1][c])       # parity-1 episodes scored by the tune-half-0 cell
    return out


res = {}
fu = assembled_cond("R1", {0: 116, 1: 119})
cfh = assembled_cond("R1", {0: 58, 1: 123}, cf=True)
assert np.array_equal((fu["a"] + fu["b"]) / 4.0, z["R1_fused_r1"])
assert np.array_equal((cfh["a"] + cfh["b"]) / 4.0, z["R1_cf_r1"])
res["R1_fused_minus_cf_per_pair_condition_pp"] = {
    f"{p}|{c}": 100 * float(np.mean((fu[c] - cfh[c])[pidx == i]) / 2.0) for i, p in enumerate(PAIRS) for c in CONDS}

# runner-up cells and V4's Delta_k with its tune-half-0 runner-up
fr = json.loads((OUT / "fr_results.json").read_text())
ctrl = {int(h): v[1] for h, v in fr["control"].items()}


def crit_table(gates):
    fr4 = np.zeros((224, E), np.int64)
    fo4 = np.zeros((224, E), np.int64)
    for i in range(224):
        t, u, a = CELLS[i]
        g = {c: z[f"{gates}_gate_{c}"][t] for c in CONDS}
        sa = {d: combine(zB[d], (g["a"][:, None] * zT["a"][d]).astype(f32), U[u], A[a]) for d in DIRS}
        sb = {d: combine(zB[d], (g["b"][:, None] * zT["b"][d]).astype(f32), U[u], A[a]) for d in DIRS}
        fr4[i] = sum(hits(sa[d], 0) + hits(sb[d], 1) for d in DIRS)
        fo4[i] = sum(hits(sb[d], 0) + hits(sa[d], 1) for d in DIRS)
    return fr4, fo4


res["runner_up"] = {}
tabs = {}
for k in ("AFF", "V4", "V2", "V24"):
    fr4, fo4 = crit_table(k)
    tabs[k] = fr4
    rr = {}
    for h in (0, 1):
        tune = parity == h
        crit = np.minimum(fr4[:, tune].sum(1) - ctrl[h], (fr4 - fo4)[:, tune].sum(1))
        order = np.argsort(-crit, kind="stable")
        rr[str(h)] = {"best": [int(order[0]), int(crit[order[0]])], "runner_up": [int(order[1]), int(crit[order[1]])]}
    res["runner_up"][k] = rr
aff_r4 = np.where(parity == 1, tabs["AFF"][39], tabs["AFF"][119])
for k in ("V4", "V2", "V24"):
    ru0 = res["runner_up"][k]["0"]["runner_up"][0]
    alt = np.where(parity == 1, tabs[k][ru0], tabs[k][119])
    chosen = np.where(parity == 1, tabs[k][39], tabs[k][119])
    assert int((chosen - aff_r4).sum()) == fr["scorers"][k]["delta_int"]
    res["runner_up"][k]["delta_if_half0_runner_up"] = int((alt - aff_r4).sum())
(OUT / "fr_extra.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
