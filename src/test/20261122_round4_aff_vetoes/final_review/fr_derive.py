"""Final review of round 4: a third, independent derivation of the seed-42 decision (rule DECISION_RULE.md §5).

Own code for: features, the A0 and A1 readers' P, T, margins and picks, v and v75, every gate (R1, AFF, IMGABST, V4, V2,
V24), z-scored terms, G_cf, the 224 cells' integer statistics, sigma*, both cross-fits, the assembly, the per-anchor
metrics, the comparators and bar comparator, the margins, gain statistics, Delta_k, the D10 clauses and the carry.
Imports only what the rule lets a re-derivation import: round 1's common.load_bundle (seed 42 only; data, B, B'(A0),
B'(A1), posteriors), rb_build.load_readers, zscore_rows, cluster_bootstrap. Not r4_*, r3_fusion, r2_fusion or rederive/.
CPU only. Writes only to final_review/out/.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/project/CoSiR")
TST = ROOT / "src/test"
R1DIR = TST / "20261117_reader_fix_csd"
HERE = TST / "20261122_round4_aff_vetoes"
OUT = HERE / "final_review/out"
OUT.mkdir(parents=True, exist_ok=True)
for p in (str(ROOT), str(R1DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import common as C1  # noqa: E402  (round 1)
import rb_build  # noqa: E402
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

assert Path(C1.__file__).resolve().parent == R1DIR, C1.__file__
assert Path(rb_build.__file__).resolve().parent == R1DIR, rb_build.__file__

f32, f64 = np.float32, np.float64
CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
A0 = ("affect", "image", "caption")
A1 = ("affect", "image", "caption", "csd")
U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CELLS = [(t, u, a) for t in range(4) for u in range(7) for a in range(8)]   # number = (t*7 + u)*8 + a
SIGMAS = sorted({x + y for x in U for y in A})
V75 = 0.021043562795966864
RULE_SHA = "cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b"
PAIRS = ("emotion__style", "emotion__genre", "style__genre")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


assert sha(HERE / "DECISION_RULE.md") == RULE_SHA
for i, (t, u, a) in enumerate(CELLS):
    assert i == (t * 7 + u) * 8 + a

# ------------------------------------------------------------------ data (round 1's loader, seed 42)
t0 = time.time()
bun = C1.load_bundle()
log(f"round-1 bundle loaded [{time.time() - t0:.0f}s]")
ctx, post = bun.ctx, bun.post
ep = ctx.pooled
E = len(ep.anchor)
assert E == 12288
parity = np.asarray(ctx.parity)
assert np.array_equal(parity, np.arange(E) % 2)
cl = np.asarray(bun.cl)
pidx = np.asarray(ctx.pair_index)


def sets(c):
    if c == "a":
        return ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt
    return ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt


def feats(parts, c):
    si, st, ci, ct = sets(c)
    cols = []
    for h in parts:
        pi, pt = post[h]["img"], post[h]["txt"]
        sup = np.einsum("nsc,nsc->ns", pi[si], pt[st])
        con = np.einsum("nsc,nsc->ns", pi[ci], pt[ct])
        S, Cn = sup.mean(axis=1), con.mean(axis=1)
        match = (pi[si].argmax(axis=-1) == pt[st].argmax(axis=-1)).mean(axis=1)
        cols += [S, Cn, S - Cn, sup.astype(f64).std(axis=1, ddof=1), con.astype(f64).std(axis=1, ddof=1), match]
    return np.stack([np.asarray(x, f64) for x in cols], axis=1)


F0 = {c: feats(A0, c) for c in CONDS}
F1 = {c: feats(A1, c) for c in CONDS}
for c in CONDS:
    assert np.array_equal(F1[c][:, :18], F0[c]), "A1's first 18 columns differ from A0's"
    assert np.isfinite(F1[c]).all()
log("features built")


def probs(cfg, X):
    pk = rb_build.load_readers(cfg, False)[0]
    ps = []
    for hlf in pk["halves"]:
        ps.append(hlf["model"].predict_proba(hlf["scaler"].transform(X)))
    return (np.asarray(ps[0], f64) + np.asarray(ps[1], f64)) / 2.0


P0 = {c: probs("A0", F0[c]) for c in CONDS}
P1 = {c: probs("A1", F1[c]) for c in CONDS}
pick0 = {c: P0[c].argmax(axis=1) for c in CONDS}
pick1 = {c: P1[c].argmax(axis=1) for c in CONDS}
srt = {c: np.sort(P0[c], axis=1) for c in CONDS}
marg = {c: srt[c][:, -1] - srt[c][:, -2] for c in CONDS}


def stack(parts):
    out = {}
    for d in DIRS:
        q, k = ("img", "txt") if d == "i2t" else ("txt", "img")
        out[d] = np.stack([np.einsum("nc,nkc->nk", post[h][q][ep.anchor], post[h][k][ep.candidates]) for h in parts],
                          axis=1)
    return out


STK = stack(A0)
Tm = {c: {d: np.einsum("nh,nhk->nk", P0[c], STK[d].astype(f64)).astype(f32) for d in DIRS} for c in CONDS}

# stored round-1 R-c reader arrays and round-2 A1 reader
z_rc = np.load(R1DIR / "results/cand_Rc_Rb_expected_A0.npz")
z_a1 = np.load(TST / "20261118_reader_fix_round2/results/cand_R1_A1.npz")
taus = json.loads((R1DIR / "results/rc_tau.json").read_text())["taus"]
chk = {}
for c in CONDS:
    chk[f"pick0_{c}"] = bool(np.array_equal(pick0[c], z_rc[f"pick__{c}"].astype(np.int64)))
    chk[f"margin_{c}"] = bool(np.array_equal(marg[c], z_rc[f"margin__{c}"]))
    for d in DIRS:
        chk[f"T_{c}_{d}"] = bool(np.array_equal(Tm[c][d], z_rc[f"T__{c}__{d}"]))
    chk[f"P1_{c}"] = bool(np.array_equal(P1[c], z_a1[f"probs__{c}"]))
    chk[f"pick1_{c}"] = bool(np.array_equal(pick1[c], z_a1[f"pick__{c}"].astype(np.int64)))
tau_re = np.percentile(np.concatenate([marg["a"], marg["b"]]), [0, 25, 50, 75])
chk["taus"] = bool(np.array_equal(tau_re, np.asarray(taus)))
# A1 tie check
for c in CONDS:
    s1 = np.sort(P1[c], axis=1)
    chk[f"A1_no_exact_tie_{c}"] = bool((s1[:, -1] > s1[:, -2]).all())
v = np.minimum(F0["a"][:, 6], F0["a"][:, 7])
vb = np.minimum(F0["b"][:, 6], F0["b"][:, 7])
chk["v_b_equals_v_a"] = bool(np.array_equal(v, vb))
chk["v75_exact"] = bool(float(np.percentile(v, 75)) == V75)
keep = (v < V75).astype(f32)
chk["keep_9216"] = int(keep.sum()) == 9216
log(f"reader checks: {chk}")
assert all(chk.values()), chk

# ------------------------------------------------------------------ gates


def g_r1():
    return [{c: (marg[c] >= taus[t]).astype(f32) for c in CONDS} for t in range(4)]


def times(g, fac):
    """fac: {c: (E,) 0/1}"""
    return [{c: (g[t][c] * np.asarray(fac[c], f32)).astype(f32) for c in CONDS} for t in range(4)]


GR1 = g_r1()
GAFF = times(GR1, {c: (pick0[c] == 0) for c in CONDS})
KEEP = {c: keep for c in CONDS}
APICK = {c: (pick1[c] == 0) for c in CONDS}
GATES = {
    "R1": GR1,
    "IMGABST": times(GR1, KEEP),
    "AFF": GAFF,
    "V4": times(GAFF, KEEP),
    "V2": times(GAFF, APICK),
    "V24": times(times(GAFF, KEEP), APICK),
}
for k in ("V4", "V2", "V24"):
    for t in range(4):
        for c in CONDS:
            assert not np.any(GATES[k][t][c] > GAFF[t][c])
open0 = {k: {c: int(GATES[k][0][c].sum()) for c in CONDS} for k in GATES}
log(f"tau_0 open counts: {open0}")

# ------------------------------------------------------------------ family


def zs(x):
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy().astype(f32)


Bsc = bun.B
for d in DIRS:
    assert np.array_equal(Bsc["a"][d], Bsc["b"][d])
zB = {d: zs(Bsc["a"][d]) for d in DIRS}
zT = {c: {d: zs(Tm[c][d]) for d in DIRS} for c in CONDS}


def hits(s, col):
    s = np.asarray(s, f32)
    tgt = s[:, col]
    oth = np.delete(s, col, axis=1)
    fin = np.isfinite(s).all(axis=1)
    return ((oth < tgt[:, None]).all(axis=1) & fin).astype(np.int64)


def metrics4(sa, sb):
    """{d: s} per condition -> integer 4*r1, 4*other, and per-anchor float arrays."""
    r4 = np.zeros(E, np.int64)
    o4 = np.zeros(E, np.int64)
    sw = np.zeros(E, f64)
    st = np.zeros(E, f64)
    for d in DIRS:
        a, b = sa[d], sb[d]
        haa, hbb, hab, hba = hits(a, 0), hits(b, 1), hits(b, 0), hits(a, 1)
        r4 += haa + hbb
        o4 += hab + hba
        fin = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        sw += ((a[:, 0] > a[:, 1]) & (b[:, 1] > b[:, 0]) & fin).astype(f64)
        st += (haa * hbb).astype(f64)
    return r4, o4, sw / 2.0, st / 2.0


def combine(base, term, lu, la):
    s = base
    if lu > 0:
        s = s + f32(lu) * base
    if la > 0:
        s = s + f32(la) * term
    return s.astype(f32)


def control():
    """sigma* per tune half: smallest sigma with the largest integer rho of (1+sigma) z(B), as zB + sigma*zB."""
    rhos = []
    for sg in SIGMAS:
        s = {d: combine(zB[d], zB[d], sg, 0.0) for d in DIRS}
        r4, _, _, _ = metrics4(s, s)
        rhos.append(r4)
    out = {}
    for h in (0, 1):
        tune = parity == h
        vals = [int(r[tune].sum()) for r in rhos]
        best = max(vals)
        i = vals.index(best)
        out[h] = (SIGMAS[i], best)
    return out


CTRL = control()
log(f"control: {CTRL}")


def family(gates):
    gated = [{c: {d: (gates[t][c][:, None] * zT[c][d]).astype(f32) for d in DIRS} for c in CONDS} for t in range(4)]
    G = [{d: (0.5 * (gated[t]["a"][d].astype(f64) + gated[t]["b"][d].astype(f64))).astype(f32) for d in DIRS}
         for t in range(4)]
    fr = np.zeros((224, E), np.int64)
    fo = np.zeros((224, E), np.int64)
    cr = np.zeros((224, E), np.int64)
    co = np.zeros((224, E), np.int64)
    extra = {}
    for i, (t, u, a) in enumerate(CELLS):
        lu, la = U[u], A[a]
        sa = {d: combine(zB[d], gated[t]["a"][d], lu, la) for d in DIRS}
        sb = {d: combine(zB[d], gated[t]["b"][d], lu, la) for d in DIRS}
        r4, o4, sw, st = metrics4(sa, sb)
        fr[i], fo[i] = r4, o4
        extra[("f", i)] = (sw, st)
        sc = {d: combine(zB[d], G[t][d], lu, la) for d in DIRS}
        r4c, o4c, swc, stc = metrics4(sc, sc)
        cr[i], co[i] = r4c, o4c
        extra[("c", i)] = (swc, stc)
    fg = fr - fo
    # cross-fits
    fpick, cpick, crit_rec = {}, {}, {}
    for h in (0, 1):
        tune = parity == h
        rho = fr[:, tune].sum(axis=1)
        gam = fg[:, tune].sum(axis=1)
        crit = np.minimum(rho - CTRL[h][1], gam)
        fpick[h] = int(np.flatnonzero(crit == crit.max())[0])
        rc = cr[:, tune].sum(axis=1)
        cpick[h] = int(np.flatnonzero(rc == rc.max())[0])
        srt_c = np.sort(crit)[::-1]
        crit_rec[h] = {"best": int(crit.max()), "second": int(srt_c[1]), "rho": int(rho[fpick[h]]),
                       "gamma": int(gam[fpick[h]]), "rho_ctrl": CTRL[h][1],
                       "n_at_max": int((crit == crit.max()).sum()), "cf_best": int(rc.max()),
                       "cf_second": int(np.sort(rc)[::-1][1])}

    def assemble(r, o, kind, picks):
        cell = np.where(parity == 1, picks[0], picks[1])        # parity-1 episodes use the tune-half-0 pick
        idx = np.arange(E)
        r4, o4 = r[cell, idx], o[cell, idx]
        sw = np.empty(E)
        st = np.empty(E)
        for h in (0, 1):
            m = parity != h
            sw[m] = extra[(kind, picks[h])][0][m]
            st[m] = extra[(kind, picks[h])][1][m]
        return {"r1": r4 / 4.0, "gain": (r4 - o4) / 4.0, "other": o4 / 4.0, "swap": sw, "strict": st,
                "r4": r4, "g4": r4 - o4, "cell": cell}

    fused = assemble(fr, fo, "f", fpick)
    cf = assemble(cr, co, "c", cpick)
    insample = {"fused_best_r4": int(fr.sum(1).max()), "fused_best_cell": int(fr.sum(1).argmax()),
                "cf_best_r4": int(cr.sum(1).max()), "cf_best_cell": int(cr.sum(1).argmax())}
    return {"fpick": fpick, "cpick": cpick, "fused": fused, "cf": cf, "crit": crit_rec, "insample": insample,
            "fr_tot": fr.sum(1), "cr_tot": cr.sum(1)}


FAM = {}
for k in ("R1", "IMGABST", "AFF", "V4", "V2", "V24"):
    t1 = time.time()
    FAM[k] = family(GATES[k])
    log(f"{k}: fused cells {FAM[k]['fpick']}, cf cells {FAM[k]['cpick']} [{time.time() - t1:.0f}s]")

# ------------------------------------------------------------------ comparators and statistics


def pa_from_scores(sc):
    r4, o4, sw, st = metrics4(sc["a"], sc["b"])
    return {"r1": r4 / 4.0, "gain": (r4 - o4) / 4.0, "other": o4 / 4.0, "swap": sw, "strict": st, "r4": r4}


pB = pa_from_scores(bun.B)
pBp0 = pa_from_scores(bun.Bp["A0"])
pBp1 = pa_from_scores(bun.Bp["A1"])
for m in ("r1", "gain", "other", "swap", "strict"):
    assert np.array_equal(pB[m], np.asarray(bun.pB[m])), m
    assert np.array_equal(pBp0[m], np.asarray(bun.pBp["A0"][m])), m
    assert np.array_equal(pBp1[m], np.asarray(bun.pBp["A1"][m])), m


def pci(x, mask=None):
    x = np.asarray(x, f64)
    c = cl
    if mask is not None:
        x, c = x[mask], cl[mask]
    r = cluster_bootstrap(x, c)
    return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}


def bar_comp(name, cf_r1):
    cands = []
    if name in ("V2", "V24"):
        cands.append(("Bprime_A1", pBp1["r1"]))
    cands += [("Bprime_A0", pBp0["r1"]), ("counterpart", cf_r1), ("B", pB["r1"])]
    means = [float(np.mean(x)) for _, x in cands]
    best = 0
    for i in range(1, len(cands)):
        if means[i] > means[best]:
            best = i
    return cands[best][0], cands[best][1], {n: 100 * m for (n, _), m in zip(cands, means)}


RES = {"bundle": {"B_r1": 100 * float(np.mean(pB["r1"])), "Bp0_r1": 100 * float(np.mean(pBp0["r1"])),
                  "Bp1_r1": 100 * float(np.mean(pBp1["r1"])),
                  "int_sums": {"B": int(pB["r4"].sum()), "Bp0": int(pBp0["r4"].sum()), "Bp1": int(pBp1["r4"].sum())}},
       "checks": chk, "control": {str(h): list(CTRL[h]) for h in (0, 1)}, "open_tau0": open0, "scorers": {}}

for k in ("R1", "IMGABST", "AFF", "V4", "V2", "V24"):
    fm = FAM[k]
    fu, cf = fm["fused"], fm["cf"]
    assert np.all(cf["gain"] == 0), f"{k}: counterpart gain not 0"
    if k in ("R1", "IMGABST"):
        cname, cr1, means = "counterpart", cf["r1"], None
        # round-1 / brainstorm comparator: max of B', counterpart, B with order B', counterpart, B
        cname, cr1, means = bar_comp("AFF", cf["r1"])
    else:
        cname, cr1, means = bar_comp(k, cf["r1"])
    bar = pci(fu["r1"] - cr1)
    rec = {"fused_r1": 100 * float(np.mean(fu["r1"])), "cf_r1": 100 * float(np.mean(cf["r1"])),
           "fused_r4": int(fu["r4"].sum()), "cf_r4": int(cf["r4"].sum()),
           "fpick": fm["fpick"], "cpick": fm["cpick"], "crit": fm["crit"], "insample": fm["insample"],
           "bar_comparator": cname, "comparator_means": means, "bar_margin": bar,
           "margin_vs_counterpart": pci(fu["r1"] - cf["r1"]),
           "gain_statistic": pci(fu["gain"] - cf["gain"]),
           "either_change": 100 * (float(np.mean(fu["r1"] + fu["other"])) - float(np.mean(cf["r1"] + cf["other"]))),
           "per_pair_bar_margin": {p: pci(fu["r1"] - cr1, pidx == i) for i, p in enumerate(PAIRS)}}
    rec["d10"] = {"c1": rec["bar_margin"]["point"] >= 0.5, "c2": rec["bar_margin"]["ci95"][0] > 0,
                  "c3": rec["gain_statistic"]["ci95"][0] > 0}
    rec["d10"]["clears"] = all(rec["d10"].values())
    if k in ("V4", "V2", "V24"):
        dk = int(fu["r4"].sum() - FAM["AFF"]["fused"]["r4"].sum())
        dd = fu["r4"] - FAM["AFF"]["fused"]["r4"]
        assert dk == int(dd.sum())
        rec["delta_int"] = dk
        rec["delta"] = pci(fu["r1"] - FAM["AFF"]["fused"]["r1"])
        rec["delta_point_from_int"] = 100 * dk / (4 * E)
        rec["better_worse"] = [int((dd > 0).sum()), int((dd < 0).sum())]
    RES["scorers"][k] = rec
    log(f"{k}: fused {rec['fused_r1']:.6f} cf {rec['cf_r1']:.6f} bar[{cname}] {bar['point']:+.6f} "
        f"[{bar['ci95'][0]:+.6f}, {bar['ci95'][1]:+.6f}] gain {rec['gain_statistic']['point']:.6f} d10 {rec['d10']}")

aff_r1 = FAM["AFF"]["fused"]["r1"]
RES["beside_aff"] = {"AFF_minus_Bprime_A1": pci(aff_r1 - pBp1["r1"]),
                     "Bprime_A1_minus_Bprime_A0": pci(pBp1["r1"] - pBp0["r1"]),
                     "Bprime_A1_minus_Bprime_A0_per_pair": {p: pci(pBp1["r1"] - pBp0["r1"], pidx == i)
                                                            for i, p in enumerate(PAIRS)},
                     "AFF_minus_Bprime_A1_per_pair": {p: pci(aff_r1 - pBp1["r1"], pidx == i)
                                                      for i, p in enumerate(PAIRS)},
                     "AFF_minus_R1": pci(aff_r1 - FAM["R1"]["fused"]["r1"])}

# carry
E_set = [k for k in ("V4", "V2", "V24") if RES["scorers"][k]["d10"]["clears"] and RES["scorers"][k]["delta_int"] > 0]
M = max([RES["scorers"][k]["delta_int"] for k in E_set], default=None)
tied = [k for k in E_set if M - RES["scorers"][k]["delta_int"] <= 24]
RES["carry"] = {"E": E_set, "M": M, "tied": tied, "carried": tied[0] if tied else None, "kill": not E_set}
log(f"carry: {RES['carry']}")

# ------------------------------------------------------------------ save arrays
arrs = {"v": v, "keep": keep, "pick0_a": pick0["a"], "pick0_b": pick0["b"], "pick1_a": pick1["a"],
        "pick1_b": pick1["b"], "marg_a": marg["a"], "marg_b": marg["b"], "cl": cl, "pair_index": pidx,
        "parity": parity}
for k in GATES:
    for c in CONDS:
        arrs[f"{k}_gate_{c}"] = np.stack([GATES[k][t][c] for t in range(4)])
    for part in ("fused", "cf"):
        for m in ("r1", "gain", "other", "swap", "strict"):
            arrs[f"{k}_{part}_{m}"] = FAM[k][part][m]
        arrs[f"{k}_{part}_cell"] = FAM[k][part]["cell"]
    arrs[f"{k}_fr_tot"] = FAM[k]["fr_tot"]
    arrs[f"{k}_cr_tot"] = FAM[k]["cr_tot"]
for nm, pa in (("B", pB), ("Bp0", pBp0), ("Bp1", pBp1)):
    for m in ("r1", "gain", "other", "swap", "strict"):
        arrs[f"{nm}_{m}"] = pa[m]
np.savez(OUT / "fr_arrays.npz", **arrs)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


(OUT / "fr_results.json").write_text(json.dumps(jsonable(RES), indent=1))
log(f"written: fr_results.json sha {sha(OUT / 'fr_results.json')}, fr_arrays.npz sha {sha(OUT / 'fr_arrays.npz')}")
log(f"total {time.time() - t0:.0f}s")
