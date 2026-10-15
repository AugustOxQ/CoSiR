"""Whole-branch final review of round 5 (idea 3): a third derivation of the seed-42 decision, in my own code.

Imports only what DECISION_RULE.md §8 lets a re-derivation import: round 1's common.load_bundle (seed 42: episodes,
parity, clusters, cosine, the CLIP heads through run_told_oracle.fit_one_head, the image and caption posteriors, B,
B'(A0), B'(A1), the method-A term), rb_build.load_readers("A0"), zscore_rows, crossfit_condition_free,
uniform_probe_scores, cluster_bootstrap, and the data modules (src.data.artelingo_splits for the aspect labels).
Nothing of r5_*, rederive/ (rd5_*), r2_*/r3_*/r4_*, rc_core, rb_eval, rb_features, round 3/4 re-derivations or the
brainstorm is imported. scikit-learn's LogisticRegression and roc_auc_score are called directly.

Own code: the GE head (draw, check rows, labels, fit, fallback rule, scatter, accuracy), the 18 features, P, T,
margins, picks, tau and tau', all gates, z-scored terms, G_cf, the 224 cells' integer statistics, sigma*, both
cross-fits, the parity assembly, per-anchor metrics, comparators and bar comparator, margins, gain statistics, either
changes, per-pair bar margins, the D10 clauses, Delta_k and the carry; the measured diagnostics (a) to (d); and the
report's section-6 numbers (pair lifts, sharpness, redundancy, affect score alone, same-painting agreement, B' weights,
per pair-condition net rankings, fixed-cell decomposition, in-sample family, G-TF's reader against AFF's).

CPU only. Writes only to final_review/out/.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ROOT = Path("/project/CoSiR")
TST = ROOT / "src/test"
R1DIR = TST / "20261117_reader_fix_csd"
HERE = TST / "20261123_idea3_goemotions"
OUT = HERE / "final_review/out"
OUT.mkdir(parents=True, exist_ok=True)
for p in (str(ROOT), str(R1DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import common as C1  # noqa: E402  (round 1: load_bundle only)
import rb_build  # noqa: E402     (load_readers only)
from src.data.artelingo_splits import artelingo_aspect_labels  # noqa: E402
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

assert Path(C1.__file__).resolve().parent == R1DIR, C1.__file__
assert Path(rb_build.__file__).resolve().parent == R1DIR, rb_build.__file__
for name in list(sys.modules):
    # rb_build imports rb_features itself (transitive); my code calls only load_readers
    assert not name.startswith(("r5_", "rd5_", "rd3_", "rd4_", "r4_", "r3_", "r2_", "rb_eval", "bs_", "run_r")), \
        f"forbidden module imported: {name}"

f32, f64 = np.float32, np.float64
CONDS, DIRS = ("a", "b"), ("i2t", "t2i")
A0 = ("affect", "image", "caption")
U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CELLS = [(t, u, a) for t in range(4) for u in range(7) for a in range(8)]
SIGMAS = sorted({x + y for x in U for y in A})
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
RULE_SHA = "19e59fc7220c05b630f4773a94578aa3858d29853dbf7d438455e1ee973d735e"
GOEMO_SHA = "f8372a89a808421772e19cce9413dbab9b83dcef5cfd28da8232d77d729e18a8"     # cache json + run log 17:48
GEPOST_SHA = "081bb19e2a9b23cc5d97612f83950f77e2c9a503919e48d42fb72d94bbd7fbf7"    # placement.json + run log 17:50
AFFPREP_SHA = "e25d2dcadadc23b33ac94659fb44c13e220110e64ce463def07b04343bfc4f2e"
TOLD_NPZ_SHA = "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366"
TOLD_JSON_SHA = "76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2"
DRAW_SHA = "7be956c09bf716547df20264435388bd3645ae963df728636774e80359cdef5c"
TAUS_RULE = (3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211)
assert len(SIGMAS) == 30
for i, (t, u, a) in enumerate(CELLS):
    assert i == (t * 7 + u) * 8 + a


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


assert sha(HERE / "DECISION_RULE.md") == RULE_SHA
assert sha(HERE / "cache/r5_goemotions_selection.npz") == GOEMO_SHA
assert sha(HERE / "cache/r5_ge_posterior.npz") == GEPOST_SHA
assert sha(TST / "20261018_affect_factor_learning/cache/affect_prepare.npz") == AFFPREP_SHA
assert sha(TST / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz") == TOLD_NPZ_SHA
assert sha(TST / "20261111_community_told_oracle/results/told_oracle.json") == TOLD_JSON_SHA

# ------------------------------------------------------------------ round 1's bundle (seed 42)
bun = C1.load_bundle()
log("round-1 bundle loaded")
ctx, post = bun.ctx, bun.post
ep = ctx.pooled
E = len(ep.anchor)
assert E == 12288
par = np.asarray(ctx.parity)
assert np.array_equal(par, np.arange(E) % 2)
cl = np.asarray(bun.cl)
pidx = np.asarray(ctx.pair_index)
assert np.array_equal(cl, np.asarray(ctx.groups)[ep.anchor])
N_ALL = len(ctx.groups)
assert N_ALL == 308_723
sel = np.asarray(ctx.selection)
st_rows = np.asarray(bun.scorer_train)


def sets(c):
    if c == "a":
        return ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt
    return ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt


def features(pst, c):
    """18 columns: per grouping in A0 order S, C, Delta (float32 means), sd_S, sd_C (ddof 1 in float64 of the float32
    agreements), arg-max match share of the support pairs."""
    si, st, ci, ct = sets(c)
    cols = []
    for h in A0:
        pi, pt = pst[h]["img"], pst[h]["txt"]
        sup = np.einsum("nsc,nsc->ns", pi[si], pt[st])
        con = np.einsum("nsc,nsc->ns", pi[ci], pt[ct])
        S, Cc = sup.mean(axis=1), con.mean(axis=1)
        mt = (pi[si].argmax(axis=-1) == pt[st].argmax(axis=-1)).mean(axis=1)
        cols += [S, Cc, S - Cc, sup.astype(f64).std(axis=1, ddof=1), con.astype(f64).std(axis=1, ddof=1), mt]
    return np.stack([np.asarray(x, f64) for x in cols], axis=1)


PK = rb_build.load_readers("A0", False)[0]


def reader(F):
    out = {}
    for c in CONDS:
        ps = []
        for hlf in PK["halves"]:
            assert np.array_equal(hlf["model"].classes_, np.arange(3))
            ps.append(np.asarray(hlf["model"].predict_proba(hlf["scaler"].transform(F[c])), f64))
        out[c] = np.stack(ps).mean(axis=0)
    pick = {c: out[c].argmax(axis=1) for c in CONDS}
    s = {c: np.sort(out[c], axis=1) for c in CONDS}
    marg = {c: s[c][:, -1] - s[c][:, -2] for c in CONDS}
    return out, pick, marg


def stack_of(pst):
    o = {}
    for d in DIRS:
        q, k = ("img", "txt") if d == "i2t" else ("txt", "img")
        o[d] = np.stack([np.einsum("nc,nkc->nk", pst[h][q][ep.anchor], pst[h][k][ep.candidates]) for h in A0], axis=1)
    return o


def term(P, STK):
    return {c: {d: np.einsum("nh,nhk->nk", P[c], STK[d].astype(f64)).astype(f32) for d in DIRS} for c in CONDS}


def zs(x):
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy().astype(f32)


def gates(marg, pick, taus, affect_only):
    g = []
    for t in range(4):
        g.append({c: ((marg[c] >= taus[t]) & ((pick[c] == 0) if affect_only else True)).astype(f32) for c in CONDS})
    return g


# CLIP placement: features, reader, stack
F = {c: features(post, c) for c in CONDS}
P, pick, marg = reader(F)
STK = stack_of(post)
T_aff = term(P, STK)

# round-1 targets
z_rc = np.load(R1DIR / "results/cand_Rc_Rb_expected_A0.npz")
taus_file = tuple(json.loads((R1DIR / "results/rc_tau.json").read_text())["taus"])
CHK = {}
for c in CONDS:
    CHK[f"R1_pick_{c}"] = bool(np.array_equal(pick[c], z_rc[f"pick__{c}"].astype(np.int64)))
    CHK[f"R1_margin_{c}"] = bool(np.array_equal(marg[c], z_rc[f"margin__{c}"]))
    for d in DIRS:
        CHK[f"R1_T_{c}_{d}"] = bool(np.array_equal(T_aff[c][d], z_rc[f"T__{c}__{d}"]))
tau_re = tuple(float(x) for x in np.percentile(np.concatenate([marg["a"], marg["b"]]).astype(f64), [0, 25, 50, 75]))
CHK["tau_recomputed_equals_file"] = tau_re == taus_file
CHK["tau_file_equals_rule"] = taus_file == TAUS_RULE
log(f"R1 reader checks: {all(CHK.values())}")
assert all(CHK.values()), CHK

# ------------------------------------------------------------------ family machinery
Bsc = bun.B
for d in DIRS:
    assert np.array_equal(Bsc["a"][d], Bsc["b"][d])
zB = {d: zs(Bsc["a"][d]) for d in DIRS}


def hits(s, col):
    s = np.asarray(s, f32)
    tgt = s[:, col]
    oth = np.delete(s, col, axis=1)
    return ((oth < tgt[:, None]).all(axis=1) & np.isfinite(s).all(axis=1)).astype(np.int64)


def combine(base, x, lu, la):
    s = base
    if lu > 0:
        s = s + f32(lu) * base
    if la > 0:
        s = s + f32(la) * x
    return np.asarray(s, f32)


def ints4(sa, sb):
    """per episode: 4*R@1 and 4*other (integers) of the scores {d: (E,13)} under conditions a and b."""
    r4 = np.zeros(E, np.int64)
    o4 = np.zeros(E, np.int64)
    for d in DIRS:
        r4 += hits(sa[d], 0) + hits(sb[d], 1)
        o4 += hits(sb[d], 0) + hits(sa[d], 1)
    return r4, o4


def per_anchor_own(sa, sb):
    """r1, gain, other, swap, strict per episode (averaged over the two directions), float64."""
    r1 = np.zeros(E)
    ot = np.zeros(E)
    sw = np.zeros(E)
    stc = np.zeros(E)
    for d in DIRS:
        a, b = np.asarray(sa[d], f32), np.asarray(sb[d], f32)
        haa, hbb, hab, hba = hits(a, 0), hits(b, 1), hits(b, 0), hits(a, 1)
        fin = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        r1 += 0.5 * (haa + hbb)
        ot += 0.5 * (hab + hba)
        sw += ((a[:, 0] > a[:, 1]) & (b[:, 1] > b[:, 0]) & fin).astype(f64)
        stc += (haa * hbb).astype(f64)
    return {"r1": r1 / 2.0, "gain": (r1 - ot) / 2.0, "other": ot / 2.0, "swap": sw / 2.0, "strict": stc / 2.0}


def control():
    rh = []
    for sg in SIGMAS:
        s = {d: combine(zB[d], zB[d], sg, 0.0) for d in DIRS}
        rh.append(ints4(s, s)[0])
    out = {}
    for h in (0, 1):
        vals = [int(r[par == h].sum()) for r in rh]
        i = vals.index(max(vals))
        out[h] = (SIGMAS[i], vals[i])
    return out


CTRL = control()
log(f"sigma*: {CTRL}")


def family(T, G):
    zT = {c: {d: zs(T[c][d]) for d in DIRS} for c in CONDS}
    gated = [{c: {d: (G[t][c][:, None] * zT[c][d]).astype(f32) for d in DIRS} for c in CONDS} for t in range(4)]
    Gcf = [{d: (0.5 * (gated[t]["a"][d].astype(f64) + gated[t]["b"][d].astype(f64))).astype(f32) for d in DIRS}
           for t in range(4)]
    fr = np.zeros((224, E), np.int64)
    fo = np.zeros((224, E), np.int64)
    cr = np.zeros((224, E), np.int64)
    co = np.zeros((224, E), np.int64)
    for i, (t, u, a) in enumerate(CELLS):
        sa = {d: combine(zB[d], gated[t]["a"][d], U[u], A[a]) for d in DIRS}
        sb = {d: combine(zB[d], gated[t]["b"][d], U[u], A[a]) for d in DIRS}
        fr[i], fo[i] = ints4(sa, sb)
        sc = {d: combine(zB[d], Gcf[t][d], U[u], A[a]) for d in DIRS}
        cr[i], co[i] = ints4(sc, sc)
    fpick, cpick, info = {}, {}, {}
    for h in (0, 1):
        tune = par == h
        rho = fr[:, tune].sum(1)
        gam = (fr - fo)[:, tune].sum(1)
        crit = np.minimum(rho - CTRL[h][1], gam)
        fpick[h] = int(np.flatnonzero(crit == crit.max())[0])
        rc_ = cr[:, tune].sum(1)
        cpick[h] = int(np.flatnonzero(rc_ == rc_.max())[0])
        sc_ = np.sort(crit)[::-1]
        sr_ = np.sort(rc_)[::-1]
        info[h] = {"crit_best": int(sc_[0]), "crit_second": int(sc_[1]), "cf_best": int(sr_[0]),
                   "cf_second": int(sr_[1]), "crit_ties_at_max": int((crit == crit.max()).sum()),
                   "cf_ties_at_max": int((rc_ == rc_.max()).sum())}

    def assemble(kind, picks):
        cell = {h: CELLS[picks[h]] for h in (0, 1)}
        out_a, out_b = {}, {}
        for d in DIRS:
            xa = np.empty((E, 13), f32)
            xb = np.empty((E, 13), f32)
            for h in (0, 1):
                t, u, a = cell[h]
                m = par != h            # the tune-half-h cell scores parity 1-h
                if kind == "fused":
                    xa[m] = combine(zB[d], gated[t]["a"][d], U[u], A[a])[m]
                    xb[m] = combine(zB[d], gated[t]["b"][d], U[u], A[a])[m]
                else:
                    sc = combine(zB[d], Gcf[t][d], U[u], A[a])
                    xa[m], xb[m] = sc[m], sc[m]
            out_a[d], out_b[d] = xa, xb
        return out_a, out_b

    fsa, fsb = assemble("fused", fpick)
    csa, csb = assemble("cf", cpick)
    pf, pc = per_anchor_own(fsa, fsb), per_anchor_own(csa, csb)
    # assembled integers agree with the cell statistics
    cellidx = np.where(par == 1, fpick[0], fpick[1])
    assert np.array_equal(fr[cellidx, np.arange(E)], np.rint(4 * pf["r1"]).astype(np.int64))
    cellidx_c = np.where(par == 1, cpick[0], cpick[1])
    assert np.array_equal(cr[cellidx_c, np.arange(E)], np.rint(4 * pc["r1"]).astype(np.int64))
    assert np.all(pc["gain"] == 0)
    return {"fpick": fpick, "cpick": cpick, "fused": pf, "cf": pc, "info": info, "fr": fr, "fo": fo, "cr": cr,
            "zT": zT, "gated": gated, "Gcf": Gcf, "scores": {"fused": (fsa, fsb), "cf": (csa, csb)}}


def pci(x, mask=None):
    x = np.asarray(x, f64)
    c = cl
    if mask is not None:
        x, c = x[mask], cl[mask]
    r = cluster_bootstrap(x, c)
    return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}


def pa_from(sc):
    return per_anchor_own(sc["a"], sc["b"])


pB = pa_from(bun.B)
pBp0 = pa_from(bun.Bp["A0"])
pBp1 = pa_from(bun.Bp["A1"])
for m in ("r1", "gain", "other", "swap", "strict"):
    assert np.array_equal(pB[m], np.asarray(bun.pB[m], f64)), m
    assert np.array_equal(pBp0[m], np.asarray(bun.pBp["A0"][m], f64)), m
    assert np.array_equal(pBp1[m], np.asarray(bun.pBp["A1"][m], f64)), m

# ------------------------------------------------------------------ R1 and AFF
G_R1 = gates(marg, pick, TAUS_RULE, False)
G_AFF = gates(marg, pick, TAUS_RULE, True)
FAM = {}
for k, G in (("R1", G_R1), ("AFF", G_AFF)):
    FAM[k] = family(T_aff, G)
    log(f"{k}: fused {FAM[k]['fpick']} cf {FAM[k]['cpick']}")


def bar_choice(cands):
    means = [float(np.mean(x)) for _, x in cands]
    b = 0
    for i in range(1, len(cands)):
        if means[i] > means[b]:
            b = i
    return cands[b][0], cands[b][1], {n: 100 * m for (n, _), m in zip(cands, means)}


def record(k, cands):
    fm = FAM[k]
    fu, cf = fm["fused"], fm["cf"]
    cname, cr1, means = bar_choice(cands)
    rec = {"fused_r1": 100 * float(np.mean(fu["r1"])), "cf_r1": 100 * float(np.mean(cf["r1"])),
           "fused_int": int(np.rint(4 * fu["r1"]).sum()), "cf_int": int(np.rint(4 * cf["r1"]).sum()),
           "fpick": fm["fpick"], "cpick": fm["cpick"], "sigma": {h: CTRL[h][0] for h in (0, 1)}, "info": fm["info"],
           "bar_comparator": cname, "comparator_means": means, "bar_margin": pci(fu["r1"] - cr1),
           "margin_vs_counterpart": pci(fu["r1"] - cf["r1"]), "gain_statistic": pci(fu["gain"] - cf["gain"]),
           "either_change": 100 * float(np.mean((fu["r1"] + fu["other"]) - (cf["r1"] + cf["other"]))),
           "per_pair_bar_margin": {p: pci(fu["r1"] - cr1, pidx == i) for i, p in enumerate(PAIRS)}}
    rec["d10"] = {"c1": bool(rec["bar_margin"]["point"] >= 0.5), "c2": bool(rec["bar_margin"]["ci95"][0] > 0),
                  "c3": bool(rec["gain_statistic"]["ci95"][0] > 0)}
    rec["d10"]["clears"] = all(rec["d10"].values())
    return rec


RES = {"checks": CHK, "sigma_control": {str(h): list(CTRL[h]) for h in (0, 1)}, "scorers": {}}
RES["scorers"]["R1"] = record("R1", [("B_prime", pBp0["r1"]), ("counterpart", FAM["R1"]["cf"]["r1"]),
                                     ("B", pB["r1"])])
RES["scorers"]["AFF"] = record("AFF", [("Bprime_A0", pBp0["r1"]), ("counterpart", FAM["AFF"]["cf"]["r1"]),
                                       ("B", pB["r1"])])
aff_f = FAM["AFF"]["fused"]
r1_f = FAM["R1"]["fused"]
RES["aff_extra"] = {
    "AFF_minus_R1_fused": pci(aff_f["r1"] - r1_f["r1"]),
    "AFF_minus_R1_bar": pci((aff_f["r1"] - pBp0["r1"]) - (r1_f["r1"] - FAM["R1"]["cf"]["r1"])),
    "AFF_minus_Bprime_A1": pci(aff_f["r1"] - pBp1["r1"]),
    "open_tau0": {c: int(G_AFF[0][c].sum()) for c in CONDS},
    "B_r1": 100 * float(np.mean(pB["r1"])), "Bp0_r1": 100 * float(np.mean(pBp0["r1"])),
    "Bp1_r1": 100 * float(np.mean(pBp1["r1"]))}
log(f"AFF: {RES['scorers']['AFF']['fused_r1']} bar {RES['scorers']['AFF']['bar_margin']}")

# ------------------------------------------------------------------ the GE head (own code, D4)
told = np.load(TST / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz")
partL = np.asarray(told["partition_L"], np.int64)
assert np.array_equal(partL, np.asarray(bun.setup.partition_L, np.int64))
assert len(partL) == len(st_rows) == 183_694
assert np.all(np.diff(st_rows) > 0)
lab = np.full(N_ALL, -1, np.int64)
lab[st_rows] = partL
aff = np.load(TST / "20261018_affect_factor_learning/cache/affect_prepare.npz")
aprobs = np.asarray(aff["affect_probs"])
assert aprobs.shape == (183_694, 28), aprobs.shape
goe = np.load(HERE / "cache/r5_goemotions_selection.npz")
g_rows, g_probs, g_sids = np.asarray(goe["rows"]), np.asarray(goe["probs"]), np.asarray(goe["sample_ids"])
CHK["goemo_rows_equal_selection"] = bool(np.array_equal(g_rows, sel))
CHK["goemo_sample_ids"] = bool(np.array_equal(g_sids, np.asarray(ctx.data.sample_ids)[sel]))
CHK["goemo_probs_f32_finite_unit"] = bool(g_probs.dtype == f32 and np.isfinite(g_probs).all()
                                          and (g_probs >= 0).all() and (g_probs <= 1).all())
CHK["selection_disjoint_scorer_train"] = bool(len(np.intersect1d(sel, st_rows)) == 0)
X = np.full((N_ALL, 28), np.nan, f32)
X[st_rows] = aprobs
X[sel] = g_probs
draw = np.random.default_rng(0).choice(st_rows, 60_000, replace=False)
rest = np.setdiff1d(st_rows, draw)
check = np.random.default_rng(1).choice(rest, 10_000, replace=False)
CHK["draw_sha"] = sha_arr(np.sort(draw)) == DRAW_SHA
CHK["draw_check_finite"] = bool(np.isfinite(X[draw]).all() and np.isfinite(X[check]).all())
clf = LogisticRegression(C=1.0, max_iter=300).fit(X[draw], lab[draw])
n_iter_first = int(clf.n_iter_[0])
fallback = n_iter_first >= 300
if fallback:
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(X[draw], lab[draw])
CHK["ge_classes"] = bool(np.array_equal(clf.classes_, np.arange(41)))
QGE = np.full((N_ALL, 41), np.nan, f32)
QGE[sel] = clf.predict_proba(X[sel])
ge_acc = 100 * float(clf.score(X[check], lab[check]))
cnt = np.bincount(lab[check])
ge_info = {"n_iter_first": n_iter_first, "fallback": bool(fallback), "n_iter": int(clf.n_iter_[0]),
           "heldout_accuracy": ge_acc, "check_majority_share": 100 * float(cnt.max() / cnt.sum()),
           "rows_sum_max_dev": float(np.abs(QGE[sel].astype(f64).sum(1) - 1).max())}
gp = np.load(HERE / "cache/r5_ge_posterior.npz")
CHK["Q_GE_equals_cache"] = bool(np.array_equal(QGE[sel], gp["post_sel"]) and np.array_equal(gp["rows"], sel)
                                and np.array_equal(gp["classes"], np.arange(41)))
CHK["Q_GE_nan_outside"] = bool(np.isnan(QGE[np.setdiff1d(np.arange(N_ALL), sel)]).all())
RES["ge_head"] = ge_info
log(f"GE head: {ge_info}; equals cache {CHK['Q_GE_equals_cache']}")
assert all(CHK.values()), {k: v for k, v in CHK.items() if not v}

# ------------------------------------------------------------------ GE extension (D5)
post_G = {"affect": {"img": post["affect"]["img"], "txt": QGE}, "image": post["image"], "caption": post["caption"]}
STK_G = stack_of(post_G)
for d in DIRS:
    CHK[f"stackG_img_cap_slices_equal_{d}"] = bool(np.array_equal(STK_G[d][:, 1:], STK[d][:, 1:]))
    CHK[f"stackG_affect_differs_{d}"] = bool(not np.array_equal(STK_G[d][:, 0], STK[d][:, 0]))
F_G = {c: features(post_G, c) for c in CONDS}
for c in CONDS:
    CHK[f"FG_cols6_17_equal_{c}"] = bool(np.array_equal(F_G[c][:, 6:], F[c][:, 6:]))
CHK["FG_delta_b_neg_a"] = bool(np.array_equal(F_G["b"][:, 2], -F_G["a"][:, 2]))
Bp_clip, Bp_clip_picks = crossfit_condition_free(ctx.cos, bun.t_n1u, uniform_probe_scores(post, ep, A0), par)
pBp_clip = pa_from(Bp_clip)
CHK["Bprime_recipe_reproduces_Bp_A0"] = all(np.array_equal(pBp_clip[m], pBp0[m]) for m in pBp0)
Bp_G, Bp_G_picks = crossfit_condition_free(ctx.cos, bun.t_n1u, uniform_probe_scores(post_G, ep, A0), par)
pBpG = pa_from(Bp_G)
CHK["BpG_condition_free"] = bool(all(np.array_equal(Bp_G["a"][d], Bp_G["b"][d]) for d in DIRS)
                                 and np.all(pBpG["gain"] == 0))
log(f"B'_G {100 * np.mean(pBpG['r1'])}; picks A0 {Bp_clip_picks} G {Bp_G_picks}")
assert all(CHK.values()), {k: v for k, v in CHK.items() if not v}

# ------------------------------------------------------------------ candidates (D6)
T_GT = term(P, STK_G)
G_GT = gates(marg, pick, TAUS_RULE, True)
P2, pick2, marg2 = reader(F_G)
allm = np.concatenate([marg2["a"], marg2["b"]]).astype(f64)
assert len(allm) == 24_576
tau_p = tuple(float(x) for x in np.percentile(allm, [0, 25, 50, 75]))
T_GTF = term(P2, STK_G)
G_GTF = gates(marg2, pick2, tau_p, True)
for k, T, G in (("G-T", T_GT, G_GT), ("G-TF", T_GTF, G_GTF)):
    FAM[k] = family(T, G)
    log(f"{k}: fused {FAM[k]['fpick']} cf {FAM[k]['cpick']}")
for k in ("G-T", "G-TF"):
    RES["scorers"][k] = record(k, [("Bprime_G", pBpG["r1"]), ("Bprime_A0", pBp0["r1"]),
                                   ("counterpart", FAM[k]["cf"]["r1"]), ("B", pB["r1"])])
    fu = FAM[k]["fused"]
    d4 = np.rint(4 * fu["r1"]).astype(np.int64) - np.rint(4 * aff_f["r1"]).astype(np.int64)
    assert np.array_equal(d4, 4 * (fu["r1"] - aff_f["r1"]))
    RES["scorers"][k]["delta_int"] = int(d4.sum())
    RES["scorers"][k]["delta"] = pci(fu["r1"] - aff_f["r1"])
    RES["scorers"][k]["delta_point_from_int"] = 100 * int(d4.sum()) / (4 * E)
    RES["scorers"][k]["better_worse"] = [int((d4 > 0).sum()), int((d4 < 0).sum())]
    RES["scorers"][k]["minus_Bprime_A1"] = pci(fu["r1"] - pBp1["r1"])
    RES["scorers"][k]["open_tau0"] = {c: int((G_GT if k == "G-T" else G_GTF)[0][c].sum()) for c in CONDS}
    RES["scorers"][k]["per_pair_minus_AFF"] = {p: pci(fu["r1"] - aff_f["r1"], pidx == i) for i, p in enumerate(PAIRS)}
RES["tau_prime"] = list(tau_p)
RES["BpG"] = {"mean_r1": 100 * float(np.mean(pBpG["r1"])), "minus_Bp0": pci(pBpG["r1"] - pBp0["r1"]),
              "minus_B": pci(pBpG["r1"] - pB["r1"]),
              "net_vs_Bp0": int((np.rint(4 * pBpG["r1"]) - np.rint(4 * pBp0["r1"])).sum()),
              "per_pair_minus_Bp0": {p: pci(pBpG["r1"] - pBp0["r1"], pidx == i) for i, p in enumerate(PAIRS)},
              "picks": {str(h): list(v) for h, v in Bp_G_picks.items()},
              "picks_Bp0": {str(h): list(v) for h, v in Bp_clip_picks.items()}}
RES["int_sums"] = {"AFF": int(np.rint(4 * aff_f["r1"]).sum()), "G-T": int(np.rint(4 * FAM["G-T"]["fused"]["r1"]).sum()),
                   "G-TF": int(np.rint(4 * FAM["G-TF"]["fused"]["r1"]).sum()),
                   "Bp1": int(np.rint(4 * pBp1["r1"]).sum()), "Bp0": int(np.rint(4 * pBp0["r1"]).sum()),
                   "BpG": int(np.rint(4 * pBpG["r1"]).sum()), "B": int(np.rint(4 * pB["r1"]).sum())}
Eset = [k for k in ("G-T", "G-TF") if RES["scorers"][k]["d10"]["clears"] and RES["scorers"][k]["delta_int"] > 0]
M = max([RES["scorers"][k]["delta_int"] for k in Eset], default=None)
tied = [k for k in Eset if M - RES["scorers"][k]["delta_int"] <= 24]
RES["carry"] = {"E": Eset, "M": M, "tied": tied, "carried": tied[0] if tied else None, "kill": not Eset}
log(f"carry {RES['carry']}")
for k in ("G-T", "G-TF"):
    r = RES["scorers"][k]
    log(f"{k}: fused {r['fused_r1']} bar[{r['bar_comparator']}] {r['bar_margin']} gain {r['gain_statistic']} "
        f"d10 {r['d10']} delta {r['delta_int']}")

# ------------------------------------------------------------------ measured diagnostics (a) to (d)
emo_pos = np.concatenate([(pidx <= 1), np.zeros(E, bool)])            # condition a of the two emotion pairs
DIAG = {"a_detection_auc": {"AFF": float(roc_auc_score(emo_pos, np.concatenate([P["a"][:, 0], P["b"][:, 0]]))),
                            "G-TF": float(roc_auc_score(emo_pos, np.concatenate([P2["a"][:, 0], P2["b"][:, 0]])))},
        "b_delta_affect_auc": {"F": float(roc_auc_score(emo_pos, np.concatenate([F["a"][:, 2], F["b"][:, 2]]))),
                               "F_G": float(roc_auc_score(emo_pos, np.concatenate([F_G["a"][:, 2], F_G["b"][:, 2]])))}}


# pair lifts through the heads (own implementation of the different-painting ordered-pair means)
def keyinv(keys, n):
    if not keys:
        return np.zeros(n, np.int64)
    return np.unique(np.stack(keys, 1), axis=0, return_inverse=True)[1].ravel()


def ssum(keys, Pi, Pt):
    inv = keyinv(keys, len(Pi))
    k = int(inv.max()) + 1
    si = np.zeros((k, Pi.shape[1]))
    st = np.zeros((k, Pt.shape[1]))
    np.add.at(si, inv, Pi)
    np.add.at(st, inv, Pt)
    c = np.bincount(inv, minlength=k).astype(f64)
    return float((si * st).sum()), float((c ** 2).sum())


def dp(keys, Pi, Pt, g):
    a = ssum(keys, Pi, Pt)
    b = ssum(keys + [g], Pi, Pt)
    return a[0] - b[0], a[1] - b[1]


labels = artelingo_aspect_labels(ctx.data)
labS = {a_: np.asarray(labels[a_])[sel] for a_ in ("emotion", "style", "genre")}
gS = np.asarray(ctx.groups)[sel]


def lift(Pi, Pt):
    m = labS["emotion"] >= 0
    x, g, pi_, pt_ = labS["emotion"][m], gS[m], Pi[m], Pt[m]
    s_all, n_all = dp([], pi_, pt_, g)
    s_same, n_same = dp([x], pi_, pt_, g)
    out = {"ratio_same_over_diff": (s_same / n_same) / ((s_all - s_same) / (n_all - n_same))}
    for A_, B_ in (("emotion", "style"), ("emotion", "genre")):
        m = (labS[A_] >= 0) & (labS[B_] >= 0)
        xa, xb, g, pi_, pt_ = labS[A_][m], labS[B_][m], gS[m], Pi[m], Pt[m]
        sab, nab = dp([xa, xb], pi_, pt_, g)
        sa, na = dp([xa], pi_, pt_, g)
        sb, nb = dp([xb], pi_, pt_, g)
        out[f"{A_}x{B_}"] = ((sa - sab) / (na - nab)) / ((sb - sab) / (nb - nab))
    return out


Pi_s = np.asarray(post["affect"]["img"])[sel].astype(f64)
Qc_s = np.asarray(post["affect"]["txt"])[sel].astype(f64)
Qg_s = QGE[sel].astype(f64)
LIFT = {"image_x_caption_CLIP": lift(Pi_s, Qc_s), "image_x_caption_GE": lift(Pi_s, Qg_s),
        "caption_x_caption_CLIP": lift(Qc_s, Qc_s), "caption_x_caption_GE": lift(Qg_s, Qg_s),
        "image_x_image": lift(Pi_s, Pi_s)}
DIAG["c_pair_lift"] = LIFT
log(f"diag a/b/c: {DIAG['a_detection_auc']} {DIAG['b_delta_affect_auc']} "
    f"{LIFT['image_x_caption_CLIP']['ratio_same_over_diff']} {LIFT['image_x_caption_GE']['ratio_same_over_diff']}")


def cond_terms(sa, sb):
    """per episode, per condition (mean over directions): R@1^c and other^c; per direction (mean over conditions)."""
    oc = {}
    for c in CONDS:
        tgt, oth = (0, 1) if c == "a" else (1, 0)
        s = sa if c == "a" else sb
        oc[c] = {"r1": np.mean([hits(s[d], tgt) for d in DIRS], axis=0),
                 "other": np.mean([hits(s[d], oth) for d in DIRS], axis=0)}
    od = {}
    for d in DIRS:
        od[d] = {"r1": 0.5 * (hits(sa[d], 0) + hits(sb[d], 1)), "other": 0.5 * (hits(sa[d], 1) + hits(sb[d], 0))}
    return oc, od


CT = {}
for k in ("AFF", "G-T", "G-TF"):
    fs, cs = FAM[k]["scores"]["fused"], FAM[k]["scores"]["cf"]
    CT[k] = {"fused": cond_terms(*fs), "cf": cond_terms(*cs)}


def three(x_r1, x_ot, mask):
    return {"r1": pci(x_r1, mask), "gain": pci(x_r1 - x_ot, mask), "either": pci(x_r1 + x_ot, mask)}


DD = {}
for k in ("AFF", "G-T", "G-TF"):
    f, c_ = CT[k]["fused"], CT[k]["cf"]
    rec = {"per_pair_condition": {}, "per_direction": {}}
    for i, p in enumerate(PAIRS):
        for c in CONDS:
            rec["per_pair_condition"][f"{p}|{c}"] = three(f[0][c]["r1"] - c_[0][c]["r1"],
                                                          f[0][c]["other"] - c_[0][c]["other"], pidx == i)
    for d in DIRS:
        rec["per_direction"][d] = three(f[1][d]["r1"] - c_[1][d]["r1"], f[1][d]["other"] - c_[1][d]["other"], None)
    if k != "AFF":
        a_ = CT["AFF"]
        rec["minus_aff"] = {"per_pair_condition": {}, "per_direction": {}}
        for i, p in enumerate(PAIRS):
            for c in CONDS:
                dr = (f[0][c]["r1"] - c_[0][c]["r1"]) - (a_["fused"][0][c]["r1"] - a_["cf"][0][c]["r1"])
                do = (f[0][c]["other"] - c_[0][c]["other"]) - (a_["fused"][0][c]["other"] - a_["cf"][0][c]["other"])
                rec["minus_aff"]["per_pair_condition"][f"{p}|{c}"] = three(dr, do, pidx == i)
        for d in DIRS:
            dr = (f[1][d]["r1"] - c_[1][d]["r1"]) - (a_["fused"][1][d]["r1"] - a_["cf"][1][d]["r1"])
            do = (f[1][d]["other"] - c_[1][d]["other"]) - (a_["fused"][1][d]["other"] - a_["cf"][1][d]["other"])
            rec["minus_aff"]["per_direction"][d] = three(dr, do, None)
    rec["either_cost_per_gain"] = -RES["scorers"][k]["either_change"] / RES["scorers"][k]["gain_statistic"]["point"]
    DD[k] = rec
DIAG["d_sharper_term"] = DD
RES["diagnostics"] = DIAG
log("diagnostic (d) done")

# ------------------------------------------------------------------ section 6 of the report
W = {}


def sharp(x):
    p = np.clip(x, 1e-12, 1.0)
    return {"mean_max": float(x.max(1).mean()), "mean_entropy": float(-(p * np.log(p)).sum(1).mean())}


W["sharpness"] = {"image": sharp(Pi_s), "caption_CLIP": sharp(Qc_s), "caption_GE": sharp(Qg_s),
                  "argmax_row_img_cap_CLIP": 100 * float((Pi_s.argmax(1) == Qc_s.argmax(1)).mean()),
                  "argmax_row_img_cap_GE": 100 * float((Pi_s.argmax(1) == Qg_s.argmax(1)).mean())}


# same painting, different emotion, against different painting, different emotion (image head x placement)
def same_painting(q):
    m = labS["emotion"] >= 0
    e_, g_, pi_, q_ = labS["emotion"][m], gS[m], Pi_s[m], q[m]
    all_s, all_n = float(pi_.sum(0) @ q_.sum(0)), float(len(pi_)) ** 2
    gp_s, gp_n = ssum([g_], pi_, q_)
    gpe_s, gpe_n = ssum([g_, e_], pi_, q_)
    e_s, e_n = ssum([e_], pi_, q_)
    sp_de = (gp_s - gpe_s) / (gp_n - gpe_n)
    dp_de = (all_s - gp_s - (e_s - gpe_s)) / (all_n - gp_n - (e_n - gpe_n))
    row = float((pi_ * q_).sum(1).mean())
    return {"ratio_same_painting": sp_de / dp_de, "ratio_same_row": row / dp_de, "n_pairs": int(gp_n - gpe_n)}


W["same_painting"] = {"CLIP": same_painting(Qc_s), "GE": same_painting(Qg_s), "image_x_image": same_painting(Pi_s)}


def rowcorr(x, y):
    a = np.asarray(x, f64)
    b = np.asarray(y, f64)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = den > 0
    return float(((a * b).sum(1)[ok] / den[ok]).mean())


red = {}
for lab_, stk in (("CLIP", STK), ("GE", STK_G)):
    red[lab_] = {}
    for j, h in enumerate(A0):
        red[lab_][h] = {d: {"B": rowcorr(zs(stk[d][:, j]), zB[d]), "cos": rowcorr(zs(stk[d][:, j]), zs(ctx.cos["a"][d]))}
                        for d in DIRS}
# gated term with z(B) where the gate is open as scored
GATES = {"AFF": G_AFF, "G-T": G_GT, "G-TF": G_GTF}
TERMS = {"AFF": T_aff, "G-T": T_GT, "G-TF": T_GTF}
asc = {}
for k in GATES:
    ti = (FAM[k]["fpick"][0] // 56, FAM[k]["fpick"][1] // 56)
    eff = np.where(par == 1, ti[0], ti[1])
    asc[k] = {"tau_index": ti}
    for c in CONDS:
        gs = np.stack([GATES[k][t][c] for t in range(4)])[eff, np.arange(E)].astype(bool)
        asc[k][c] = {"open": int(gs.sum()), "mask": gs}
        for d in DIRS:
            asc[k][c][d] = rowcorr(zs(TERMS[k][c][d][gs]), zs(Bsc[c][d][gs]))
            if k != "AFF":
                asc[k][c][f"{d}_vs_AFF_term"] = rowcorr(zs(TERMS[k][c][d][gs]), zs(T_aff[c][d][gs]))
W["redundancy"] = red
W["gated_term"] = {k: {kk: (vv if kk == "tau_index" else {x: y for x, y in vv.items() if x != "mask"})
                       for kk, vv in v.items()} for k, v in asc.items()}

# shut values: does G-T differ from AFF where the as-scored gate is shut (both conditions, per ranking)?
shut_diff = 0
for c in CONDS:
    shut = ~asc["AFF"][c]["mask"]
    assert np.array_equal(asc["AFF"][c]["mask"], asc["G-T"][c]["mask"])
    tgt = 0 if c == "a" else 1
    for d in DIRS:
        idx = 0 if c == "a" else 1
        ha = hits(FAM["AFF"]["scores"]["fused"][idx][d], tgt)
        hg = hits(FAM["G-T"]["scores"]["fused"][idx][d], tgt)
        shut_diff += int((ha != hg)[shut].sum())
W["GT_vs_AFF_hit_differences_where_gate_shut"] = shut_diff

# affect score alone
alone = {}
for lab_, stk in (("CLIP", STK), ("GE", STK_G)):
    alone[lab_] = {}
    for d in DIRS:
        s = stk[d][:, 0, :]
        fa, fb = hits(s, 0), hits(s, 1)
        ab = (s[:, 0] > s[:, 1])
        for i, p in enumerate(PAIRS):
            m = pidx == i
            alone[lab_][f"{p}|{d}"] = {"pA_first": 100 * float(fa[m].mean()), "pB_first": 100 * float(fb[m].mean()),
                                       "pA_above_pB": 100 * float(ab[m].mean())}
W["affect_alone"] = alone

# per pair and condition: net rankings against AFF (own cells), and at fixed cells
per_cell = {}
for k in ("G-T", "G-TF"):
    per_cell[k] = {}
    for i, p in enumerate(PAIRS):
        for c in CONDS:
            dr = CT[k]["fused"][0][c]["r1"] - CT["AFF"]["fused"][0][c]["r1"]
            m = pidx == i
            per_cell[k][f"{p}|{c}"] = {"net": int(np.rint(2 * dr[m]).sum()), "r1": pci(dr, m)}
W["per_cell_minus_AFF"] = per_cell


def at_cells(k, cells_):
    zT, gated = FAM[k]["zT"], FAM[k]["gated"]
    sa, sb = {}, {}
    for d in DIRS:
        xa = np.empty((E, 13), f32)
        xb = np.empty((E, 13), f32)
        for h in (0, 1):
            t, u, a = CELLS[cells_[h]]
            m = par != h
            xa[m] = combine(zB[d], gated[t]["a"][d], U[u], A[a])[m]
            xb[m] = combine(zB[d], gated[t]["b"][d], U[u], A[a])[m]
        sa[d], sb[d] = xa, xb
    return sa, sb


fixed = {}
aff_e = aff_f["r1"] + aff_f["other"]
for lab_, k, cc in (("G-T@AFF", "G-T", (39, 119)), ("G-TF@AFF", "G-TF", (39, 119)), ("AFF@G-T", "AFF", (46, 117)),
                    ("AFF@G-TF", "AFF", (12, 117))):
    sa, sb = at_cells(k, cc)
    pa = per_anchor_own(sa, sb)
    oc, _ = cond_terms(sa, sb)
    row = {"fused_r1": 100 * float(np.mean(pa["r1"])),
           "net": int((np.rint(4 * pa["r1"]) - np.rint(4 * aff_f["r1"])).sum()),
           "minus_AFF": pci(pa["r1"] - aff_f["r1"]), "gain": pci(pa["gain"] - aff_f["gain"]),
           "either": pci(pa["r1"] + pa["other"] - aff_e), "per_cell": {}}
    for i, p in enumerate(PAIRS):
        for c in CONDS:
            row["per_cell"][f"{p}|{c}"] = int(np.rint(2 * (oc[c]["r1"] - CT["AFF"]["fused"][0][c]["r1"])[pidx == i]).sum())
    fixed[lab_] = row
for k in ("G-T", "G-TF"):
    fu = FAM[k]["fused"]
    fixed[f"{k}@own"] = {"gain": pci(fu["gain"] - aff_f["gain"]), "either": pci(fu["r1"] + fu["other"] - aff_e),
                         "other": pci(fu["other"] - aff_f["other"])}
W["fixed_cells"] = fixed

# in-sample family: whole-seed fused R@1 per cell, best lambda_u at each lambda_a, per tau index
fam_is = {}
for k in ("AFF", "G-T", "G-TF"):
    tot = 100 * FAM[k]["fr"].sum(1) / (4 * E)
    gtot = 100 * (FAM[k]["fr"] - FAM[k]["fo"]).sum(1) / (4 * E)
    curves = {}
    gcurves = {}
    for t in range(4):
        curves[t] = [float(max(tot[(t * 7 + u) * 8 + a] for u in range(7))) for a in range(8)]
        gcurves[t] = [[float(gtot[(t * 7 + u) * 8 + a]) for u in range(7)] for a in range(8)]
    fam_is[k] = {"best": float(tot.max()), "best_cell": int(tot.argmax()), "curves": curves, "gain_cells": gcurves,
                 "fused_r1_cells": tot.tolist(), "gain_cells_flat": gtot.tolist()}
W["in_sample"] = fam_is

# G-TF's reader against AFF's
same_pick = np.concatenate([pick2["a"] == pick["a"], pick2["b"] == pick["b"]])
gtf_rd = {"same_pick_pct": 100 * float(same_pick.mean()), "per_pair_condition": {}}
for i, p in enumerate(PAIRS):
    m = pidx == i
    for c in CONDS:
        gtf_rd["per_pair_condition"][f"{p}|{c}"] = {
            "same_pick_pct": 100 * float((pick2[c] == pick[c])[m].mean()),
            "affect_share_AFF": 100 * float((pick[c] == 0)[m].mean()),
            "affect_share_GTF": 100 * float((pick2[c] == 0)[m].mean()),
            "open_as_scored_AFF": int(asc["AFF"][c]["mask"][m].sum()),
            "open_as_scored_GTF": int(asc["G-TF"][c]["mask"][m].sum())}
W["GTF_reader"] = gtf_rd
RES["why"] = W
log("section-6 numbers done")


# ------------------------------------------------------------------ save
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


arrs = {"QGE_sel": QGE[sel], "tau_prime": np.asarray(tau_p), "taus": np.asarray(TAUS_RULE)}
for k, key in (("AFF", "aff"), ("G-T", "gt"), ("G-TF", "gtf"), ("R1", "r1")):
    for part in ("fused", "cf"):
        for m in ("r1", "gain", "other", "swap", "strict"):
            arrs[f"{key}_{part}__{m}"] = FAM[k][part][m]
    arrs[f"{key}_fused_cells"] = np.asarray([FAM[k]["fpick"][0], FAM[k]["fpick"][1]])
    arrs[f"{key}_cf_cells"] = np.asarray([FAM[k]["cpick"][0], FAM[k]["cpick"][1]])
for k, key, G in (("AFF", "aff", G_AFF), ("G-T", "gt", G_GT), ("G-TF", "gtf", G_GTF), ("R1", "r1", G_R1)):
    for c in CONDS:
        arrs[f"{key}_gate__{c}"] = np.stack([G[t][c] for t in range(4)])
for c in CONDS:
    arrs[f"aff_P__{c}"], arrs[f"aff_pick__{c}"], arrs[f"aff_margin__{c}"] = P[c], pick[c], marg[c]
    arrs[f"gtf_P__{c}"], arrs[f"gtf_pick__{c}"], arrs[f"gtf_margin__{c}"] = P2[c], pick2[c], marg2[c]
for nm, pa in (("B", pB), ("Bp0", pBp0), ("Bp1", pBp1), ("BpG", pBpG)):
    for m in ("r1", "gain", "other", "swap", "strict"):
        arrs[f"{nm}__{m}"] = pa[m]
for d in DIRS:
    arrs[f"stackG__{d}"] = STK_G[d]
for c in CONDS:
    arrs[f"FG__{c}"] = F_G[c]
    for d in DIRS:
        arrs[f"T_GT__{c}__{d}"] = T_GT[c][d]
        arrs[f"T_GTF__{c}__{d}"] = T_GTF[c][d]
arrs["cl"], arrs["pair_index"], arrs["parity"] = cl, pidx, par
np.savez(OUT / "fr5_arrays.npz", **arrs)
RES["checks"] = CHK
(OUT / "fr5_results.json").write_text(json.dumps(jsonable(RES), indent=1))
log(f"written fr5_results.json {sha(OUT / 'fr5_results.json')} fr5_arrays.npz {sha(OUT / 'fr5_arrays.npz')}")
