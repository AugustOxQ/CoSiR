"""Round 5 re-derivation: tests of the own code on synthetic data only (no GoEmotions data, no real GE head).

Run: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
     /root/miniconda3/envs/CoSiR/bin/python src/test/20261123_idea3_goemotions/rederive/rd5_test_synthetic.py
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd5_paths as paths  # noqa: E402
import rd5_core as core  # noqa: E402
import rd5_stats as stats  # noqa: E402
import rd5_placement as pl  # noqa: E402
import rd5_candidates as cands  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PASSED = []


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL {name}")
    PASSED.append(name)
    print(f"PASS {name}")


def raises(fn, exc=AssertionError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- metrics and ties

def test_first_place():
    S = np.array([[2, 1, 1], [1, 1, 0], [np.nan, 0, 0], [0, 3, 1]], np.float32)
    check("first_place strict, ties miss, NaN miss", core.first_place(S, 0).tolist() == [True, False, False, False])
    check("first_place other column", core.first_place(S, 1).tolist() == [False, False, False, True])


def test_per_anchor():
    a = np.array([[3, 1, 0], [0, 2, 1]], np.float32)
    b = np.array([[1, 3, 0], [2, 1, 0]], np.float32)
    pa = core.per_anchor({"a": {"i2t": a, "t2i": a}, "b": {"i2t": b, "t2i": b}})
    # ep0: a target col0 wins, b target col1 wins -> r1 1, other 0, swap 1, strict 1
    # ep1: a col1 wins (other), b col0 wins (other) -> r1 0, other 1, gain -1, swap 0
    check("per_anchor values", pa["r1"].tolist() == [1, 0] and pa["other"].tolist() == [0, 1]
          and pa["gain"].tolist() == [1, -1] and pa["swap"].tolist() == [1, 0] and pa["strict"].tolist() == [1, 0])
    check("as_int4 refuses non-multiples", raises(lambda: core.as_int4(np.array([0.1]))))


def test_cells():
    check("cell numbering", core.cell_index(1, 0, 3) == 59 and core.cell_params(116) == (2, 0.0, 2.0)
          and core.cell_params(119) == (2, 0.0, 16.0) and core.cell_params(39) == (0, 4.0, 16.0)
          and core.cell_params(149) == (2, 4.0, 4.0) and core.cell_params(10) == (0, 0.5, 0.5)
          and core.cell_params(58) == (1, 0.0, 0.5) and core.cell_params(123) == (2, 0.5, 1.0))


def _zrows():
    from src.model.aspect_rule import zscore_rows
    return zscore_rows


def test_family_ties_and_control():
    """A family where every cell gives the same statistics: ties go to cell 0 and sigma* to 0; the criterion is
    min(rho - rho_ctrl, gamma) against a hand computation."""
    rng = np.random.default_rng(0)
    E = 8
    B = rng.normal(size=(E, 13)).astype(np.float32)
    Bd = {c: {d: B.copy() for d in core.DIRECTIONS} for c in core.CONDITIONS}
    T = {c: {d: rng.normal(size=(E, 13)).astype(np.float32) for d in core.DIRECTIONS} for c in core.CONDITIONS}
    zeros = {c: np.zeros((4, E), np.float32) for c in core.CONDITIONS}
    parity = np.arange(E) % 2
    fam = core.run_family(Bd, T, zeros, parity, _zrows())
    check("closed gates: every fused cell ties -> lowest cell (0) and sigma* 0",
          all(fam.picks[h]["fused_cell"] == 0 and fam.picks[h]["cf_cell"] == 0 and fam.picks[h]["sigma"] == 0.0
              for h in (0, 1)))
    # hand computation of the criterion for one cell with open gates
    ones = {c: np.ones((4, E), np.float32) for c in core.CONDITIONS}
    fam = core.Family(Bd, T, ones, parity, _zrows()).statistics()
    h = 0
    tune = parity == h
    zb = fam.zB["i2t"]
    ctrl = {c: {d: zb for d in core.DIRECTIONS} for c in core.CONDITIONS}
    rho_ctrl = int(core.int4_counts({c: {d: ctrl[c][d][tune] for d in core.DIRECTIONS} for c in core.CONDITIONS})[0].sum())
    cell = core.cell_index(3, 2, 5)
    S = {c: {d: (zb + np.float32(1.0) * zb) + np.float32(4.0) * (np.float32(1) * fam.zT[c][d]) for d in core.DIRECTIONS}
         for c in core.CONDITIONS}
    r4, o4 = core.int4_counts({c: {d: S[c][d][tune] for d in core.DIRECTIONS} for c in core.CONDITIONS})
    fam.crossfit()
    crit_hand = min(int(r4.sum()) - rho_ctrl, int((r4 - o4).sum()))
    rho = fam.f_r4[cell, tune].astype(int).sum()
    gam = fam.f_g4[cell, tune].astype(int).sum()
    check("rho_ctrl and the criterion match a hand computation",
          fam.picks[h]["rho_ctrl"] == rho_ctrl and min(rho - rho_ctrl, gam) == crit_hand)
    # constructed exact ties for the pick: P with equal top two -> argmax first, margin 0
    P = np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.3, 0.3, 0.4]])
    check("pick ties to the first grouping, margin 0 on exact ties",
          np.argmax(P, axis=1).tolist() == [0, 1, 2] and core.top_two_margin(P).tolist()[:2] == [0.0, 0.0])
    g = core.gates_aff(np.array([0.5, 0.5, 0.1]), np.array([0, 1, 0]), np.array([0.0, 0.2, 0.5, 0.7]))
    check("AFF gate = 1[m >= tau] * 1[pick = affect], float32",
          g.dtype == np.float32 and g[:, 0].tolist() == [1, 1, 1, 0] and g[:, 1].tolist() == [0, 0, 0, 0]
          and g[:, 2].tolist() == [1, 0, 0, 0])
    check("tau' = percentile of margins, condition a first",
          np.array_equal(core.taus_from_margins(np.array([3.0, 1.0]), np.array([2.0, 0.0])),
                         np.percentile(np.array([3.0, 1.0, 2.0, 0.0]), [0, 25, 50, 75])))


# ---------------------------------------------------------------- stats

def _rec(point, lo, glo, delta):
    return {"bar_margin": {"point": point, "ci95": [lo, 1.0]}, "gain_statistic": {"point": 1.0, "ci95": [glo, 2.0]},
            "delta_vs_aff": {"delta_int": delta}}


def test_d10_and_carry():
    r = _rec(0.5, 1e-3, 1e-3, 1)
    check("D10: bar point exactly 0.5 passes", stats.d10(r)["c1_bar_point_ge_0.5"])
    check("D10: 0.49 fails", not stats.d10(_rec(0.49, 1e-3, 1e-3, 1))["c1_bar_point_ge_0.5"])
    check("D10: lower bound exactly 0 fails", not stats.d10(_rec(0.6, 0.0, 1e-3, 1))["c2_bar_lower_gt_0"])
    check("D10: clause 3 reads the gain statistic's lower bound",
          not stats.d10(_rec(0.6, 1e-3, 0.0, 1))["c3_gain_lower_gt_0"] and stats.d10(_rec(0.6, -1.0, 1e-3, 1))["c3_gain_lower_gt_0"])
    check("D10 boundary flag at 1e-12", stats.d10(_rec(0.5 + 5e-13, 1e-3, 1e-3, 1))["boundary"]["c1_within_1e-12"])

    def recs(dT, dTF, clearT=True, clearTF=True):
        out = {}
        for k, dd, cl in (("G-T", dT, clearT), ("G-TF", dTF, clearTF)):
            r = _rec(0.6 if cl else 0.4, 1e-3, 1e-3, dd)
            r["D10"] = stats.d10(r)
            out[k] = r
        return out
    check("carry: band 24 inclusive -> tie to G-T", stats.carry(recs(10, 34))["carried"] == "G-T")
    check("carry: gap 25 exclusive -> G-TF", stats.carry(recs(10, 35))["carried"] == "G-TF")
    check("carry: gap of exactly 24 flagged", stats.carry(recs(10, 34))["boundary"]["gap_24"] == ["G-T"])
    check("carry: delta 0 not in E", stats.carry(recs(0, 0))["decision"] == "KILL"
          and stats.carry(recs(0, 0))["boundary"]["delta_zero"] == ["G-T", "G-TF"])
    check("carry: E needs all three clauses", stats.carry(recs(50, 10, clearT=False))["carried"] == "G-TF")
    check("carry: empty E is a kill", stats.carry(recs(-3, 5, clearTF=False))["decision"] == "KILL")


def test_bar_comparator():
    a = {"r1": np.array([0.5, 0.5])}
    b = {"r1": np.array([0.5, 0.5])}
    c = {"r1": np.array([0.75, 0.25])}
    check("bar comparator: ties to the earliest", stats.bar_comparator([("B'_Q", a), ("B'(A0)", b), ("counterpart", c)])[0] == "B'_Q")
    d = {"r1": np.array([0.75, 0.5])}
    check("bar comparator: strict maximum", stats.bar_comparator([("B'_Q", a), ("B'(A0)", b), ("counterpart", d)])[0] == "counterpart")


def test_sensitivity():
    d = np.array([1.0, 0.0, 0.5, 0.25, 1.0, 0.0, 0.75])
    g = np.array([0, 0, 1, 1, 1, 2, 3])
    r = stats.sensitivity(d, g)
    # hand computation
    groups = {0: [1.0, 0.0], 1: [0.5, 0.25, 1.0], 2: [0.0], 3: [0.75]}
    n, Pn = 7, 4
    grand = d.mean()
    ssw = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in groups.values())
    ssb = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
    msw, msb = ssw / (n - Pn), ssb / (Pn - 1)
    sm2 = sum(len(v) ** 2 for v in groups.values())
    n0 = (n - sm2 / n) / (Pn - 1)
    sa2 = max(0.0, (msb - msw) / n0)
    se = np.sqrt((sa2 * (9 * sm2 - 6 * n) + msw * 3 * n) / (3 * n) ** 2)
    check("sensitivity: SE and x = 2.80 SE on a hand-computed example",
          abs(r["SE"] - se) < 1e-15 and abs(r["x_2.80SE"] - 2.80 * se) < 1e-15 and r["P"] == 4)


# ---------------------------------------------------------------- placement, GE head, swap

def _synthetic_rows(rng, N=3000, K=5, F=6):
    labels_all = rng.integers(0, K, N)
    centers = rng.normal(size=(K, F)) * 2
    X = (centers[labels_all] + rng.normal(size=(N, F))).astype(np.float32)
    perm = rng.permutation(N)
    st = np.sort(perm[:2000])
    rows = np.sort(perm[2000:2600])
    return X, labels_all, st, rows


def test_placement_equals_direct_fit():
    rng = np.random.default_rng(1)
    X, y, st, rows = _synthetic_rows(rng)
    lab = pl.global_labels(y[st], st, len(X))
    r = pl.placement(X, pl.identity, lab, st, rows, 300, n_draw=800, n_check=300)
    draw = np.random.default_rng(0).choice(st, 800, replace=False)
    rest = np.setdiff1d(st, draw)
    chk = np.random.default_rng(1).choice(rest, 300, replace=False)
    clf = LogisticRegression(C=1.0, max_iter=300).fit(X[draw], lab[draw])
    full = np.full((len(X), 5), np.nan, np.float32)
    full[rows] = clf.predict_proba(X[rows])
    check("placement = direct LogisticRegression fit (same draw, check rows, scatter)",
          np.array_equal(r["post"], full, equal_nan=True) and r["accuracy"] == 100 * float(clf.score(X[chk], lab[chk]))
          and np.array_equal(r["draw"], draw) and np.array_equal(r["check"], chk))
    check("placement: float32, NaN outside rows", r["post"].dtype == np.float32
          and np.isnan(np.delete(r["post"], rows, axis=0)).all() and np.isfinite(r["post"][rows]).all())
    lab2 = lab.copy()
    lab2[st] = lab2[st] * 2                          # classes 0, 2, 4, ...: not 0..K-1
    check("placement: classes other than 0..K-1 refused",
          raises(lambda: pl.placement(X, pl.identity, lab2, st, rows, 300, n_draw=800, n_check=300)))
    Xn = X.copy()
    Xn[draw[0]] = np.nan
    check("placement: non-finite draw rows refused",
          raises(lambda: pl.placement(Xn, pl.identity, lab, st, rows, 300, n_draw=800, n_check=300)))


def test_ge_input_and_head():
    rng = np.random.default_rng(2)
    X, y, st, rows = _synthetic_rows(rng)
    probs_st, probs_sel = X[st], X[rows]
    Xg = pl.ge_input(len(X), st, probs_st, rows, probs_sel)
    check("GE input: X[scorer_train[i]] = affect_probs[i], X[rows] = probs, NaN elsewhere",
          all(np.array_equal(Xg[st[i]], probs_st[i]) for i in (0, 7, 1999)) and np.array_equal(Xg[rows], probs_sel)
          and np.isnan(np.delete(Xg, np.concatenate([st, rows]), axis=0)).all() and Xg.dtype == np.float32)
    check("GE input: unsorted scorer_train refused", raises(lambda: pl.ge_input(len(X), st[::-1], probs_st, rows, probs_sel)))
    check("GE input: overlap with selection refused",
          raises(lambda: pl.ge_input(len(X), st, probs_st, np.concatenate([rows[:-1], st[:1]]), probs_sel)))
    draw = np.random.default_rng(0).choice(st, 800, replace=False)
    check("mapping: scorer_train[pos] == draw", pl.check_mapping(st, draw))
    check("mapping: a shifted draw is detected", not pl.check_mapping(st, draw + 1) or not np.isin(draw + 1, st).all())
    lab = pl.global_labels(y[st], st, len(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        r = pl.ge_head(Xg, lab, st, rows, n_draw=800, n_check=300, max_iter=2, fallback_max_iter=3000)
        check("fallback: reaching max_iter refits once with the larger cap and records both counts",
              r["fallback_used"] and r["n_iter_first"] == 2 and r["n_iter_fallback"] < 3000 and r["head"] is not None
              and r["head"]["max_iter"] == 3000)
        r2 = pl.ge_head(Xg, lab, st, rows, n_draw=800, n_check=300, max_iter=2, fallback_max_iter=3)
        check("fallback: a second cap stops (no posterior)", r2["head"] is None and r2["fallback_used"] and "stop" in r2)
    r3 = pl.ge_head(Xg, lab, st, rows, n_draw=800, n_check=300)
    check("GE head without fallback when it converges", not r3["fallback_used"] and r3["head"]["n_iter"] < 300
          and np.allclose(r3["head"]["post"][rows].sum(1), 1, atol=1e-5))


def _synthetic_bundle(rng, E=40, N=600, K=(41, 7, 6)):
    """A small bundle shaped like the real one (selection rows, 3 groupings, readers on 18 features)."""
    from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores
    groups = np.arange(N) // 3
    sel = np.sort(rng.choice(N, 300, replace=False))
    in_sel = np.zeros(N, bool)
    in_sel[sel] = True

    def posterior(k):
        p = np.full((N, k), np.nan, np.float32)
        x = rng.random((len(sel), k)).astype(np.float32) ** 3
        p[sel] = x / x.sum(1, keepdims=True)
        return p
    post = {h: {"img": posterior(k), "txt": posterior(k)} for h, k in zip(core.A0, K)}
    pick = lambda shape: rng.choice(sel, size=shape)  # noqa: E731
    ep = SimpleNamespace(anchor=pick(E), candidates=pick((E, 13)), pairs_a_img=pick((E, 4)), pairs_a_txt=pick((E, 4)),
                         pairs_b_img=pick((E, 4)), pairs_b_txt=pick((E, 4)))
    cs = {d: rng.normal(size=(E, 13)).astype(np.float32) for d in core.DIRECTIONS}
    cos = {c: {d: cs[d].copy() for d in core.DIRECTIONS} for c in core.CONDITIONS}
    tu = rng.normal(size=(E, 13)).astype(np.float32)
    t_n1u = {c: {d: tu for d in core.DIRECTIONS} for c in core.CONDITIONS}
    parity = np.arange(E) % 2
    ctx = SimpleNamespace(groups=groups, selection=sel, in_sel=in_sel, cos=cos, parity=parity, pooled=ep)
    halves = []
    for j in range(2):
        Xb = rng.normal(size=(300, 18))
        yb = rng.integers(0, 3, 300)
        sc = StandardScaler().fit(Xb)
        halves.append({"scaler": sc, "model": LogisticRegression(max_iter=500).fit(sc.transform(Xb), yb)})
    Bp0 = crossfit_condition_free(cos, t_n1u, uniform_probe_scores(post, ep, core.A0), parity)[0]
    B = crossfit_condition_free(cos, t_n1u, t_n1u, parity)[0]
    return SimpleNamespace(ctx=ctx, ep=ep, post=post, t_n1u=t_n1u, Bp0=Bp0, pBp0=core.per_anchor(Bp0), B=B,
                           pB=core.per_anchor(B), stack=core.grouping_stack(post, ep), F=core.reader_features(post, ep),
                           halves=halves, parity=parity, heads=post["affect"])


def test_extension_and_candidates():
    from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores
    from src.model.aspect_rule import zscore_rows
    A = {"crossfit_condition_free": crossfit_condition_free, "uniform_probe_scores": uniform_probe_scores,
         "zscore_rows": zscore_rows}
    rng = np.random.default_rng(3)
    b = _synthetic_bundle(rng)
    fp0 = pl.fingerprint(b.post)
    cands.Guard.released = False
    Qge = b.post["affect"]["txt"].copy()
    Qge[b.ctx.selection] = np.roll(Qge[b.ctx.selection], 1, axis=1)
    check("guard refuses a GE placement before release", raises(lambda: cands.extend(b, Qge, "ge", A)))
    check("a 'clip' placement that is not Q_CLIP is refused", raises(lambda: cands.extend(b, Qge, "clip", A)))
    ext = cands.extend(b, b.post["affect"]["txt"], "clip", A)
    check("CLIP extension: stack, F and B' equal the bundle's", all(np.array_equal(ext.stack[d], b.stack[d]) for d in core.DIRECTIONS)
          and all(np.array_equal(ext.F[c], b.F[c]) for c in core.CONDITIONS)
          and all(np.array_equal(ext.B[c][d], b.Bp0[c][d]) for c in core.CONDITIONS for d in core.DIRECTIONS))
    rd = core.reader(b.F, b.stack, b.halves)
    taus = core.taus_from_margins(rd["a"]["m"], rd["b"]["m"])
    out = cands.candidates(b, ext, rd, taus, A)
    g_aff = {c: core.gates_aff(rd[c]["m"], rd[c]["pi"], taus) for c in core.CONDITIONS}
    check("CLIP: G-T and G-TF gates and tau' equal AFF's", all(np.array_equal(out[k]["gates"][c], g_aff[c])
                                                                for k in out for c in core.CONDITIONS)
          and np.array_equal(out["G-TF"]["taus"], taus))
    cands.Guard.released = True
    ext_g = cands.extend(b, Qge, "ge", A)
    check("GE-like extension: only the affect slice and columns 0..5 change",
          any(not np.array_equal(ext_g.stack[d][:, 0], b.stack[d][:, 0]) for d in core.DIRECTIONS)
          and any(not np.array_equal(ext_g.F[c][:, :6], b.F[c][:, :6]) for c in core.CONDITIONS)
          and all(np.array_equal(ext_g.stack[d][:, 1:], b.stack[d][:, 1:]) for d in core.DIRECTIONS))
    pi_, Q = b.post["affect"]["img"], Qge
    ind = {"i2t": np.einsum("nc,nkc->nk", pi_[b.ep.anchor], Q[b.ep.candidates]),
           "t2i": np.einsum("nc,nkc->nk", Q[b.ep.anchor], pi_[b.ep.candidates])}
    check("positive check: affect slice equals the independent recomputation from Q and p_img",
          all(np.array_equal(ext_g.stack[d][:, 0], ind[d]) for d in core.DIRECTIONS))
    check("the bundle's post is unchanged after both extensions (identity and value)", pl.fingerprint(b.post) == fp0)
    cands.Guard.released = False
    # a swap that assigns into the shared dict is caught by the fingerprint
    bad = dict(b.post)
    bad_aff = b.post["affect"]
    old = bad_aff["txt"]
    bad_aff["txt"] = Qge
    check("a mutation of the shared post is detected by the fingerprint", pl.fingerprint(b.post) != fp0)
    bad_aff["txt"] = old
    check("fingerprint restored", pl.fingerprint(b.post) == fp0)
    new = pl.swap_caption_side(b.post, Qge)
    check("swap returns a new dict and leaves post untouched", new is not b.post and new["affect"] is not b.post["affect"]
          and b.post["affect"]["txt"] is old and new["image"] is b.post["image"])
    del bad


def test_ge_file_checks():
    """load_ge_file on synthetic files in a temporary folder (no GoEmotions model call)."""
    import json
    import tempfile
    import rd5_goemotions as gm
    rng = np.random.default_rng(4)
    N = 400_000
    sel = np.sort(rng.choice(N, gm.N_SEL, replace=False))
    rest = np.setdiff1d(np.arange(N), sel)
    splits = SimpleNamespace(selection=sel, scorer_train=rest[:1000], held=rest[1000:2000])
    sids = rng.permutation(N).astype(np.int64)
    ctx = SimpleNamespace(selection=sel, data=SimpleNamespace(sample_ids=sids))
    probs = rng.random((gm.N_SEL, 28)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        saved = (gm.GE_FILE, gm.GE_RECORD, gm.RUN_LOG)
        try:
            gm.GE_FILE, gm.GE_RECORD, gm.RUN_LOG = td / "f.npz", td / "f.json", td / "log.md"

            def write(rows, s_ids, record_sha=True, log_sha=True):
                np.savez(gm.GE_FILE, probs=probs, rows=rows, sample_ids=s_ids)
                sha = paths.sha_file(gm.GE_FILE)
                gm.GE_RECORD.write_text(json.dumps({"npz_sha256": sha if record_sha else "0" * 64,
                                                    "probs_sha256": paths.sha_array(probs)}))
                gm.RUN_LOG.write_text(f"line {sha if log_sha else 'x'}\n")
            write(sel.astype(np.int64), sids[sel])
            r = gm.load_ge_file(ctx, splits)
            check("GE file: good file accepted", all(r["checks"].values()))
            write(sel.astype(np.int64), sids[sel], record_sha=False)
            check("GE file: SHA not in its record refused", raises(lambda: gm.load_ge_file(ctx, splits), SystemExit))
            write(sel.astype(np.int64), sids[sel], log_sha=False)
            check("GE file: SHA not on a run-log line refused", raises(lambda: gm.load_ge_file(ctx, splits), SystemExit))
            write(np.roll(sel, 1).astype(np.int64), sids[sel])
            check("GE file: rows not equal to the selection refused", raises(lambda: gm.load_ge_file(ctx, splits), SystemExit))
            write(sel.astype(np.int64), np.roll(sids[sel], 1))
            check("GE file: misaligned sample ids refused", raises(lambda: gm.load_ge_file(ctx, splits), SystemExit))
        finally:
            gm.GE_FILE, gm.GE_RECORD, gm.RUN_LOG = saved
    pos = gm.spot_positions()
    check("spot positions: sorted, 1,024 unique, from rng(6)", len(np.unique(pos)) == 1024 and np.all(np.diff(pos) > 0)
          and np.array_equal(pos, np.sort(np.random.default_rng(6).choice(32_413, size=1_024, replace=False))))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"{len(PASSED)} checks passed")
