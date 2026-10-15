"""Phase 1, step 3: seed 42 on selection rows with the refit posteriors (rd_heads.py) and the rebuilt episodes
(rd_episodes.py): B, B0 = B'(A0), B1 = B'(A1) with their picks (rule §6 item 3); AFF, CF, R1 and R1's counterpart
assembled from the frozen cells (rule §5 item 4); cosine, RCA and the nine PM scorers fused at the recorded λ picks;
every per-anchor array against round 3's seed42_arrays.npz and E1's per_anchor_seed42.npz; the regression numbers of
rule §6 item 4. Writes rd_seed42.json and out/rd_arrays_seed42.npz (per-anchor arrays for rd_stats.py).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/rederive/rd_seed42.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as TF  # noqa: E402

from src.eval.aspect_episodes import AspectEpisodes, episodes_sha256  # noqa: E402
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free, uniform_probe_scores  # noqa: E402
from src.eval.aspect_scorers import fused_scores  # noqa: E402
from src.eval.pair_metric_baselines import fit_pair_scaler, fit_pca_basis, rca_term  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

sys.path.insert(0, str(R.T / "20261108_new_method_quick_checks"))
import run_checks as rc  # noqa: E402

sys.path.insert(0, str(R.R1DIR))
import rb_build  # noqa: E402

CELLS = {"aff_fused": (39, 119), "aff_cf": (149, 10), "r1_fused": (116, 119), "r1_cf": (58, 123)}
LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf")]
EDGE = [32.0, 64.0]
CONTROL_SUMS = sorted({u + a for u in R.NESTED_U for a in R.NESTED_A})


def cell_params(cell):
    t, rem = divmod(int(cell), 56)
    return t, R.NESTED_U[rem // 8], R.NESTED_A[rem % 8]


def full(sel_arr, n_rows, sel):
    out = np.full((n_rows, sel_arr.shape[1]), np.nan, dtype=np.float32)
    out[sel] = sel_arr
    return out


def eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.shape == b.shape and np.array_equal(a, b))


# ---------------------------------------------------------------- reader (own code)

def features(post, parts, si, st, ci, ct):
    cols = []
    for h in parts:
        pi, pt = post[h]["img"], post[h]["txt"]
        ai, at = pi[si], pt[st]
        sup = np.einsum("nsc,nsc->ns", ai, at)
        con = np.einsum("nsc,nsc->ns", pi[ci], pt[ct])
        s, c = sup.mean(axis=1), con.mean(axis=1)
        cols += [s, c, s - c, sup.astype(np.float64).std(axis=1, ddof=1), con.astype(np.float64).std(axis=1, ddof=1),
                 (ai.argmax(axis=-1) == at.argmax(axis=-1)).mean(axis=1)]
    return np.stack([np.asarray(x, dtype=np.float64) for x in cols], axis=1)


def stack_dots(post, ep, parts):
    out = {}
    for d in R.DIRS:
        q, c = ("img", "txt") if d == "i2t" else ("txt", "img")
        out[d] = np.stack([np.einsum("nc,nkc->nk", post[h][q][ep.anchor], post[h][c][ep.candidates]) for h in parts],
                          axis=1)
    return out


def zdict(x):
    return {c: {d: zscore_rows(torch.as_tensor(np.asarray(x[c][d]), dtype=torch.float32)) for d in R.DIRS}
            for c in R.CONDS}


def combine(zB, term, u, a):
    out = {c: {} for c in R.CONDS}
    for c in R.CONDS:
        for d in R.DIRS:
            s = zB[c][d]
            if u > 0:
                s = s + u * zB[c][d]
            if a > 0:
                s = s + a * term[c][d]
            out[c][d] = s.numpy().astype(np.float32)
    return out


def condition_free(s):
    return all(np.array_equal(s["a"][d], s["b"][d], equal_nan=True) for d in R.DIRS)


# ---------------------------------------------------------------- PM terms (own code, the formulas of E1's scorers)

def ex_pairs(inp, ep, cond):
    si, st, ci, ct, _ = ep.condition(cond)
    return inp.img[si], inp.txt[st], inp.img[ci], inp.txt[ct]


def qc_sides(inp, ep, d):
    if d == "i2t":
        return inp.img[ep.anchor], inp.txt[ep.candidates], "img", "txt"
    return inp.txt[ep.anchor], inp.img[ep.candidates], "txt", "img"


def proj_pairs(inp, ep, cond, basis):
    out = []
    for a, m in zip(ex_pairs(inp, ep, cond), ("img", "txt", "img", "txt")):
        out.append(basis.project(a.reshape(-1, a.shape[-1]), m).reshape(a.shape[0], a.shape[1], -1))
    return out


def proj_sides(inp, ep, d, basis):
    q, c, qm, cm = qc_sides(inp, ep, d)
    n, k, dim = c.shape
    return basis.project(q, qm), basis.project(c.reshape(-1, dim), cm).reshape(n, k, -1)


def each(fn):
    return {c: {d: fn(c, d) for d in R.DIRS} for c in R.CONDS}


def t32(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def pm_terms(inp, ep, basis, scaler):
    terms = {}

    def diag(relu):
        def fn(cond, d):
            sx, sy, cx, cy = ex_pairs(inp, ep, cond)
            w = (sx * sy).mean(1) - (cx * cy).mean(1)
            if relu:
                w = np.maximum(w, 0.0)
            q, c, _, _ = qc_sides(inp, ep, d)
            return np.einsum("nd,nkd->nk", q * w, c)
        return each(fn)

    terms["diag"], terms["diag_relu"] = diag(False), diag(True)
    R.log("diag terms")

    def bil(cond, d):
        sx, sy, cx, cy = proj_pairs(inp, ep, cond, basis)
        m = np.einsum("nsi,nsj->nij", sx, sy) / sx.shape[1] - np.einsum("nsi,nsj->nij", cx, cy) / cx.shape[1]
        m = 0.5 * (m + m.transpose(0, 2, 1))
        q, c = proj_sides(inp, ep, d, basis)
        return np.einsum("ni,nij,nkj->nk", q, m, c)
    terms["bilinear"] = each(bil)
    R.log("bilinear")

    def cov(delta):
        r = delta.shape[-1]
        return np.einsum("nsi,nsj->nij", delta, delta) / delta.shape[1] + np.eye(r, dtype=np.float32)

    def kis(cond, d):
        sx, sy, cx, cy = proj_pairs(inp, ep, cond, basis)
        m = np.linalg.inv(cov(sx - sy)) - np.linalg.inv(cov(cx - cy))
        q, c = proj_sides(inp, ep, d, basis)
        diff = q[:, None, :] - c
        return -np.einsum("nki,nij,nkj->nk", diff, m, diff)
    terms["kissme"] = each(kis)
    R.log("kissme")

    xing_a = {}
    for cond in R.CONDS:
        sx, sy, cx, cy = (t32(a) for a in proj_pairs(inp, ep, cond, basis))
        ds, dc = (sx - sy) ** 2, (cx - cy) ** 2
        a = torch.ones(ds.shape[0], ds.shape[-1], requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(100):
            loss = ((ds * a[:, None]).sum(-1).sum(-1)
                    - torch.log(torch.sqrt((dc * a[:, None]).sum(-1) + 1e-8).sum(-1))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(min=0.0)
        xing_a[cond] = a.detach()

    def xing(cond, d):
        q, c = (t32(v) for v in proj_sides(inp, ep, d, basis))
        return (-(((q[:, None] - c) ** 2) * xing_a[cond][:, None]).sum(-1)).numpy()
    terms["xing"] = each(xing)
    R.log("xing")

    wang_w = {}
    for cond in R.CONDS:
        sx, sy, cx, cy = (t32(a) for a in ex_pairs(inp, ep, cond))
        ps, pc = sx * sy, cx * cy
        w = torch.ones(ps.shape[0], ps.shape[-1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        for _ in range(50):
            sim_s = (ps * (w ** 2)[:, None]).sum(-1)
            sim_c = (pc * (w ** 2)[:, None]).sum(-1)
            loss = TF.softplus(sim_c[:, None, :] - sim_s[:, :, None]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        wang_w[cond] = w.detach()

    def wang(cond, d):
        q, c, _, _ = qc_sides(inp, ep, d)
        return (t32(q)[:, None] * t32(c) * (wang_w[cond] ** 2)[:, None]).sum(-1).numpy()
    terms["wang"] = each(wang)
    R.log("wang")

    mean, std = t32(scaler.mean), t32(scaler.std)
    probe_p = {}
    for cond in R.CONDS:
        sx, sy, cx, cy = (t32(a) for a in ex_pairs(inp, ep, cond))
        z = (torch.cat([sx * sy, cx * cy], dim=1) - mean) / std
        y = torch.cat([torch.ones(sx.shape[:2]), torch.zeros(cx.shape[:2])], dim=1)
        beta = torch.zeros(z.shape[0], z.shape[-1], requires_grad=True)
        bias = torch.zeros(z.shape[0], requires_grad=True)
        opt = torch.optim.Adam([beta, bias], lr=0.05)
        for _ in range(300):
            logits = (z * beta[:, None]).sum(-1) + bias[:, None]
            per_ep = TF.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1) \
                + 0.5 * 1.0 * (beta ** 2).sum(-1) / z.shape[1]
            opt.zero_grad()
            per_ep.sum().backward()
            opt.step()
        probe_p[cond] = (beta.detach(), bias.detach())

    def probe(cond, d):
        q, c, _, _ = qc_sides(inp, ep, d)
        qc = (t32(q)[:, None] * t32(c) - mean) / std
        beta, bias = probe_p[cond]
        return ((qc * beta[:, None]).sum(-1) + bias[:, None]).numpy()
    terms["probe"] = each(probe)
    R.log("probe")

    def tip(cond, d):
        sx, sy, cx, cy = ex_pairs(inp, ep, cond)
        q, c, _, _ = qc_sides(inp, ep, d)
        qc = q[:, None] * c
        qc /= np.linalg.norm(qc, axis=-1, keepdims=True) + 1e-12

        def aff(px, py):
            k = px * py
            k /= np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12
            return np.exp(-5.0 * (1.0 - np.einsum("nkd,nsd->nks", qc, k))).sum(-1)
        return aff(sx, sy) - aff(cx, cy)
    terms["tip"] = each(tip)

    def vproto(cond, d):
        si, st, ci, ct, _ = ep.condition(cond)
        q, c, _, cm = qc_sides(inp, ep, d)
        feats = inp.txt if cm == "txt" else inp.img
        sup = np.concatenate([feats[si], feats[st]], axis=1).mean(1)
        con = np.concatenate([feats[ci], feats[ct]], axis=1).mean(1)
        sup /= np.linalg.norm(sup, axis=1, keepdims=True)
        con /= np.linalg.norm(con, axis=1, keepdims=True)
        return np.einsum("nkd,nd->nk", c, sup) - np.einsum("nkd,nd->nk", c, con)
    terms["value_prototype"] = each(vproto)
    R.log("tip, value_prototype")
    return terms


def own_crossfit_lambda(cos, term, parity):
    """Own copy of the λ rule (pick by mean(R@1, gain) on each tune half, max keeps the first, grid extended with 32
    and 64 for a half whose pick is 16), to check the recorded key convention."""
    cache = {}

    def crit(lam, rows):
        if lam not in cache:
            cache[lam] = R.metrics(fused_scores(cos, term, lam))
        m = cache[lam]
        return 0.5 * (m["r1"][rows].mean() + m["gain"][rows].mean())
    picks = {}
    for h in (0, 1):
        tune = parity == h
        cands = list(LAMBDA_GRID)
        best = max(cands, key=lambda lam: crit(lam, tune))
        if best == 16.0:
            cands += EDGE
            best = max(cands, key=lambda lam: crit(lam, tune))
        picks[h] = best
    return picks


def point_ci(v, cl, draws):
    r = cluster_bootstrap(v, cl)
    b = draws.boots(v)
    mine = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    if mine != r["ci95"]:
        raise AssertionError("own draws differ from cluster_bootstrap")
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]]}


def main():
    rec = {"time_start": R.now_ams()}
    data, sp, labels = R.load_data()
    ctx = R.SelCtx(data, sp)
    st = np.asarray(sp.scorer_train)
    sel, groups, nrows = ctx.selection, ctx.groups, len(ctx.groups)
    R.log("data loaded")

    # episodes (rd_episodes.py's rebuild; hashes checked again)
    z = np.load(R.OUT / "rd_episodes_seed42.npz")
    base = json.loads(R.need(R.E1 / "baselines_seed42.json").read_text())
    parts = []
    for a, b, _ in R.PAIRS:
        ep = AspectEpisodes(a, b, *(z[f"{a}__{b}__{f}"].astype(np.int64) for f in R.FIELDS))
        if episodes_sha256(ep) != base["episodes_sha256"][f"{a}__{b}"]:
            raise AssertionError(f"{a}__{b}: rebuilt episode hash differs")
        if not ctx.in_sel[ep.rows()].all():
            raise AssertionError("an episode row outside selection")
        parts.append(ep)
    ep = R.concat(parts)
    n = len(ep.anchor)
    pair_index = np.repeat(np.arange(3), n // 3)
    cl = groups[ep.anchor]
    parity = np.arange(n) % 2
    img_u, txt_u = R.unit32(ctx.img), R.unit32(ctx.txt)
    cos = R.cosine(img_u, txt_u, ep)
    R.log(f"{n} episodes, cosine")

    # refit posteriors (rd_heads.py)
    pz = np.load(R.OUT / "rd_post.npz")
    if not np.array_equal(pz["selection"], sel):
        raise AssertionError("posterior selection differs")
    P = {g: {m: full(pz[f"{g}__{m}"], nrows, sel) for m in ("img", "txt")}
         for g in ("affect", "affect_km", "image", "caption", "csd")}
    post_e2 = {"affect": P["affect_km"], "image": P["image"], "caption": P["caption"]}
    post_a1 = {"affect": P["affect"], "image": P["image"], "caption": P["caption"], "csd": P["csd"]}

    # T_N1u, B, B0, B1
    inputs, _, prov = rc.model_inputs(ctx, "A3", st, False)
    t_n1u = centered_term(inputs, ep, uniform=True)
    del inputs
    R.log("T_N1u")
    sc = {}
    rec["B_picks"] = {}
    for name, post_, parts_ in (("B", post_e2, ("affect", "image", "caption")), ("B0", post_a1, R.A0),
                                ("B1", post_a1, R.A1)):
        s, pk = crossfit_condition_free(cos, t_n1u, uniform_probe_scores(post_, ep, parts_), parity)
        sc[name] = s
        pa = R.metrics(s)
        rec["B_picks"][name] = {"picks": {str(h): [float(x) for x in pk[h]] for h in (0, 1)},
                                "mean_r1": 100 * float(np.mean(pa["r1"])), "hits4": int(np.rint(4 * pa["r1"]).sum())}
        R.log(f"{name}: {rec['B_picks'][name]}")
    target = {"B": 18.341064453125, "B0": 18.436686197916664, "B1": 18.804931640625}
    rec["B_means_exact"] = {k: rec["B_picks"][k]["mean_r1"] == v for k, v in target.items()}

    # reader on A0 (round 1's frozen half-readers)
    pk = rb_build.load_readers("A0", False)[0]
    post_a0 = {h: post_a1[h] for h in R.A0}
    Pc, pick, marg = {}, {}, {}
    for cond in R.CONDS:
        si, st_, ci, ct, _ = ep.condition(cond)
        X = features(post_a0, R.A0, si, st_, ci, ct)
        probs = []
        for half in pk["halves"]:
            if not np.array_equal(half["model"].classes_, np.arange(3)):
                raise AssertionError("half-reader classes")
            probs.append(np.asarray(half["model"].predict_proba(half["scaler"].transform(X)), dtype=np.float64))
        Pc[cond] = np.stack(probs).mean(axis=0)
        pick[cond] = Pc[cond].argmax(axis=1)
        srt = np.sort(Pc[cond], axis=1)
        marg[cond] = srt[:, -1] - srt[:, -2]
    taus = json.loads(R.need(R.R1DIR / "results/rc_tau.json").read_text())["taus"]
    stk = stack_dots(post_a0, ep, R.A0)
    Tc = {c: {d: np.einsum("nh,nhk->nk", Pc[c], stk[d].astype(np.float64)).astype(np.float32) for d in R.DIRS}
          for c in R.CONDS}
    zT = zdict(Tc)
    gates = {"r1": {t: {c: (marg[c] >= tau).astype(np.float32) for c in R.CONDS} for t, tau in enumerate(taus)},
             "aff": {t: {c: ((marg[c] >= tau) & (pick[c] == 0)).astype(np.float32) for c in R.CONDS}
                     for t, tau in enumerate(taus)}}
    gated, gcf = {}, {}
    for r in ("r1", "aff"):
        gated[r], gcf[r] = {}, {}
        for t in range(4):
            gated[r][t] = {c: {d: torch.as_tensor(gates[r][t][c])[:, None] * zT[c][d] for d in R.DIRS}
                           for c in R.CONDS}
            m = {d: (0.5 * (gated[r][t]["a"][d].numpy().astype(np.float64)
                            + gated[r][t]["b"][d].numpy().astype(np.float64))).astype(np.float32) for d in R.DIRS}
            gcf[r][t] = {c: {d: torch.as_tensor(m[d].copy()) for d in R.DIRS} for c in R.CONDS}
    zB = zdict(sc["B"])
    R.log("reader, gates, terms")

    scores = {"cosine": cos, "B": sc["B"], "B0": sc["B0"], "B1": sc["B1"]}
    rec["frozen_cells"] = {}
    for key, (c0, c1) in CELLS.items():
        reader = key.split("_")[0]
        by = {}
        for h, cell in ((0, c0), (1, c1)):
            t, u, a = cell_params(cell)
            term = gated[reader][t] if key.endswith("fused") else gcf[reader][t]
            by[h] = combine(zB, term, u, a)
            if key.endswith("cf") and not condition_free(by[h]):
                raise AssertionError(f"{key} cell {cell} is not condition-free")
        scores[key] = R.by_parity(by, parity)
        rec["frozen_cells"][key] = {str(h): {"cell": c, "tau_index": cell_params(c)[0], "lambda_u": cell_params(c)[1],
                                             "lambda_a": cell_params(c)[2]} for h, c in ((0, c0), (1, c1))}
    # σ* of the nested control (depends on B only), for the record
    ctrl = {s_: R.metrics(combine(zB, zB, s_, 0.0))["r1"] for s_ in CONTROL_SUMS}
    rec["sigma_star"] = {str(h): float(max(CONTROL_SUMS, key=lambda s_: float(ctrl[s_][parity == h].mean())))
                         for h in (0, 1)}
    R.log("AFF, CF, R1 assembled")

    # RCA and PM
    fit_rows = np.random.default_rng(0).choice(st, 60000, replace=False)
    rec["fit_rows_sha256"] = R.sha_bytes(np.ascontiguousarray(fit_rows, dtype=np.int64))
    rec["fit_rows_sha_equal"] = rec["fit_rows_sha256"] == base["fit_rows_sha256"]
    if ctx.in_sel[fit_rows].any():
        raise AssertionError("a fit row is a selection row")
    fi, ft = (np.asarray(x[fit_rows], np.float32) for x in (data.img_features, data.txt_features))
    fi, ft = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (fi, ft))
    basis, scaler = fit_pca_basis(fi, ft), fit_pair_scaler(fi, ft)
    inp = SimpleNamespace(img=img_u, txt=txt_u)
    terms = {"rca": rca_term(inp, ep, basis)}
    R.log("rca term")
    terms.update(pm_terms(inp, ep, basis, scaler))
    rec["lambda"] = {}
    for name, term in terms.items():
        lp = base["scorers"][name]["lambda_picks"]
        lam = {h: (float("inf") if lp[str(h)] == "inf" else float(lp[str(h)])) for h in (0, 1)}
        mine = own_crossfit_lambda(cos, term, parity)
        scores[name] = R.by_parity({h: fused_scores(cos, term, lam[h]) for h in (0, 1)}, parity)
        rec["lambda"][name] = {"recorded": {str(h): lam[h] for h in (0, 1)},
                               "own_crossfit": {str(h): mine[h] for h in (0, 1)},
                               "equal": all(mine[h] == lam[h] for h in (0, 1))}
    R.log("RCA and PM fused")

    # per-anchor arrays and the stored comparisons
    pa = {k: R.metrics(v) for k, v in scores.items()}
    r3 = np.load(R.need(R.R3DIR / "results/seed42_arrays.npz"))
    e1 = np.load(R.need(R.E1 / "per_anchor_seed42.npz"))
    cmp = {}
    for key in ("aff_fused", "aff_cf", "r1_fused", "r1_cf"):
        for m in R.METRICS:
            cmp[f"seed42_arrays/{key}__{m}"] = eq(pa[key][m], r3[f"{key}__{m}"])
        cmp[f"seed42_arrays/{key}_cells"] = eq(np.array(CELLS[key]), r3[f"{key}_cells"])
    for r in ("aff", "r1"):
        for c in R.CONDS:
            cmp[f"seed42_arrays/{r}_gate__{c}"] = eq(np.stack([gates[r][t][c] for t in range(4)]), r3[f"{r}_gate__{c}"])
        cmp[f"seed42_arrays/{r}_sigma"] = eq(np.array([float(rec["sigma_star"]["0"]), float(rec["sigma_star"]["1"])]),
                                             r3[f"{r}_sigma"])
    for c in R.CONDS:
        cmp[f"seed42_arrays/P__{c}"] = eq(Pc[c], r3[f"P__{c}"])
        cmp[f"seed42_arrays/margin__{c}"] = eq(marg[c], r3[f"margin__{c}"])
        cmp[f"seed42_arrays/pick__{c}"] = eq(pick[c], r3[f"pick__{c}"])
    cmp["seed42_arrays/taus"] = eq(np.array(taus), r3["taus"])
    cmp["seed42_arrays/cl"] = eq(cl, r3["cl"])
    cmp["seed42_arrays/parity"] = eq(parity, r3["parity"])
    cmp["seed42_arrays/pair_index"] = eq(pair_index, r3["pair_index"])
    cmp["per_anchor_seed42/anchor_group"] = eq(cl, e1["anchor_group"])
    cmp["per_anchor_seed42/pair_index"] = eq(pair_index, e1["pair_index"])
    for name in ("cosine", "rca", *R.PM):
        for m in R.METRICS:
            cmp[f"per_anchor_seed42/{name}__{m}"] = eq(pa[name][m], e1[f"{name}__{m}"])

    # regression numbers (rule §6 item 4)
    draws = R.Draws(cl)
    if not all((pa[k]["gain"] == 0).all() for k in ("cosine", "B", "B0", "B1", "aff_cf", "r1_cf")):
        raise AssertionError("a condition-free scorer has a non-zero gain")
    means = {k: float(np.mean(pa[k]["r1"])) for k in ("B0", "aff_cf", "B")}
    comp = "B0"
    for k in ("aff_cf", "B"):
        if means[k] > means[comp]:
            comp = k
    reg = {"aff_fused_r1": 100 * float(np.mean(pa["aff_fused"]["r1"])),
           "cf_r1": 100 * float(np.mean(pa["aff_cf"]["r1"])), "bar_comparator": comp,
           "bar_margin": point_ci(pa["aff_fused"]["r1"] - pa[comp]["r1"], cl, draws),
           "margin_vs_cf": point_ci(pa["aff_fused"]["r1"] - pa["aff_cf"]["r1"], cl, draws),
           "gain_statistic": point_ci(pa["aff_fused"]["gain"] - pa["aff_cf"]["gain"], cl, draws),
           "aff_minus_r1": point_ci(pa["aff_fused"]["r1"] - pa["r1_fused"]["r1"], cl, draws),
           "r1_margin_vs_cf": point_ci(pa["r1_fused"]["r1"] - pa["r1_cf"]["r1"], cl, draws),
           "r1_gain_statistic": point_ci(pa["r1_fused"]["gain"] - pa["r1_cf"]["gain"], cl, draws),
           "aff_minus_b1": point_ci(pa["aff_fused"]["r1"] - pa["B1"]["r1"], cl, draws)}
    cmp["bar_v/aff"] = eq(pa["aff_fused"]["r1"] - pa[comp]["r1"], r3["aff_bar_v"])
    rec["regression"] = reg
    want = {"aff_fused_r1": 19.136555989583336, "cf_r1": 18.39599609375,
            "bar_margin": (0.6998697916666667, 0.4598852740816973, 0.9371680126852968),
            "gain_statistic": (3.110758463541667, 2.780005709854805, 3.4559584315470384),
            "aff_minus_r1": (0.21769205729166666, 0.06425880757348419, 0.3709597330984391),
            "r1_margin_vs_cf": (0.4435221354166667, 0.21646171563312194, 0.6735669710776852),
            "r1_gain_statistic": (2.667236328125, 2.325087836946873, 3.012361650695922),
            "aff_minus_b1": (0.33162434895833337, 0.048231414333532084, 0.6246158772581268)}
    vs = {}
    for k, w in want.items():
        if isinstance(w, tuple):
            got = (reg[k]["point"], *reg[k]["ci95"])
            vs[k] = {"rule": list(w), "mine": list(got), "max_abs_diff": max(abs(x - y) for x, y in zip(got, w))}
        else:
            vs[k] = {"rule": w, "mine": reg[k], "max_abs_diff": abs(reg[k] - w)}
        vs[k]["within_1e-9"] = vs[k]["max_abs_diff"] <= 1e-9
        vs[k]["exact"] = vs[k]["max_abs_diff"] == 0.0
    vs["bar_comparator_is_B0"] = comp == "B0"
    rec["regression_vs_rule"] = vs
    rec["array_checks"] = cmp
    rec["array_checks_all_equal"] = all(cmp.values())
    rec["n_array_checks"] = len(cmp)
    np.savez(R.OUT / "rd_arrays_seed42.npz", cl=cl, pair_index=pair_index, parity=parity,
             **{f"{k}__{m}": pa[k][m] for k in pa for m in R.METRICS},
             **{f"{r}_gate__{c}": np.stack([gates[r][t][c] for t in range(4)]) for r in ("aff", "r1") for c in R.CONDS})
    rec["time_end"] = R.now_ams()
    R.write_json(R.RD / "rd_seed42.json", rec)
    bad = [k for k, v in cmp.items() if not v]
    R.log(f"array checks {len(cmp)}, unequal {bad}; B means exact {rec['B_means_exact']}; "
          f"regression within 1e-9 {all(v['within_1e-9'] for k, v in vs.items() if isinstance(v, dict))}")


if __name__ == "__main__":
    main()
