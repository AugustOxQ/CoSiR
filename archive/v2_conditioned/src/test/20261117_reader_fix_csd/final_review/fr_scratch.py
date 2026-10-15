"""Final review: from-scratch re-derivation of R-b expected/A0, R-b arg-max/A1 and R-c (parent R-b expected/A0).

Written by the final reviewer without using common.py, rb_*.py, rc_core.py or rederive/. Uses only: the step-1 setup
(run_sweep.setup: episodes, cosine, parity, B), the library (crossfit_nested, crossfit_condition_free,
uniform_probe_scores, centered_term, per_anchor, cluster_bootstrap, zscore_rows), run_told_oracle.fit_one_head (the D2
recipe) and raw input files. R-b probabilities come (i) from the stored half-readers applied to features computed here,
and (ii) for A0 from half-readers retrained here from the stored banks and cross-fitted heads with features and labels
computed here. Outputs: final_review/out/fr_scratch.{json,npz}.
"""
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/project/CoSiR")
sys.path.insert(0, str(ROOT))
FR = Path(__file__).resolve().parent
RD = FR.parent
RES = RD / "results"
OUT = FR / "out"
T = ROOT / "src/test"

from src.eval.aspect_metrics import per_anchor, cluster_bootstrap  # noqa: E402
from src.eval.aspect_nested import crossfit_nested, NESTED_U, NESTED_A  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores, centered_term  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

COND, DIRS = ("a", "b"), ("i2t", "t2i")
CFG = {"A0": ("affect", "image", "caption"), "A1": ("affect", "image", "caption", "csd")}
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


spec = importlib.util.spec_from_file_location("run_step1", T / "20261116_grouping_step1_style/run_step1.py")
rs1 = importlib.util.module_from_spec(spec)
sys.modules["run_step1"] = rs1
spec.loader.exec_module(rs1)
rsw, rto, rc, n6 = rs1.rsw, rs1.rto, rs1.rc, rs1.n6

S = rsw.setup()
ctx = S.ctx
ep, cl, par, pidx = ctx.pooled, np.asarray(ctx.anchor_group), np.asarray(ctx.parity), np.asarray(ctx.pair_index)
n = len(cl)
checks = {"parity_is_episode_index_parity": bool(np.array_equal(par, np.arange(n) % 2)),
          "n_episodes": int(n), "n_clusters": int(len(np.unique(cl)))}
B, pB = S.B, S.pB
log("setup done")

# ------------------------------------------------------------------ posteriors (D2)
sel = np.asarray(ctx.selection)
NR = len(ctx.groups)


def full(a):
    f = np.full((NR, a.shape[1]), np.nan, np.float32)
    f[sel] = np.asarray(a, np.float32)
    return f


zn6 = np.load(T / "20261108_new_method_quick_checks/results/n6_posteriors.npz")
zst = np.load(T / "20261116_grouping_step1_style/results/step1_heads_style.npz")
assert np.array_equal(zn6["selection"], sel) and np.array_equal(zst["selection"], sel)
post = {"image": {"img": full(zn6["image__img"]), "txt": full(zn6["image__txt"])},
        "caption": {"img": full(zn6["caption__img"]), "txt": full(zn6["caption__txt"])},
        "csd": {"img": full(zst["style_csd__img"]), "txt": full(zst["style_csd__txt"])}}
pl = np.load(T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz")["partition_L"]
checks["partition_L_equals_setup"] = bool(np.array_equal(np.asarray(pl), np.asarray(S.partition_L)))
lab = rto.global_labels(np.asarray(pl, np.int64), S.scorer_train, NR)
post["affect"], prov_aff = rto.fit_one_head(ctx, lab, S.scorer_train, n6.HEAD_ROWS)
checks["affect_head_heldout"] = prov_aff["heldout_accuracy"]
log("posteriors ready")

# ------------------------------------------------------------------ B' from scratch (D9)
inp, _, _ = rc.model_inputs(ctx, "A3", S.scorer_train, False)
t_n1u = centered_term(inp, ep, uniform=True)
del inp
z1 = np.load(T / "20261116_grouping_step1_style/results/step1_eval_style.npz")
pBp = {}
for c, parts in CFG.items():
    bp, _ = crossfit_condition_free(ctx.cos, t_n1u, uniform_probe_scores(post, ep, parts), par)
    pBp[c] = per_anchor(bp)
    checks[f"Bprime_{c}_equals_step1_stored"] = bool(all(np.array_equal(pBp[c][m], z1[f"{c}__Bprime__{m}"])
                                                         for m in ("r1", "gain", "other")))
checks["B_equals_step1_stored"] = bool(all(np.array_equal(pB[m], z1[f"B__{m}"]) for m in ("r1", "gain", "other")))
log(f"B' done {checks}")


# ------------------------------------------------------------------ features (my own code)
def agree(h, ii, tt, dt):
    return np.einsum("nsc,nsc->ns", post_[h]["img"][ii].astype(dt), post_[h]["txt"][tt].astype(dt))


def feats(postd, parts, sa_i, sa_t, sb_i, sb_t, dt=np.float32):
    """six features per grouping for supports (sa) and contrasts (sb)."""
    global post_
    post_ = postd
    cols = []
    for h in parts:
        s = agree(h, sa_i, sa_t, dt)
        c = agree(h, sb_i, sb_t, dt)
        S_, C_ = s.mean(1), c.mean(1)
        mt = (postd[h]["img"][sa_i].argmax(-1) == postd[h]["txt"][sa_t].argmax(-1)).mean(1)
        cols += [S_, C_, S_ - C_, s.astype(np.float64).std(1, ddof=1), c.astype(np.float64).std(1, ddof=1), mt]
    return np.stack([np.asarray(x, np.float64) for x in cols], 1)


def feats_both(postd, parts, e, dt=np.float32):
    return {"a": feats(postd, parts, e.pairs_a_img, e.pairs_a_txt, e.pairs_b_img, e.pairs_b_txt, dt),
            "b": feats(postd, parts, e.pairs_b_img, e.pairs_b_txt, e.pairs_a_img, e.pairs_a_txt, dt)}


def stack_s(parts):
    """{dir: (E, H, 13)} s_h(q, k) = p_h(q) . p_h(k), each item with its own modality's head."""
    out = {}
    for d in DIRS:
        qa, ka = ("img", "txt") if d == "i2t" else ("txt", "img")
        out[d] = np.stack([np.einsum("nc,nkc->nk", post[h][qa][ep.anchor], post[h][ka][ep.candidates])
                           for h in parts], 1)
    return out


def apply_readers(halves, X):
    return np.mean([h["model"].predict_proba(h["scaler"].transform(X)) for h in halves], axis=0)


def retrain(config):
    """A half-reader per half from the stored banks and cross-fitted heads; my own features, labels and CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    parts = CFG[config]
    hv = np.load(RES / "rb_halves.npz")
    cfp = {}
    for h in parts:
        z = np.load(RES / f"rb_heads_{h}.npz")
        assert np.array_equal(z["filled_by_half"], 1 - hv["half_of_local_row"])
        cfp[h] = {"img": z["img"], "txt": z["txt"]}
    # bank labels straight from the grouping files (not from the block position) to check the layout
    labs = {"affect": np.asarray(pl, np.int64),
            "image": np.load(T / "20261031_pseudo_partitions/results/partitions.npz")["image"].astype(np.int64),
            "caption": np.load(T / "20261031_pseudo_partitions/results/partitions.npz")["caption"].astype(np.int64),
            "csd": np.load(T / "20261116_grouping_step1_style/results/step1_group_style.npz")["style_csd"].astype(np.int64)}
    halves, info = [], {}
    for j in (0, 1):
        z = np.load(RES / f"rb_bank_{config}_half{j}.npz")
        bp = [tuple(s.split("__")) for s in z["block_pairs"].tolist()]
        nb = int(z["block_size"])
        e = type("E", (), {k: z[k].astype(np.int64) for k in ("pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt",
                                                               "anchor", "candidates")})
        rows = np.concatenate([e.anchor[:, None], e.candidates, e.pairs_a_img, e.pairs_a_txt, e.pairs_b_img,
                               e.pairs_b_txt], 1)
        assert (hv["half_of_local_row"][rows] == j).all()
        ya = np.empty(len(e.anchor), np.int64)
        yb = np.empty(len(e.anchor), np.int64)
        lay_ok = True
        for i, (ga, gb) in enumerate(bp):
            sl = slice(i * nb, (i + 1) * nb)
            lay_ok &= bool((labs[ga][e.pairs_a_img[sl]] == labs[ga][e.pairs_a_txt[sl]]).all())
            lay_ok &= bool((labs[gb][e.pairs_b_img[sl]] == labs[gb][e.pairs_b_txt[sl]]).all())
            lay_ok &= bool((labs[ga][e.candidates[sl, 0]] == labs[ga][e.anchor[sl]]).all())
            ya[sl], yb[sl] = parts.index(ga), parts.index(gb)
        F = feats_both(cfp, parts, e)
        X = np.vstack([F["a"], F["b"]])
        y = np.concatenate([ya, yb])
        epi = np.concatenate([np.arange(len(ya))] * 2)
        fold = np.empty(len(ya), np.int64)
        for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(ya)))):
            fold[te] = f
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        losses = []
        for Cv in (0.01, 0.1, 1.0, 10.0, 100.0):
            L = []
            for f in range(5):
                tr = fold[epi] != f
                m = LogisticRegression(C=Cv, solver="lbfgs", max_iter=2000).fit(Xs[tr], y[tr])
                L.append(log_loss(y[~tr], m.predict_proba(Xs[~tr]), labels=np.arange(len(parts))))
            losses.append(float(np.mean(L)))
        best = int(np.argmin(losses))  # argmin keeps the first (smaller C) on exact ties
        Cb = (0.01, 0.1, 1.0, 10.0, 100.0)[best]
        m = LogisticRegression(C=Cb, solver="lbfgs", max_iter=2000).fit(Xs, y)
        halves.append({"scaler": sc, "model": m})
        info[j] = {"layout_ok": lay_ok, "chosen_C": Cb, "cv_losses": losses, "balanced": np.bincount(y).tolist()}
        log(f"retrained {config} half {j}: C {Cb}, losses {losses}")
    return halves, info


# ------------------------------------------------------------------ evaluation (library cross-fits, own bar)
def boot(v):
    r = cluster_bootstrap(np.asarray(v, np.float64), cl)
    return [100 * r["point"], 100 * r["ci95"][0], 100 * r["ci95"][1]]


def bar_and_gain(pn, pc, config):
    comps = [("B_prime", pBp[config]), ("counterpart", pc), ("B", pB)]
    means = [float(np.mean(np.asarray(p["r1"], np.float64))) for _, p in comps]
    k = 0
    for i in (1, 2):
        if means[i] > means[k]:
            k = i
    bar_v = np.asarray(pn["r1"], np.float64) - np.asarray(comps[k][1]["r1"], np.float64)
    bar = boot(bar_v)
    gain = boot(np.asarray(pn["gain"], np.float64) - np.asarray(pc["gain"], np.float64))
    margin = boot(np.asarray(pn["r1"], np.float64) - np.asarray(pc["r1"], np.float64))
    clears = bool(bar[0] >= 0.5 and bar[1] > 0 and gain[1] > 0)
    return {"comparator": comps[k][0], "means": {nm: 100 * m for (nm, _), m in zip(comps, means)},
            "fused_r1": 100 * float(np.mean(pn["r1"])), "bar": bar, "gain_stat": gain, "margin": margin,
            "clears": clears,
            "per_pair_bar": {i: boot_sub(bar_v, pidx == i) for i in range(3)}}, bar_v


def boot_sub(v, m):
    r = cluster_bootstrap(np.asarray(v, np.float64)[m], cl[m])
    return [100 * r["point"], 100 * r["ci95"][0], 100 * r["ci95"][1]]


def cf(Tm):
    return {c: {d: (0.5 * (Tm["a"][d].astype(np.float64) + Tm["b"][d].astype(np.float64))).astype(np.float32)
                for d in DIRS} for c in COND}


def evaluate(Tm, config):
    nested, _, tp = crossfit_nested(B, B, Tm, par)
    cfs, cp = crossfit_condition_free(B, B, cf(Tm), par)
    pn, pc = per_anchor(nested), per_anchor(cfs)
    res, bar_v = bar_and_gain(pn, pc, config)
    res["T_picks"], res["cf_picks"] = tp, cp
    return res, pn, pc, bar_v


result = {"checks": checks}
arrays = {}
stored = {}
for config in ("A0", "A1"):
    parts = CFG[config]
    F = feats_both(post, parts, ep)
    F64 = feats_both(post, parts, ep, np.float64)
    with open(RES / f"rb_reader_{config}.pkl", "rb") as f:
        pk = pickle.load(f)
    P = {c: apply_readers(pk["halves"], F[c]) for c in COND}
    P64 = {c: apply_readers(pk["halves"], F64[c]) for c in COND}
    st = np.load(RES / f"cand_Rb_expected_{config}.npz")
    stored[config] = st
    result[f"{config}_P_equals_stored"] = bool(np.array_equal(P["a"], st["extra__probs_a"])
                                              and np.array_equal(P["b"], st["extra__probs_b"]))
    result[f"{config}_P_maxabs_vs_stored"] = float(max(np.abs(P[c] - st[f"extra__probs_{c}"]).max() for c in COND))
    result[f"{config}_pick_changes_float64_features"] = int(sum((P64[c].argmax(1) != P[c].argmax(1)).sum() for c in COND))
    if config == "A0":
        halves_rt, info_rt = retrain("A0")
        Prt = {c: apply_readers(halves_rt, F[c]) for c in COND}
        result["A0_retrain"] = {"info": info_rt,
                                "P_maxabs_vs_stored": float(max(np.abs(Prt[c] - st[f"extra__probs_{c}"]).max()
                                                                for c in COND)),
                                "pick_changes_vs_stored": int(sum((Prt[c].argmax(1) != st[f"pick__{c}"]).sum()
                                                                  for c in COND))}
    picks = {c: P[c].argmax(1) for c in COND}
    sP = {c: np.sort(P[c], 1) for c in COND}
    marg = {c: sP[c][:, -1] - sP[c][:, -2] for c in COND}
    result[f"{config}_picks_equal_stored"] = bool(all(np.array_equal(picks[c], st[f"pick__{c}"]) for c in COND))
    s = stack_s(parts)
    rows = np.arange(n)
    Targ = {c: {d: s[d][rows, picks[c]].astype(np.float32) for d in DIRS} for c in COND}
    Texp = {c: {d: np.einsum("nh,nhk->nk", P[c].astype(np.float64), s[d].astype(np.float64)).astype(np.float32)
                for d in DIRS} for c in COND}
    result[f"{config}_Texp_equals_stored"] = bool(all(np.array_equal(Texp[c][d], st[f"T__{c}__{d}"])
                                                     for c in COND for d in DIRS))
    todo = {"A0": [("Rb_expected_A0", Texp)], "A1": [("Rb_argmax_A1", Targ)]}[config]
    for name, Tm in todo:
        res, pn, pc, bar_v = evaluate(Tm, config)
        sz = np.load(RES / f"cand_{name}.npz")
        res["arrays_equal_stored"] = {
            "fused": bool(all(np.array_equal(pn[m], sz[f"fused__{m}"]) for m in ("r1", "gain", "other"))),
            "cf": bool(all(np.array_equal(pc[m], sz[f"cf__{m}"]) for m in ("r1", "gain", "other"))),
            "bar_v": bool(np.array_equal(bar_v, sz["bar_v"]))}
        result[name] = res
        log(f"{name}: {json.dumps({k: res[k] for k in ('comparator', 'bar', 'gain_stat', 'clears', 'arrays_equal_stored')})}")
    if config == "A0":
        # ---------------------------------------------------------- R-c from scratch on R-b expected / A0
        m_all = np.concatenate([marg["a"], marg["b"]])
        taus = [float(x) for x in np.percentile(m_all, [0, 25, 50, 75])]
        gates = [{c: (marg[c] >= t).astype(np.float32) for c in COND} for t in taus]
        zB = {c: {d: zscore_rows(torch.as_tensor(np.asarray(B[c][d]), dtype=torch.float32)) for d in DIRS} for c in COND}
        zT = {c: {d: zscore_rows(torch.as_tensor(Texp[c][d], dtype=torch.float32)) for d in DIRS} for c in COND}
        gated = [{c: {d: torch.as_tensor(g[c])[:, None] * zT[c][d] for d in DIRS} for c in COND} for g in gates]
        Gcf = []
        for gk in gated:
            m = {d: torch.as_tensor((0.5 * (gk["a"][d].numpy().astype(np.float64)
                                            + gk["b"][d].numpy().astype(np.float64))).astype(np.float32)) for d in DIRS}
            Gcf.append({c: m for c in COND})

        def score(term, u, a):
            out = {}
            for c in COND:
                out[c] = {}
                for d in DIRS:
                    x = zB[c][d]
                    if u > 0:
                        x = x + u * zB[c][d]
                    if a > 0:
                        x = x + a * term[c][d]
                    out[c][d] = x.numpy().astype(np.float32)
            return out

        cells = [(k, u, a) for k in range(4) for u in NESTED_U for a in NESTED_A]
        stats_f, stats_c, scores_f, scores_c = [], [], {}, {}
        for i, (k, u, a) in enumerate(cells):
            pf = per_anchor(score(gated[k], u, a))
            pcc = per_anchor(score(Gcf[k], u, a))
            stats_f.append((np.asarray(pf["r1"]), np.asarray(pf["gain"])))
            stats_c.append(np.asarray(pcc["r1"]))
        sums = sorted({u + a for u in NESTED_U for a in NESTED_A})
        ctrl_r1 = {sg: np.asarray(per_anchor(score(zB, sg, 0.0) if False else
                                             {c: {d: (zB[c][d] + sg * zB[c][d]).numpy().astype(np.float32)
                                                  if sg > 0 else zB[c][d].numpy().astype(np.float32)
                                                  for d in DIRS} for c in COND})["r1"]) for sg in sums}
        fpick, cpick, ctrl = {}, {}, {}
        for half in (0, 1):
            tune = par == half
            best_s = sums[0]
            for sg in sums:
                if ctrl_r1[sg][tune].mean() > ctrl_r1[best_s][tune].mean():
                    best_s = sg
            rc_ = float(ctrl_r1[best_s][tune].mean())
            ctrl[half] = (best_s, rc_)
            bi, bv = 0, None
            for i in range(len(cells)):
                v = min(float(stats_f[i][0][tune].mean()) - rc_, float(stats_f[i][1][tune].mean()))
                if bv is None or v > bv:
                    bi, bv = i, v
            fpick[half] = bi
            ci_, cv_ = 0, None
            for i in range(len(cells)):
                v = float(stats_c[i][tune].mean())
                if cv_ is None or v > cv_:
                    ci_, cv_ = i, v
            cpick[half] = ci_
        pn = {m: np.empty(n) for m in ("r1", "gain")}
        pc = {m: np.empty(n) for m in ("r1", "gain")}
        for half in (0, 1):
            ap = par != half
            pn["r1"][ap], pn["gain"][ap] = stats_f[fpick[half]][0][ap], stats_f[fpick[half]][1][ap]
            pc["r1"][ap] = stats_c[cpick[half]][ap]
        pc["gain"][:] = 0.0
        # counterpart gain must be exactly 0 (condition-free)
        for half in (0, 1):
            k, u, a = cells[cpick[half]]
            g_c = np.asarray(per_anchor(score(Gcf[k], u, a))["gain"])
            assert (g_c == 0).all()
        res, bar_v = bar_and_gain(pn, pc, "A0")
        sz = np.load(RES / "cand_Rc_Rb_expected_A0.npz")
        res.update(taus=taus, taus_equal_stored=bool(taus == json.loads((RES / "rc_tau.json").read_text())["taus"]),
                   fused_cells={h: cells[i] for h, i in fpick.items()}, cf_cells={h: cells[i] for h, i in cpick.items()},
                   control=ctrl, open_share=[100 * float(np.mean(np.concatenate([g["a"], g["b"]]))) for g in gates],
                   arrays_equal_stored={"fused_r1": bool(np.array_equal(pn["r1"], sz["fused__r1"])),
                                        "fused_gain": bool(np.array_equal(pn["gain"], sz["fused__gain"])),
                                        "cf_r1": bool(np.array_equal(pc["r1"], sz["cf__r1"])),
                                        "bar_v": bool(np.array_equal(bar_v, sz["bar_v"]))})
        # alternative comparators, for the reading of the null: B' only, and the product g_bar*z(T_cf)
        res["vs_Bprime_only"] = boot(np.asarray(pn["r1"]) - np.asarray(pBp["A0"]["r1"], np.float64))
        result["Rc_Rb_expected_A0"] = res
        log(f"R-c: {json.dumps({k: res[k] for k in ('comparator', 'means', 'bar', 'gain_stat', 'clears', 'taus', 'fused_cells', 'cf_cells', 'arrays_equal_stored')}, default=str)}")


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


(OUT / "fr_scratch.json").write_text(json.dumps(js(result), indent=1))
log("done")
