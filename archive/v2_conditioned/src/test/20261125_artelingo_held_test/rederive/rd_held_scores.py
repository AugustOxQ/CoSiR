"""Phase 2, step 2: the held context and the frozen-pick assembly (rule §5 items 4 and 5), own code.

1. Head refits (rule §6 item 2) with the allowed fitters, every fitted LogisticRegression kept (numerics unchanged,
   as rd_heads.py); selection posteriors checked bit for bit against the stored arrays and phase 1's rd_post.npz; the
   same fitted objects then predict held rows (the fitters' own call: predict_proba of the unit-normalised CLIP
   features, cast to float32, NaN outside held rows).
2. Own scoring function: cosine, T_N1u, B, B0, B1 by the frozen seed-42 picks of phase 1 (own nested score, the
   z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6u) of round 3 D10, D11 and round 4 D4, written here), the A0 reader, gates, AFF,
   CF, R1 and its counterpart at the rule's cells, RCA at the recorded λ. Regression: run in selection mode on seed 42
   it must reproduce phase 1's per-anchor arrays exactly (which equal round 3's and E1's stored arrays).
3. Held seeds 52, 53, 54 (episodes from rd_held_episodes.py), each scored as one 12,288-episode block (pairs in
   run_baselines order, parity = episode index mod 2); per-anchor arrays concatenated in seed order to
   out/rd_held_arrays.npz. No per-seed or per-pair metric is computed or printed.

  cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/rederive/rd_held_scores.py \
  > src/test/20261125_artelingo_held_test/rederive/out/rd_held_scores.log 2>&1
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rd_heads as H  # noqa: E402  (RecLR, coef_hashes, global_labels, fit; imports run_n6 and run_told_oracle)
import rd_seed42 as S  # noqa: E402  (features, stack_dots, zdict, combine, condition_free, cell_params, CELLS)

from src.eval.aspect_episodes import AspectEpisodes, episodes_sha256  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, uniform_probe_scores  # noqa: E402
from src.eval.aspect_scorers import fused_scores  # noqa: E402
from src.eval.pair_metric_baselines import fit_pair_scaler, fit_pca_basis, rca_term  # noqa: E402

n6, rto, rc, rb_build = H.n6, H.rto, S.rc, S.rb_build
HELD_SEEDS = (52, 53, 54)
SCORERS = ("cosine", "rca", "B", "B0", "B1", "aff_fused", "aff_cf", "r1_fused", "r1_cf")
GROUPINGS = ("affect", "affect_km", "image", "caption", "csd")


class HeldCtx:
    """EvalContext's fields with held rows in place of selection rows: CLIP features NaN outside held rows and finite
    on them, A3 codes the same (encode_rows over the held rows, as EvalContext.encode does for selection)."""

    def __init__(self, data, sp):
        self.data = data
        self.groups = np.asarray(sp.groups)
        self.rows = np.asarray(sp.held)
        self.in_rows = np.zeros(len(self.groups), dtype=bool)
        self.in_rows[self.rows] = True
        self.img = self.masked(data.img_features)
        self.txt = self.masked(data.txt_features)

    def masked(self, values):
        out = np.full(values.shape, np.nan, dtype=np.float32)
        out[self.rows] = values[self.rows]
        if not (np.isnan(out[~self.in_rows]).all() and np.isfinite(out[self.in_rows]).all()):
            raise AssertionError("held masking failed")
        return out

    def encode(self, ckpt):
        from src.train.train_factors import encode_rows, load_factor_checkpoint
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        ic, tc = encode_rows(model, self.data.img_features, self.data.txt_features, rows=self.rows, device="cpu")
        res = []
        for codes in (ic, tc):
            full = np.full((len(self.groups), codes.shape[1]), np.nan, dtype=np.float32)
            full[self.rows] = codes
            if not (np.isnan(full[~self.in_rows]).all() and np.isfinite(full[self.in_rows]).all()):
                raise AssertionError("held codes must be finite on held rows and NaN elsewhere")
            res.append(full)
        return res[0], res[1]


def bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


# ---------------------------------------------------------------- heads

def refit_heads(data, sp, ctx_sel):
    """Fit every head once (rd_heads.py's calls); return {grouping: {img, txt: classifier}}, records."""
    n6.LogisticRegression = H.RecLR
    rto.LogisticRegression = H.RecLR
    st = np.asarray(sp.scorer_train)
    n = len(ctx_sel.groups)
    sel = ctx_sel.selection
    if tuple(n6.PARTS) != ("affect", "image", "caption"):
        raise AssertionError(f"run_n6.PARTS is {n6.PARTS}")
    clf, post_sel, rec = {}, {}, {"bitwise": {}, "coef_sha256": {}, "convergence_warnings": {}}

    pl = np.load(R.need(R.T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz"))["partition_L"]
    (p, prov), fitted, w = H.fit("affect", lambda: rto.fit_one_head(ctx_sel, H.global_labels(pl, st, n), st,
                                                                     H.HEAD_ROWS))
    told = json.loads(R.need(R.T / "20261111_community_told_oracle/results/told_oracle.json").read_text())
    rec["affect_identity"] = json.loads(json.dumps(prov)) == told["arms"]["L"]["head"]
    clf["affect"], post_sel["affect"] = dict(zip(("img", "txt"), fitted)), p
    rec["convergence_warnings"]["affect"] = w
    R.log("affect heads")

    z = np.load(R.need(R.T / "20261031_pseudo_partitions/results/partitions.npz"))
    if not np.array_equal(z["local_groups"], np.unique(ctx_sel.groups[st], return_inverse=True)[1]):
        raise AssertionError("E2 partitions not aligned with scorer_train")
    labs = {h: H.global_labels(z[h], st, n) for h in ("affect", "image", "caption")}
    (pe, _), fitted, w = H.fit("e2", lambda: n6.fit_heads(ctx_sel, labs, st, H.HEAD_ROWS))
    for i, (h, g) in enumerate((("affect", "affect_km"), ("image", "image"), ("caption", "caption"))):
        clf[g], post_sel[g] = dict(zip(("img", "txt"), fitted[2 * i:2 * i + 2])), pe[h]
    rec["convergence_warnings"]["e2"] = w
    stored = np.load(R.need(R.T / "20261108_new_method_quick_checks/results/n6_posteriors.npz"))
    for h, g in (("affect", "affect_km"), ("image", "image"), ("caption", "caption")):
        for m in ("img", "txt"):
            rec["bitwise"][f"n6_posteriors/{h}__{m}"] = bits_equal(post_sel[g][m][sel].astype(np.float32),
                                                                   stored[f"{h}__{m}"])
    R.log("E2 heads")

    g_ = np.load(R.need(R.T / "20261116_grouping_step1_style/results/step1_group_style.npz"))
    if not np.array_equal(g_["scorer_train"], st):
        raise AssertionError("step 1 scorer_train differs")
    (pc, _), fitted, w = H.fit("csd", lambda: rto.fit_one_head(ctx_sel, H.global_labels(g_["style_csd"], st, n), st,
                                                                H.HEAD_ROWS))
    clf["csd"], post_sel["csd"] = dict(zip(("img", "txt"), fitted)), pc
    rec["convergence_warnings"]["csd"] = w
    hs = np.load(R.need(R.T / "20261116_grouping_step1_style/results/step1_heads_style.npz"))
    for m in ("img", "txt"):
        rec["bitwise"][f"step1_heads_style/style_csd__{m}"] = bits_equal(pc[m][sel].astype(np.float32),
                                                                         hs[f"style_csd__{m}"])
    R.log("csd heads")

    rp = np.load(R.OUT / "rd_post.npz")
    for g in GROUPINGS:
        for m in ("img", "txt"):
            rec["bitwise"][f"phase1_rd_post/{g}__{m}"] = bits_equal(post_sel[g][m][sel], rp[f"{g}__{m}"])
            rec["coef_sha256"].setdefault(g, {})[m] = H.coef_hashes(clf[g][m])["coef_then_intercept"]
            # the kept object reproduces the fitter's own selection posteriors
            feats = data.img_features if m == "img" else data.txt_features
            own = np.full_like(post_sel[g][m], np.nan)
            own[sel] = clf[g][m].predict_proba(rc.unit(feats[sel]))
            rec["bitwise"][f"kept_object_reproduces_fitter/{g}__{m}"] = bits_equal(own, post_sel[g][m])
    rec["all_bitwise_equal"] = all(rec["bitwise"].values())
    rec["n_bitwise"] = len(rec["bitwise"])
    if not (rec["all_bitwise_equal"] and rec["affect_identity"]):
        raise AssertionError(f"head refit check failed: {[k for k, v in rec['bitwise'].items() if not v]}")
    return clf, post_sel, rec


def held_posteriors(data, hctx, clf):
    post = {}
    for g in GROUPINGS:
        post[g] = {}
        for m in ("img", "txt"):
            feats = data.img_features if m == "img" else data.txt_features
            c = clf[g][m]
            full = np.full((len(hctx.groups), len(c.classes_)), np.nan, dtype=np.float32)
            full[hctx.rows] = c.predict_proba(rc.unit(feats[hctx.rows]))
            if not (np.isfinite(full[hctx.rows]).all() and np.isnan(full[~hctx.in_rows]).all()):
                raise AssertionError(f"{g}/{m}: held posteriors must be finite on held rows and NaN elsewhere")
            post[g][m] = full
    return post


# ---------------------------------------------------------------- own nested score and the scoring function

def nested(cos, t_u, t_c, lam_u, lam_a):
    """z(cos) + λ_u·z(t_u) + λ_a·z(t_c), a weight-0 term left out (round 3 D10)."""
    if not (math.isfinite(lam_u) and math.isfinite(lam_a)) or lam_u < 0 or lam_a < 0:
        raise ValueError("weights")
    zc, zu, za = S.zdict(cos), S.zdict(t_u), S.zdict(t_c)
    out = {c: {} for c in R.CONDS}
    for c in R.CONDS:
        for d in R.DIRS:
            s = zc[c][d]
            if lam_u > 0:
                s = s + lam_u * zu[c][d]
            if lam_a > 0:
                s = s + lam_a * za[c][d]
            out[c][d] = s.numpy().astype(np.float32)
    return out


def score_block(ctx, P, ep, inputs, frozen, basis, readers, taus):
    """Per-anchor arrays of one 12,288-episode block with the frozen picks. ctx: SelCtx or HeldCtx."""
    n = len(ep.anchor)
    parity = np.arange(n) % 2
    img_u, txt_u = R.unit32(ctx.img), R.unit32(ctx.txt)
    cos = R.cosine(img_u, txt_u, ep)
    t_n1u = centered_term(inputs, ep, uniform=True)
    post_e2 = {"affect": P["affect_km"], "image": P["image"], "caption": P["caption"]}
    post_a1 = {"affect": P["affect"], "image": P["image"], "caption": P["caption"], "csd": P["csd"]}
    scores = {"cosine": cos}
    for name, post_, parts_ in (("B", post_e2, ("affect", "image", "caption")), ("B0", post_a1, R.A0),
                                ("B1", post_a1, R.A1)):
        tc = uniform_probe_scores(post_, ep, parts_)
        pk = frozen["B_picks"][name]
        scores[name] = R.by_parity({h: nested(cos, t_n1u, tc, *pk[h]) for h in (0, 1)}, parity)

    post_a0 = {h: post_a1[h] for h in R.A0}
    Pc, pick, marg = {}, {}, {}
    for cond in R.CONDS:
        si, st_, ci, ct, _ = ep.condition(cond)
        X = S.features(post_a0, R.A0, si, st_, ci, ct)
        probs = []
        for half in readers["halves"]:
            if not np.array_equal(half["model"].classes_, np.arange(3)):
                raise AssertionError("half-reader classes")
            probs.append(np.asarray(half["model"].predict_proba(half["scaler"].transform(X)), dtype=np.float64))
        Pc[cond] = np.stack(probs).mean(axis=0)
        pick[cond] = Pc[cond].argmax(axis=1)
        srt = np.sort(Pc[cond], axis=1)
        marg[cond] = srt[:, -1] - srt[:, -2]
    stk = S.stack_dots(post_a0, ep, R.A0)
    Tc = {c: {d: np.einsum("nh,nhk->nk", Pc[c], stk[d].astype(np.float64)).astype(np.float32) for d in R.DIRS}
          for c in R.CONDS}
    zT = S.zdict(Tc)
    gates = {"r1": {t: {c: (marg[c] >= tau).astype(np.float32) for c in R.CONDS} for t, tau in enumerate(taus)},
             "aff": {t: {c: ((marg[c] >= tau) & (pick[c] == 0)).astype(np.float32) for c in R.CONDS}
                     for t, tau in enumerate(taus)}}
    gated, gcf = {}, {}
    for r in ("r1", "aff"):
        gated[r], gcf[r] = {}, {}
        for t in range(len(taus)):
            gated[r][t] = {c: {d: torch.as_tensor(gates[r][t][c])[:, None] * zT[c][d] for d in R.DIRS}
                           for c in R.CONDS}
            m = {d: (0.5 * (gated[r][t]["a"][d].numpy().astype(np.float64)
                            + gated[r][t]["b"][d].numpy().astype(np.float64))).astype(np.float32) for d in R.DIRS}
            gcf[r][t] = {c: {d: torch.as_tensor(m[d].copy()) for d in R.DIRS} for c in R.CONDS}
    zB = S.zdict(scores["B"])
    for key, (c0, c1) in S.CELLS.items():
        reader = key.split("_")[0]
        by = {}
        for h, cell in ((0, c0), (1, c1)):
            t, u, a = S.cell_params(cell)
            term = gated[reader][t] if key.endswith("fused") else gcf[reader][t]
            by[h] = S.combine(zB, term, u, a)
            if key.endswith("cf") and not S.condition_free(by[h]):
                raise AssertionError(f"{key} cell {cell} is not condition-free")
        scores[key] = R.by_parity(by, parity)

    term = rca_term(SimpleNamespace(img=img_u, txt=txt_u), ep, basis)
    lam = frozen["rca_lambda"]
    scores["rca"] = R.by_parity({h: fused_scores(cos, term, lam[h]) for h in (0, 1)}, parity)

    pa = {k: R.metrics(scores[k]) for k in SCORERS}
    for k in ("cosine", "B", "B0", "B1", "aff_cf", "r1_cf"):
        if not (pa[k]["gain"] == 0).all():
            raise AssertionError(f"{k}: condition-free scorer with non-zero gain")
    extra = {f"{r}_gate__{c}": np.stack([gates[r][t][c] for t in range(len(taus))]) for r in ("aff", "r1")
             for c in R.CONDS}
    for c in R.CONDS:
        extra[f"P__{c}"], extra[f"margin__{c}"], extra[f"pick__{c}"] = Pc[c], marg[c], pick[c]
    return pa, extra, parity


def episodes_from(z, prefix):
    parts = []
    for a, b, _ in R.PAIRS:
        parts.append(AspectEpisodes(a, b, *(z[f"{prefix}{a}__{b}__{f}"].astype(np.int64) for f in R.FIELDS)))
    return parts


def main():
    rec = {"time_start": R.now_ams()}
    data, sp, labels = R.load_data()
    R.check_split(sp)
    ctx_sel = R.SelCtx(data, sp)
    hctx = HeldCtx(data, sp)
    st = np.asarray(sp.scorer_train)
    R.log("data, selection and held contexts")

    clf, post_sel, rec["heads"] = refit_heads(data, sp, ctx_sel)
    R.log(f"heads: bitwise {rec['heads']['all_bitwise_equal']} ({rec['heads']['n_bitwise']} checks)")
    post_held = held_posteriors(data, hctx, clf)
    R.log("held posteriors")

    # frozen picks (rule §5 item 4): B, B0, B1 from phase 1's own seed-42 cross-fit (equal to the runner's and the
    # rule's mean R@1); cells from the rule; RCA λ from baselines_seed42.json (phase 1's own λ cross-fit equal)
    p1 = json.loads((R.RD / "rd_seed42.json").read_text())
    base = json.loads(R.need(R.E1 / "baselines_seed42.json").read_text())
    lp = base["scorers"]["rca"]["lambda_picks"]
    frozen = {"B_picks": {b: {h: tuple(float(x) for x in p1["B_picks"][b]["picks"][str(h)]) for h in (0, 1)}
                          for b in ("B", "B0", "B1")},
              "rca_lambda": {h: (float("inf") if lp[str(h)] == "inf" else float(lp[str(h)])) for h in (0, 1)},
              "cells": {k: {h: S.CELLS[k][h] for h in (0, 1)} for k in S.CELLS},
              "sigma_star": p1["sigma_star"]}
    if not all(p1["B_means_exact"].values()) or not p1["lambda"]["rca"]["equal"]:
        raise AssertionError("phase-1 picks not confirmed")
    rule_cells = {"aff_fused": (39, 119), "aff_cf": (149, 10), "r1_fused": (116, 119), "r1_cf": (58, 123)}
    if {k: tuple(v) for k, v in S.CELLS.items()} != rule_cells:
        raise AssertionError("cells differ from the rule's table")
    rec["frozen"] = {"B_picks": {b: {f"tune_half_{h}_scores_parity_{1 - h}": list(v[h]) for h in (0, 1)}
                                 for b, v in frozen["B_picks"].items()},
                     "rca_lambda": {f"tune_half_{h}_scores_parity_{1 - h}": frozen["rca_lambda"][h] for h in (0, 1)},
                     "cells": {k: {f"tune_half_{h}_scores_parity_{1 - h}": {
                         "cell": c, "tau_index": S.cell_params(c)[0], "lambda_u": S.cell_params(c)[1],
                         "lambda_a": S.cell_params(c)[2]} for h, c in v.items()} for k, v in frozen["cells"].items()},
                     "sigma_star_seed42": frozen["sigma_star"]}

    readers = rb_build.load_readers("A0", False)[0]
    taus = json.loads(R.need(R.R1DIR / "results/rc_tau.json").read_text())["taus"]
    R.need(R.T / "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt")
    fit_rows = np.random.default_rng(0).choice(st, 60000, replace=False)
    rec["fit_rows_sha256"] = R.sha_bytes(np.ascontiguousarray(fit_rows, dtype=np.int64))
    if rec["fit_rows_sha256"] != base["fit_rows_sha256"]:
        raise AssertionError("fit rows")
    fi, ft = (np.asarray(x[fit_rows], np.float32) for x in (data.img_features, data.txt_features))
    fi, ft = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (fi, ft))
    basis = fit_pca_basis(fi, ft)
    _ = fit_pair_scaler  # PM scorers are not computed before the verdict (rule §8.3)
    R.log("readers, taus, RCA basis")

    # regression in selection mode on seed 42 against phase 1's arrays
    z42 = np.load(R.OUT / "rd_episodes_seed42.npz")
    parts = episodes_from(z42, "")
    for ep_ in parts:
        if episodes_sha256(ep_) != base["episodes_sha256"][f"{ep_.aspect_a}__{ep_.aspect_b}"]:
            raise AssertionError("seed-42 episode hash")
    inputs_sel = rc.model_inputs(ctx_sel, "A3", st, False)[0]
    pa42, ex42, par42 = score_block(ctx_sel, post_sel, R.concat(parts), inputs_sel, frozen, basis, readers, taus)
    del inputs_sel
    a42 = np.load(R.OUT / "rd_arrays_seed42.npz")
    reg = {f"{k}__{m}": bits_equal(pa42[k][m], a42[f"{k}__{m}"]) for k in SCORERS for m in R.METRICS}
    reg.update({k: bits_equal(v, a42[k]) for k, v in ex42.items() if k in a42.files})
    reg["parity"] = bool(np.array_equal(par42, a42["parity"]))
    rec["regression_seed42"] = {"checks": reg, "n": len(reg), "all_equal": all(reg.values())}
    if not rec["regression_seed42"]["all_equal"]:
        raise AssertionError(f"selection-mode regression failed: {[k for k, v in reg.items() if not v]}")
    R.log(f"seed-42 regression: {len(reg)} arrays equal phase 1's")

    # held seeds
    zh = np.load(R.OUT / "rd_held_episodes.npz")
    eprec = json.loads((R.RD / "rd_held_episodes.json").read_text())
    inputs_held = rc.model_inputs(hctx, "A3", st, False)[0]
    R.log("held A3 codes")
    pooled = {f"{k}__{m}": [] for k in SCORERS for m in R.METRICS}
    extras = {}
    cl, pair_index, seed_index, parity_all = [], [], [], []
    rec["held_blocks"] = {}
    for i, s in enumerate(HELD_SEEDS):
        parts = episodes_from(zh, f"seed{s}__")
        for ep_ in parts:
            k = f"{ep_.aspect_a}__{ep_.aspect_b}"
            if episodes_sha256(ep_) != eprec["seeds"][str(s)]["episodes_sha256"][k]:
                raise AssertionError("held episode hash")
            if not hctx.in_rows[ep_.rows()].all():
                raise AssertionError("an episode row outside held rows")
        ep = R.concat(parts)
        pa, ex, par = score_block(hctx, post_held, ep, inputs_held, frozen, basis, readers, taus)
        for k in SCORERS:
            for m in R.METRICS:
                pooled[f"{k}__{m}"].append(pa[k][m])
        for k, v in ex.items():
            extras.setdefault(k, []).append(v)
        n = len(ep.anchor)
        cl.append(hctx.groups[ep.anchor])
        pair_index.append(np.repeat(np.arange(3), n // 3))
        seed_index.append(np.full(n, i, dtype=np.int64))
        parity_all.append(par)
        rec["held_blocks"][str(s)] = {"n_episodes": int(n), "all_rows_held": True}
        R.log(f"held seed {s} scored")
    arrays = {k: np.concatenate(v).astype(np.float64) for k, v in pooled.items()}
    arrays.update({k: np.concatenate(v, axis=1 if "_gate__" in k else 0) for k, v in extras.items()})
    arrays["cl"] = np.concatenate(cl).astype(np.int64)
    arrays["pair_index"] = np.concatenate(pair_index).astype(np.int64)
    arrays["seed_index"] = np.concatenate(seed_index).astype(np.int64)
    arrays["parity"] = np.concatenate(parity_all).astype(np.int64)
    np.savez(R.OUT / "rd_held_arrays.npz", **arrays)
    rec["arrays_npz"] = str((R.OUT / "rd_held_arrays.npz").relative_to(R.ROOT))
    rec["arrays_npz_sha256"] = R.sha_file(R.OUT / "rd_held_arrays.npz")
    rec["array_sha256"] = {k: R.sha_bytes(v) for k, v in arrays.items()}
    rec["array_shapes"] = {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()}
    rec["n_pooled"] = int(len(arrays["cl"]))
    rec["time_end"] = R.now_ams()
    R.write_json(R.RD / "rd_held_scores.json", rec)
    R.log(f"pooled arrays written ({rec['n_pooled']} episodes)")


if __name__ == "__main__":
    main()
