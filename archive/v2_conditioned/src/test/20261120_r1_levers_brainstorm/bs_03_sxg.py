"""Exploratory, seed 42, decides nothing. Style x genre and picks: where R1 loses, what declared oracles would gain, and
whether label-free reader features can tell the s x g signature apart.

  1. R1 at its chosen cells: R@1, other-aspect rate and either rate per aspect pair x condition, fused vs counterpart,
     and by pick (labels describe only).
  2. Declared oracles (evaluation labels used; upper bounds, not methods), each in R1's tied 224-cell family:
       O_sxg_closed : gates closed in both conditions on s x g episodes (fused = counterpart = B-family there)
       O_sxg_same   : on s x g, P^a = P^b = their mean (no condition contrast there), R1's gates
       O_sxg_told   : on s x g, P = told one-hot (image, image); elsewhere R1
       O_emo_told   : on the two emotion pairs, P = told one-hot; s x g R1
       O_told_all   : P = told one-hot everywhere (gate always open: margin 1)
       O_sup18      : a supervised 18-feature reader (multinomial LR on the told grouping, trained on one parity half,
                      applied to the other): the feature ceiling of the reader inputs
       O_sup36      : the same with both conditions' features (36), i.e. joint decoding with labels
  3. Label-free separability of the s x g signature: AUC of each of the 18 condition-a features (and a few contrasts
     between them) for "s x g" against "emotion pairs", condition a and condition b.
Writes results/bs_03_sxg.json.
"""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import bs_lib as L

FEATS = [f"{h}:{f}" for h in ("affect", "image", "caption", "csd") for f in ("S", "C", "D", "sdS", "sdC", "match")]


def tied(data, P, label, gate_override=None, taus=None):
    MDs, taus_, zT, g, m = L.standard_MDs(data, P, taus)
    if gate_override is not None:
        MDs = [L.MD(zT, gate_override(g[t])) for t in range(len(g))]
    fam = L.Family(data, MDs, tied=True).stats()
    fp, cp = fam.crossfit()
    pn, pc = fam.assemble(fp, cp)
    r, _ = L.evaluate(data, pn, pc, label)
    r["fused_cells"] = [fam.cells[fp[h]] for h in (0, 1)]
    r["cf_cells"] = [fam.cf_cells[cp[h]] for h in (0, 1)]
    print(L.fmt(r), r["fused_cells"], r["cf_cells"], flush=True)
    return r, fam, fp, cp


def assembled_scores(fam, fp, cp):
    data = fam.data
    par = data.parity
    S = {"fused": {c: {d: np.empty((data.E, 13)) for d in L.DIRECTIONS} for c in L.CONDITIONS},
         "cf": {c: {d: np.empty((data.E, 13)) for d in L.DIRECTIONS} for c in L.CONDITIONS}}
    for half in (0, 1):
        ap = par != half
        t, lu, lm, ld = fam.cells[fp[half]]
        M, D = fam.MDs[t]
        s = L.score(fam.zB64, M, D, lu, lm, ld)
        t2, lu2, lm2 = fam.cf_cells[cp[half]]
        M2, _ = fam.MDs[t2]
        s2 = L.score(fam.zB64, M2, None, lu2, lm2, 0.0)
        for c in L.CONDITIONS:
            for d in L.DIRECTIONS:
                S["fused"][c][d][ap] = s[c][d][ap]
                S["cf"][c][d][ap] = s2[c][d][ap]
    return S


def first(s):
    mx = s.max(axis=1, keepdims=True)
    uniq = (s == mx).sum(axis=1) == 1
    return np.where(uniq, s.argmax(axis=1), -1)


def per_condition(data, S, picks):
    """R@1 / other / either per pair x condition (mean over directions), fused and cf, and by the condition's pick."""
    out = {}
    for j, (c, col, oth) in enumerate((("a", 0, 1), ("b", 1, 0))):
        for who in ("fused", "cf"):
            hit = np.mean([first(S[who][c][d]) == col for d in L.DIRECTIONS], axis=0)
            other = np.mean([first(S[who][c][d]) == oth for d in L.DIRECTIONS], axis=0)
            for i, p in enumerate(L.PAIRS):
                mk = data.pi == i
                out.setdefault(p, {}).setdefault(c, {})[who] = {"r1": 100 * hit[mk].mean(), "other": 100 * other[mk].mean(),
                                                                "either": 100 * (hit[mk] + other[mk]).mean()}
                for h in range(3):
                    mh = mk & (picks[c] == h)
                    out[p][c].setdefault(f"pick_{L.A0[h]}", {})[who] = {
                        "share": 100 * mh.sum() / mk.sum(), "r1": 100 * hit[mh].mean(), "either": 100 * (hit[mh] + other[mh]).mean()}
    return out


def sup_reader(data, n_feat, joint=False):
    """Declared oracle: multinomial LR predicting the told grouping, trained on one parity half, applied to the other."""
    Xa, Xb = data.feat["a"][:, :n_feat], data.feat["b"][:, :n_feat]
    if joint:
        Xa, Xb = np.hstack([Xa, Xb]), np.hstack([Xb, Xa])
    P = {c: np.zeros((data.E, 3)) for c in L.CONDITIONS}
    acc = {}
    for half in (0, 1):
        tr, te = data.parity == half, data.parity != half
        X = np.vstack([Xa[tr], Xb[tr]])
        y = np.concatenate([data.told["a"][tr], data.told["b"][tr]])
        sc = StandardScaler().fit(X)
        mdl = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), y)
        for c, Xc in (("a", Xa), ("b", Xb)):
            pr = mdl.predict_proba(sc.transform(Xc[te]))
            full = np.zeros((te.sum(), 3))
            full[:, mdl.classes_] = pr
            P[c][te] = full
    for c in L.CONDITIONS:
        acc[c] = 100 * float(np.mean(P[c].argmax(1) == data.told[c]))
    return P, acc


def main():
    t0 = time.time()
    data = L.Data()
    out = {"note": "exploratory, seed 42, decides nothing; O_* rows use evaluation labels as declared oracles"}
    P1 = data.P("R1")
    sxg = data.pi == 2
    emo = ~sxg

    # ---- 1. R1 per pair x condition
    r, fam, fp, cp = tied(data, P1, "R1")
    out["R1"] = r
    S = assembled_scores(fam, fp, cp)
    picks = {c: P1[c].argmax(1) for c in L.CONDITIONS}
    out["per_condition"] = per_condition(data, S, picks)
    for p, v in out["per_condition"].items():
        for c in L.CONDITIONS:
            f, k = v[c]["fused"], v[c]["cf"]
            print(f"  {p} cond {c}: R@1 {f['r1']:.2f} vs {k['r1']:.2f}; other {f['other']:.2f} vs {k['other']:.2f}; "
                  f"either {f['either']:.2f} vs {k['either']:.2f}  | " + "; ".join(
                      f"{h}: {v[c][h]['fused']['share']:.0f}% R@1 {v[c][h]['fused']['r1']:.1f}/{v[c][h]['cf']['r1']:.1f} "
                      f"either {v[c][h]['fused']['either']:.1f}/{v[c][h]['cf']['either']:.1f}"
                      for h in ("pick_affect", "pick_image", "pick_caption")))

    # ---- 2. declared oracles
    taus1 = L.thresholds(L.margins_of(P1))
    out["oracles"] = {}

    def close_sxg(gt):
        return {c: (gt[c] * (~sxg)).astype(np.float32) for c in L.CONDITIONS}
    out["oracles"]["O_sxg_closed"] = tied(data, P1, "O_sxg_closed (gates shut on s×g)", close_sxg, taus1)[0]

    Pm = {c: P1[c].copy() for c in L.CONDITIONS}
    mean = 0.5 * (P1["a"] + P1["b"])
    for c in L.CONDITIONS:
        Pm[c][sxg] = mean[sxg]

    def same_gate_sxg(gt):                      # also make the two gates equal on s x g (use R1's own gates' max)
        gm = np.maximum(gt["a"], gt["b"])
        return {c: np.where(sxg, gm, gt[c]).astype(np.float32) for c in L.CONDITIONS}
    out["oracles"]["O_sxg_same"] = tied(data, Pm, "O_sxg_same (P^a=P^b, equal gates on s×g)", same_gate_sxg, taus1)[0]

    def onehot(idx):
        o = np.zeros((data.E, 3))
        o[np.arange(data.E), idx] = 1.0
        return o
    Pt = {c: onehot(data.told[c]) for c in L.CONDITIONS}
    Ps = {c: np.where(sxg[:, None], Pt[c], P1[c]) for c in L.CONDITIONS}

    def open_where_told(mask):
        def f(gt):
            return {c: np.where(mask, 1.0, gt[c]).astype(np.float32) for c in L.CONDITIONS}
        return f
    out["oracles"]["O_sxg_told"] = tied(data, Ps, "O_sxg_told (told on s×g, gates open there)", open_where_told(sxg), taus1)[0]
    Pe = {c: np.where(emo[:, None], Pt[c], P1[c]) for c in L.CONDITIONS}
    out["oracles"]["O_emo_told"] = tied(data, Pe, "O_emo_told (told on emotion pairs, open)", open_where_told(emo), taus1)[0]
    out["oracles"]["O_told_all"] = tied(data, Pt, "O_told_all (told everywhere, open)",
                                        open_where_told(np.ones(data.E, bool)), taus1)[0]
    for nf, joint, nm in ((18, False, "O_sup18"), (18, True, "O_sup36_joint")):
        Ps_, acc = sup_reader(data, nf, joint)
        r_, *_ = tied(data, Ps_, f"{nm} (supervised reader, cross-fitted)")
        r_["told_accuracy"] = acc
        r_["pick_accuracy_per_pair"] = {p: {c: 100 * float(np.mean(Ps_[c][data.pi == i].argmax(1) == data.told[c][data.pi == i]))
                                            for c in L.CONDITIONS} for i, p in enumerate(L.PAIRS)}
        print("   accuracy", acc, r_["pick_accuracy_per_pair"])
        out["oracles"][nm] = r_
    # same with the csd features added (24, 48): does csd make the s x g signature readable?
    for nf, joint, nm in ((24, False, "O_sup24_with_csd_features"), (24, True, "O_sup48_joint_with_csd_features")):
        Ps_, acc = sup_reader(data, nf, joint)
        r_, *_ = tied(data, Ps_, f"{nm}")
        r_["told_accuracy"] = acc
        r_["pick_accuracy_per_pair"] = {p: {c: 100 * float(np.mean(Ps_[c][data.pi == i].argmax(1) == data.told[c][data.pi == i]))
                                            for c in L.CONDITIONS} for i, p in enumerate(L.PAIRS)}
        print("   accuracy", acc, r_["pick_accuracy_per_pair"])
        out["oracles"][nm] = r_

    # ---- 3. AUC of single label-free features: s x g vs emotion pairs (per condition)
    auc = {}
    for c in L.CONDITIONS:
        X = data.feat[c]
        auc[c] = {}
        for k, nm in enumerate(FEATS):
            a = roc_auc_score(sxg, X[:, k])
            auc[c][nm] = float(a)
        # composite: image both elevated
        S_img, C_img = X[:, 6], X[:, 7]
        auc[c]["min(S_image, C_image)"] = float(roc_auc_score(sxg, np.minimum(S_img, C_img)))
        auc[c]["S_image + C_image"] = float(roc_auc_score(sxg, S_img + C_img))
        auc[c]["csd S + C"] = float(roc_auc_score(sxg, X[:, 18] + X[:, 19]))
    out["auc_sxg_vs_emotion"] = auc
    for c in L.CONDITIONS:
        top = sorted(auc[c].items(), key=lambda kv: -abs(kv[1] - 0.5))[:8]
        print(f"  AUC s×g vs emotion pairs, condition {c}: " + ", ".join(f"{k} {v:.3f}" for k, v in top))
    # pairwise-learned separability (declared diagnostic): cross-fitted LR on the 36 / 48 features, s x g vs emotion
    for nf in (18, 24):
        Xj = np.hstack([data.feat["a"][:, :nf], data.feat["b"][:, :nf]])
        Xj_sym = np.hstack([Xj, np.hstack([data.feat["b"][:, :nf], data.feat["a"][:, :nf]])])
        pr = np.zeros(data.E)
        for half in (0, 1):
            tr, te = data.parity == half, data.parity != half
            sc = StandardScaler().fit(Xj[tr])
            mdl = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xj[tr]), sxg[tr])
            # symmetric score: average over the two orderings so the detector cannot use which side is condition a
            Xs = np.hstack([data.feat["b"][:, :nf], data.feat["a"][:, :nf]])
            pr[te] = 0.5 * (mdl.predict_proba(sc.transform(Xj[te]))[:, 1] + mdl.predict_proba(sc.transform(Xs[te]))[:, 1])
        out[f"auc_sxg_learned_{2 * nf}"] = float(roc_auc_score(sxg, pr))
        print(f"  learned (labels) s×g detector on {2 * nf} features, symmetrised, cross-fitted: AUC {out[f'auc_sxg_learned_{2 * nf}']:.3f}")
    out["runtime_s"] = round(time.time() - t0)
    (L.HERE / "results" / "bs_03_sxg.json").write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
