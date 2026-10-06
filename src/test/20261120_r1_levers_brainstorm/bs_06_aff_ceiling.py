"""Exploratory, seed 42, decides nothing. How far is the affect-only gate (AFF) from what an emotion detector could do,
and do softer label-free forms of "steer only with what B lacks" behave like it?

  sanity   : gate_sets with H = all groupings reproduces R1 exactly.
  AFF_pure : AFF with the term replaced by z(s_affect) where the gate is open (no mixture).
  AFF_R2/R3: the affect-only gate on R2's and R3's probabilities.
  RESID    : T^c = sum_h P^c(h) r_h with r_h = z(s_h) minus its per-row least-squares fit on z(B) (label-free).
  SOFTW    : T^c = sum_h P^c(h) (1 - rho_h) z(s_h), rho_h = the seed-42 mean row correlation of z(s_h) and z(B).
  Declared oracles (evaluation labels):
  O_emo_gate     : gate open iff the condition's aspect is emotion, R1's term.
  O_emo_gate_aff : the same with z(s_affect) as the term.
  O_sup_emo{18,24}: a supervised binary detector "the supports show emotion" (LR on the 18 or 24 reader features,
                   trained on one parity half, applied to the other); gate open iff its probability >= the t-th
                   percentile (t = 0, 25, 50, 75) of its own values; R1's term.
Writes results/bs_06_aff_ceiling.json.
"""
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import bs_lib as L
from bs_05_aff import gate_sets, run_tied


def main():
    t0 = time.time()
    data = L.Data()
    out = {"note": "exploratory, seed 42, decides nothing; O_* rows use evaluation labels as declared oracles"}
    P1 = data.P("R1")
    MDs, taus, zT, g, m = L.standard_MDs(data, P1)
    res = {}
    r_all = run_tied(data, [L.MD(zT, gg) for gg in gate_sets(P1, g, [0, 1, 2])], "sanity: H = all")
    res["sanity_equals_R1"] = bool(abs(r_all[0]["fused_r1"] - 18.918863932291664) < 1e-9
                                  and abs(r_all[0]["bar"]["point"] - 0.4435221354166667) < 1e-9)
    print("sanity equals R1:", res["sanity_equals_R1"])

    # AFF with a pure affect term
    zA = {c: {d: L.zrows(data.stack[d][:, 0]) for d in L.DIRECTIONS} for c in L.CONDITIONS}
    res["AFF_pure"] = run_tied(data, [L.MD(zA, gg) for gg in gate_sets(P1, g, [0])], "AFF_pure: z(s_affect) where open")[0]
    for nm in ("R2", "R3"):
        Pn = data.P(nm)
        MDn, _, zTn, gn, _ = L.standard_MDs(data, Pn)
        res[f"AFF_{nm}"] = run_tied(data, [L.MD(zTn, gg) for gg in gate_sets(Pn, gn, [0])], f"AFF on {nm}")[0]

    # residual-to-B and soft-weighted terms
    zB = {d: data.zB["a"][d].numpy().astype(np.float64) for d in L.DIRECTIONS}
    zS = {d: np.stack([L.zrows(data.stack[d][:, h]).numpy().astype(np.float64) for h in range(3)], 1) for d in L.DIRECTIONS}
    resid = {}
    rho = np.zeros(3)
    for d in L.DIRECTIONS:
        b = zB[d][:, None, :]
        beta = (zS[d] * b).sum(-1, keepdims=True) / np.maximum((b * b).sum(-1, keepdims=True), 1e-12)
        resid[d] = (zS[d] - beta * b).astype(np.float32)
        rho += 0.5 * np.mean((zS[d] * b).sum(-1) / 13.0, axis=0)
    out["rho"] = rho.tolist()
    for nm, stk in (("RESID", resid), ("SOFTW", {d: (zS[d] * (1 - rho)[None, :, None]).astype(np.float32) for d in L.DIRECTIONS})):
        T = L.term(stk, P1)
        zTx = L.zterm(T)
        res[nm] = run_tied(data, [L.MD(zTx, g[t]) for t in range(4)], f"{nm} (R1's gates)")[0]

    # declared oracles
    emo = {c: np.array([L.ASPECTS[i][j] == "emotion" for i in data.pi]) for j, c in enumerate(L.CONDITIONS)}
    ge = [{c: emo[c].astype(np.float32) for c in L.CONDITIONS}]
    res["O_emo_gate"] = run_tied(data, [L.MD(zT, gg) for gg in ge], "O_emo_gate (open iff emotion condition)")[0]
    res["O_emo_gate_aff"] = run_tied(data, [L.MD(zA, gg) for gg in ge], "O_emo_gate_aff (z s_affect iff emotion)")[0]
    for nf in (18, 24):
        pr = {c: np.zeros(data.E) for c in L.CONDITIONS}
        for half in (0, 1):
            tr, te = data.parity == half, data.parity != half
            X = np.vstack([data.feat["a"][tr, :nf], data.feat["b"][tr, :nf]])
            y = np.concatenate([emo["a"][tr], emo["b"][tr]])
            sc = StandardScaler().fit(X)
            mdl = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), y)
            for c in L.CONDITIONS:
                pr[c][te] = mdl.predict_proba(sc.transform(data.feat[c][te, :nf]))[:, 1]
        allp = np.concatenate([pr["a"], pr["b"]])
        auc = roc_auc_score(np.concatenate([emo["a"], emo["b"]]), allp)
        auc_r1 = roc_auc_score(np.concatenate([emo["a"], emo["b"]]), np.concatenate([P1["a"][:, 0], P1["b"][:, 0]]))
        ths = np.percentile(allp, [0, 25, 50, 75])
        gl = [{c: (pr[c] >= t).astype(np.float32) for c in L.CONDITIONS} for t in ths]
        r = run_tied(data, [L.MD(zT, gg) for gg in gl], f"O_sup_emo{nf} (supervised emotion gate)")[0]
        r["auc_emotion_detector"] = auc
        r["auc_R1_P_affect"] = auc_r1
        print(f"   AUC supervised emotion detector ({nf} features) {auc:.3f}; R1's P(affect) as detector {auc_r1:.3f}")
        res[f"O_sup_emo{nf}"] = r
    out["results"] = res
    out["runtime_s"] = round(time.time() - t0)
    (L.HERE / "results" / "bs_06_aff_ceiling.json").write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
