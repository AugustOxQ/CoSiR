"""Exploratory, seed 42, decides nothing. The affect-only gate reframes the reader as a binary emotion detector: "do the
support pairs share the affect grouping?". How good are label-free detectors, and what does detection quality buy?

Detectors of "supports share affect" (each gives d^c in [0, 1] per (episode, condition)):
  R1aff, R2aff, R3aff : the stored readers' P^c(affect)
  BANK_LR18           : binary LR (affect vs rest) on round 1's A0 bank features (18), per bank half, averaged (bank
                        labels come from the groupings, no evaluation labels)
  BANK_GB18           : the same with HistGradientBoosting
  BANK_LR36           : binary LR on the own condition's 18 features plus the other condition's 18 (joint view)
  SUP18 (oracle)      : LR on seed-42 features with the evaluation label "condition aspect = emotion", cross-fitted on
                        parity halves: the feature ceiling for these 18 inputs
For each: AUC for emotion conditions (diagnostic), and the gated family with gate g^c = 1[d^c >= q-th percentile of d]
(q in 0, 50, 75, 90; chosen by the cross-fit like tau) and term z(s_affect) (pure) in R1's tied 224-cell layout.
Writes results/bs_07_detector.json.
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import bs_lib as L
from bs_05_aff import run_tied

BANK = L.HERE.parent / "20261117_reader_fix_csd" / "results" / "rb_reader_A0.npz"
PCTS = (0, 50, 75, 90)


def bank_detector(kind, joint=False):
    z = np.load(BANK)
    models = []
    for j in (0, 1):
        X, y, ep = z[f"half{j}__X"], z[f"half{j}__y"], z[f"half{j}__episode"]
        if joint:
            # rows: condition a's N episodes then condition b's (rb_features.stack_conditions); pair each row with
            # the other condition's row of the same episode
            N = len(X) // 2
            other = np.concatenate([np.arange(N, 2 * N), np.arange(0, N)])
            assert np.array_equal(ep[:N], ep[N:])
            X = np.hstack([X, X[other]])
        yb = (y == 0).astype(int)
        sc = StandardScaler().fit(X)
        if kind == "lr":
            mdl = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), yb)
        else:
            mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(sc.transform(X), yb)
        models.append((sc, mdl))
    return models


def apply_bank(models, data, joint=False):
    out = {}
    for c, o in (("a", "b"), ("b", "a")):
        X = data.feat[c][:, :18]
        if joint:
            X = np.hstack([X, data.feat[o][:, :18]])
        out[c] = np.mean([mdl.predict_proba(sc.transform(X))[:, 1] for sc, mdl in models], axis=0)
    return out


def main():
    t0 = time.time()
    data = L.Data()
    out = {"note": "exploratory, seed 42, decides nothing; SUP18 uses evaluation labels as a declared oracle"}
    emo = {c: np.array([L.ASPECTS[i][j] == "emotion" for i in data.pi]) for j, c in enumerate(L.CONDITIONS)}
    yall = np.concatenate([emo["a"], emo["b"]])
    det = {}
    for nm in ("R1", "R2", "R3"):
        P = data.P(nm)
        det[f"{nm}aff"] = {c: P[c][:, 0] for c in L.CONDITIONS}
    det["BANK_LR18"] = apply_bank(bank_detector("lr"), data)
    det["BANK_GB18"] = apply_bank(bank_detector("gb"), data)
    det["BANK_LR36"] = apply_bank(bank_detector("lr", joint=True), data, joint=True)
    pr = {c: np.zeros(data.E) for c in L.CONDITIONS}
    for half in (0, 1):
        tr, te = data.parity == half, data.parity != half
        X = np.vstack([data.feat["a"][tr, :18], data.feat["b"][tr, :18]])
        y = np.concatenate([emo["a"][tr], emo["b"][tr]])
        sc = StandardScaler().fit(X)
        mdl = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), y)
        for c in L.CONDITIONS:
            pr[c][te] = mdl.predict_proba(sc.transform(data.feat[c][te, :18]))[:, 1]
    det["SUP18_oracle"] = pr
    zA = {c: {d: L.zrows(data.stack[d][:, 0]) for d in L.DIRECTIONS} for c in L.CONDITIONS}
    res = {}
    for nm, d in det.items():
        allv = np.concatenate([d["a"], d["b"]])
        auc = float(roc_auc_score(yall, allv))
        # share of open values (at the median threshold) that are emotion conditions, per pair x condition
        ths = np.percentile(allv, PCTS)
        gl = [{c: (d[c] >= t).astype(np.float32) for c in L.CONDITIONS} for t in ths]
        r = run_tied(data, [L.MD(zA, gg) for gg in gl], f"{nm} gate, z(s_affect) term")[0]
        r["auc_emotion"] = auc
        med = ths[1]
        r["open_share_at_median_per_pair_condition"] = {
            p: {c: round(100 * float(np.mean(d[c][data.pi == i] >= med)), 1) for c in L.CONDITIONS}
            for i, p in enumerate(L.PAIRS)}
        print(f"   {nm}: AUC (emotion condition) {auc:.3f}; open share at median per pair (a/b): "
              + "; ".join(f"{p} {v['a']}/{v['b']}" for p, v in r["open_share_at_median_per_pair_condition"].items()),
              flush=True)
        res[nm] = r
    out["results"] = res
    out["runtime_s"] = round(time.time() - t0)
    (L.HERE / "results" / "bs_07_detector.json").write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
