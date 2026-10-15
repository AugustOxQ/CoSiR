"""Exploratory, seed 42, decides nothing. CSD as evidence for the emotion detector rather than as a steering grouping.

Detectors of "supports share affect", gate g^c = 1[d^c >= q-th percentile] (q in 0, 50, 75, 90), term z(s_affect), in
R1's tied 224-cell layout (comparator floor B'(A0), since the steering term is A0's affect score only):
  R1A1aff       : round 1's A1 reader (affect, image, caption, csd) P^c(affect)
  BANK_A1_LR24  : binary LR (affect vs rest) on round 1's A1 bank features (24, csd included), per half, averaged
  BANK_A1_GB24  : the same with HistGradientBoosting
  SUP24 (oracle): LR on seed-42's 24 features with the evaluation label "condition aspect = emotion", cross-fitted
Writes results/bs_08_csd_evidence.json.
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

BANK1 = L.HERE.parent / "20261117_reader_fix_csd" / "results" / "rb_reader_A1.npz"
PCTS = (0, 50, 75, 90)


def bank_detector(kind):
    z = np.load(BANK1)
    models = []
    for j in (0, 1):
        X, y = z[f"half{j}__X"], z[f"half{j}__y"]
        yb = (y == 0).astype(int)
        sc = StandardScaler().fit(X)
        if kind == "lr":
            mdl = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), yb)
        else:
            mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(sc.transform(X), yb)
        models.append((sc, mdl))
    return models


def main():
    t0 = time.time()
    data = L.Data()
    out = {"note": "exploratory, seed 42, decides nothing; SUP24 uses evaluation labels as a declared oracle"}
    emo = {c: np.array([L.ASPECTS[i][j] == "emotion" for i in data.pi]) for j, c in enumerate(L.CONDITIONS)}
    yall = np.concatenate([emo["a"], emo["b"]])
    det = {"R1A1aff": {c: data.P("R1A1")[c][:, 0] for c in L.CONDITIONS}}
    for kind in ("lr", "gb"):
        mods = bank_detector(kind)
        det[f"BANK_A1_{kind.upper()}24"] = {c: np.mean([m.predict_proba(s.transform(data.feat[c]))[:, 1] for s, m in mods], 0)
                                           for c in L.CONDITIONS}
    pr = {c: np.zeros(data.E) for c in L.CONDITIONS}
    for half in (0, 1):
        tr, te = data.parity == half, data.parity != half
        X = np.vstack([data.feat["a"][tr], data.feat["b"][tr]])
        y = np.concatenate([emo["a"][tr], emo["b"][tr]])
        sc = StandardScaler().fit(X)
        mdl = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), y)
        for c in L.CONDITIONS:
            pr[c][te] = mdl.predict_proba(sc.transform(data.feat[c][te]))[:, 1]
    det["SUP24_oracle"] = pr
    zA = {c: {d: L.zrows(data.stack[d][:, 0]) for d in L.DIRECTIONS} for c in L.CONDITIONS}
    res = {}
    for nm, d in det.items():
        allv = np.concatenate([d["a"], d["b"]])
        auc = float(roc_auc_score(yall, allv))
        ths = np.percentile(allv, PCTS)
        gl = [{c: (d[c] >= t).astype(np.float32) for c in L.CONDITIONS} for t in ths]
        r = run_tied(data, [L.MD(zA, gg) for gg in gl], f"{nm} gate, z(s_affect) term")[0]
        r["auc_emotion"] = auc
        med = ths[1]
        r["open_share_at_median_per_pair_condition"] = {
            p: {c: round(100 * float(np.mean(d[c][data.pi == i] >= med)), 1) for c in L.CONDITIONS}
            for i, p in enumerate(L.PAIRS)}
        print(f"   {nm}: AUC {auc:.3f}; open at median (a/b): "
              + "; ".join(f"{p} {v['a']}/{v['b']}" for p, v in r["open_share_at_median_per_pair_condition"].items()),
              flush=True)
        res[nm] = r
    out["results"] = res
    out["runtime_s"] = round(time.time() - t0)
    (L.HERE / "results" / "bs_08_csd_evidence.json").write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
