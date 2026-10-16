"""Check 1 probe: 5-fold painting-grouped logistic regression, caption emotion from t, r_mean, r_proj, CLIP image v."""
import sys, json, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from apc_data import *

S = load_sample(min_caps=3)
t, v, pid, P, emo = S["t"], S["v"], S["pid"], S["n_paint"], S["emo"]
m = paint_mean(t, pid, P)
feats = {"t": t, "r_mean": l2(t - m[pid]), "r_proj": l2(t - (t * v).sum(1, keepdims=True) * v), "v_clip_image": v}
folds = list(GroupKFold(5).split(t, emo, pid))

def fit(name, X, i):
    tr, te = folds[i]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=300).fit((X[tr] - mu) / sd, emo[tr])
    return name, i, float((clf.predict((X[te] - mu) / sd) == emo[te]).mean())

t0 = time.time()
out = Parallel(n_jobs=5)(delayed(fit)(n, X, i) for n, X in feats.items() for i in range(5))
res = {"n_captions": int(len(t)), "majority_class_rate": float(np.bincount(emo).max() / len(emo)),
       "classifier": "StandardScaler + LogisticRegression(C=0.1, max_iter=300), GroupKFold(5) by painting", "acc": {}}
for n in feats:
    a = [x[2] for x in out if x[0] == n]
    res["acc"][n] = [float(np.mean(a)), float(np.std(a))]
print(res, time.time() - t0)
json.dump(res, open(HERE / "results_check1_probe.json", "w"), indent=1)
