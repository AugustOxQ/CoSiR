"""Check 4: caption-level emotion AMI of k-means on cached GoEmotions 28-d probabilities (archive caches), same sample as check 1."""
import sys, json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from apc_data import *

A = Path("/project/CoSiR/archive/v2_conditioned/src/test")
S = load_sample(min_caps=3)
sp = S["sp"]; n = len(S["d"].sample_ids)
P = np.full((n, 28), np.nan, dtype=np.float32)
z = np.load(A / "20261018_affect_factor_learning/cache/affect_prepare.npz")
assert len(z["affect_probs"]) == len(sp.scorer_train)
P[np.sort(sp.scorer_train)] = z["affect_probs"]       # stored in ascending scorer_train row order (asserted in r5_goemo.regression_sample)
g = np.load(A / "20261123_idea3_goemotions/cache/r5_goemotions_selection.npz")
assert np.array_equal(g["rows"], np.sort(sp.selection)) and np.array_equal(g["sample_ids"], S["d"].sample_ids[g["rows"]])
P[g["rows"]] = g["probs"]
X = P[S["rows"]]
assert np.isfinite(X).all()
emo, pid, Pn = S["emo"], S["pid"], S["n_paint"]
emo_p = paint_majority(emo, pid, Pn, 9)
res = {"source": "28-d GoEmotions classifier probabilities (not hidden features): archive/v2_conditioned/src/test/20261018_affect_factor_learning/cache/affect_prepare.npz (scorer_train rows) + 20261123_idea3_goemotions/cache/r5_goemotions_selection.npz (selection rows)",
       "n_captions": int(len(X)), "n_paintings": int(Pn), "prob_row_sum_mean": float(X.sum(1).mean()), "cluster": {}}
for k in (9, 20, 40):
    a, b = [], []
    for seed in (0, 1, 2):
        lab = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit_predict(X)
        a.append(ami(emo, lab)); b.append(ami(emo_p, paint_majority(lab, pid, Pn, k)))
    res["cluster"][f"k{k}"] = {"caption_AMI_emotion": [float(np.mean(a)), float(np.std(a))], "painting_AMI_emotion_majority": [float(np.mean(b)), float(np.std(b))]}
    print(k, res["cluster"][f"k{k}"], flush=True)
json.dump(res, open(HERE / "results_check4.json", "w"), indent=1)
