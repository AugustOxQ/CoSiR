"""Check 1: within-painting caption residual (clustering AMI + probe)."""
import sys, json, time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from apc_data import *

S = load_sample(min_caps=3)
t, v, pid, P = S["t"], S["v"], S["pid"], S["n_paint"]
emo, gen_c, sty_c = S["emo"], S["gen"], S["sty"]
gen_p, sty_p = paint_first(gen_c, pid, P), paint_first(sty_c, pid, P)
emo_p = paint_majority(emo, pid, P, 9)
cap_per_p = np.bincount(pid)
m = paint_mean(t, pid, P)
r_mean = l2(t - m[pid])           # t is unit-norm; subtract the painting mean of t, re-normalise
r_proj = l2(t - (t * v).sum(1, keepdims=True) * v)
feats = {"a_t": t, "b_r_mean": r_mean, "b2_r_proj": r_proj, "c_clip_image": v}
res = {"n_captions": int(len(t)), "n_paintings": int(P), "captions_per_painting_mean": float(cap_per_p.mean()),
       "min_captions_per_painting": int(cap_per_p.min()),
       "emotion_majority_rate_caption": float(np.bincount(emo).max() / len(emo)),
       "genre_labelled_paintings": int((gen_p >= 0).sum()), "style_labelled_paintings": int((sty_p >= 0).sum()),
       "n_genre_classes": int(len(np.unique(gen_p[gen_p >= 0]))), "n_style_classes": int(len(np.unique(sty_p[sty_p >= 0]))),
       "cos(t, r_mean) mean": float((t * r_mean).sum(1).mean()), "cluster": {}}
gm = gen_p >= 0
for name, X in feats.items():
    for k in (9, 20, 40):
        a_emo, a_gen, a_sty, a_emop = [], [], [], []
        for seed in (0, 1, 2):
            lab = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit_predict(X)
            a_emo.append(ami(emo, lab))
            pl = paint_majority(lab, pid, P, k)
            a_emop.append(ami(emo_p, pl))
            a_gen.append(ami(gen_p[gm], pl[gm]))
            a_sty.append(ami(sty_p, pl))
        res["cluster"][f"{name}_k{k}"] = {
            "caption_AMI_emotion": [float(np.mean(a_emo)), float(np.std(a_emo))],
            "painting_AMI_emotion_majority": [float(np.mean(a_emop)), float(np.std(a_emop))],
            "painting_AMI_genre": [float(np.mean(a_gen)), float(np.std(a_gen))],
            "painting_AMI_style": [float(np.mean(a_sty)), float(np.std(a_sty))]}
        print(name, k, res["cluster"][f"{name}_k{k}"], flush=True)
json.dump(res, open(HERE / "results_check1_cluster.json", "w"), indent=1)
