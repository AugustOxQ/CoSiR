"""Check 2: painting-level k-means (k=20) on four views; AMI to labels and pairwise AMI."""
import sys, json, itertools
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from apc_data import *

S = load_sample(min_caps=1)
pid, P, rows = S["pid"], S["n_paint"], S["rows"]
first = np.full(P, -1); first[pid[::-1]] = np.arange(len(pid))[::-1]   # first caption index of each painting
fr = rows[first]
img_idx = {}
for s in ("style_csd_vitl", "style_vgg19_gram"):
    r2i = np.load(STYLE_ROOT / s / "row_to_image.npy")
    emb = np.load(STYLE_ROOT / s / "embeddings.npy", mmap_mode="r")
    ii = r2i[rows]
    # consistency: all captions of a painting map to one image; distinct paintings to distinct images
    assert (ii == ii[first][pid]).all(), s
    assert len(np.unique(ii[first])) == P, s
    img_idx[s] = np.asarray(emb[ii[first]], dtype=np.float32)
views = {"clip_image": S["v"][first], "mean_clip_text": l2(paint_mean(S["t"], pid, P)),
         "csd_style": l2(img_idx["style_csd_vitl"]), "vgg_gram": l2(img_idx["style_vgg19_gram"])}
emo_p = paint_majority(S["emo"], pid, P, 9)
gen_p, sty_p = paint_first(S["gen"], pid, P), paint_first(S["sty"], pid, P)
gm = gen_p >= 0
K = 20
res = {"n_captions": int(len(rows)), "n_paintings": int(P), "n_genre_labelled": int(gm.sum()), "k": K,
       "n_style_classes": int(len(np.unique(sty_p))), "label_AMI": {}, "pairwise_AMI_seed0": {}, "same_view_seed0_vs_seed1": {}}
labs = {}
for n, X in views.items():
    ls = [MiniBatchKMeans(K, random_state=s, n_init=3, batch_size=4096).fit_predict(X) for s in (0, 1, 2)]
    labs[n] = ls
    res["label_AMI"][n] = {
        "emotion_majority": float(np.mean([ami(emo_p, l) for l in ls])),
        "genre": float(np.mean([ami(gen_p[gm], l[gm]) for l in ls])),
        "style": float(np.mean([ami(sty_p, l) for l in ls]))}
    res["same_view_seed0_vs_seed1"][n] = float(ami(ls[0], ls[1]))
    print(n, res["label_AMI"][n], flush=True)
for a, b in itertools.combinations(views, 2):
    res["pairwise_AMI_seed0"][f"{a}__{b}"] = float(ami(labs[a][0], labs[b][0]))
res["label_AMI"]["_chance_note"] = "emotion_majority uses 9 classes incl. 'something else'; ties -> smallest id; mean of 3 k-means seeds"
print(json.dumps(res["pairwise_AMI_seed0"], indent=1))
json.dump(res, open(HERE / "results_check2.json", "w"), indent=1)
