"""Re-derive CLIP R@1 on the aspect episodes and add a label-supervised probe ceiling (diagnostic only)."""
import importlib.util, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
ROOT = Path("/project/CoSiR"); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ra", ROOT / "src/test/20261018_affect_factor_learning/run_affect.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
from src.data.artelingo import load_artelingo
data = load_artelingo(); cache, *_ = ra.grid.load_grid()
st, sl = cache["scorer_train"], cache["selection"]
ep = np.load(ROOT / "src/test/20261023_aspect_episode_spike/results/aspect_episodes.npz")
A, C = ep["anchor"], ep["cands"]
rows = np.unique(np.concatenate([A, C.ravel()]))
assert np.isin(rows, sl).all()
emo, sty = np.asarray(data.emotions), np.asarray(data.art_styles)
# independent constraint check on candidates
e, s = emo[A], sty[A]
assert (emo[C[:, 0]] == e).all() and (sty[C[:, 0]] != s).all()
assert (sty[C[:, 1]] == s).all() and (emo[C[:, 1]] != e).all()
assert (sty[C[:, 2:]] != s[:, None]).all() and (emo[C[:, 2:]] != e[:, None]).all()
def nrm(x): x = np.asarray(x, np.float32); return x / np.linalg.norm(x, axis=1, keepdims=True)
img, txt = data.img_features, data.txt_features
def feats(F, idx): return nrm(F[idx])
r1 = lambda S, col: float(np.mean((np.delete(S, col, axis=1) < S[:, [col]]).all(axis=1)) * 100)
qi, qt = feats(img, A), feats(txt, A)
ci = nrm(img[C.ravel()]).reshape(*C.shape, -1); ct = nrm(txt[C.ravel()]).reshape(*C.shape, -1)
cos_i2t, cos_t2i = np.einsum("nd,nkd->nk", qi, ct), np.einsum("nd,nkd->nk", qt, ci)
for name, col in (("emotion", 0), ("style", 1)):
    print(f"CLIP {name}: i2t {r1(cos_i2t, col):.2f} t2i {r1(cos_t2i, col):.2f}")
rng = np.random.default_rng(0); tr = rng.choice(st, 60000, replace=False)
probes = {}
for lab_name, lab in (("emotion", emo), ("style", sty)):
    for mod, F in (("img", img), ("txt", txt)):
        clf = LogisticRegression(C=1.0, max_iter=300).fit(nrm(F[tr]), lab[tr])
        acc = clf.score(nrm(F[sl]), lab[sl])
        probes[(lab_name, mod)] = clf
        print(f"probe {lab_name} from {mod}: selection accuracy {100*acc:.1f}")
def post(lab_name, mod, idx): return probes[(lab_name, mod)].predict_proba(nrm((img if mod == "img" else txt)[idx]))
for name, col in (("emotion", 0), ("style", 1)):
    P = {m: post(name, m, C.ravel()).reshape(*C.shape, -1) for m in ("img", "txt")}
    Q = {m: post(name, m, A) for m in ("img", "txt")}
    cross_i2t = np.einsum("nc,nkc->nk", Q["img"], P["txt"]); cross_t2i = np.einsum("nc,nkc->nk", Q["txt"], P["img"])
    same_ii = np.einsum("nc,nkc->nk", Q["img"], P["img"]); same_tt = np.einsum("nc,nkc->nk", Q["txt"], P["txt"])
    print(f"CEILING {name}: cross i2t {r1(cross_i2t, col):.2f} cross t2i {r1(cross_t2i, col):.2f} | "
          f"same img-img {r1(same_ii, col):.2f} same txt-txt {r1(same_tt, col):.2f}")
