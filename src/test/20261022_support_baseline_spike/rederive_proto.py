"""Independent re-derivation of proto (lam=4) and CLIP-only R@1 on the selection episodes."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/project/CoSiR"); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ra", ROOT / "src/test/20261018_affect_factor_learning/run_affect.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
from src.data.artelingo import load_artelingo
data = load_artelingo(); cache, *_ = ra.grid.load_grid()
eps, _, _ = ra.selection_episodes(data, cache)
sl = cache["selection"]
def norm(x):
    out = np.full(x.shape, np.nan, np.float32); v = x[sl]; out[sl] = v / np.linalg.norm(v, axis=1, keepdims=True); return out
img, txt = norm(np.asarray(data.img_features, np.float32)), norm(np.asarray(data.txt_features, np.float32))
def z(a):
    s = a.std(axis=1, keepdims=True); return np.where(s > 0, (a - a.mean(axis=1, keepdims=True)) / np.where(s > 0, s, 1), 0)
stored = np.load(ROOT / "src/test/20261018_affect_factor_learning/results/selection_ranks.npz")
res = {}
for label, e in eps.items():
    cand = np.concatenate([e.positive[:, None], e.distractors], axis=1)
    for d, (Q, C) in {"i2t": (img, txt), "t2i": (txt, img)}.items():
        q, c = Q[e.anchor], C[cand]
        cos = np.einsum("nd,nkd->nk", q, c)
        mp, mn = C[e.supports].mean(1), C[e.contrasts].mean(1)
        mp /= np.linalg.norm(mp, axis=1, keepdims=True); mn /= np.linalg.norm(mn, axis=1, keepdims=True)
        T = np.einsum("nkd,nd->nk", c, mp) - np.einsum("nkd,nd->nk", c, mn)
        for name, s in (("clip", cos), ("proto4", z(cos) + 4 * z(T)), ("proto_only", T)):
            hit = (s[:, 1:] < s[:, :1]).all(axis=1)          # strict win; ties count as miss here
            res[(name, label, d)] = hit
        se = stored[f"naive__SE__0.3__{label}__{d}"] <= 1
        res[("SE", label, d)] = se
for name in ("clip", "proto4", "proto_only", "SE"):
    per = {l: 50 * (res[(name, l, "i2t")].mean() + res[(name, l, "t2i")].mean()) for l in eps}
    pooled = 50 * np.mean([np.concatenate([res[(name, l, d)] for l in eps]).mean() for d in ("i2t", "t2i")]) * 2 / 2
    print(f"{name:11s} pooled {np.mean(list(per.values())):.2f}  emotion {per['emotion']:.2f}  style {per['art_style']:.2f}")

from src.eval.condition_eval import paired_bootstrap
def per_ep(name, label):
    return 0.5 * (res[(name, label, "i2t")].astype(float) + res[(name, label, "t2i")].astype(float))
out = {}
for a, b in (("proto_only", "SE"), ("proto4", "proto_only"), ("proto4", "SE")):
    for scope in ("emotion", "art_style", "pooled"):
        labels = list(eps) if scope == "pooled" else [scope]
        diff = np.concatenate([per_ep(a, l) - per_ep(b, l) for l in labels])
        r = paired_bootstrap(diff, 5000, 42)
        print(f"{a} - {b} [{scope}]: {100*r['point']:+.2f} [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}]")
for name in ("clip", "proto4", "proto_only", "SE"):
    for label in eps:
        print(name, label, {d: round(100 * res[(name, label, d)].mean(), 2) for d in ("i2t", "t2i")})
