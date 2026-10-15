"""Final review: cache-path vs frozen image cosines over all val and selection rows (the report writer's note), the
identity of the six LB/LoRA epoch-0 files, and whether low-cosine rows are rows whose frozen image features differ
within their painting."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_splits
JOBS = ROOT / "res/cluster_jobs"
data = load_artelingo(); sp = artelingo_splits(data)
paint = np.asarray(data.paintings)
runs = [l.split() for l in (ROOT / ".superpowers/sdd/2026-10-07-clip-lightweight-ft/runs.txt").read_text().splitlines() if l.strip()]
out = {}
digests = {}
for node, v, lr, ok, tag in [r[:5] for r in runs]:
    d = next((JOBS / tag / "code/outputs/clipft").iterdir())
    z = np.load(d / "features_epoch0.npz")
    digests[f"{v}_{lr}"] = hashlib.sha256(z["img"].tobytes() + z["txt"].tobytes() + z["rows"].tobytes()).hexdigest()[:16]
out["epoch0_digests"] = digests
z = np.load(JOBS / "20261007-071151-65bb2f4/code/outputs/clipft/LB_lr3e-5/features_epoch0.npz")
rows = z["rows"]
def ucos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True) * b / np.linalg.norm(b, axis=1, keepdims=True)).sum(1)
c_img = ucos(z["img"], data.img_features[rows]); c_txt = ucos(z["txt"], data.txt_features[rows])
# within-painting deviation of the frozen features, over val+selection rows: cosine of each row to its painting's mean
p = paint[rows]
uniq, inv = np.unique(p, return_inverse=True)
F = data.img_features[rows].astype(np.float64); F /= np.linalg.norm(F, axis=1, keepdims=True)
mean = np.zeros((len(uniq), F.shape[1])); np.add.at(mean, inv, F)
cnt = np.bincount(inv)
# max pairwise deviation proxy: min cosine of a row to the first row of its painting
first = np.zeros(len(uniq), int); first[inv[::-1]] = np.arange(len(rows))[::-1]
c_within = (F * F[first[inv]]).sum(1)
for name, m in (("selection", np.isin(rows, sp.selection)), ("val", np.isin(rows, sp.val))):
    low = m & (c_img < 0.999)
    out[name] = {"n": int(m.sum()), "img_mean": float(c_img[m].mean()), "img_min": float(c_img[m].min()),
                 "img_p0_1": float(np.quantile(c_img[m], 0.001)), "below_0_999_rows": int(low.sum()),
                 "below_0_999_paintings": int(len(np.unique(p[low]))),
                 "txt_min": float(c_txt[m].min()), "txt_max_abs": float(np.abs(z["txt"][m] - data.txt_features[rows[m]]).max()),
                 "rows_frozen_differs_within_painting": int((m & (c_within < 1 - 1e-6)).sum()),
                 "low_rows_also_within_painting_differ": int((low & (c_within < 1 - 1e-6)).sum()),
                 "min_within_painting_cos": float(c_within[m].min())}
    # cache-path cosine on the painting's first row vs the low row
print(json.dumps(out, indent=1))
(HERE / "fr_cachecos.json").write_text(json.dumps(out, indent=1))
