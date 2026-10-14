"""Final review B1: near-duplicate counts against every non-held ArtELingo painting (49,121), beside the 912 the
user decision fixed. Image features only (zip images vs ArtELingo store images); no label, no score.
Writes fr_b1_neardup_wide.json beside this file."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/project/CoSiR-wt/r9-fr-b")
from src.data.artelingo import load_artelingo
HERE = Path(__file__).resolve().parent
data = load_artelingo()
pa = np.asarray(data.paintings)
with np.load("/project/CoSiR/src/test/20261014_stage_d_final/cache/held_codes.npz") as z:
    held = set(pa[z["held_rows"]])
first = {}
for i, p in enumerate(pa):
    first.setdefault(p, i)
side = sorted(set(first) - held)
E = np.asarray(data.img_features[[first[p] for p in side]], dtype=np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
idx = json.loads(Path("/data/SSD2/pre_extract/artelingo28/clip_b32/index.json").read_text())
names = [im["painting"] for im in idx["images"]]
Z = np.asarray(np.load("/data/SSD2/pre_extract/artelingo28/clip_b32/img.npy"), dtype=np.float32)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
S = Z @ E.T
mx, arg = S.max(axis=1), S.argmax(axis=1)
out = {"train_side_all": len(side), **{f"cos_{k}": int((mx >= v).sum()) for k, v in (("099", .99), ("098", .98), ("092", .92))},
       "top": sorted([[float(m), names[i], side[a]] for i, (m, a) in enumerate(zip(mx, arg))], reverse=True)[:6]}
(HERE / "fr_b1_neardup_wide.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out)[:900])
