"""Final review (descriptive, decides nothing; selection was fixed before): pooled seeds 49-51 R@1 of every run's
best-epoch features.npz, to see how much the conclusion depends on which grid point val selection picked."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fr_episodes as E
from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_splits
data = load_artelingo(); sp = artelingo_splits(data)
groups = np.asarray(sp.groups); n = len(groups); sel = np.asarray(sp.selection)
selmask = np.zeros(n, bool); selmask[sel] = True
eps = []
for s in (49, 50, 51):
    z = np.load(E.E1 / f"episodes_seed{s}.npz")
    for a, b in E.PAIRS:
        eps.append((z[f"{a}__{b}__anchor"].astype(np.int64), z[f"{a}__{b}__candidates"].astype(np.int64)))
anchor = np.concatenate([e[0] for e in eps]); cand = np.concatenate([e[1] for e in eps])
bp0 = np.concatenate([np.load(E.R3 / f"go_seed{s}.npz")["Bp__r1"] for s in (49, 50, 51)])
out = {}
for line in (E.ROOT / ".superpowers/sdd/2026-10-07-clip-lightweight-ft/runs.txt").read_text().splitlines():
    if not line.strip():
        continue
    node, v, lr, ok, tag = line.split()[:5]
    d = next((E.JOBS / tag / "code/outputs/clipft").iterdir())
    z = np.load(d / "features.npz")
    rows = z["rows"]; k = selmask[rows]
    fi = np.full((n, 512), np.nan, np.float32); ft = np.full((n, 512), np.nan, np.float32)
    fi[rows[k]] = z["img"][k]; ft[rows[k]] = z["txt"][k]
    m = E.cf_metrics(fi, ft, anchor, cand)
    out[f"{v} {lr}"] = {"r1": 100 * m["r1"].mean(), "minus_Bp0": 100 * (m["r1"] - bp0).mean()}
    print(v, lr, "pooled R@1 %.3f, minus B'(A0) %.3f" % (out[f"{v} {lr}"]["r1"], out[f"{v} {lr}"]["minus_Bp0"]), flush=True)
(HERE / "fr_allruns.json").write_text(json.dumps(out, indent=1))
