"""Exploratory, seed 42, decides nothing. Is the affect-only gate special, or does any narrower gate help?
Each singleton and pair of groupings as the set H of picks that may open R1's gate, plus two random controls that open
R1's gate on a random subset of values with AFF's open share per condition (seeds 0 and 1). R1's term, tied 224 cells.
Writes results/bs_10_subsets.json."""
import json

import numpy as np

import bs_lib as L
from bs_05_aff import gate_sets, run_tied

data = L.Data()
P1 = data.P("R1")
MDs, taus, zT, g, m = L.standard_MDs(data, P1)
res = {}
for H in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
    nm = "H=" + "+".join(L.A0[h] for h in H)
    gs = gate_sets(P1, g, H)
    r = run_tied(data, [L.MD(zT, gg) for gg in gs], nm)[0]
    r["open_share_tau0"] = {c: 100 * float(gs[0][c].mean()) for c in L.CONDITIONS}
    res[nm] = r
aff = gate_sets(P1, g, (0,))
for seed in (0, 1):
    rng = np.random.default_rng(seed)
    share = {c: float(aff[0][c].mean()) for c in L.CONDITIONS}
    keep = {c: (rng.random(data.E) < share[c]).astype(np.float32) for c in L.CONDITIONS}
    gs = [{c: (g[t][c] * keep[c]).astype(np.float32) for c in L.CONDITIONS} for t in range(4)]
    res[f"random_seed{seed}"] = run_tied(data, [L.MD(zT, gg) for gg in gs], f"random gate, AFF's open share (seed {seed})")[0]
print({k: v.get("open_share_tau0") for k, v in res.items()})
(L.HERE / "results" / "bs_10_subsets.json").write_text(json.dumps(L.C.jsonable(res), indent=1))
