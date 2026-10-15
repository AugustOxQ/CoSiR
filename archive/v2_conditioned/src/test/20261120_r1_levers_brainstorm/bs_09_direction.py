"""Exploratory, seed 42, decides nothing. Per-direction R@1 / either of R1 and of the affect-only gate (AFF, R1's pick
gate with R1's term; AFFp, gate on R1's P(affect) percentile with the pure z(s_affect) term), fused vs counterpart.
Writes results/bs_09_direction.json."""
import json

import numpy as np

import bs_lib as L
from bs_03_sxg import assembled_scores, first
from bs_05_aff import gate_sets, run_tied


def per_dir(data, S):
    out = {}
    for d in L.DIRECTIONS:
        for who in ("fused", "cf"):
            hit = np.mean([first(S[who][c][d]) == col for c, col in (("a", 0), ("b", 1))], axis=0)
            oth = np.mean([first(S[who][c][d]) == col for c, col in (("a", 1), ("b", 0))], axis=0)
            out.setdefault(d, {})[who] = {"r1": 100 * hit.mean(), "either": 100 * (hit + oth).mean()}
            for i, p in enumerate(L.PAIRS):
                mk = data.pi == i
                out[d].setdefault(p, {})[who] = {"r1": 100 * hit[mk].mean(), "either": 100 * (hit[mk] + oth[mk]).mean()}
    return out


data = L.Data()
P1 = data.P("R1")
MDs, taus, zT, g, m = L.standard_MDs(data, P1)
zA = {c: {d: L.zrows(data.stack[d][:, 0]) for d in L.DIRECTIONS} for c in L.CONDITIONS}
allv = np.concatenate([P1["a"][:, 0], P1["b"][:, 0]])
gl = [{c: (P1[c][:, 0] >= t).astype(np.float32) for c in L.CONDITIONS} for t in np.percentile(allv, (0, 50, 75, 90))]
res = {}
for nm, mds in (("R1", MDs), ("AFF", [L.MD(zT, gg) for gg in gate_sets(P1, g, [0])]), ("AFFp", [L.MD(zA, gg) for gg in gl])):
    r = run_tied(data, mds, nm)
    res[nm] = per_dir(data, assembled_scores(r[1], r[2], r[3]))
    for d in L.DIRECTIONS:
        f, k = res[nm][d]["fused"], res[nm][d]["cf"]
        print(f"   {nm} {d}: R@1 {f['r1']:.3f} vs {k['r1']:.3f} (diff {f['r1'] - k['r1']:+.3f}); either {f['either']:.3f} vs "
              f"{k['either']:.3f}; per pair diff " + " / ".join(
                  f"{res[nm][d][p]['fused']['r1'] - res[nm][d][p]['cf']['r1']:+.2f}" for p in L.PAIRS))
(L.HERE / "results" / "bs_09_direction.json").write_text(json.dumps(L.C.jsonable(res), indent=1))
