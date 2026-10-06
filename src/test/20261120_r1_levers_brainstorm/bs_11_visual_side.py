"""Exploratory, seed 42, decides nothing. One-sided steering chosen by the strong visual signal: steer with z(s_affect)
on the side whose contrasts are more image-coherent than its supports (Delta_image^c < 0), never on the other side.
Delta_image^b = -Delta_image^a, so at most one side is steered. Thresholds: |Delta_image| >= its q-th percentile
(q = 0, 25, 50, 75; chosen by the cross-fit like tau). Variants: the image grouping alone; image + caption Delta;
R1's term in place of z(s_affect); AND-combined with R1's affect pick. Tied 224-cell layout, matched counterpart.
Writes results/bs_11_visual_side.json."""
import json

import numpy as np

import bs_lib as L
from bs_05_aff import gate_sets, run_tied

data = L.Data()
P1 = data.P("R1")
MDs, taus, zT, g, m = L.standard_MDs(data, P1)
zA = {c: {d: L.zrows(data.stack[d][:, 0]) for d in L.DIRECTIONS} for c in L.CONDITIONS}
res = {}


def side_gates(delta_a, pcts=(0, 25, 50, 75)):
    mag = np.abs(delta_a)
    ths = np.percentile(np.concatenate([mag, mag]), pcts)
    out = []
    for t in ths:
        ok = mag >= t
        out.append({"a": ((delta_a < 0) & ok).astype(np.float32), "b": ((delta_a > 0) & ok).astype(np.float32)})
    return out


fa = data.feat["a"]
d_img = fa[:, 8]                       # Delta_image under condition a (feature 2 of grouping 1)
d_vis = fa[:, 8] + fa[:, 14]           # image + caption Delta
for nm, delta, term in (("VIS_img_aff", d_img, zA), ("VIS_imgcap_aff", d_vis, zA), ("VIS_img_R1term", d_img, zT)):
    gs = side_gates(delta)
    r = run_tied(data, [L.MD(term, gg) for gg in gs], nm)[0]
    r["open_share_q0"] = {c: 100 * float(gs[0][c].mean()) for c in L.CONDITIONS}
    r["open_share_q0_per_pair"] = {p: {c: round(100 * float(gs[0][c][data.pi == i].mean()), 1) for c in L.CONDITIONS}
                                   for i, p in enumerate(L.PAIRS)}
    print("   open share (q0) per pair a/b:", r["open_share_q0_per_pair"])
    res[nm] = r
aff = gate_sets(P1, g, (0,))
gs = side_gates(d_img)
both = [{c: (aff[t][c] * gs[0][c]).astype(np.float32) for c in L.CONDITIONS} for t in range(4)]
res["AFF_and_VIS"] = run_tied(data, [L.MD(zA, gg) for gg in both], "AFF pick AND Delta_image^c < 0, z(s_affect)")[0]
(L.HERE / "results" / "bs_11_visual_side.json").write_text(json.dumps(L.C.jsonable(res), indent=1))
