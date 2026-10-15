"""Final review: a label-free look at the report's "side detector" reading (§8): how often the affect pick goes with
the visual groupings' Delta < 0 (contrasts agree more than supports), per pair and condition, pooled over 49-51."""
import json
import numpy as np
import fr3_lib as L

rows = {}
F = {c: [] for c in L.CONDS}; PK = {c: [] for c in L.CONDS}; PI = []
for s in (49, 50, 51):
    z = np.load(L.OUT / f"fr3_seed{s}.npz")
    for c in L.CONDS:
        F[c].append(z[f"F__{c}"]); PK[c].append(z[f"pick__{c}"])
    PI.append(z["pair_index"])
PI = np.concatenate(PI)
out = {}
for c in L.CONDS:
    f = np.concatenate(F[c]); pk = np.concatenate(PK[c])
    d_aff, d_img, d_cap = f[:, 2], f[:, 8], f[:, 14]
    vis_neg = (d_img < 0) & (d_cap < 0)
    for i, p in enumerate(L.PAIRS):
        m = PI == i
        out[f"{p}.{c}"] = {"share_visual_delta_both_negative": 100 * vis_neg[m].mean(),
                           "P(affect | both negative)": 100 * (pk[m & vis_neg] == 0).mean(),
                           "P(affect | not both negative)": 100 * (pk[m & ~vis_neg] == 0).mean(),
                           "affect_pick_share": 100 * (pk[m] == 0).mean(),
                           "mean_delta_affect": float(d_aff[m].mean()), "mean_delta_image": float(d_img[m].mean()),
                           "mean_delta_caption": float(d_cap[m].mean())}
for k, v in out.items():
    print(k, {kk: round(vv, 4) for kk, vv in v.items()})
(L.OUT / "fr3_side.json").write_text(json.dumps(out, indent=1))
