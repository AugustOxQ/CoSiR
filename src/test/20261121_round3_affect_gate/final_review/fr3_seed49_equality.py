"""Final review: is seed 49's bar margin equal to seed 42's by coincidence? Episode overlap and per-pair net counts."""
import json
import numpy as np
import fr3_lib as L

z42 = np.load(L.OUT / "fr3_seed42.npz"); z49 = np.load(L.OUT / "fr3_seed49.npz")
P = np.load(L.OUT / "fr3_perepisode.npz")
out = {}
a42, a49 = z42["anchor"], z49["anchor"]
out["same_anchor_same_index"] = int((a42 == a49).sum())
out["anchor_sets_overlap"] = int(len(np.intersect1d(a42, a49)))
out["paintings_overlap"] = int(len(np.intersect1d(z42["cl"], z49["cl"])))
out["n_paintings"] = {"42": int(len(np.unique(z42["cl"]))), "49": int(len(np.unique(z49["cl"])))}
for s in (50, 51):
    zz = np.load(L.OUT / f"fr3_seed{s}.npz")
    out[f"same_anchor_same_index_42_{s}"] = int((a42 == zz["anchor"]).sum())
# per-pair net rankings AFF fused minus B' on seed 49 (my arrays)
f49, b49 = P["s49__aff__r1"], P["s49__Bp__r1"]
pi = z49["pair_index"]
out["net49_per_pair"] = [float(4 * (f49 - b49)[pi == i].sum()) for i in range(3)]
out["net49_total"] = float(4 * (f49 - b49).sum())
out["hits49"] = {"aff": float(4 * f49.sum()), "Bp": float(4 * b49.sum())}
# seed 42 from the regression (rule numbers): per-pair bar margins 0.9765625, 1.45263671875, -0.32958984375 (pp, 4096 eps)
out["net42_per_pair"] = [round(v / 100 * 4 * 4096, 6) for v in (0.9765625, 1.45263671875, -0.32958984375)]
out["hits42"] = {"aff": 19.136555989583336 / 100 * 49152, "Bp": 18.436686197916664 / 100 * 49152}
print(json.dumps(out, indent=1))
(L.OUT / "fr3_seed49_equality.json").write_text(json.dumps(out, indent=1))
