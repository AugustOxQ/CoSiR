"""Final review: seed-42 numbers the report quotes from the brainstorm (§2, §8): R1 fused minus its counterpart per
pair and condition, and B's R@1 on the emotion side (condition a of the two emotion pairs)."""
import json
import numpy as np
import fr3_lib as L

z = np.load(L.OUT / "fr3_seed42.npz")
sc = lambda k: {c: {d: z[f"{k}__{c}__{d}"] for d in L.DIRS} for c in L.CONDS}  # noqa: E731
B, T = sc("B"), sc("T")
m = {c: z[f"m__{c}"] for c in L.CONDS}; pk = {c: z[f"pick__{c}"] for c in L.CONDS}
pi, par = z["pair_index"], z["parity"]
fam = L.Family(B, T, L.gates(m, pk, "R1"), par)
fp, _ = fam.pick_fused(); cp, _ = fam.pick_cf()


def assembled(picks, cf):
    n = len(par)
    out = {c: {d: np.empty((n, 13), np.float32) for d in L.DIRS} for c in L.CONDS}
    for h in (0, 1):
        S = fam.scores_of(picks[h], cf)
        for c in L.CONDS:
            for d in L.DIRS:
                out[c][d][par != h] = S[c][d][par != h]
    return out


Sf, Sc = assembled(fp, False), assembled(cp, True)
res = {}
for i, p in enumerate(L.PAIRS):
    mk = pi == i
    for c, col in (("a", 0), ("b", 1)):
        hf = np.mean([L.first(Sf[c][d][mk], col).mean() for d in L.DIRS])
        hc = np.mean([L.first(Sc[c][d][mk], col).mean() for d in L.DIRS])
        hb = np.mean([L.first(B[c][d][mk], col).mean() for d in L.DIRS])
        res[f"{p}.{c}"] = {"R1_fused_minus_cf": 100 * (hf - hc), "B_r1": 100 * hb}
for k, v in res.items():
    print(k, {kk: round(vv, 3) for kk, vv in v.items()})
(L.OUT / "fr3_cites.json").write_text(json.dumps(res, indent=1))
