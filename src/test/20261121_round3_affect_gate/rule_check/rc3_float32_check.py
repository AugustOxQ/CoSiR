"""Rule check (read only, seed 42, CPU): AFF and R1 through the rule's float32 path (round 2's r2_fusion on 224 cells,
round 1's rc_core gates / G_cf, aspect_nested._combine), from the brainstorm cache, compared with the brainstorm's
recorded float64 numbers (bs_04_readers.json results.AFF, bs_05_aff.json) and round-1 R-c's stored arrays.
Also: D7 redundancy, the open-share counts, and a cell-by-cell count of float32 vs float64 disagreements.

Writes only rule_check/rc3_float32_check.json. Run from /project/CoSiR:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/rule_check/rc3_float32_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parents[1]
R2DIR = T / "20261118_reader_fix_round2"
BS = T / "20261120_r1_levers_brainstorm"
R1DIR = T / "20261117_reader_fix_csd"
sys.path.insert(0, str(R2DIR))
import r2_common as R  # noqa: E402  (round 1 on sys.path; no rule assertion on import)
import r2_fusion as F  # noqa: E402

C, K = R.C, R.K
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import NESTED_A, NESTED_U, _zdict  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

z = np.load(BS / "cache" / "bs_cache.npz")
E = len(z["anchor_group"])
cl, pi, par = z["anchor_group"], z["pair_index"], z["parity"]
Bd = {c: {d: z[f"B__{d}"] for d in DIRECTIONS} for c in CONDITIONS}
stack = {d: z[f"stack4__{d}"][:, :3] for d in DIRECTIONS}
P = {c: z[f"P_R1__{c}"] for c in CONDITIONS}
pB = {m: z[f"pB__{m}"] for m in METRICS}
pBp = {m: z[f"pBpA0__{m}"] for m in METRICS}
taus = json.loads((R1DIR / "results/rc_tau.json").read_text())["taus"]
rc_ = np.load(R1DIR / "results/cand_Rc_Rb_expected_A0.npz")
out = {}

# ---------------------------------------------------------------- reader quantities (D5) vs stored round-1 R-c
Tt = C.expected_term(stack, P)
m = {c: C.top_two_margin(P[c]) for c in CONDITIONS}
pk = {c: np.asarray(P[c]).argmax(1) for c in CONDITIONS}
out["T_equal_stored"] = all(np.array_equal(Tt[c][d], rc_[f"T__{c}__{d}"]) for c in CONDITIONS for d in DIRECTIONS)
out["margins_equal_stored"] = all(np.array_equal(m[c], rc_[f"margin__{c}"]) for c in CONDITIONS)
out["picks_equal_stored"] = all(np.array_equal(pk[c], rc_[f"pick__{c}"].astype(np.int64)) for c in CONDITIONS)
tau_re = K.thresholds(m)[0]
out["tau_recomputed_equal_rc_tau"] = [a == b for a, b in zip(tau_re, taus)]
out["margin_dtype"] = str(m["a"].dtype)
out["min_margin_equals_tau0"] = bool(min(m["a"].min(), m["b"].min()) == taus[0])

# ---------------------------------------------------------------- the rule's float32 family
zB, zT = _zdict(Bd), _zdict(Tt)
info = F.rank_info(Bd)
g_R1 = K.gates(m, taus)
g_AFF = {t: {c: (g_R1[t][c] * (pk[c] == 0)).astype(np.float32) for c in CONDITIONS} for t in g_R1}
ctrl = F.control_choice(zB, par)
out["control"] = {str(h): ctrl[h] for h in ctrl}


def cellv(i):
    _, t, lu, la = F.cell_values(i)
    return [int(t), float(lu), float(la)]


def run(g, label):
    gated = {t: K.gated_terms(zT, g[t]) for t in g}
    G = {t: K.g_cf(gated[t]) for t in g}
    fri, fgi, cri = F.cell_statistics(zB, info, gated, G, n_kappa=1)
    fp, cp = F.select_fused(fri, fgi, ctrl, par), F.select_cf(cri, par)
    fused = F.assemble(zB, info, gated, fp, par)
    cfs = F.assemble(zB, info, G, cp, par)
    pn, pc = per_anchor(fused), per_anchor(cfs)
    bar_v, bar = C.bar_info(pn, pc, pBp, pB, cl, pi)
    d = C.diff3(pn, pc, cl)
    res = {"fused_cells": {h: [fp[h]] + cellv(fp[h]) for h in (0, 1)},
           "cf_cells": {h: [cp[h]] + cellv(cp[h]) for h in (0, 1)},
           "fused_r1": 100 * float(np.mean(pn["r1"])), "cf_r1": 100 * float(np.mean(pc["r1"])),
           "comparator": bar["comparator"], "bar": bar["r1"], "margin": d["r1"], "gain": d["gain"],
           "either": d["either"], "per_pair_bar": {p: v["point"] for p, v in bar["per_pair_r1"].items()},
           "open_share_tau0_counts": {c: int(g[0][c].sum()) for c in CONDITIONS},
           "open_share_tau0_float32_mean_pct": {c: 100 * float(g[0][c].mean()) for c in CONDITIONS},
           "open_share_tau0_float64_mean_pct": {c: 100 * float(np.mean(g[0][c], dtype=np.float64)) for c in CONDITIONS}}
    # integer criteria at the maximum and the tie sets (float32 path)
    ties = {}
    for h in (0, 1):
        tune = par == h
        crit, rho, gam = F.fused_criterion(fri, fgi, tune, ctrl[h][1], np.arange(len(fri)))
        ties[f"fused_h{h}"] = {"max": int(crit.max()), "tied_cells": np.flatnonzero(crit == crit.max()).tolist(),
                               "second": int(np.sort(np.unique(crit))[-2])}
        rr = cri[:, tune].sum(1, dtype=np.int64)
        ties[f"cf_h{h}"] = {"max": int(rr.max()), "tied_cells": np.flatnonzero(rr == rr.max()).tolist(),
                            "second": int(np.sort(np.unique(rr))[-2])}
    res["ties"] = ties
    print(label, json.dumps({k: res[k] for k in ("fused_cells", "cf_cells", "fused_r1", "cf_r1", "comparator")}),
          flush=True)
    return res, (fri, fgi, cri), (pn, pc, bar_v)


out["R1"], mats_R1, arr_R1 = run(g_R1, "R1")
out["AFF"], mats_AFF, arr_AFF = run(g_AFF, "AFF")

# R1 per-anchor arrays vs round-1 R-c stored
pn1, pc1, bv1 = arr_R1
out["R1_arrays_equal_stored"] = {
    **{f"fused__{k}": bool(np.array_equal(np.asarray(pn1[k]), rc_[f"fused__{k}"])) for k in METRICS},
    **{f"cf__{k}": bool(np.array_equal(np.asarray(pc1[k]), rc_[f"cf__{k}"])) for k in METRICS},
    "bar_v": bool(np.array_equal(bv1, rc_["bar_v"]))}

# AFF minus R1 (paired)
pnA, pcA, bvA = arr_AFF
out["AFF_minus_R1"] = {"fused_r1": C.point_ci(np.asarray(pnA["r1"], float) - np.asarray(pn1["r1"], float), cl),
                       "bar_margin": C.point_ci(bvA - bv1, cl)}

# ---------------------------------------------------------------- the brainstorm's float64 family, cell by cell
zB64 = {d: zB["a"][d].numpy().astype(np.float64) for d in DIRECTIONS}


def f64_stats(g):
    """bs_lib.Family(tied=True).stats() logic: fused = (1+lu) zB + la*M + la*D, cf = (1+lu) zB + la*M, float64."""
    fri = np.zeros((224, E), np.int8)
    fgi = np.zeros((224, E), np.int8)
    cri = np.zeros((224, E), np.int8)
    for t in range(4):
        ga = np.asarray(g[t]["a"], np.float64)[:, None]
        gb = np.asarray(g[t]["b"], np.float64)[:, None]
        M, D = {}, {}
        for d in DIRECTIONS:
            xa = ga * zT["a"][d].numpy().astype(np.float64)
            xb = gb * zT["b"][d].numpy().astype(np.float64)
            M[d] = 0.5 * (xa + xb)
            D[d] = 0.5 * (xa - xb)
        for u, lu in enumerate(NESTED_U):
            for a, la in enumerate(NESTED_A):
                i = (t * 7 + u) * 8 + a
                sf, sc = {}, {}
                for c, sgn in (("a", 1.0), ("b", -1.0)):
                    sf[c], sc[c] = {}, {}
                    for d in DIRECTIONS:
                        s = (1.0 + lu) * zB64[d]
                        sc[c][d] = s + la * M[d] if la else s
                        if la:
                            s = s + la * M[d]
                            s = s + la * (sgn * D[d])
                        sf[c][d] = s
                pa = per_anchor(sf)
                fri[i], fgi[i] = F.as_int4(pa["r1"]), F.as_int4(pa["gain"])
                cri[i] = F.as_int4(per_anchor(sc)["r1"])
    return fri, fgi, cri


for nm, g, mats in (("R1", g_R1, mats_R1), ("AFF", g_AFF, mats_AFF)):
    f64 = f64_stats(g)
    diff = {}
    for lab, a32, a64 in zip(("fused_r1", "fused_gain", "cf_r1"), mats, f64):
        bad = a32 != a64
        diff[lab] = {"cells_with_any_difference": int(bad.any(1).sum()), "entries_differing": int(bad.sum())}
    fp64 = F.select_fused(f64[0], f64[1], ctrl, par)
    cp64 = F.select_cf(f64[2], par)
    diff["picks_float64"] = {"fused": [fp64[0], fp64[1]], "cf": [cp64[0], cp64[1]]}
    out[f"{nm}_float32_vs_float64_cellstats"] = diff
    print(nm, "float32 vs float64:", json.dumps(diff), flush=True)

# ---------------------------------------------------------------- D7 redundancy
def row_corr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
    ok = den > 0
    return float(np.mean((x * y).sum(1)[ok] / den[ok])), int((~ok).sum())


red = {}
for h, nm in enumerate(("affect", "image", "caption")):
    red[nm] = {}
    for d in DIRECTIONS:
        zs = zscore_rows(torch.as_tensor(stack[d][:, h], dtype=torch.float32)).numpy().astype(np.float64)
        v, n0 = row_corr(zs, zB64[d])
        red[nm][d] = {"value": v, "rows_left_out": n0}
out["redundancy"] = red

# ---------------------------------------------------------------- compare with the brainstorm's records
b4 = json.loads((BS / "results/bs_04_readers.json").read_text())["results"]["AFF"]
b5 = json.loads((BS / "results/bs_05_aff.json").read_text())
b10 = json.loads((BS / "results/bs_10_subsets.json").read_text())
A = out["AFF"]
cmp = {
    "fused_r1": A["fused_r1"] == b4["fused_r1"], "cf_r1": A["cf_r1"] == b4["cf_r1"],
    "comparator": A["comparator"] == b4["comparator"],
    "bar": A["bar"]["point"] == b4["bar"]["point"] and A["bar"]["ci95"] == b4["bar"]["ci95"],
    "margin": A["margin"]["point"] == b4["margin"]["point"] and A["margin"]["ci95"] == b4["margin"]["ci95"],
    "gain": A["gain"]["point"] == b4["gain"]["point"] and A["gain"]["ci95"] == b4["gain"]["ci95"],
    "either": A["either"]["point"] == b4["either"]["point"],
    "per_pair_bar": A["per_pair_bar"] == b4["per_pair_bar"],
    "fused_cells": [A["fused_cells"][h][1:] for h in (0, 1)] == [x[:3] for x in b4["fused_cells"]],
    "cf_cells": [A["cf_cells"][h][1:] for h in (0, 1)] == b4["cf_cells"],
    "AFF_minus_R1_fused": out["AFF_minus_R1"]["fused_r1"] == b5["A0"]["AFF_minus_R1"]["fused_r1"],
    "AFF_minus_R1_bar": out["AFF_minus_R1"]["bar_margin"] == b5["A0"]["AFF_minus_R1"]["bar_margin"],
    "redundancy": all(red[h][d]["value"] == b5["corr_zs_h_with_zB"][h][d] for h in red for d in DIRECTIONS),
    "open_share_float32_equals_bs10": A["open_share_tau0_float32_mean_pct"] == b10["H=affect"]["open_share_tau0"],
    "open_share_float64_equals_bs10": A["open_share_tau0_float64_mean_pct"] == b10["H=affect"]["open_share_tau0"],
    "R1_bar": out["R1"]["bar"]["point"] == 0.4435221354166667
              and out["R1"]["bar"]["ci95"] == [0.21646171563312194, 0.6735669710776852],
    "R1_gain": out["R1"]["gain"]["point"] == 2.667236328125
               and out["R1"]["gain"]["ci95"] == [2.325087836946873, 3.012361650695922],
}
out["compare_with_records"] = cmp
print(json.dumps(cmp, indent=1))
(HERE / "rc3_float32_check.json").write_text(json.dumps(C.jsonable(out), indent=1))
print("wrote", HERE / "rc3_float32_check.json")
