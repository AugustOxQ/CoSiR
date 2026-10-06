"""Final review: every decision quantity and the report's numbers re-derived from the stored per-anchor arrays
(results/cand_*.npz, step1_eval_style.npz), with my own comparator / bar / clause code and the library bootstrap.
Writes final_review/out/fr_stored.json and prints a table."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/project/CoSiR")
sys.path.insert(0, str(ROOT))
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

FR = Path(__file__).resolve().parent
RES = FR.parent / "results"
OUT = FR / "out"
z1 = np.load(ROOT / "src/test/20261116_grouping_step1_style/results/step1_eval_style.npz")
ex = np.load(RES / "cand_Rb_expected_A0.npz")
cl, pidx = ex["extra__anchor_group"], ex["extra__pair_index"]
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
CFG = {"A0": ("affect", "image", "caption"), "A1": ("affect", "image", "caption", "csd"),
       "AR": ("affect", "image", "caption", "rand")}
TOLD = {"A0": {"emotion": "affect", "style": "image", "genre": "image"},
        "A1": {"emotion": "affect", "style": "csd", "genre": "image"},
        "AR": {"emotion": "affect", "style": "rand", "genre": "image"}}
M = ("r1", "gain", "other")


def bt(v, mask=None):
    v = np.asarray(v, np.float64)
    c = cl
    if mask is not None:
        v, c = v[mask], cl[mask]
    r = cluster_bootstrap(v, c)
    return (100 * r["point"], 100 * r["ci95"][0], 100 * r["ci95"][1])


pB = {m: z1[f"B__{m}"].astype(np.float64) for m in M}
pBp = {a: {m: z1[f"{a}__Bprime__{m}"].astype(np.float64) for m in M} for a in CFG}
cands = {}
for a in CFG:
    cands[f"argmax_{a}"] = dict(cfg=a, fused={m: z1[f"{a}__reader__fused__{m}"].astype(np.float64) for m in M},
                                cf={m: z1[f"{a}__reader__cf__{m}"].astype(np.float64) for m in M},
                                pick={c: z1[f"{a}__reader_pick__{c}"].astype(int) for c in "ab"})
for nm in ("Ra_A1", "Rb_argmax_A1", "Rb_expected_A1", "Ra_A0", "Rb_argmax_A0", "Rb_expected_A0", "Rc_Rb_expected_A0",
           "Ra_AR", "Rb_argmax_AR", "Rb_expected_AR"):
    z = np.load(RES / f"cand_{nm}.npz")
    cands[nm] = dict(cfg=nm.rsplit("_", 1)[-1], fused={m: z[f"fused__{m}"].astype(np.float64) for m in M},
                     cf={m: z[f"cf__{m}"].astype(np.float64) for m in M}, pick={c: z[f"pick__{c}"].astype(int) for c in "ab"},
                     bar_v_stored=z["bar_v"], z=z)

out = {}
for nm, d in cands.items():
    a = d["cfg"]
    f, c = d["fused"], d["cf"]
    comps = [("B_prime", pBp[a]), ("counterpart", c), ("B", pB)]
    means = [float(np.mean(p["r1"])) for _, p in comps]
    k = int(np.argmax(means))            # argmax keeps the first of equal maxima: order B', counterpart, B
    bar_v = f["r1"] - comps[k][1]["r1"]
    either_f = f["r1"] + f["other"]
    r = {"cfg": a, "fused": 100 * f["r1"].mean(), "cf": 100 * c["r1"].mean(), "B": 100 * pB["r1"].mean(),
         "Bp": 100 * pBp[a]["r1"].mean(), "comparator": comps[k][0], "bar": bt(bar_v),
         "margin": bt(f["r1"] - c["r1"]), "gain_stat": bt(f["gain"] - c["gain"]),
         "either_vs_cf": bt(either_f - (c["r1"] + c["other"])),
         "either_vs_comp": bt(either_f - (comps[k][1]["r1"] + comps[k][1]["other"])),
         "vs_Bp": bt(f["r1"] - pBp[a]["r1"]), "vs_B": bt(f["r1"] - pB["r1"]),
         "per_pair_bar": [bt(bar_v, pidx == i) for i in range(3)],
         "per_pair_gain_margin": [bt(f["gain"] - c["gain"], pidx == i) for i in range(3)],
         "per_pair_either_margin": [bt(either_f - c["r1"] - c["other"], pidx == i) for i in range(3)],
         "cf_gain_all_zero": bool((c["gain"] == 0).all())}
    r["clauses"] = (r["bar"][0] >= 0.5, r["bar"][1] > 0, r["gain_stat"][1] > 0)
    r["clears"] = all(r["clauses"])
    if "bar_v_stored" in d:
        r["bar_v_equal_stored"] = bool(np.array_equal(bar_v, d["bar_v_stored"]))
    # pick accuracy, told mapping
    parts = CFG[a]
    ti = {cnd: np.array([parts.index(TOLD[a][PAIRS[i][j]]) for i in pidx]) for j, cnd in enumerate("ab")}
    corr = {cnd: d["pick"][cnd] == ti[cnd] for cnd in "ab"}
    r["pick_acc"] = bt(0.5 * (corr["a"].astype(float) + corr["b"].astype(float)))
    r["pick_acc_pp"] = {f"{PAIRS[i][0]}x{PAIRS[i][1]}": [100 * corr[cnd][pidx == i].mean() for cnd in "ab"]
                        for i in range(3)}
    r["share"] = {h: [100 * (d["pick"][cnd] == j).mean() for cnd in "ab"] for j, h in enumerate(parts)}
    r["share_pp"] = {f"{PAIRS[i][0]}x{PAIRS[i][1]}": {h: [100 * (d["pick"][cnd][pidx == i] == j).mean() for cnd in "ab"]
                                                     for j, h in enumerate(parts)} for i in range(3)}
    r["same_pick_both_conditions"] = 100 * float((d["pick"]["a"] == d["pick"]["b"]).mean())
    d["bar_v"] = bar_v
    out[nm] = r

# paired differences vs the step-1 arg-max reader; A1 - A0
for nm, d in cands.items():
    if nm.startswith("argmax"):
        continue
    ref = cands[f"argmax_{d['cfg']}"]
    out[nm]["vs_argmax_margin"] = bt((d["fused"]["r1"] - d["cf"]["r1"]) - (ref["fused"]["r1"] - ref["cf"]["r1"]))
    out[nm]["vs_argmax_bar"] = bt(d["bar_v"] - ref["bar_v"])
    out[nm]["vs_argmax_fused"] = bt(d["fused"]["r1"] - ref["fused"]["r1"])
for rd in ("argmax", "Ra", "Rb_argmax", "Rb_expected"):
    x, y = cands[f"{rd}_A1"], cands[f"{rd}_A0"]
    out[f"A1-A0_{rd}"] = {"fused": bt(x["fused"]["r1"] - y["fused"]["r1"]), "bar": bt(x["bar_v"] - y["bar_v"]),
                          "margin": bt((x["fused"]["r1"] - x["cf"]["r1"]) - (y["fused"]["r1"] - y["cf"]["r1"]))}

# report-specific descriptive numbers
ext = {}
ext["Rc_net_rankings_bar"] = float(cands["Rc_Rb_expected_A0"]["bar_v"].sum() * 4)
ext["Rb_argmax_A1_gain_rankings"] = float((cands["Rb_argmax_A1"]["fused"]["gain"] - cands["Rb_argmax_A1"]["cf"]["gain"]).sum() * 4)
ext["Rb_expected_A0_gain_rankings"] = float((cands["Rb_expected_A0"]["fused"]["gain"] - cands["Rb_expected_A0"]["cf"]["gain"]).sum() * 4)
ext["Rb_expected_A0_vs_argmax_A0_fused_identical_arrays"] = bool(np.array_equal(cands["Rb_expected_A0"]["fused"]["r1"],
                                                                               cands["argmax_A0"]["fused"]["r1"]))
for cfg in ("A0", "A1"):
    z = cands[f"Rb_expected_{cfg}"]["z"]
    Pbar = 0.5 * (z["extra__probs_a"] + z["extra__probs_b"])
    ext[f"Pbar_mean_{cfg}"] = dict(zip(CFG[cfg], (100 * Pbar.mean(0)).tolist()))
    corr = {}
    for d in ("i2t", "t2i"):
        A, Bm = z[f"T__a__{d}"].astype(np.float64), z[f"T__b__{d}"].astype(np.float64)
        A = A - A.mean(1, keepdims=True)
        Bm = Bm - Bm.mean(1, keepdims=True)
        den = np.sqrt((A * A).sum(1) * (Bm * Bm).sum(1))
        ok = den > 0
        corr[d] = float(((A * Bm).sum(1)[ok] / den[ok]).mean())
        corr[d + "_n_const_rows"] = int((~ok).sum())
    ext[f"TaTb_corr_{cfg}"] = corr
    ext[f"top_prob_mean_seed42_{cfg}"] = float(np.concatenate([z["extra__probs_a"].max(1), z["extra__probs_b"].max(1)]).mean())
json.dump({"candidates": out, "extra": ext}, open(OUT / "fr_stored.json", "w"), indent=1, default=float)


def f3(t):
    return f"{t[0]:+.4f} [{t[1]:+.3f}, {t[2]:+.3f}]"


print(f"{'name':20s} {'fused':>7s} {'cf':>7s} {'Bp':>7s} {'comp':11s} {'margin':>26s} {'bar':>26s} {'gain':>26s} {'eith_cf':>8s} clears")
for nm, r in out.items():
    if nm.startswith("A1-A0"):
        continue
    print(f"{nm:20s} {r['fused']:7.3f} {r['cf']:7.3f} {r['Bp']:7.3f} {r['comparator']:11s} {f3(r['margin']):>26s} "
          f"{f3(r['bar']):>26s} {f3(r['gain_stat']):>26s} {r['either_vs_cf'][0]:+8.3f} {r['clears']} {r.get('bar_v_equal_stored', '')}")
for nm, r in out.items():
    if nm.startswith("A1-A0"):
        print(nm, {k: f3(v) for k, v in r.items()})
print(json.dumps(ext, indent=1))
