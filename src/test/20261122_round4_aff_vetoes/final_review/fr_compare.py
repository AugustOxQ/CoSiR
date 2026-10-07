"""Final review: compare the third derivation (out/fr_results.json, out/fr_arrays.npz) with the implementation's
results (results/dev_seed42.json, carry.json, seed42_arrays.npz) and the rule's stated targets, under rule §8's
agreement (discrete identical, arrays exact, continuous within 1e-9 pp). Writes out/fr_agreement.json."""
import json
from pathlib import Path

import numpy as np

HERE = Path("/project/CoSiR/src/test/20261122_round4_aff_vetoes")
OUT = HERE / "final_review/out"
mine = json.loads((OUT / "fr_results.json").read_text())
za = np.load(OUT / "fr_arrays.npz")
dev = json.loads((HERE / "results/dev_seed42.json").read_text())
car = json.loads((HERE / "results/carry.json").read_text())
zi = np.load(HERE / "results/seed42_arrays.npz")

rows = []


def cmp(name, a, b, tol=0.0):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        ok = len(a) == len(b) and all(abs(float(x) - float(y)) <= tol if isinstance(x, float) or isinstance(y, float)
                                      else x == y for x, y in zip(a, b))
        diff = max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0) if ok or len(a) == len(b) else None
    elif isinstance(a, float) or isinstance(b, float):
        diff = abs(float(a) - float(b))
        ok = diff <= tol
    else:
        diff = None
        ok = a == b
    rows.append({"name": name, "mine": a, "impl": b, "ok": bool(ok), "abs_diff": diff})


def ci3(r):
    return [r["point"], *r["ci95"]]


TOL = 1e-9
for k in ("V4", "V2", "V24", "AFF"):
    m = mine["scorers"][k]
    d = dev["aff"] if k == "AFF" else dev["candidates"][k]
    cmp(f"{k}.fused_r1", m["fused_r1"], d["fused_r1"], TOL)
    cmp(f"{k}.cf_r1", m["cf_r1"], d["cf_r1"], TOL)
    cmp(f"{k}.fpick", [m["fpick"]["0"], m["fpick"]["1"]], [d["cells"]["fpick"]["0"], d["cells"]["fpick"]["1"]])
    cmp(f"{k}.cpick", [m["cpick"]["0"], m["cpick"]["1"]], [d["cells"]["cpick"]["0"], d["cells"]["cpick"]["1"]])
    cmp(f"{k}.sigma", [mine["control"]["0"][0], mine["control"]["1"][0]], [d["sigma"]["0"], d["sigma"]["1"]])
    cmp(f"{k}.bar_comparator", m["bar_comparator"], d["bar_comparator"])
    cmp(f"{k}.bar_margin", ci3(m["bar_margin"]), ci3(d["bar_margin"]), TOL)
    cmp(f"{k}.margin_vs_counterpart", ci3(m["margin_vs_counterpart"]), ci3(d["margin_vs_counterpart"]), TOL)
    cmp(f"{k}.gain_statistic", ci3(m["gain_statistic"]), ci3(d["gain_statistic"]), TOL)
    cmp(f"{k}.either_change", m["either_change"], d["either_change"], TOL)
    cmp(f"{k}.d10", m["d10"], d["d10"])
    for p in ("emotion__style", "emotion__genre", "style__genre"):
        cmp(f"{k}.per_pair_bar_margin.{p}", ci3(m["per_pair_bar_margin"][p]), ci3(d["per_pair_bar_margin"][p]), TOL)
    if k != "AFF":
        cmp(f"{k}.delta_int", m["delta_int"], d["delta_int"])
        cmp(f"{k}.delta_int(carry.json)", m["delta_int"], car["delta_int"][k])
        cmp(f"{k}.delta", ci3(m["delta"]), ci3(d["delta"]), TOL)
        cmp(f"{k}.delta_point_from_int", m["delta_point_from_int"], d["delta"]["point"], 0.0)
        cmp(f"{k}.d10(carry.json)", m["d10"], car["d10"][k])
    cmp(f"{k}.open_tau0", mine["open_tau0"][k], dev["open_tau0_counts"][k])
cmp("carry.E", mine["carry"]["E"], car["carry"]["E"])
cmp("carry.M", mine["carry"]["M"], car["carry"]["M"])
cmp("carry.tied", mine["carry"]["tied"], car["carry"]["tied"])
cmp("carry.carried", mine["carry"]["carried"], car["carry"]["carried"])
cmp("carry.kill", mine["carry"]["kill"], car["kill"])
b = dev["beside_aff"]
cmp("beside.Bprime_A1_mean_r1", mine["bundle"]["Bp1_r1"], b["Bprime_A1_mean_r1"], 0.0)
cmp("beside.Bprime_A0_mean_r1", mine["bundle"]["Bp0_r1"], b["Bprime_A0_mean_r1"], 0.0)
cmp("beside.B_mean_r1", mine["bundle"]["B_r1"], b["B_mean_r1"], 0.0)
cmp("beside.AFF_minus_Bprime_A1", ci3(mine["beside_aff"]["AFF_minus_Bprime_A1"]), ci3(b["AFF_minus_Bprime_A1"]), TOL)

# rule targets (round 3 §5 items 2 and 3; this rule §5 item 3), compared exactly
RULE = {
    "R1": {"fused": [116, 119], "cf": [58, 123], "bar": [0.4435221354166667, 0.21646171563312194, 0.6735669710776852],
           "gain": [2.667236328125, 2.325087836946873, 3.012361650695922], "comp": "counterpart"},
    "AFF": {"fused": [39, 119], "cf": [149, 10], "bar": [0.6998697916666667, 0.4598852740816973, 0.9371680126852968],
            "gain": [3.110758463541667, 2.780005709854805, 3.4559584315470384], "comp": "Bprime_A0",
            "fused_r1": 19.136555989583336, "cf_r1": 18.39599609375, "either": -1.629638671875,
            "margin": [0.7405598958333333, 0.5196896694963071, 0.9598857494832738]},
    "IMGABST": {"fused": [117, 119], "cf": [58, 67], "bar": [0.5655924479166667, 0.3448683992591827, 0.79821625538382],
                "gain": [2.878824869791667, 2.554983173204304, 3.2105685950938248], "comp": "counterpart",
                "fused_r1": 19.059244791666664, "cf_r1": 18.49365234375, "either": -1.7476399739583333},
}
for k, t in RULE.items():
    m = mine["scorers"][k]
    cmp(f"rule.{k}.fused_cells", [m["fpick"]["0"], m["fpick"]["1"]], t["fused"])
    cmp(f"rule.{k}.cf_cells", [m["cpick"]["0"], m["cpick"]["1"]], t["cf"])
    cmp(f"rule.{k}.bar_margin", ci3(m["bar_margin"]), t["bar"], 0.0)
    cmp(f"rule.{k}.gain_statistic", ci3(m["gain_statistic"]), t["gain"], 0.0)
    cmp(f"rule.{k}.comparator", m["bar_comparator"], t["comp"])
    for f in ("fused_r1", "cf_r1"):
        if f in t:
            cmp(f"rule.{k}.{f}", m[f], t[f], 0.0)
    if "either" in t:
        cmp(f"rule.{k}.either", m["either_change"], t["either"], 0.0)
    if "margin" in t:
        cmp(f"rule.{k}.margin", ci3(m["margin_vs_counterpart"]), t["margin"], 0.0)
cmp("rule.AFF_minus_R1_fused", ci3(mine["beside_aff"]["AFF_minus_R1"]),
    [0.21769205729166666, 0.06425880757348419, 0.3709597330984391], 0.0)
cmp("rule.IMGABST.per_pair", [mine["scorers"]["IMGABST"]["per_pair_bar_margin"][p]["point"]
                              for p in ("emotion__style", "emotion__genre", "style__genre")],
    [0.677490234375, 1.28173828125, -0.262451171875], 0.0)
cmp("rule.AFF.per_pair", [mine["scorers"]["AFF"]["per_pair_bar_margin"][p]["point"]
                          for p in ("emotion__style", "emotion__genre", "style__genre")],
    [0.9765625, 1.45263671875, -0.32958984375], 0.0)
cmp("rule.Bp1_mean", mine["bundle"]["Bp1_r1"], 18.804931640625, 0.0)
cmp("rule.Bp0_mean", mine["bundle"]["Bp0_r1"], 18.436686197916664, 0.0)
cmp("rule.B_mean", mine["bundle"]["B_r1"], 18.341064453125, 0.0)

# arrays, exactly
MAP = {"R1": "r1", "IMGABST": "imgabst", "AFF": "aff", "V4": "v4", "V2": "v2", "V24": "v24"}
arr_rows = 0
for k, ik in MAP.items():
    for part in ("fused", "cf"):
        for m in ("r1", "gain", "other", "swap", "strict"):
            a, b2 = za[f"{k}_{part}_{m}"], zi[f"{ik}_{part}__{m}"]
            cmp(f"arr.{ik}_{part}__{m}", bool(np.array_equal(a, b2)), True)
            arr_rows += 1
    for c in "ab":
        cmp(f"arr.{ik}_gate__{c}", bool(np.array_equal(za[f"{k}_gate_{c}"], zi[f"{ik}_gate__{c}"])
                                         and za[f"{k}_gate_{c}"].dtype == zi[f"{ik}_gate__{c}"].dtype), True)
        arr_rows += 1
    cmp(f"arr.{ik}_fused_cells", [int(x) for x in zi[f"{ik}_fused_cells"]],
        [mine["scorers"][k]["fpick"]["0"], mine["scorers"][k]["fpick"]["1"]])
    cmp(f"arr.{ik}_cf_cells", [int(x) for x in zi[f"{ik}_cf_cells"]],
        [mine["scorers"][k]["cpick"]["0"], mine["scorers"][k]["cpick"]["1"]])
for nm, ik in (("B", "B"), ("Bp0", "Bp0"), ("Bp1", "Bp1")):
    for m in ("r1", "gain", "other", "swap", "strict"):
        cmp(f"arr.{ik}__{m}", bool(np.array_equal(za[f"{nm}_{m}"], zi[f"{ik}__{m}"])), True)
for k1, k2 in (("v", "v"), ("keep", "keep"), ("cl", "cl"), ("pair_index", "pair_index"), ("parity", "parity"),
               ("pick0_a", "pick__a"), ("pick0_b", "pick__b"), ("pick1_a", "pick_A1__a"), ("pick1_b", "pick_A1__b"),
               ("marg_a", "margin__a"), ("marg_b", "margin__b")):
    cmp(f"arr.{k2}", bool(np.array_equal(za[k1], zi[k2])), True)

bad = [r for r in rows if not r["ok"]]
rec = {"n": len(rows), "n_fail": len(bad), "failures": bad,
       "max_abs_diff_continuous": max((r["abs_diff"] for r in rows if r["abs_diff"] is not None), default=0.0),
       "rows": rows}
(OUT / "fr_agreement.json").write_text(json.dumps(rec, indent=1, default=str))
print(f"compared {len(rows)}; failures {len(bad)}; max abs diff {rec['max_abs_diff_continuous']}")
for r in bad:
    print("FAIL", r["name"], r["mine"], r["impl"], r["abs_diff"])
