"""Final review: recompute the draft report's descriptive numbers (§6, Summary, Tables 3 to 7) from the third
derivation's own arrays (out/fr_arrays.npz), not from seed42_arrays.npz or figure_data.json, then compare with
figure_data.json. Writes out/fr_descriptive.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/project/CoSiR")
sys.path.insert(0, str(ROOT))
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

HERE = ROOT / "src/test/20261122_round4_aff_vetoes"
OUT = HERE / "final_review/out"
z = np.load(OUT / "fr_arrays.npz")
fr = json.loads((OUT / "fr_results.json").read_text())
fd = json.loads((ROOT / "docs/reports/assets/2026-11-22_round4_aff_vetoes/figure_data.json").read_text())
E = 12288
cl, pidx, par = z["cl"], z["pair_index"], z["parity"]
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
CANDS = ("V4", "V2", "V24")
ar = np.arange(E)
eff = np.where(par == 1, 0, 2)          # parity-1 episodes scored by the tune-half-0 cell 39 (tau_0), parity 0 by 119


def pci(x, m=None):
    x = np.asarray(x, float)
    c = cl
    if m is not None:
        x, c = x[m], cl[m]
    r = cluster_bootstrap(x, c)
    return [100 * r["point"], 100 * r["ci95"][0], 100 * r["ci95"][1]]


def gate(k, c, t):
    g = z[f"{k}_gate_{c}"]
    return (g[eff, ar] if t == "s" else g[t]).astype(bool)


def i4(x):
    return np.rint(4 * np.asarray(x)).astype(np.int64)


for k in ("AFF",) + CANDS:
    assert tuple(fr["scorers"][k]["fpick"].values()) == (39, 119)

R = {}
aff = {m: z[f"AFF_fused_{m}"] for m in ("r1", "gain", "other")}
B = {m: z[f"B_{m}"] for m in ("r1", "gain", "other")}
# closure shares
clo = {}
for k in CANDS:
    clo[k] = {}
    for t in (0, 2, "s"):
        row = {}
        for i, p in enumerate(PAIRS):
            for c in "ab":
                m = pidx == i
                ao = gate("AFF", c, t) & m
                cc = ao & ~gate(k, c, t)
                row[f"{p}|{c}"] = [int(cc.sum()), int(ao.sum()), round(100 * cc.sum() / ao.sum(), 2)]
        emo = [row[f"{p}|a"] for p in PAIRS[:2]]
        oth = [row[f"{p}|b"] for p in PAIRS[:2]] + [row[f"{PAIRS[2]}|{c}"] for c in "ab"]
        row["emotion_side"] = [sum(r[0] for r in emo), sum(r[1] for r in emo)]
        row["other"] = [sum(r[0] for r in oth), sum(r[1] for r in oth)]
        for kk in ("emotion_side", "other"):
            row[kk].append(round(100 * row[kk][0] / row[kk][1], 2))
        clo[k][str(t)] = row
R["closure"] = clo

# net rankings by class, changed episodes, better/worse, fully closed = B
nr = {}
for k in CANDS:
    d4 = i4(z[f"{k}_fused_r1"]) - i4(aff["r1"])
    ca = gate("AFF", "a", "s") & ~gate(k, "a", "s")
    cb = gate("AFF", "b", "s") & ~gate(k, "b", "s")
    ch = ca | cb
    full_k = ~gate(k, "a", "s") & ~gate(k, "b", "s")
    row = {"delta": int(d4.sum()), "changed": int(ch.sum()), "d_zero_outside": bool(np.all(d4[~ch] == 0)),
           "up": int((d4 > 0).sum()), "down": int((d4 < 0).sum()),
           "full_and_changed": int((ch & full_k).sum()),
           "full_all": int(full_k.sum()),
           "full_all_equals_B": bool(np.array_equal(z[f"{k}_fused_r1"][full_k], B["r1"][full_k])),
           "n_full_all_differs_from_B": int((z[f"{k}_fused_r1"][full_k] != B["r1"][full_k]).sum())}
    emo = pidx < 2
    row["emo_a_only"] = [int((emo & ca & ~cb).sum()), int(d4[emo & ca & ~cb].sum())]
    row["emo_b_only"] = [int((emo & cb & ~ca).sum()), int(d4[emo & cb & ~ca].sum())]
    row["emo_both"] = [int((emo & ca & cb).sum()), int(d4[emo & ca & cb].sum())]
    sg = (pidx == 2) & ch
    row["sg_any"] = [int(sg.sum()), int(d4[sg].sum())]
    row["per_pair_minus_AFF"] = {p: pci(z[f"{k}_fused_r1"] - aff["r1"], pidx == i) for i, p in enumerate(PAIRS)}
    row["gain_minus_AFF"] = pci(z[f"{k}_fused_gain"] - aff["gain"])
    row["other_minus_AFF"] = pci(z[f"{k}_fused_other"] - aff["other"])
    row["either_minus_AFF"] = pci((z[f"{k}_fused_r1"] + z[f"{k}_fused_other"]) - (aff["r1"] + aff["other"]))
    nr[k] = row
R["net"] = nr
fullA = ~gate("AFF", "a", "s") & ~gate("AFF", "b", "s")
R["AFF_full_closed_equals_B"] = [int(fullA.sum()), bool(np.array_equal(aff["r1"][fullA], B["r1"][fullA]))]
R["V4_vs_V2"] = {"r1_differ": int((z["V4_fused_r1"] != z["V2_fused_r1"]).sum()),
                 "gate_a_differ_tau0": int((z["V4_gate_a"][0] != z["V2_gate_a"][0]).sum()),
                 "gate_b_differ_tau0": int((z["V4_gate_b"][0] != z["V2_gate_b"][0]).sum()),
                 "gates_differ_any_tau": int(sum((z["V4_gate_" + c][t] != z["V2_gate_" + c][t]).sum()
                                                 for c in "ab" for t in range(4))),
                 "V4_equals_AFF_gate_everywhere": bool(all(np.array_equal(z["V4_gate_" + c], z["AFF_gate_" + c])
                                                           for c in "ab")),
                 "V2_equals_AFF_gate_everywhere": bool(all(np.array_equal(z["V2_gate_" + c], z["AFF_gate_" + c])
                                                           for c in "ab")),
                 "V4_r1_equals_AFF": bool(np.array_equal(z["V4_fused_r1"], aff["r1"])),
                 "V2_r1_equals_AFF": bool(np.array_equal(z["V2_fused_r1"], aff["r1"])),
                 "other_differ": int((z["V4_fused_other"] != z["V2_fused_other"]).sum()),
                 "gain_V4_V2": [fr["scorers"]["V4"]["gain_statistic"]["point"],
                                fr["scorers"]["V2"]["gain_statistic"]["point"]]}

# V4 closed vs kept: AFF minus B
ca = gate("AFF", "a", "s") & ~gate("V4", "a", "s")
cb = gate("AFF", "b", "s") & ~gate("V4", "b", "s")
closed = ca | cb
kept = gate("V4", "a", "s") | gate("V4", "b", "s")
dB = aff["r1"] - B["r1"]
R["V4_closed_kept"] = {p: {"closed": pci(dB, (pidx == i) & closed) + [int(((pidx == i) & closed).sum())],
                           "kept": pci(dB, (pidx == i) & kept) + [int(((pidx == i) & kept).sum())]}
                       for i, p in enumerate(PAIRS)}

# comparator gaps
gaps = {}
for lab, v in (("AFF_minus_B", aff["r1"] - B["r1"]), ("Bp0_minus_B", z["Bp0_r1"] - B["r1"]),
               ("Bp1_minus_Bp0", z["Bp1_r1"] - z["Bp0_r1"]), ("AFF_minus_Bp1", aff["r1"] - z["Bp1_r1"])):
    gaps[lab] = {"pooled": pci(v), **{p: pci(v, pidx == i) for i, p in enumerate(PAIRS)}}
R["gaps"] = gaps

# R1 abstention table at tau_2 and as scored
rr = {}
for c in "ab":
    r1o, iao, afo, v4o = (z[f"{k}_gate_{c}"][2].astype(bool) for k in ("R1", "IMGABST", "AFF", "V4"))
    cr = r1o & ~iao
    rr[c] = {"R1_open": int(r1o.sum()), "abst_closes_R1": int(cr.sum()), "AFF_already_shut": int((cr & ~afo).sum()),
             "AFF_open": int(afo.sum()), "abst_closes_AFF": int((afo & ~v4o).sum())}
    # AFF as scored (tau_0 on parity 1, tau_2 on parity 0)
    afs, v4s = gate("AFF", c, "s"), gate("V4", c, "s")
    rr[c]["AFF_open_as_scored"] = int(afs.sum())
    rr[c]["abst_closes_AFF_as_scored"] = int((afs & ~v4s).sum())
    r10 = z[f"R1_gate_{c}"][0].astype(bool)
    rr[c]["AFF_open_tau0"] = int(z[f"AFF_gate_{c}"][0].sum())
    rr[c]["abst_closes_AFF_tau0"] = int((z[f"AFF_gate_{c}"][0].astype(bool) & ~z[f"V4_gate_{c}"][0].astype(bool)).sum())
R["R1_abst_tau2"] = rr
R["R1_fused"] = 100 * float(np.mean(z["R1_fused_r1"]))
R["IMG_fused"] = 100 * float(np.mean(z["IMGABST_fused_r1"]))
R["R1_per_pair_bar_vs_cf"] = {p: 100 * float(np.mean((z["R1_fused_r1"] - z["R1_cf_r1"])[pidx == i]))
                              for i, p in enumerate(PAIRS)}
ins = {k: fr["scorers"][k]["insample"] for k in ("R1", "IMGABST", "AFF", "V4", "V2", "V24")}
R["insample"] = {k: {"fused_best": 100 * v["fused_best_r4"] / 4 / E, "fused_cell": v["fused_best_cell"],
                     "cf_best": 100 * v["cf_best_r4"] / 4 / E, "cf_cell": v["cf_best_cell"],
                     "margin": 100 * (v["fused_best_r4"] - v["cf_best_r4"]) / 4 / E} for k, v in ins.items()}
R["insample_change_IMG_minus_R1"] = {"margin": R["insample"]["IMGABST"]["margin"] - R["insample"]["R1"]["margin"],
                                     "fused_best": R["insample"]["IMGABST"]["fused_best"]
                                     - R["insample"]["R1"]["fused_best"]}
R["insample_change_cand_minus_AFF"] = {k: {"margin": R["insample"][k]["margin"] - R["insample"]["AFF"]["margin"],
                                           "fused_best": R["insample"][k]["fused_best"]
                                           - R["insample"]["AFF"]["fused_best"]} for k in CANDS}
R["crit"] = {k: fr["scorers"][k]["crit"] for k in ("AFF",) + CANDS}

# A1 pick agreement on AFF's open emotion-side values at tau_0
a0 = z["AFF_gate_a"][0].astype(bool) & (pidx < 2)
R["A1_agrees_affect_emo_tau0"] = round(100 * float((z["pick1_a"][a0] == 0).mean()), 2)

(OUT / "fr_descriptive.json").write_text(json.dumps(R, indent=1))

# ---- compare with figure_data.json
D = fd["descriptive"]
diffs = []


def chk(name, mine, theirs, tol=1e-12):
    if isinstance(mine, (list, tuple)):
        ok = len(mine) == len(theirs) and all(abs(a - b) <= tol for a, b in zip(mine, theirs))
    elif isinstance(mine, float):
        ok = abs(mine - theirs) <= tol
    else:
        ok = mine == theirs
    if not ok:
        diffs.append((name, mine, theirs))


for k in CANDS:
    for t in ("0", "s"):
        tk = "as_scored" if t == "s" else t
        for key in [f"{p}|{c}" for p in PAIRS for c in "ab"]:
            chk(f"closure.{k}.{t}.{key}", clo[k][t][key][:2],
                [D["closure"][k][tk][key]["closed"], D["closure"][k][tk][key]["aff_open"]])
        for a, b in (("emotion_side", "emotion_side"), ("other", "other_values")):
            chk(f"closure.{k}.{t}.{a}", clo[k][t][a][:2], [D["closure"][k][tk][b]["closed"], D["closure"][k][tk][b]["aff_open"]])
    n = D["net_rankings"][k]
    chk(f"net.{k}.changed", nr[k]["changed"], n["changed_episodes"])
    chk(f"net.{k}.up", nr[k]["up"], n["up"])
    chk(f"net.{k}.down", nr[k]["down"], n["down"])
    chk(f"net.{k}.emo_a_only", nr[k]["emo_a_only"], [sum(n[p]["a_only"]["episodes"] for p in PAIRS[:2]),
                                                     sum(n[p]["a_only"]["net_rankings"] for p in PAIRS[:2])])
    chk(f"net.{k}.emo_b_only", nr[k]["emo_b_only"], [sum(n[p]["b_only"]["episodes"] for p in PAIRS[:2]),
                                                     sum(n[p]["b_only"]["net_rankings"] for p in PAIRS[:2])])
    chk(f"net.{k}.emo_both", nr[k]["emo_both"], [sum(n[p]["both"]["episodes"] for p in PAIRS[:2]),
                                                 sum(n[p]["both"]["net_rankings"] for p in PAIRS[:2])])
    chk(f"net.{k}.sg", nr[k]["sg_any"], [sum(n[PAIRS[2]][c]["episodes"] for c in ("a_only", "b_only", "both")),
                                         sum(n[PAIRS[2]][c]["net_rankings"] for c in ("a_only", "b_only", "both"))])
    for p in PAIRS:
        m = D["minus_AFF"][k][p]
        chk(f"minusAFF.{k}.{p}", nr[k]["per_pair_minus_AFF"][p], [m["point"], *m["ci95"]])
    for a in ("gain", "other", "either"):
        m = D["minus_AFF"][k][a]
        chk(f"minusAFF.{k}.{a}", nr[k][f"{a}_minus_AFF"], [m["point"], *m["ci95"]])
for p in PAIRS:
    for w in ("closed", "kept"):
        m = D["V4_closed_vs_kept_AFF_minus_B"][p][w]
        chk(f"V4ck.{p}.{w}", R["V4_closed_kept"][p][w], [m["point"], *m["ci95"], m["episodes"]])
for lab, theirs in (("AFF_minus_B", "AFF_minus_B"), ("Bp0_minus_B", "Bprime_A0_minus_B"),
                    ("Bp1_minus_Bp0", "Bprime_A1_minus_Bprime_A0"), ("AFF_minus_Bp1", "AFF_minus_Bprime_A1")):
    for p in ("pooled",) + PAIRS:
        m = D["comparator_gaps"][theirs][p]
        chk(f"gaps.{lab}.{p}", gaps[lab][p], [m["point"], *m["ci95"]])
for c in "ab":
    t = D["R1_abstention_vs_AFF_tau2"][c]["all"]
    chk(f"R1abst.{c}", [rr[c][k] for k in ("R1_open", "abst_closes_R1", "AFF_already_shut", "AFF_open", "abst_closes_AFF")],
        [t[k] for k in ("R1_open", "abstention_closes_on_R1", "of_which_AFF_gate_already_shut", "AFF_open",
                        "abstention_closes_on_AFF")])
chk("V4_vs_V2.differ", R["V4_vs_V2"]["r1_differ"], D["V4_vs_V2"]["episodes_differ"])
print(json.dumps({"n_diffs": len(diffs), "diffs": diffs}, indent=1, default=str))
