"""Final review of round 5: compare the third derivation (out/fr5_results.json, out/fr5_arrays.npz) with the rule's
stated targets, the implementation's files (results/, cache/), round 4's seed42_arrays.npz, the re-derivation's
stage-B files and the report's section-6 rebuild (why_rebuild.json). Rule §8 tolerances: discrete and arrays
identical; tau within 1e-15 absolute or 1e-9 relative; every other number within 1e-9 pp. Writes out/fr5_agreement.json.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ROOT = Path("/project/CoSiR")
HERE = ROOT / "src/test/20261123_idea3_goemotions"
OUT = HERE / "final_review/out"
RES = HERE / "results"
ASSETS = ROOT / "docs/reports/assets/2026-11-23_idea3_goemotions"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


mine = json.loads((OUT / "fr5_results.json").read_text())
ma = np.load(OUT / "fr5_arrays.npz")
dev = json.loads((RES / "dev_seed42.json").read_text())
car = json.loads((RES / "carry.json").read_text())
plc = json.loads((RES / "placement.json").read_text())
dia = json.loads((RES / "diagnostics_seed42.json").read_text())
ia = np.load(RES / "seed42_arrays.npz")
r4a = np.load(ROOT / "src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz")
rdj = json.loads((HERE / "rederive/results/rd5_stageB.json").read_text())
rda = np.load(HERE / "rederive/results/rd5_stageB_arrays.npz")
why = json.loads((ASSETS / "why_rebuild.json").read_text())
fig = json.loads((ASSETS / "figure_data.json").read_text())

ROWS = []


def num(name, a, b, kind="pp"):
    a, b = float(a), float(b)
    d = abs(a - b)
    ok = d <= (1e-9 if kind == "pp" else max(1e-15, 1e-9 * abs(b)))
    ROWS.append({"kind": kind, "name": name, "mine": a, "theirs": b, "diff": d, "ok": bool(ok)})


def disc(name, a, b):
    ROWS.append({"kind": "discrete", "name": name, "mine": a, "theirs": b, "diff": None, "ok": bool(a == b)})


def arr(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = a.shape == b.shape and np.array_equal(a.astype(np.float64), b.astype(np.float64), equal_nan=True)
    ROWS.append({"kind": "array", "name": name, "mine": f"{a.dtype}{list(a.shape)}", "theirs": f"{b.dtype}{list(b.shape)}",
                 "diff": None if ok else float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                 if a.shape == b.shape else "shape", "ok": bool(ok)})


def ci(name, m, t):
    num(f"{name}.point", m["point"], t["point"])
    num(f"{name}.lo", m["ci95"][0], t["ci95"][0])
    num(f"{name}.hi", m["ci95"][1], t["ci95"][1])


S = mine["scorers"]
# ------------------------------------------------------------------ rule targets (§5 item 1, D8 beside, D4 accuracy)
R1, AF = S["R1"], S["AFF"]
for h, v in (("0", 116), ("1", 119)):
    disc(f"rule.R1.fpick{h}", R1["fpick"][h], v)
for h, v in (("0", 58), ("1", 123)):
    disc(f"rule.R1.cpick{h}", R1["cpick"][h], v)
ci("rule.R1.bar", R1["bar_margin"], {"point": 0.4435221354166667, "ci95": [0.21646171563312194, 0.6735669710776852]})
disc("rule.R1.bar_comparator", R1["bar_comparator"], "counterpart")
ci("rule.R1.gain", R1["gain_statistic"], {"point": 2.667236328125, "ci95": [2.325087836946873, 3.012361650695922]})
num("rule.AFF.fused", AF["fused_r1"], 19.136555989583336)
num("rule.AFF.cf", AF["cf_r1"], 18.39599609375)
disc("rule.AFF.bar_comparator", AF["bar_comparator"], "Bprime_A0")
ci("rule.AFF.bar", AF["bar_margin"], {"point": 0.6998697916666667, "ci95": [0.4598852740816973, 0.9371680126852968]})
ci("rule.AFF.margin", AF["margin_vs_counterpart"], {"point": 0.7405598958333333,
                                                    "ci95": [0.5196896694963071, 0.9598857494832738]})
ci("rule.AFF.gain", AF["gain_statistic"], {"point": 3.110758463541667, "ci95": [2.780005709854805, 3.4559584315470384]})
num("rule.AFF.either", AF["either_change"], -1.629638671875)
for p, v in (("emotion__style", 0.9765625), ("emotion__genre", 1.45263671875), ("style__genre", -0.32958984375)):
    num(f"rule.AFF.per_pair_bar.{p}", AF["per_pair_bar_margin"][p]["point"], v)
for h, v in (("0", 39), ("1", 119)):
    disc(f"rule.AFF.fpick{h}", AF["fpick"][h], v)
for h, v in (("0", 149), ("1", 10)):
    disc(f"rule.AFF.cpick{h}", AF["cpick"][h], v)
for h in ("0", "1"):
    disc(f"rule.sigma{h}", float(mine["sigma_control"][h][0]), 0.0)
ax = mine["aff_extra"]
ci("rule.AFF_minus_R1.fused", ax["AFF_minus_R1_fused"], {"point": 0.21769205729166666,
                                                         "ci95": [0.06425880757348419, 0.3709597330984391]})
ci("rule.AFF_minus_R1.bar", ax["AFF_minus_R1_bar"], {"point": 0.25634765625,
                                                     "ci95": [0.04280778303598444, 0.46195041633015954]})
disc("rule.AFF.open_tau0", [ax["open_tau0"]["a"], ax["open_tau0"]["b"]], [9941, 3627])
ci("rule.AFF_minus_Bprime_A1", ax["AFF_minus_Bprime_A1"], {"point": 0.33162434895833337,
                                                           "ci95": [0.048231414333532084, 0.6246158772581268]})
num("rule.Bp1_mean", ax["Bp1_r1"], 18.804931640625)
num("rule.B_mean", ax["B_r1"], 18.341064453125)
num("rule.Bp0_mean", ax["Bp0_r1"], 18.436686197916664)
num("rule.AUC_AFF", mine["diagnostics"]["a_detection_auc"]["AFF"], 0.7870951145887375, kind="tau")
L = mine["diagnostics"]["c_pair_lift"]["image_x_caption_CLIP"]
num("rule.pair_ratio_CLIP", L["ratio_same_over_diff"], 1.1445184466303795, kind="tau")
num("rule.pair_exs_CLIP", L["emotionxstyle"], 1.070827615209179, kind="tau")
num("rule.pair_exg_CLIP", L["emotionxgenre"], 1.0375160447836334, kind="tau")
red = mine["why"]["redundancy"]["CLIP"]
for h, vi, vt in (("affect", 0.35348060377541385, 0.3828024789253903), ("image", 0.7145397990123284, 0.7090878258485419),
                  ("caption", 0.6182295729609555, 0.665152773464146)):
    num(f"rule.D7.{h}.i2t", red[h]["i2t"]["B"], vi, kind="tau")
    num(f"rule.D7.{h}.t2i", red[h]["t2i"]["B"], vt, kind="tau")
# round 4's stored arrays for R1 and AFF
for k in ("r1", "aff"):
    for part in ("fused", "cf"):
        for m in ("r1", "gain", "other", "swap", "strict"):
            arr(f"r4arrays.{k}_{part}__{m}", ma[f"{k}_{part}__{m}"], r4a[f"{k}_{part}__{m}"])
        arr(f"r4arrays.{k}_{part}_cells", ma[f"{k}_{part}_cells"], r4a[f"{k}_{part}_cells"])
    for c in ("a", "b"):
        arr(f"r4arrays.{k}_gate__{c}", ma[f"{k}_gate__{c}"], r4a[f"{k}_gate__{c}"])

# ------------------------------------------------------------------ implementation: placement.json and the GE file
g = mine["ge_head"]
disc("impl.placement.accuracy", g["heldout_accuracy"], plc["ge_head"]["heldout_accuracy"])
disc("impl.placement.n_iter", g["n_iter"], plc["ge_head"]["n_iter"])
disc("impl.placement.fallback", g["fallback"], plc["ge_head"]["fallback_used"])
disc("impl.placement.majority", g["check_majority_share"], plc["ge_head"]["check_majority_share"])
arr("impl.Q_GE_sel", ma["QGE_sel"], np.load(HERE / "cache/r5_ge_posterior.npz")["post_sel"])

# ------------------------------------------------------------------ implementation: dev_seed42.json and carry.json
NAMEMAP = {"Bprime_G": "Bprime_G", "Bprime_A0": "Bprime_A0", "counterpart": "counterpart", "B": "B"}
for k in ("G-T", "G-TF"):
    m, t = S[k], dev["candidates"][k]
    num(f"impl.{k}.fused", m["fused_r1"], t["fused_r1"])
    num(f"impl.{k}.cf", m["cf_r1"], t["cf_r1"])
    for h in ("0", "1"):
        disc(f"impl.{k}.fpick{h}", m["fpick"][h], t["cells"]["fpick"][h])
        disc(f"impl.{k}.cpick{h}", m["cpick"][h], t["cells"]["cpick"][h])
        disc(f"impl.{k}.sigma{h}", float(m["sigma"][h]), t["sigma"][h])
    disc(f"impl.{k}.bar_comparator", m["bar_comparator"], t["bar_comparator"])
    for n, v in m["comparator_means"].items():
        num(f"impl.{k}.mean.{n}", v, t["comparator_means"][NAMEMAP[n]])
    ci(f"impl.{k}.bar", m["bar_margin"], t["bar_margin"])
    ci(f"impl.{k}.margin", m["margin_vs_counterpart"], t["margin_vs_counterpart"])
    ci(f"impl.{k}.gain", m["gain_statistic"], t["gain_statistic"])
    num(f"impl.{k}.either", m["either_change"], t["either_change"])
    for p in ("emotion__style", "emotion__genre", "style__genre"):
        ci(f"impl.{k}.per_pair_bar.{p}", m["per_pair_bar_margin"][p], t["per_pair_bar_margin"][p])
    disc(f"impl.{k}.delta_int", m["delta_int"], t["delta_int"])
    ci(f"impl.{k}.delta", m["delta"], t["delta"])
    num(f"impl.{k}.delta_point_from_int", m["delta_point_from_int"], t["delta"]["point"])
    for cl_ in ("c1", "c2", "c3", "clears"):
        disc(f"impl.{k}.d10.{cl_}", m["d10"][cl_], t["d10"][cl_])
        disc(f"impl.carry.{k}.d10.{cl_}", m["d10"][cl_], car["candidates"][k]["d10"][cl_])
    disc(f"impl.carry.{k}.delta", m["delta_int"], car["candidates"][k]["delta_int"])
    ci(f"impl.{k}.minus_Bp1", m["minus_Bprime_A1"], t["beside_Bprime_A1"]["candidate_minus"])
    disc(f"impl.{k}.open_tau0", m["open_tau0"], dev["open_tau0_counts"][k])
ci("impl.BpG_minus_Bp0", mine["BpG"]["minus_Bp0"], dev["comparators"]["Bprime_G_minus_Bprime_A0"])
num("impl.BpG_mean", mine["BpG"]["mean_r1"], dev["comparators"]["Bprime_G_mean_r1"])
for i, v in enumerate(mine["tau_prime"]):
    num(f"impl.tau_prime[{i}]", v, dev["tau_prime"][i], kind="tau")
for key in ("E", "M", "tied", "carried", "kill"):
    disc(f"impl.carry.{key}", mine["carry"][key], car[key])
disc("impl.carry.dev_sha", car["dev_seed42_sha256"], sha(RES / "dev_seed42.json"))
disc("impl.carry.arrays_sha", car["seed42_arrays_sha256"], sha(RES / "seed42_arrays.npz"))
# AFF block of dev_seed42
num("impl.AFF.fused", AF["fused_r1"], dev["aff"]["fused_r1"])
ci("impl.AFF.bar", AF["bar_margin"], dev["aff"]["bar_margin"])
ci("impl.AFF.gain", AF["gain_statistic"], dev["aff"]["gain_statistic"])
ci("impl.AFF.margin", AF["margin_vs_counterpart"], dev["aff"]["margin_vs_counterpart"])

# ------------------------------------------------------------------ implementation: seed42_arrays.npz (every key)
MAP = {}
for k in ("aff", "gt", "gtf"):
    for part in ("fused", "cf"):
        for m in ("r1", "gain", "other", "swap", "strict"):
            MAP[f"{k}_{part}__{m}"] = f"{k}_{part}__{m}"
        MAP[f"{k}_{part}_cells"] = f"{k}_{part}_cells"
    for c in ("a", "b"):
        MAP[f"{k}_gate__{c}"] = f"{k}_gate__{c}"
for k in ("aff", "gtf"):
    for c in ("a", "b"):
        for q in ("P", "pick", "margin"):
            MAP[f"{k}_{q}__{c}"] = f"{k}_{q}__{c}"
for nm in ("B", "Bp0", "Bp1", "BpG"):
    for m in ("r1", "gain", "other", "swap", "strict"):
        MAP[f"{nm}__{m}"] = f"{nm}__{m}"
for x in ("tau_prime", "taus", "cl", "pair_index", "parity"):
    MAP[x] = x
for theirs, ours in MAP.items():
    arr(f"impl.arrays.{theirs}", ma[ours], ia[theirs])
for k in ("aff", "gt", "gtf"):
    arr(f"impl.arrays.{k}_sigma", np.asarray([0.0, 0.0]), ia[f"{k}_sigma"])
gate_dtypes_ok = all(ia[f"{k}_gate__{c}"].dtype == np.float32 for k in ("aff", "gt", "gtf") for c in "ab")
disc("impl.arrays.gates_float32", gate_dtypes_ok, True)
unmatched = sorted(set(ia.files) - set(MAP) - {"aff_sigma", "gt_sigma", "gtf_sigma"})

# ------------------------------------------------------------------ implementation: diagnostics_seed42.json
D, Di = mine["diagnostics"], dia
num("impl.diag.a.GTF", D["a_detection_auc"]["G-TF"], Di["a_detection_auc"]["G-TF"], kind="tau")
num("impl.diag.a.AFF", D["a_detection_auc"]["AFF"], Di["a_detection_auc"]["AFF_recomputed"], kind="tau")
num("impl.diag.b.F", D["b_delta_affect_auc"]["F"], Di["b_delta_affect_auc"]["F"], kind="tau")
num("impl.diag.b.F_G", D["b_delta_affect_auc"]["F_G"], Di["b_delta_affect_auc"]["F_G"], kind="tau")
LG = D["c_pair_lift"]["image_x_caption_GE"]
num("impl.diag.c.GE.ratio", LG["ratio_same_over_diff"], Di["c_pair_lift"]["GE"]["ratio_same_over_diff"], kind="tau")
num("impl.diag.c.GE.exs", LG["emotionxstyle"], Di["c_pair_lift"]["GE"]["emotionxstyle"], kind="tau")
num("impl.diag.c.GE.exg", LG["emotionxgenre"], Di["c_pair_lift"]["GE"]["emotionxgenre"], kind="tau")
num("impl.diag.c.CLIP.ratio", L["ratio_same_over_diff"], Di["c_pair_lift"]["CLIP_item4"]["ratio_same_over_diff"],
    kind="tau")
for k in ("AFF", "G-T", "G-TF"):
    md, td = D["d_sharper_term"][k], Di["d_sharper_term"][k]
    for key, v in md["per_pair_condition"].items():
        p, c = key.split("|")
        for q in ("r1", "gain", "either"):
            ci(f"impl.diag.d.{k}.{p}.{c}.{q}", v[q], td["per_pair_condition"][p][c][q])
    for d_, v in md["per_direction"].items():
        for q in ("r1", "gain", "either"):
            ci(f"impl.diag.d.{k}.{d_}.{q}", v[q], td["per_direction"][d_][q])
    if k != "AFF":
        for key, v in md["minus_aff"]["per_pair_condition"].items():
            p, c = key.split("|")
            for q in ("r1", "gain", "either"):
                ci(f"impl.diag.d.{k}.minus_aff.{p}.{c}.{q}", v[q], td["minus_aff"]["per_pair_condition"][p][c][q])
        for d_, v in md["minus_aff"]["per_direction"].items():
            for q in ("r1", "gain", "either"):
                ci(f"impl.diag.d.{k}.minus_aff.{d_}.{q}", v[q], td["minus_aff"]["per_direction"][d_][q])
    for part, pk in (("fused", "fpick"), ("cf", "cpick")):
        for h in ("0", "1"):
            disc(f"impl.diag.d.{k}.cells.{part}{h}", S[k][pk][h], td["cells"][part][h]["cell"])
    key = "AFF_recomputed" if k == "AFF" else k
    num(f"impl.diag.d.either_cost.{k}", md["either_cost_per_gain"], Di["d_either_cost_per_unit_gain"][key], kind="tau")
# orientation: AFF fused minus counterpart per direction equals bs_09_direction.json (AFF.i2t, AFF.t2i)
bs9 = json.loads((ROOT / "src/test/20261120_r1_levers_brainstorm/results/bs_09_direction.json").read_text())
for d_ in ("i2t", "t2i"):
    v = bs9["AFF"][d_]["fused"]["r1"] - bs9["AFF"][d_]["cf"]["r1"]
    num(f"orientation.bs09.AFF.{d_}", D["d_sharper_term"]["AFF"]["per_direction"][d_]["r1"]["point"], v, kind="pp_loose")

# ------------------------------------------------------------------ re-derivation stage B
rec = rdj["records"]
for k in ("G-T", "G-TF"):
    num(f"rederive.{k}.fused", S[k]["fused_r1"], rec[k]["fused_r1"])
    ci(f"rederive.{k}.bar", S[k]["bar_margin"], rec[k]["bar_margin"])
    ci(f"rederive.{k}.gain", S[k]["gain_statistic"], rec[k]["gain_statistic"])
    disc(f"rederive.{k}.delta", S[k]["delta_int"], rec[k]["delta_vs_aff"]["delta_int"])
    rk = "GT" if k == "G-T" else "GTF"
    ik = "gt" if k == "G-T" else "gtf"
    for part in ("fused", "cf"):
        for m in ("r1", "gain", "other", "swap", "strict"):
            arr(f"rederive.arrays.{rk}_{part}__{m}", ma[f"{ik}_{part}__{m}"], rda[f"{rk}_{part}__{m}"])
    for c in ("a", "b"):
        arr(f"rederive.arrays.{rk}_gate__{c}", ma[f"{ik}_gate__{c}"], rda[f"{rk}_gate__{c}"])
arr("rederive.arrays.Q_GE_sel", ma["QGE_sel"], rda["Q_GE_sel"])
for m in ("r1", "gain", "other", "swap", "strict"):
    arr(f"rederive.arrays.BpG__{m}", ma[f"BpG__{m}"], rda[f"BpG__{m}"])
for i, v in enumerate(mine["tau_prime"]):
    num(f"rederive.tau_prime[{i}]", v, rda["GTF_taus"][i], kind="tau")

# ------------------------------------------------------------------ the report's section-6 rebuild
W = mine["why"]
for k in ("G-T", "G-TF"):
    for key, v in W["per_cell_minus_AFF"][k].items():
        disc(f"why.per_cell.{k}.{key}.net", v["net"], why["per_cell_minus_AFF"][k][key]["net_rankings"])
        ci(f"why.per_cell.{k}.{key}.r1", v["r1"], why["per_cell_minus_AFF"][k][key]["r1"])
FX = {"G-T@AFF": "G-T_at_AFF_cells", "G-TF@AFF": "G-TF_at_AFF_cells", "AFF@G-T": "AFF_at_G-T_cells",
      "AFF@G-TF": "AFF_at_G-TF_cells"}
for lab, theirs in FX.items():
    v, t = W["fixed_cells"][lab], why["fixed_cells"][theirs]
    num(f"why.fixed.{lab}.fused", v["fused_r1"], t["fused_r1"])
    disc(f"why.fixed.{lab}.net", v["net"], t["net_vs_AFF"])
    ci(f"why.fixed.{lab}.minus", v["minus_AFF"], t["minus_AFF"])
    ci(f"why.fixed.{lab}.gain", v["gain"], t["gain_minus_AFF"])
    ci(f"why.fixed.{lab}.either", v["either"], t["either_minus_AFF"])
    for key, n in v["per_cell"].items():
        disc(f"why.fixed.{lab}.cell.{key}", n, t["per_cell_net_vs_AFF"][key])
for lab in ("CLIP", "GE"):
    for key, v in W["affect_alone"][lab].items():
        for q in ("pA_first", "pB_first", "pA_above_pB"):
            num(f"why.alone.{lab}.{key}.{q}", v[q], why["affect_score_alone"][lab][key][q])
    for h in ("affect", "image", "caption"):
        for d_ in ("i2t", "t2i"):
            num(f"why.red.{lab}.{h}.{d_}.B", W["redundancy"][lab][h][d_]["B"], why["redundancy"][f"{lab}_D7"][h][d_],
                kind="tau")
            num(f"why.red.{lab}.{h}.{d_}.cos", W["redundancy"][lab][h][d_]["cos"],
                why["redundancy"]["with_cosine"][lab][h][d_], kind="tau")
for k in ("AFF", "G-T", "G-TF"):
    for c in ("a", "b"):
        t = why["redundancy"]["gated_term_with_B"][k][c]
        disc(f"why.gated.{k}.{c}.open", W["gated_term"][k][c]["open"], t["open_as_scored"])
        for d_ in ("i2t", "t2i"):
            num(f"why.gated.{k}.{c}.{d_}", W["gated_term"][k][c][d_], t[d_], kind="tau")
            if k != "AFF":
                num(f"why.gated.{k}.{c}.{d_}.vsAFF", W["gated_term"][k][c][f"{d_}_vs_AFF_term"],
                    t[f"{d_}_corr_with_AFF_term"], kind="tau")
for lab in ("image_x_caption_CLIP", "image_x_caption_GE", "caption_x_caption_CLIP", "caption_x_caption_GE",
            "image_x_image"):
    for q in ("ratio_same_over_diff", "emotionxstyle", "emotionxgenre"):
        num(f"why.lift.{lab}.{q}", D["c_pair_lift"][lab][q], why["pair_lifts"][lab][q], kind="tau")
for lab in ("CLIP", "GE", "image_x_image"):
    v, t = W["same_painting"][lab], why["same_painting"][lab]
    num(f"why.same_painting.{lab}.ratio", v["ratio_same_painting"], t["ratio_same_painting_over_diff"], kind="tau")
    num(f"why.same_painting.{lab}.row", v["ratio_same_row"], t["ratio_same_row_over_diff"], kind="tau")
    disc(f"why.same_painting.{lab}.n", v["n_pairs"], t["n_same_painting_diff_emotion_pairs"])
for lab, theirs in (("image", "image_head"), ("caption_CLIP", "caption_CLIP"), ("caption_GE", "caption_GE")):
    num(f"why.sharp.{lab}.max", W["sharpness"][lab]["mean_max"], why["sharpness"][theirs]["mean_max"], kind="tau")
    num(f"why.sharp.{lab}.ent", W["sharpness"][lab]["mean_entropy"], why["sharpness"][theirs]["mean_entropy_nats"],
        kind="tau")
num("why.sharp.argmax_CLIP", W["sharpness"]["argmax_row_img_cap_CLIP"],
    why["sharpness"]["argmax_same_row_image_vs_caption_CLIP"])
num("why.sharp.argmax_GE", W["sharpness"]["argmax_row_img_cap_GE"],
    why["sharpness"]["argmax_same_row_image_vs_caption_GE"])
for h in ("0", "1"):
    disc(f"why.Bp_picks.G.{h}", mine["BpG"]["picks"][h], list(why["Bprime_picks"]["Bprime_G"][h]))
    disc(f"why.Bp_picks.A0.{h}", mine["BpG"]["picks_Bp0"][h], list(why["Bprime_picks"]["Bprime_A0"][h]))
for k in ("AFF", "G-T", "G-TF"):
    v = why["in_sample_family"][k]["fused_r1"]
    arr(f"why.in_sample.{k}.fused_r1", np.asarray(W["in_sample"][k]["fused_r1_cells"]), np.asarray(v))
    arr(f"why.in_sample.{k}.gain", np.asarray(W["in_sample"][k]["gain_cells_flat"]),
        np.asarray(why["in_sample_family"][k]["fused_gain"]))

kinds = {}
for r in ROWS:
    kinds.setdefault(r["kind"], [0, 0])
    kinds[r["kind"]][0] += 1
    kinds[r["kind"]][1] += int(not r["ok"])
failed = [r for r in ROWS if not r["ok"]]
num_rows = [r for r in ROWS if r["kind"] in ("pp", "tau") and r["diff"] is not None]
largest = max(num_rows, key=lambda r: r["diff"])
out = {"n": len(ROWS), "by_kind": kinds, "failed": failed, "largest_numeric": largest,
       "implementation_array_keys_not_compared": unmatched, "rows": ROWS,
       "mine_sha256": {"fr5_results.json": sha(OUT / "fr5_results.json"), "fr5_arrays.npz": sha(OUT / "fr5_arrays.npz")}}
(OUT / "fr5_agreement.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({"n": len(ROWS), "by_kind": kinds, "n_failed": len(failed), "largest": largest,
                  "unmatched_impl_arrays": unmatched}, indent=1, default=str))
for r in failed[:40]:
    print("FAIL", r)
