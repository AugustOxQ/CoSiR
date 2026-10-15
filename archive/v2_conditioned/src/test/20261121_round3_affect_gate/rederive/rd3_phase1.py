"""Phase 1 of the round-3 independent re-derivation (rule §8): seed 42 only.

§5 item 1 (bundle against stored references, D7 redundancy), item 2 (R1 = round-1 R-c), item 3 (AFF = the
brainstorm's recorded numbers), item 4 (D13 clauses for AFF), and the §6.1 sensitivity projection of the seven GO
checks and the secondary check. Every target value is copied from the rule's text. Needs out/rd3_bundle_seed42.npz
(rd3_bundle.py --seed 42). Writes out/phase1.json and out/phase1_arrays.npz.
"""
import json
import time

import numpy as np

import rd3_core as K
import rd3_family as FAM
from rd3_core import COND, DIRS, METRICS, T

R1DIR = T / "20261117_reader_fix_csd"
STORED = R1DIR / "results/cand_Rc_Rb_expected_A0"
STEP1 = T / "20261116_grouping_step1_style/results/step1_eval_style.npz"
PA42 = T / "20261030_aspect_baselines/results/per_anchor_seed42.npz"
EPI42 = T / "20261030_aspect_baselines/results/episodes_seed42.npz"
N6_POST = T / "20261108_new_method_quick_checks/results/n6_posteriors.npz"
RD2_CACHE = T / "20261118_reader_fix_round2/rederive/out/rd2_seed42_cache.npz"
TOLD_JSON = T / "20261111_community_told_oracle/results/told_oracle.json"
NEEDED = ["20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz",
          "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json",
          "20261117_reader_fix_csd/results/rc_tau.json",
          "20261117_reader_fix_csd/results/rb_reader_A0.pkl", "20261117_reader_fix_csd/results/rb_reader_A0.json",
          "20261030_aspect_baselines/results/per_anchor_seed42.npz",
          "20261030_aspect_baselines/results/episodes_seed42.npz",
          "20261108_new_method_quick_checks/results/n6_posteriors.npz",
          "20261111_community_told_oracle/results/told_oracle.json"]

# ---- targets, copied from the rule's text
TGT = {
    "B_r1": 18.341064453125, "Bprime_r1": 18.436686197916664,
    "redundancy": {"affect": {"i2t": 0.35348060377541385, "t2i": 0.3828024789253903},
                   "image": {"i2t": 0.7145397990123284, "t2i": 0.7090878258485419},
                   "caption": {"i2t": 0.6182295729609555, "t2i": 0.665152773464146}},
    "R1": {"fused_cells": {"0": 116, "1": 119}, "counterpart_cells": {"0": 58, "1": 123},
           "sigma_star": {"0": 0.0, "1": 0.0}, "bar_comparator": "counterpart",
           "bar_margin": (0.4435221354166667, 0.21646171563312194, 0.6735669710776852),
           "gain_statistic": (2.667236328125, 2.325087836946873, 3.012361650695922),
           "fused_cell_settings": {"0": (2, 0.0, 2.0), "1": (2, 0.0, 16.0)},
           "counterpart_cell_settings": {"0": (1, 0.0, 0.5), "1": (2, 0.5, 1.0)}},
    "AFF": {"fused_r1": 19.136555989583336, "counterpart_r1": 18.39599609375, "bar_comparator": "B_prime",
            "bar_margin": (0.6998697916666667, 0.4598852740816973, 0.9371680126852968),
            "margin": (0.7405598958333333, 0.5196896694963071, 0.9598857494832738),
            "gain_statistic": (3.110758463541667, 2.780005709854805, 3.4559584315470384),
            "either_change": -1.629638671875,
            "per_pair_bar": {"emotion__style": 0.9765625, "emotion__genre": 1.45263671875,
                             "style__genre": -0.32958984375},
            "fused_cells": {"0": 39, "1": 119}, "counterpart_cells": {"0": 149, "1": 10},
            "sigma_star": {"0": 0.0, "1": 0.0},
            "fused_cell_settings": {"0": (0, 4.0, 16.0), "1": (2, 0.0, 16.0)},
            "counterpart_cell_settings": {"0": (2, 4.0, 4.0), "1": (0, 0.5, 0.5)},
            "AFF_minus_R1_fused": (0.21769205729166666, 0.06425880757348419, 0.3709597330984391),
            "AFF_minus_R1_bar": (0.25634765625, 0.04280778303598444, 0.46195041633015954),
            "tau0_open_counts": {"a": 9941, "b": 3627},
            "tau0_open_pct": {"a": 80.90006510416667, "b": 29.5166015625}},
    "R1_pick_accuracy": 51.261393229166664,
}
TOL_NUM = 1e-9         # rule §8: margins, gain statistics, check points and bounds (pp); redundancy values (abs)


def near(a, b, tol=TOL_NUM):
    return bool(abs(float(a) - float(b)) <= tol)


def pci_cmp(got, tgt):
    """got: {'point', 'ci95'}; tgt: (point, lo, hi)."""
    vals = [got["point"], got["ci95"][0], got["ci95"][1]]
    return {"ours": vals, "rule": list(tgt), "abs_diff": [abs(a - b) for a, b in zip(vals, tgt)],
            "exact": [a == b for a, b in zip(vals, tgt)], "agree": all(near(a, b) for a, b in zip(vals, tgt))}


def settings(rec, kind, h):
    c = rec[f"{kind}_cells"][h]
    return (c["tau_index"], c["lambda_u"], c["lambda_a"])


def main():
    t0 = time.time()
    K.assert_rule()
    K.guard_seed(42)
    inputs = K.assert_inputs(NEEDED)
    bjson = json.loads((K.OUT / "rd3_bundle_seed42.json").read_text())
    if K.sha_file(K.OUT / "rd3_bundle_seed42.npz") != bjson["cache_sha256"]:
        raise SystemExit("bundle cache changed since it was written")
    import rd3_bundle as BU    # noqa: E402  (imports the allowed loaders; rb_build for the readers)
    bd = BU.load(42)
    cl, parity, pair_index = bd["cl"], bd["parity"], bd["pair_index"]
    E = len(parity)
    out = {"what": "round-3 independent re-derivation, phase 1 (seed 42): rule §5 items 1-4, D7, §6.1",
           "rule_sha256": K.RULE_SHA, "inputs_sha256": inputs, "bundle_record": bjson}
    ok = {}

    # ================================================================ item 1: bundle
    i1 = {}
    z_ep = np.load(EPI42)
    anchors = np.concatenate([z_ep[f"{p}__anchor"].astype(np.int64) for p in K.PAIR_NAMES])
    cands = np.concatenate([z_ep[f"{p}__candidates"].astype(np.int64) for p in K.PAIR_NAMES])
    s1 = np.load(STEP1)
    pa = np.load(PA42)
    rd2 = np.load(RD2_CACHE)
    i1["anchors_equal_episodes_file"] = bool(np.array_equal(bd["anchor"], anchors))
    i1["candidates_equal_episodes_file"] = bool(np.array_equal(bd["candidates"], cands))
    i1["parity_equal_index_mod_2"] = bool(np.array_equal(parity, np.arange(E) % 2))
    i1["parity_equal_rd2_cache"] = bool(np.array_equal(parity, rd2["parity"]))
    i1["anchor_group_equal_step1"] = bool(np.array_equal(cl, s1["anchor_group"]))
    i1["anchor_group_equal_rd2_cache"] = bool(np.array_equal(cl, rd2["anchor_group"]))
    i1["pair_index_equal_step1"] = bool(np.array_equal(pair_index, s1["pair_index"]))
    i1["pair_index_equal_rd2_cache"] = bool(np.array_equal(pair_index, rd2["pair_index"]))
    i1["n_episodes"], i1["n_paintings"] = int(E), int(len(np.unique(cl)))
    # §4 item 2: per_anchor_seed42.npz
    pcos = K.metrics(bd["cos"])
    i1["per_anchor_seed42_anchor_group_equal"] = bool(np.array_equal(pa["anchor_group"], cl))
    i1["per_anchor_seed42_pair_index_equal"] = bool(np.array_equal(pa["pair_index"], pair_index))
    i1["cosine_per_anchor_equal_stored"] = {m: bool(np.array_equal(pcos[m], pa[f"cosine__{m}"])) for m in METRICS}
    prca = {m: pa[f"rca__{m}"].astype(np.float64) for m in METRICS}
    # B, B'
    pB, pBp = K.metrics(bd["B"]), K.metrics(bd["Bp"])
    i1["B_scores_equal_rd2_cache"] = {f"{c}__{d}": bool(np.array_equal(bd["B"][c][d], rd2[f"B__{c}__{d}"]))
                                      for c in COND for d in DIRS}
    i1["B_scores_dtype"] = str(bd["B"]["a"]["i2t"].dtype)
    i1["B_per_anchor_equal_step1"] = {m: bool(np.array_equal(pB[m], s1[f"B__{m}"])) for m in METRICS}
    i1["B_per_anchor_equal_rd2_cache"] = {m: bool(np.array_equal(pB[m], rd2[f"pB__{m}"])) for m in METRICS}
    i1["Bprime_per_anchor_equal_step1"] = {m: bool(np.array_equal(pBp[m], s1[f"A0__Bprime__{m}"])) for m in METRICS}
    i1["Bprime_per_anchor_equal_rd2_cache"] = {m: bool(np.array_equal(pBp[m], rd2[f"pBp_A0__{m}"])) for m in METRICS}
    i1["B_r1"], i1["Bprime_r1"] = K.mean_pp(pB), K.mean_pp(pBp)
    i1["B_r1_equal_rule"] = i1["B_r1"] == TGT["B_r1"]
    i1["Bprime_r1_equal_rule"] = i1["Bprime_r1"] == TGT["Bprime_r1"]
    tj = json.loads(TOLD_JSON.read_text())
    i1["B_picks"] = bjson["picks"]["B"]
    i1["B_picks_equal_told_oracle_json"] = bool(json.loads(json.dumps(bjson["picks"]["B"])) == tj["B_picks"])
    i1["Bprime_picks"] = bjson["picks"]["Bprime"]
    i1["Bprime_picks_equal_told_oracle_L"] = bool(json.loads(json.dumps(bjson["picks"]["Bprime"]))
                                                  == tj["arms"]["L"]["B_prime"]["picks"])
    # posteriors
    zp = np.load(N6_POST)
    i1["image_caption_posteriors_equal_n6_file"] = {
        f"{h}__{m}": bool(np.array_equal(bd["post"][h][m], zp[f"{h}__{m}"])) for h in ("image", "caption")
        for m in ("img", "txt")}
    i1["selection_equal_n6_file"] = bool(np.array_equal(bd["selection"], zp["selection"]))
    i1["affect_head_equals_told_oracle_L"] = bool(bjson["affect_head_equals_told_oracle_L"])
    # grouping scores and features against round 2's re-derivation cache (bit-equal there to round 1's
    # common.grouping_stack and rb_eval.seed42_features on round 1's bundle)
    g2 = [str(x) for x in rd2["stack_groupings"]]
    idx = [g2.index(h) for h in K.A0]
    i1["grouping_scores_equal_rd2_cache"] = {d: bool(np.array_equal(bd["stack"][d], rd2[f"stack__{d}"][:, idx, :]))
                                             for d in DIRS}
    i1["features_equal_rd2_cache"] = {c: bool(np.array_equal(bd["F"][c], rd2[f"F_A0__{c}"])) for c in COND}
    i1["features_max_abs_diff_rd2_cache"] = float(max(np.abs(bd["F"][c] - rd2[f"F_A0__{c}"]).max() for c in COND))

    # D7 redundancy
    red, kept = K.redundancy(bd["stack"], bd["B"], K.A0)
    i1["redundancy"] = red
    i1["redundancy_rows_kept"] = kept
    i1["redundancy_abs_diff"] = {h: {d: abs(red[h][d] - TGT["redundancy"][h][d]) for d in DIRS} for h in K.A0}
    i1["redundancy_exact"] = {h: {d: red[h][d] == TGT["redundancy"][h][d] for d in DIRS} for h in K.A0}
    i1["redundancy_agree_1e-9"] = all(i1["redundancy_abs_diff"][h][d] <= TOL_NUM for h in K.A0 for d in DIRS)
    i1["redundancy_order"] = {d: sorted(K.A0, key=lambda h: red[h][d]) for d in DIRS}
    i1["affect_smallest_both_directions"] = all(i1["redundancy_order"][d][0] == "affect" for d in DIRS)

    def allv(x):
        if isinstance(x, dict):
            return all(allv(v) for v in x.values())
        return bool(x) if isinstance(x, (bool, np.bool_)) else True

    i1_bools = {k: v for k, v in i1.items() if (isinstance(v, (bool, dict)) and k.endswith(
        ("_equal", "_equal_stored", "_equal_rule", "_equal_rd2_cache", "_equal_step1", "_equal_episodes_file",
         "_equal_index_mod_2", "_equal_n6_file", "_equal_told_oracle_json", "_equal_told_oracle_L",
         "_equals_told_oracle_L", "_both_directions", "agree_1e-9")))}
    i1["checks_counted_for_PASSED"] = sorted(i1_bools)
    ok["item1"] = all(allv(v) for v in i1_bools.values())
    i1["PASSED"] = ok["item1"]
    out["item1_bundle"] = i1
    K.log(f"item 1 {'PASSED' if ok['item1'] else 'FAILED'}; redundancy {red}", t0)
    if not ok["item1"]:
        K.save_json(K.OUT / "phase1.json", out)
        raise SystemExit("item 1 failed: " + json.dumps(K.jsonable(i1_bools)))

    # ================================================================ item 2: R1 = round-1 R-c
    pk, _, _ = BU.rbb.load_readers("A0", False)
    P, halves = K.reader_probs(pk, bd["F"])
    pm = {c: K.picks_margins(P[c]) for c in COND}
    picks = {c: pm[c][0] for c in COND}
    margins = {c: pm[c][1] for c in COND}
    Tr = K.weighted_term(bd["stack"], P)
    sz = np.load(f"{STORED}.npz")
    sj = json.loads((R1DIR / "results/cand_Rc_Rb_expected_A0.json").read_text())
    st = json.loads((R1DIR / "results/rc_tau.json").read_text())
    taus_file = [float(x) for x in st["taus"]]
    i2 = {"reader_C": [float(h["model"].C) for h in pk["halves"]]}
    i2["T_equal_stored"] = {f"{c}__{d}": bool(np.array_equal(Tr[c][d], sz[f"T__{c}__{d}"])) for c in COND for d in DIRS}
    i2["margin_equal_stored"] = {c: bool(np.array_equal(margins[c], sz[f"margin__{c}"])) for c in COND}
    i2["pick_equal_stored"] = {c: bool(np.array_equal(picks[c], sz[f"pick__{c}"].astype(np.int64))) for c in COND}
    taus = K.thresholds(margins["a"], margins["b"])
    i2["taus_recomputed"] = taus
    i2["taus_rc_tau_json"] = taus_file
    i2["taus_equal_rc_tau_json"] = [a == b for a, b in zip(taus, taus_file)]
    i2["taus_equal_rule_text"] = [a == b for a, b in zip(taus_file, K.RULE_TAUS)]
    i2["tau0_is_min_margin"] = bool(taus[0] == min(margins["a"].min(), margins["b"].min()))
    gR1 = FAM.gates_r1(margins, taus_file)
    i2["gates_equal_stored"] = {c: bool(np.array_equal(np.stack([g[c] for g in gR1]).astype(np.float32),
                                                       sz[f"extra__gate_{c}"])) for c in COND}
    recR1, pnR1, pcR1, exR1 = FAM.run_family(bd["B"], Tr, gR1, taus_file, parity)
    K.log(f"R1 family: fused {[recR1['fused_cells'][h]['cell'] for h in '01']}, cf "
          f"{[recR1['counterpart_cells'][h]['cell'] for h in '01']}", t0)
    i2["family"] = recR1
    i2["fused_cells"] = {h: recR1["fused_cells"][h]["cell"] for h in ("0", "1")}
    i2["counterpart_cells"] = {h: recR1["counterpart_cells"][h]["cell"] for h in ("0", "1")}
    i2["sigma_star"] = {h: recR1["control"][h]["sigma_star"] for h in ("0", "1")}
    i2["cells_equal_rule"] = (i2["fused_cells"] == TGT["R1"]["fused_cells"]
                              and i2["counterpart_cells"] == TGT["R1"]["counterpart_cells"]
                              and i2["sigma_star"] == TGT["R1"]["sigma_star"])
    i2["cell_settings_equal_rule"] = all(
        settings(recR1, k, h) == TGT["R1"][f"{k}_cell_settings"][h] for k in ("fused", "counterpart") for h in "01")
    i2["per_anchor_equal_stored"] = {**{f"fused__{m}": bool(np.array_equal(pnR1[m], sz[f"fused__{m}"])) for m in METRICS},
                                     **{f"cf__{m}": bool(np.array_equal(pcR1[m], sz[f"cf__{m}"])) for m in METRICS}}
    evR1, barR1, gainR1 = K.evaluate(pnR1, pcR1, pB, pBp, cl, pair_index)
    i2["bar_v_equal_stored"] = bool(np.array_equal(barR1, sz["bar_v"]))
    i2["evaluation"] = evR1
    i2["bar_comparator"] = evR1["bar_comparator"]
    i2["bar_margin_vs_rule"] = pci_cmp(evR1["bar_margin"], TGT["R1"]["bar_margin"])
    i2["gain_statistic_vs_rule"] = pci_cmp(evR1["gain_statistic"], TGT["R1"]["gain_statistic"])
    i2["bar_equal_round1_json"] = bool(evR1["bar_margin"] == sj["bar"]["r1"] and evR1["bar_comparator"] == sj["bar"]["comparator"])
    tellR1 = np.stack([np.asarray([K.TOLD_IDX[K.PAIR_NAMES[i]][j] for i in pair_index]) for j in (0, 1)])
    pacc = 0.5 * ((picks["a"] == tellR1[0]).astype(np.float64) + (picks["b"] == tellR1[1]))
    i2["pick_accuracy_D14"] = K.point_ci(pacc, cl)
    i2["pick_accuracy_equal_rule"] = i2["pick_accuracy_D14"]["point"] == TGT["R1_pick_accuracy"]
    ok["item2"] = bool(all(i2["T_equal_stored"].values()) and all(i2["margin_equal_stored"].values())
                       and all(i2["pick_equal_stored"].values()) and all(i2["taus_equal_rc_tau_json"])
                       and all(i2["taus_equal_rule_text"]) and all(i2["gates_equal_stored"].values())
                       and i2["cells_equal_rule"] and i2["cell_settings_equal_rule"]
                       and all(i2["per_anchor_equal_stored"].values()) and i2["bar_v_equal_stored"]
                       and i2["bar_comparator"] == TGT["R1"]["bar_comparator"]
                       and i2["bar_margin_vs_rule"]["agree"] and i2["gain_statistic_vs_rule"]["agree"])
    i2["PASSED"] = ok["item2"]
    out["item2_R1"] = i2
    K.log(f"item 2 {'PASSED' if ok['item2'] else 'FAILED'}: bar {evR1['bar_margin']}, gain {evR1['gain_statistic']}", t0)
    if not ok["item2"]:
        K.save_json(K.OUT / "phase1.json", out)
        raise SystemExit("item 2 failed")

    # ================================================================ item 3: AFF
    gA = FAM.gates_aff(margins, picks, taus_file)
    recA, pnA, pcA, exA = FAM.run_family(bd["B"], Tr, gA, taus_file, parity)
    evA, barA, gainA = K.evaluate(pnA, pcA, pB, pBp, cl, pair_index)
    i3 = {"family": recA, "evaluation": evA}
    i3["fused_r1"], i3["counterpart_r1"] = K.mean_pp(pnA), K.mean_pp(pcA)
    i3["fused_r1_vs_rule"] = {"ours": i3["fused_r1"], "rule": TGT["AFF"]["fused_r1"],
                              "exact": i3["fused_r1"] == TGT["AFF"]["fused_r1"],
                              "agree": near(i3["fused_r1"], TGT["AFF"]["fused_r1"])}
    i3["counterpart_r1_vs_rule"] = {"ours": i3["counterpart_r1"], "rule": TGT["AFF"]["counterpart_r1"],
                                    "exact": i3["counterpart_r1"] == TGT["AFF"]["counterpart_r1"],
                                    "agree": near(i3["counterpart_r1"], TGT["AFF"]["counterpart_r1"])}
    i3["bar_comparator"] = evA["bar_comparator"]
    i3["bar_margin_vs_rule"] = pci_cmp(evA["bar_margin"], TGT["AFF"]["bar_margin"])
    i3["margin_vs_rule"] = pci_cmp(evA["margin_vs_counterpart"], TGT["AFF"]["margin"])
    i3["gain_statistic_vs_rule"] = pci_cmp(evA["gain_statistic"], TGT["AFF"]["gain_statistic"])
    ec = evA["either_change_vs_counterpart"]["point"]
    i3["either_change_vs_rule"] = {"ours": ec, "rule": TGT["AFF"]["either_change"],
                                   "exact": ec == TGT["AFF"]["either_change"], "agree": near(ec, TGT["AFF"]["either_change"])}
    i3["per_pair_bar_vs_rule"] = {p: {"ours": evA["per_pair"][p]["bar_margin"]["point"], "rule": v,
                                      "exact": evA["per_pair"][p]["bar_margin"]["point"] == v,
                                      "agree": near(evA["per_pair"][p]["bar_margin"]["point"], v)}
                                  for p, v in TGT["AFF"]["per_pair_bar"].items()}
    i3["fused_cells"] = {h: recA["fused_cells"][h]["cell"] for h in ("0", "1")}
    i3["counterpart_cells"] = {h: recA["counterpart_cells"][h]["cell"] for h in ("0", "1")}
    i3["sigma_star"] = {h: recA["control"][h]["sigma_star"] for h in ("0", "1")}
    i3["cells_equal_rule"] = (i3["fused_cells"] == TGT["AFF"]["fused_cells"]
                              and i3["counterpart_cells"] == TGT["AFF"]["counterpart_cells"]
                              and i3["sigma_star"] == TGT["AFF"]["sigma_star"])
    i3["cell_settings_equal_rule"] = all(
        settings(recA, k, h) == TGT["AFF"][f"{k}_cell_settings"][h] for k in ("fused", "counterpart") for h in "01")
    dF = pnA["r1"] - pnR1["r1"]
    dBar = barA - barR1
    i3["AFF_minus_R1_fused_r1"] = K.point_ci(dF, cl)
    i3["AFF_minus_R1_bar_margin"] = K.point_ci(dBar, cl)
    i3["AFF_minus_R1_fused_vs_rule"] = pci_cmp(i3["AFF_minus_R1_fused_r1"], TGT["AFF"]["AFF_minus_R1_fused"])
    i3["AFF_minus_R1_bar_vs_rule"] = pci_cmp(i3["AFF_minus_R1_bar_margin"], TGT["AFF"]["AFF_minus_R1_bar"])
    cnt = {c: int(gA[0][c].sum()) for c in COND}
    i3["tau0_open_counts"] = cnt
    i3["tau0_open_pct"] = {c: 100 * cnt[c] / E for c in COND}
    i3["tau0_open_counts_equal_rule"] = cnt == TGT["AFF"]["tau0_open_counts"]
    i3["tau0_open_pct_float32_mean_a"] = 100 * float(np.mean(gA[0]["a"].astype(np.float32)))
    i3["gate_open_counts_all_tau"] = {f"tau_{t}": {c: int(gA[t][c].sum()) for c in COND} for t in range(4)}
    i3["R1_gate_open_counts_all_tau"] = {f"tau_{t}": {c: int(gR1[t][c].sum()) for c in COND} for t in range(4)}
    ok["item3"] = bool(i3["fused_r1_vs_rule"]["agree"] and i3["counterpart_r1_vs_rule"]["agree"]
                       and i3["bar_comparator"] == TGT["AFF"]["bar_comparator"]
                       and i3["bar_margin_vs_rule"]["agree"] and i3["margin_vs_rule"]["agree"]
                       and i3["gain_statistic_vs_rule"]["agree"] and i3["either_change_vs_rule"]["agree"]
                       and all(v["agree"] for v in i3["per_pair_bar_vs_rule"].values())
                       and i3["cells_equal_rule"] and i3["cell_settings_equal_rule"]
                       and i3["AFF_minus_R1_fused_vs_rule"]["agree"] and i3["AFF_minus_R1_bar_vs_rule"]["agree"]
                       and i3["tau0_open_counts_equal_rule"])
    i3["PASSED"] = ok["item3"]
    out["item3_AFF"] = i3
    K.log(f"item 3 {'PASSED' if ok['item3'] else 'FAILED'}: bar {evA['bar_margin']} ({evA['bar_comparator']})", t0)

    # ================================================================ item 4: D13 for AFF
    out["item4_D13_AFF"] = {"clauses": evA["D13_clauses"], "bar_margin": evA["bar_margin"],
                            "gain_statistic": evA["gain_statistic"], "clears_bar": evA["D13_clauses"]["clears_bar"]}
    ok["item4"] = bool(evA["D13_clauses"]["clears_bar"])

    # ================================================================ §6.1 sensitivity
    pp = lambda x: 100.0 * np.asarray(x, np.float64)  # noqa: E731
    diffs = {
        "R1_AFF_minus_cosine": pnA["r1"] - pcos["r1"],
        "R1_AFF_minus_RCA": pnA["r1"] - prca["r1"],
        "R1_AFF_minus_B": pnA["r1"] - pB["r1"],
        "R1_AFF_minus_Bprime": pnA["r1"] - pBp["r1"],
        "R1_AFF_minus_counterpart": pnA["r1"] - pcA["r1"],
        "gain_statistic": pnA["gain"] - pcA["gain"],
        "gain_AFF_minus_RCA": pnA["gain"] - prca["gain"],
        "secondary_AFF_minus_R1_fused_R1": pnA["r1"] - pnR1["r1"],
    }
    sens = {}
    for k, v in diffs.items():
        s = K.sensitivity(pp(v), cl)
        b = K.point_ci(v, cl)
        s["seed42_point_bootstrap"] = b["point"]
        s["seed42_ci95"] = b["ci95"]
        s["seed42_bootstrap_half_width"] = (b["ci95"][1] - b["ci95"][0]) / 2.0
        sens[k] = s
    out["sensitivity_6_1"] = sens
    ok["sensitivity_computed"] = True

    out["ok"] = ok
    out["all_items_passed"] = bool(ok["item1"] and ok["item2"] and ok["item3"] and ok["item4"])

    def reader_summary(rec, ev):
        return {"fused_cells": {h: K.cell_desc(rec["fused_cells"][h]["cell"], taus_file) for h in ("0", "1")},
                "counterpart_cells": {h: K.cell_desc(rec["counterpart_cells"][h]["cell"], taus_file) for h in ("0", "1")},
                "sigma_star": {h: rec["control"][h]["sigma_star"] for h in ("0", "1")},
                "r1_means": ev["r1_means"], "bar_comparator": ev["bar_comparator"], "bar_margin": ev["bar_margin"],
                "gain_statistic": ev["gain_statistic"], "margin_vs_counterpart": ev["margin_vs_counterpart"],
                "either_change_vs_counterpart": ev["either_change_vs_counterpart"],
                "per_pair_bar_margin": {p: ev["per_pair"][p]["bar_margin"] for p in K.PAIR_NAMES},
                "D13_clauses": ev["D13_clauses"]}

    out["summary"] = {
        "taus": taus_file, "R1": reader_summary(recR1, evR1), "AFF": reader_summary(recA, evA),
        "AFF_minus_R1": {"fused_r1": i3["AFF_minus_R1_fused_r1"], "bar_margin": i3["AFF_minus_R1_bar_margin"]},
        "AFF_tau0_open_counts": i3["tau0_open_counts"], "redundancy": red,
        "redundancy_affect_smallest_both_directions": i1["affect_smallest_both_directions"],
        "D13_AFF": evA["D13_clauses"],
        "sensitivity": {k: {f: s[f] for f in ("SE", "half_width", "x", "seed42_bootstrap_half_width",
                                                "seed42_point_bootstrap", "seed42_ci95")} for k, s in sens.items()}}
    out["runtime_s"] = round(time.time() - t0, 1)
    out["written_amsterdam"] = K.now_ams()
    K.save_json(K.OUT / "phase1.json", out)

    npz = {"anchor_group": cl, "pair_index": pair_index, "parity": parity, "taus": np.asarray(taus_file)}
    for m in METRICS:
        npz[f"R1__fused__{m}"], npz[f"R1__cf__{m}"] = pnR1[m], pcR1[m]
        npz[f"AFF__fused__{m}"], npz[f"AFF__cf__{m}"] = pnA[m], pcA[m]
        npz[f"B__{m}"], npz[f"Bprime__{m}"] = pB[m], pBp[m]
        npz[f"cosine__{m}"] = pcos[m]
    npz["R1__bar_v"], npz["AFF__bar_v"] = barR1, barA
    npz["R1__gain_v"], npz["AFF__gain_v"] = gainR1, gainA
    for c in COND:
        npz[f"pick__{c}"] = picks[c]
        npz[f"margin__{c}"] = margins[c]
        npz[f"P__{c}"] = P[c]
        npz[f"R1__gate__{c}"] = np.stack([g[c] for g in gR1])            # (4, E) bool, tau index 0..3
        npz[f"AFF__gate__{c}"] = np.stack([g[c] for g in gA])
        for d in DIRS:
            npz[f"T__{c}__{d}"] = Tr[c][d]
    for name, ex in (("R1", exR1), ("AFF", exA)):
        for k in ("rho_f", "gam_f", "rho_c", "crit_f", "ctrl_rho"):
            npz[f"{name}__{k}"] = ex[k]
    np.savez(K.OUT / "phase1_arrays.npz", **npz)
    K.log(f"phase 1 done: {ok}", t0)


if __name__ == "__main__":
    main()
