"""Phase 1 of the round-4 independent re-derivation (round-4 rule §8), seed 42 only.

Order (rule §5): items 1 to 5 (regression checks) first; the script stops with SystemExit, before any candidate result
of V4, V2 or V24 is computed, if any of them fails. Then item 6 (development numbers of V4, V2, V24), item 7 (D10),
item 8 (the carry) or item 9 (the kill), and, for a carried candidate, the §6.1 sensitivity projection of its GO checks
(§6.5). Every target value is copied from the rules' text. Needs out/rd4_bundle_seed42.npz (rd4_bundle.py) and
out/rd4_ref42.npz (rd4_ref42.py). Writes out/rd4_phase1.json, out/rd4_phase1_arrays.npz, out/rd4_phase1.sha256.
Usage: python rd4_phase1.py [--force]
"""
import argparse
import json
import time

import numpy as np

import rd4_bundle as BU
import rd4_core as K
from rd4_core import COND, DIRS, F3, K3, METRICS, PAIR_NAMES, T

R1DIR = T / "20261117_reader_fix_csd"
STORED_RC = R1DIR / "results/cand_Rc_Rb_expected_A0"
STORED_A1 = T / "20261118_reader_fix_round2/results/cand_R1_A1.npz"
STEP1 = T / "20261116_grouping_step1_style/results/step1_eval_style.npz"
PA42 = T / "20261030_aspect_baselines/results/per_anchor_seed42.npz"
EPI42 = T / "20261030_aspect_baselines/results/episodes_seed42.npz"
N6_POST = T / "20261108_new_method_quick_checks/results/n6_posteriors.npz"
RD2_CACHE = T / "20261118_reader_fix_round2/rederive/out/rd2_seed42_cache.npz"
TOLD_JSON = T / "20261111_community_told_oracle/results/told_oracle.json"
BS04 = T / "20261120_r1_levers_brainstorm/results/bs_04_readers.json"
NEEDED = ["20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz",
          "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json",
          "20261117_reader_fix_csd/results/rc_tau.json",
          "20261117_reader_fix_csd/results/rb_reader_A0.pkl", "20261117_reader_fix_csd/results/rb_reader_A0.json",
          "20261117_reader_fix_csd/results/rb_reader_A1.pkl", "20261117_reader_fix_csd/results/rb_reader_A1.json",
          "20261117_reader_fix_csd/results/rb_reader_A1.npz",
          "20261118_reader_fix_round2/results/cand_R1_A1.npz", "20261118_reader_fix_round2/results/cand_R1_A1.json",
          "20261030_aspect_baselines/results/per_anchor_seed42.npz",
          "20261030_aspect_baselines/results/episodes_seed42.npz",
          "20261108_new_method_quick_checks/results/n6_posteriors.npz",
          "20261111_community_told_oracle/results/told_oracle.json",
          "20261116_grouping_step1_style/results/step1_eval_style.npz",
          "20261120_r1_levers_brainstorm/results/bs_04_readers.json",
          "20261118_reader_fix_round2/rederive/out/rd2_seed42_cache.npz"]
OUTJSON = K.OUT / "rd4_phase1.json"
OUTNPZ = K.OUT / "rd4_phase1_arrays.npz"
OUTSHA = K.OUT / "rd4_phase1.sha256"

# ---- targets, copied from the rules' text (round 3 §5 items 1 to 3, D7, D10, D11; round 4 D4, D5, §5 items 2, 3)
TGT = {
    "B_r1": 18.341064453125, "Bprime_A0_r1": 18.436686197916664, "Bprime_A1_r1": 18.804931640625,
    "redundancy": {"affect": {"i2t": 0.35348060377541385, "t2i": 0.3828024789253903},
                   "image": {"i2t": 0.7145397990123284, "t2i": 0.7090878258485419},
                   "caption": {"i2t": 0.6182295729609555, "t2i": 0.665152773464146}},
    "R1": {"fused_cells": {"0": 116, "1": 119}, "counterpart_cells": {"0": 58, "1": 123},
           "sigma_star": {"0": 0.0, "1": 0.0}, "bar_comparator": "counterpart",
           "bar_margin": (0.4435221354166667, 0.21646171563312194, 0.6735669710776852),
           "gain_statistic": (2.667236328125, 2.325087836946873, 3.012361650695922),
           "fused_cell_settings": {"0": (2, 0.0, 2.0), "1": (2, 0.0, 16.0)},
           "counterpart_cell_settings": {"0": (1, 0.0, 0.5), "1": (2, 0.5, 1.0)}},
    "AFF": {"fused_r1": 19.136555989583336, "counterpart_r1": 18.39599609375, "bar_comparator": "Bprime_A0",
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
            "tau0_open_counts": {"a": 9941, "b": 3627}},
    "IMGABST": {"v75": 0.021043562795966864, "a_v_count": 9216,
                "fused_r1": 19.059244791666664, "counterpart_r1": 18.49365234375, "bar_comparator": "counterpart",
                "bar_margin": (0.5655924479166667, 0.3448683992591827, 0.79821625538382),
                "gain_statistic": (2.878824869791667, 2.554983173204304, 3.2105685950938248),
                "either_change": -1.7476399739583333,
                "per_pair_bar": {"emotion__style": 0.677490234375, "emotion__genre": 1.28173828125,
                                 "style__genre": -0.262451171875},
                "fused_cells": {"0": 117, "1": 119}, "counterpart_cells": {"0": 58, "1": 67},
                "sigma_star": {"0": 0.0, "1": 0.0},
                "fused_cell_settings": {"0": (2, 0.0, 4.0), "1": (2, 0.0, 16.0)},
                "counterpart_cell_settings": {"0": (1, 0.0, 0.5), "1": (1, 0.5, 1.0)}},
}
TOL = 1e-9          # rule §8: margins, gain statistics, check points and bounds (pp); v75 within 1e-15 abs / 1e-9 rel


def near(a, b, tol=TOL):
    return bool(abs(float(a) - float(b)) <= tol)


def pci_cmp(got, tgt):
    vals = [got["point"], got["ci95"][0], got["ci95"][1]]
    return {"ours": vals, "rule": list(tgt), "abs_diff": [abs(a - b) for a, b in zip(vals, tgt)],
            "exact": [a == b for a, b in zip(vals, tgt)], "agree": all(near(a, b) for a, b in zip(vals, tgt))}


def val_cmp(got, tgt):
    return {"ours": float(got), "rule": float(tgt), "abs_diff": abs(float(got) - float(tgt)),
            "exact": float(got) == float(tgt), "agree": near(got, tgt)}


def settings(rec, kind, h):
    c = rec[f"{kind}_cells"][h]
    return (c["tau_index"], c["lambda_u"], c["lambda_a"])


def cells_of(rec):
    return ({h: rec["fused_cells"][h]["cell"] for h in ("0", "1")},
            {h: rec["counterpart_cells"][h]["cell"] for h in ("0", "1")},
            {h: rec["control"][h]["sigma_star"] for h in ("0", "1")})


def check_against(name, rec, ev, pn, pc, tgt):
    """Compare a family (cells, sigma*) and its evaluation with the rule's targets."""
    fc, cc, ss = cells_of(rec)
    r = {"fused_cells": fc, "counterpart_cells": cc, "sigma_star": ss,
         "cells_equal_rule": fc == tgt["fused_cells"] and cc == tgt["counterpart_cells"] and ss == tgt["sigma_star"],
         "cell_settings_equal_rule": all(settings(rec, k, h) == tgt[f"{k}_cell_settings"][h]
                                         for k in ("fused", "counterpart") for h in "01"),
         "bar_comparator": ev["bar_comparator"],
         "bar_comparator_equal_rule": ev["bar_comparator"] == tgt["bar_comparator"]}
    if "fused_r1" in tgt:
        r["fused_r1"] = val_cmp(K3.mean_pp(pn), tgt["fused_r1"])
        r["counterpart_r1"] = val_cmp(K3.mean_pp(pc), tgt["counterpart_r1"])
    r["bar_margin"] = pci_cmp(ev["bar_margin"], tgt["bar_margin"])
    r["gain_statistic"] = pci_cmp(ev["gain_statistic"], tgt["gain_statistic"])
    if "margin" in tgt:
        r["margin"] = pci_cmp(ev["margin_vs_counterpart"], tgt["margin"])
    if "either_change" in tgt:
        r["either_change"] = val_cmp(ev["either_change_vs_counterpart"]["point"], tgt["either_change"])
    if "per_pair_bar" in tgt:
        r["per_pair_bar"] = {p: val_cmp(ev["per_pair"][p]["bar_margin"]["point"], v)
                             for p, v in tgt["per_pair_bar"].items()}
    agree = [r["cells_equal_rule"], r["cell_settings_equal_rule"], r["bar_comparator_equal_rule"]]
    for k in ("fused_r1", "counterpart_r1", "bar_margin", "gain_statistic", "margin", "either_change"):
        if k in r:
            agree.append(r[k]["agree"])
    if "per_pair_bar" in r:
        agree += [v["agree"] for v in r["per_pair_bar"].values()]
    r["PASSED"] = bool(all(agree))
    return r


def summary_of(rec, ev, taus, pn, pc):
    return {"fused_cells": {h: K3.cell_desc(rec["fused_cells"][h]["cell"], taus) for h in ("0", "1")},
            "counterpart_cells": {h: K3.cell_desc(rec["counterpart_cells"][h]["cell"], taus) for h in ("0", "1")},
            "sigma_star": {h: rec["control"][h]["sigma_star"] for h in ("0", "1")},
            "fused_r1": K3.mean_pp(pn), "counterpart_r1": K3.mean_pp(pc),
            "r1_means": ev["r1_means"], "r1_int4_sums": ev["r1_int4_sums"], "bar_comparator": ev["bar_comparator"],
            "bar_margin": ev["bar_margin"], "margin_vs_counterpart": ev["margin_vs_counterpart"],
            "gain_statistic": ev["gain_statistic"], "either_change_vs_counterpart": ev["either_change_vs_counterpart"],
            "D10_clauses": ev["D10_clauses"],
            "per_pair_bar_margin": {p: ev["per_pair"][p]["bar_margin"] for p in PAIR_NAMES}}


def stop(out, msg):
    out["STOPPED"] = msg
    out["written_amsterdam"] = K.now_ams()
    K.save_json(OUTJSON, out)
    raise SystemExit(msg)


def allv(x):
    if isinstance(x, dict):
        return all(allv(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(allv(v) for v in x)
    return bool(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing out/rd4_phase1.json")
    a = ap.parse_args()
    if OUTJSON.exists() and not a.force:
        raise SystemExit(f"{OUTJSON} exists; pass --force to overwrite")
    t0 = time.time()
    K.guard_seed(42)
    out = {"what": "round-4 independent re-derivation, phase 1 (seed 42): rule §5 items 1-9, §6.1 for the carried "
                   "candidate", "rules": K.assert_rules(), "inputs_sha256": K.assert_inputs(NEEDED)}
    bd, brec = BU.load()
    ref = np.load(K.OUT / "rd4_ref42.npz")
    rrec = json.loads((K.OUT / "rd4_ref42.json").read_text())
    if K.sha_file(K.OUT / "rd4_ref42.npz") != rrec["cache_sha256"]:
        raise SystemExit("rd4_ref42.npz changed since it was written")
    out["bundle_record"], out["ref42_record"] = brec, rrec
    cl, parity, pair_index = bd["cl"], bd["parity"], bd["pair_index"]
    E = len(parity)
    if E != K.N_EP_SEED42:
        raise SystemExit(f"{E} episodes, expected {K.N_EP_SEED42}")
    ok = {}

    # ================================================================ item 1: bundle
    i1 = {}
    z_ep = np.load(EPI42)
    anchors = np.concatenate([z_ep[f"{p}__anchor"].astype(np.int64) for p in PAIR_NAMES])
    cands = np.concatenate([z_ep[f"{p}__candidates"].astype(np.int64) for p in PAIR_NAMES])
    s1, pa, rd2, zp = np.load(STEP1), np.load(PA42), np.load(RD2_CACHE), np.load(N6_POST)
    eq = lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y)))  # noqa: E731
    # A0 part (round 3 §5 item 1) against the stored references and round 1's load_bundle
    i1["anchors_equal_episodes_file"] = eq(bd["anchor"], anchors)
    i1["candidates_equal_episodes_file"] = eq(bd["candidates"], cands)
    i1["parity_equal_index_mod_2"] = eq(parity, np.arange(E) % 2)
    i1["anchor_group_equal_step1"] = eq(cl, s1["anchor_group"])
    i1["pair_index_equal_step1"] = eq(pair_index, s1["pair_index"])
    i1["parity_anchor_group_pair_index_equal_rd2_cache"] = (eq(parity, rd2["parity"]) and eq(cl, rd2["anchor_group"])
                                                            and eq(pair_index, rd2["pair_index"]))
    i1["episodes_equal_load_bundle"] = {k: eq(bd[k2], ref[k]) for k, k2 in
                                        (("anchor", "anchor"), ("candidates", "candidates"), ("parity", "parity"),
                                         ("anchor_group", "cl"), ("pair_index", "pair_index"),
                                         ("selection", "selection"))}
    pcos = K3.metrics(bd["cos"])
    i1["cosine_scores_equal_load_bundle"] = {f"{c}__{d}": eq(bd["cos"][c][d], ref[f"cos__{c}__{d}"])
                                             for c in COND for d in DIRS}
    i1["per_anchor_seed42_anchor_group_and_pair_index_equal"] = eq(pa["anchor_group"], cl) and eq(pa["pair_index"],
                                                                                                    pair_index)
    i1["cosine_per_anchor_equal_stored"] = {m: eq(pcos[m], pa[f"cosine__{m}"]) for m in METRICS}
    prca = {m: pa[f"rca__{m}"].astype(np.float64) for m in METRICS}
    pB, pBp0, pBp1 = K3.metrics(bd["B"]), K3.metrics(bd["Bp0"]), K3.metrics(bd["Bp1"])
    i1["B_scores_equal_load_bundle"] = {f"{c}__{d}": eq(bd["B"][c][d], ref[f"B__{c}__{d}"]) for c in COND for d in DIRS}
    i1["B_scores_equal_rd2_cache"] = {f"{c}__{d}": eq(bd["B"][c][d], rd2[f"B__{c}__{d}"]) for c in COND for d in DIRS}
    i1["B_per_anchor_equal_load_bundle"] = {m: eq(pB[m], ref[f"pB__{m}"]) for m in METRICS}
    i1["B_per_anchor_equal_step1"] = {m: eq(pB[m], s1[f"B__{m}"]) for m in METRICS}
    i1["Bprime_A0_scores_equal_load_bundle"] = {f"{c}__{d}": eq(bd["Bp0"][c][d], ref[f"Bp0__{c}__{d}"])
                                                for c in COND for d in DIRS}
    i1["Bprime_A0_per_anchor_equal_load_bundle"] = {m: eq(pBp0[m], ref[f"pBp0__{m}"]) for m in METRICS}
    i1["Bprime_A0_per_anchor_equal_step1"] = {m: eq(pBp0[m], s1[f"A0__Bprime__{m}"]) for m in METRICS}
    i1["Bprime_A0_per_anchor_equal_rd2_cache"] = {m: eq(pBp0[m], rd2[f"pBp_A0__{m}"]) for m in METRICS}
    i1["B_r1"], i1["Bprime_A0_r1"] = K3.mean_pp(pB), K3.mean_pp(pBp0)
    i1["B_r1_equal_rule"] = i1["B_r1"] == TGT["B_r1"]
    i1["Bprime_A0_r1_equal_rule"] = i1["Bprime_A0_r1"] == TGT["Bprime_A0_r1"]
    tj = json.loads(TOLD_JSON.read_text())
    i1["B_picks_equal_told_oracle_json"] = json.loads(json.dumps(K.jsonable(brec["picks"]["B"]))) == tj["B_picks"]
    i1["Bprime_A0_picks_equal_told_oracle_L"] = (json.loads(json.dumps(K.jsonable(brec["picks"]["Bprime_A0"])))
                                                 == tj["arms"]["L"]["B_prime"]["picks"])
    i1["posteriors_equal_load_bundle"] = {f"{h}__{m}": eq(bd["post"][h][m], ref[f"post__{h}__{m}"])
                                          for h in K.A1 for m in ("img", "txt")}
    i1["image_caption_posteriors_equal_n6_file"] = {f"{h}__{m}": eq(bd["post"][h][m], zp[f"{h}__{m}"])
                                                    for h in ("image", "caption") for m in ("img", "txt")}
    i1["selection_equal_n6_file"] = eq(bd["selection"], zp["selection"])
    i1["affect_head_equals_told_oracle_L"] = bool(brec["affect_head_equals_told_oracle_L"])
    g2 = [str(x) for x in rd2["stack_groupings"]]
    if tuple(g2) != K.A1:
        raise SystemExit(f"rd2 cache stack groupings {g2} are not A1")
    i1["grouping_scores_A0_equal_rd2_cache"] = {d: eq(bd["stack0"][d], rd2[f"stack__{d}"][:, :3, :]) for d in DIRS}
    i1["grouping_scores_A1_equal_rd2_cache"] = {d: eq(bd["stack1"][d], rd2[f"stack__{d}"]) for d in DIRS}
    i1["features_A0_equal_rd2_cache"] = {c: eq(bd["F0"][c], rd2[f"F_A0__{c}"]) for c in COND}
    # A1 extension (round 4 §5 item 1)
    i1["csd_full_sha256_equal_load_bundle"] = brec["csd_full_sha256"] == rrec["csd_full_sha256"]
    i1["csd_shape_dtype_equal_load_bundle"] = brec["csd_shape_dtype"] == rrec["csd_shape_dtype"]
    i1["Bprime_A1_scores_equal_load_bundle"] = {f"{c}__{d}": eq(bd["Bp1"][c][d], ref[f"Bp1__{c}__{d}"])
                                                for c in COND for d in DIRS}
    i1["Bprime_A1_per_anchor_equal_load_bundle"] = {m: eq(pBp1[m], ref[f"pBp1__{m}"]) for m in METRICS}
    i1["Bprime_A1_per_anchor_equal_step1"] = {m: eq(pBp1[m], s1[f"A1__Bprime__{m}"]) for m in METRICS}
    i1["Bprime_A1_per_anchor_equal_rd2_cache"] = {m: eq(pBp1[m], rd2[f"pBp_A1__{m}"]) for m in METRICS}
    i1["Bprime_A1_r1"] = K3.mean_pp(pBp1)
    i1["Bprime_A1_r1_equal_rule"] = i1["Bprime_A1_r1"] == TGT["Bprime_A1_r1"]
    i1["features_A1_equal_rd2_cache"] = {c: eq(bd["F1"][c], rd2[f"F_A1__{c}"]) for c in COND}
    i1["features_A1_first18_equal_A0"] = {c: eq(bd["F1"][c][:, :18], bd["F0"][c]) for c in COND}
    i1["features_A1_shape"] = list(bd["F1"]["a"].shape)
    i1["Bprime_A1_picks"] = brec["picks"]["Bprime_A1"]
    # D7 redundancy (round 3)
    red, kept = K3.redundancy(bd["stack0"], bd["B"], K.A0)
    i1["redundancy"] = red
    i1["redundancy_rows_kept"] = kept
    i1["redundancy_abs_diff"] = {h: {d: abs(red[h][d] - TGT["redundancy"][h][d]) for d in DIRS} for h in K.A0}
    i1["redundancy_exact"] = {h: {d: red[h][d] == TGT["redundancy"][h][d] for d in DIRS} for h in K.A0}
    i1["redundancy_agree_1e-9"] = all(i1["redundancy_abs_diff"][h][d] <= TOL for h in K.A0 for d in DIRS)
    i1["redundancy_order"] = {d: sorted(K.A0, key=lambda h: red[h][d]) for d in DIRS}
    i1["affect_smallest_both_directions"] = all(i1["redundancy_order"][d][0] == "affect" for d in DIRS)
    counted = [k for k, v in i1.items() if isinstance(v, (bool, dict)) and (
        "_equal" in k or k.endswith(("_both_directions", "agree_1e-9", "_equals_told_oracle_L")))
        and k not in ("redundancy_exact",)]
    i1["checks_counted_for_PASSED"] = sorted(counted)
    ok["item1"] = all(allv(i1[k]) for k in counted)
    i1["PASSED"] = ok["item1"]
    out["item1_bundle"] = i1
    K.log(f"item 1 {'PASSED' if ok['item1'] else 'FAILED'}", t0)
    if not ok["item1"]:
        stop(out, "item 1 failed: " + json.dumps({k: i1[k] for k in counted if not allv(i1[k])}, default=str))

    # ================================================================ item 2: R1 = round-1 R-c, AFF = round 3's targets
    pk0, _, _ = BU.rbb.load_readers("A0", False)
    P = K.reader_probs(pk0, bd["F0"], K.A0)
    pm = {c: K3.picks_margins(P[c]) for c in COND}
    picks = {c: pm[c][0] for c in COND}
    margins = {c: pm[c][1] for c in COND}
    Tr = K3.weighted_term(bd["stack0"], P)
    sz = np.load(f"{STORED_RC}.npz")
    st = json.loads((R1DIR / "results/rc_tau.json").read_text())
    taus = [float(x) for x in st["taus"]]
    i2 = {"R1": {"reader_C": [float(h["model"].C) for h in pk0["halves"]]}}
    r = i2["R1"]
    r["T_equal_stored"] = {f"{c}__{d}": eq(Tr[c][d], sz[f"T__{c}__{d}"]) for c in COND for d in DIRS}
    r["margin_equal_stored"] = {c: eq(margins[c], sz[f"margin__{c}"]) for c in COND}
    r["pick_equal_stored"] = {c: eq(picks[c], sz[f"pick__{c}"].astype(np.int64)) for c in COND}
    taus_re = K3.thresholds(margins["a"], margins["b"])
    r["taus_recomputed"], r["taus_rc_tau_json"] = taus_re, taus
    r["taus_equal_rc_tau_json"] = [a_ == b_ for a_, b_ in zip(taus_re, taus)]
    r["taus_equal_rule_text"] = [a_ == b_ for a_, b_ in zip(taus, K.RULE_TAUS)]
    gR1 = K.gates_r1(margins, taus)
    r["gates_equal_stored"] = {c: eq(np.stack([g[c] for g in gR1]), sz[f"extra__gate_{c}"]) for c in COND}
    recR1, pnR1, pcR1, exR1 = F3.run_family(bd["B"], Tr, gR1, taus, parity)
    order_A0 = lambda pc: [("Bprime_A0", pBp0), ("counterpart", pc), ("B", pB)]  # noqa: E731  round 3 D12
    evR1, arrR1 = K.evaluate(pnR1, pcR1, order_A0(pcR1), cl, pair_index)
    r["per_anchor_equal_stored"] = {**{f"fused__{m}": eq(pnR1[m], sz[f"fused__{m}"]) for m in METRICS},
                                    **{f"cf__{m}": eq(pcR1[m], sz[f"cf__{m}"]) for m in METRICS}}
    r["bar_v_equal_stored"] = eq(arrR1["bar_v"], sz["bar_v"])
    r["vs_rule"] = check_against("R1", recR1, evR1, pnR1, pcR1, TGT["R1"])
    r["PASSED"] = bool(allv(r["T_equal_stored"]) and allv(r["margin_equal_stored"]) and allv(r["pick_equal_stored"])
                       and all(r["taus_equal_rc_tau_json"]) and all(r["taus_equal_rule_text"])
                       and allv(r["gates_equal_stored"]) and allv(r["per_anchor_equal_stored"])
                       and r["bar_v_equal_stored"] and r["vs_rule"]["PASSED"])
    # AFF through the candidates' gate function with both factors set to 1
    gA_def = K.gates_aff(margins, picks, taus)
    gA = K.apply_factors(gA_def, K.factor_ones(E), K.factor_ones(E))
    i2["AFF"] = {"candidate_gate_function_with_unit_factors_equals_AFF_gates": K.gates_equal(gA, gA_def),
                 "AFF_gates_float32": all(gA[t][c].dtype == np.float32 for t in range(4) for c in COND)}
    recA, pnA, pcA, exA = F3.run_family(bd["B"], Tr, gA, taus, parity)
    evA, arrA = K.evaluate(pnA, pcA, order_A0(pcA), cl, pair_index)
    ra = i2["AFF"]
    ra["vs_rule"] = check_against("AFF", recA, evA, pnA, pcA, TGT["AFF"])
    ra["AFF_minus_R1_fused"] = pci_cmp(K.point_ci(pnA["r1"] - pnR1["r1"], cl), TGT["AFF"]["AFF_minus_R1_fused"])
    ra["AFF_minus_R1_bar"] = pci_cmp(K.point_ci(arrA["bar_v"] - arrR1["bar_v"], cl), TGT["AFF"]["AFF_minus_R1_bar"])
    ra["tau0_open_counts"] = {c: int(gA[0][c].sum()) for c in COND}
    ra["tau0_open_counts_equal_rule"] = ra["tau0_open_counts"] == TGT["AFF"]["tau0_open_counts"]
    ra["PASSED"] = bool(ra["candidate_gate_function_with_unit_factors_equals_AFF_gates"] and ra["AFF_gates_float32"]
                        and ra["vs_rule"]["PASSED"] and ra["AFF_minus_R1_fused"]["agree"]
                        and ra["AFF_minus_R1_bar"]["agree"] and ra["tau0_open_counts_equal_rule"])
    ok["item2"] = bool(r["PASSED"] and ra["PASSED"])
    i2["PASSED"] = ok["item2"]
    i2["summary"] = {"R1": summary_of(recR1, evR1, taus, pnR1, pcR1), "AFF": summary_of(recA, evA, taus, pnA, pcA)}
    out["item2_R1_AFF"] = i2
    K.log(f"item 2 {'PASSED' if ok['item2'] else 'FAILED'} (R1 {r['PASSED']}, AFF {ra['PASSED']})", t0)
    if not ok["item2"]:
        stop(out, "item 2 failed")

    # ================================================================ item 3: R1 x a_v = IMGABST_q75
    v = np.asarray(bd["v"], np.float64)
    v_re, vb = K.abstention_signal(bd["F0"]["a"], bd["F0"]["b"])
    v75 = float(np.percentile(v, 75))
    i3 = {"v_equal_bundle": eq(v_re, v), "v_b_equal_v_a": eq(vb, v), "v75_recomputed": v75,
          "v75_rule": TGT["IMGABST"]["v75"], "v75_exact": v75 == TGT["IMGABST"]["v75"],
          "v75_abs_diff": abs(v75 - TGT["IMGABST"]["v75"])}
    i3["v75_agree"] = bool(i3["v75_abs_diff"] <= 1e-15 or i3["v75_abs_diff"] <= 1e-9 * abs(TGT["IMGABST"]["v75"]))
    a_v = K.factor_abstention(v, K.V75)          # the frozen v75 of D5
    i3["a_v_count"] = int(a_v["a"].sum())
    i3["a_v_count_equal_rule"] = i3["a_v_count"] == TGT["IMGABST"]["a_v_count"]
    i3["a_v_same_both_conditions"] = eq(a_v["a"], a_v["b"])
    gI = K.apply_factors(gR1, a_v)               # the same function that builds V4's gate (AFF's gates in place of R1's)
    recI, pnI, pcI, exI = F3.run_family(bd["B"], Tr, gI, taus, parity)
    evI, arrI = K.evaluate(pnI, pcI, order_A0(pcI), cl, pair_index)
    i3["vs_rule"] = check_against("IMGABST", recI, evI, pnI, pcI, TGT["IMGABST"])
    # the brainstorm's record itself (provenance cross-check; the rule's text is the target)
    bs = json.loads(BS04.read_text())["results"]["IMGABST_q75"]
    i3["bs04_record_cross_check"] = {
        "fused_r1": val_cmp(K3.mean_pp(pnI), bs["fused_r1"]), "cf_r1": val_cmp(K3.mean_pp(pcI), bs["cf_r1"]),
        "comparator": evI["bar_comparator"] == "counterpart" and bs["comparator"] == "counterpart",
        "bar": pci_cmp(evI["bar_margin"], (bs["bar"]["point"], *bs["bar"]["ci95"])),
        "margin": pci_cmp(evI["margin_vs_counterpart"], (bs["margin"]["point"], *bs["margin"]["ci95"])),
        "gain": pci_cmp(evI["gain_statistic"], (bs["gain"]["point"], *bs["gain"]["ci95"])),
        "either": pci_cmp(evI["either_change_vs_counterpart"], (bs["either"]["point"], *bs["either"]["ci95"])),
        "per_pair_bar": {p: val_cmp(evI["per_pair"][p]["bar_margin"]["point"], bs["per_pair_bar"][p])
                         for p in PAIR_NAMES},
        "per_pair_gain": {p: val_cmp(evI["per_pair"][p]["gain_statistic"]["point"], bs["per_pair_gain"][p])
                          for p in PAIR_NAMES},
        "fused_cells_tau_lu_la": [list(settings(recI, "fused", h)) for h in "01"],
        "bs_fused_cells": bs["fused_cells"], "bs_cf_cells": bs["cf_cells"],
        "fused_cells_match_bs": all(list(settings(recI, "fused", h)) == [float(x) if i else int(x) for i, x in
                                                                         enumerate(bs["fused_cells"][int(h)][:3])]
                                    for h in "01"),
        "cf_cells_match_bs": all(list(settings(recI, "counterpart", h)) == [float(x) if i else int(x) for i, x in
                                                                            enumerate(bs["cf_cells"][int(h)][:3])]
                                 for h in "01")}
    ok["item3"] = bool(i3["v_equal_bundle"] and i3["v_b_equal_v_a"] and i3["v75_agree"] and i3["a_v_count_equal_rule"]
                       and i3["a_v_same_both_conditions"] and i3["vs_rule"]["PASSED"])
    i3["PASSED"] = ok["item3"]
    i3["summary"] = summary_of(recI, evI, taus, pnI, pcI)
    out["item3_IMGABST_q75"] = i3
    K.log(f"item 3 {'PASSED' if ok['item3'] else 'FAILED'} (v75 exact {i3['v75_exact']}, a_v {i3['a_v_count']})", t0)
    if not ok["item3"]:
        stop(out, "item 3 failed")

    # ================================================================ item 4: the A1 reader = cand_R1_A1.npz
    pk1, _, _ = BU.rbb.load_readers("A1", False)
    P1 = K.reader_probs(pk1, bd["F1"], K.A1)
    pick1 = {c: np.argmax(P1[c], axis=1).astype(np.int64) for c in COND}
    za = np.load(STORED_A1)
    i4 = {"reader_C": [float(h["model"].C) for h in pk1["halves"]],
          "probs_equal_stored": {c: eq(P1[c], za[f"probs__{c}"]) for c in COND},
          "probs_max_abs_diff": {c: float(np.abs(P1[c] - za[f"probs__{c}"]).max()) for c in COND},
          "pick_equal_stored": {c: eq(pick1[c], za[f"pick__{c}"].astype(np.int64)) for c in COND},
          "stored_pick_dtype": {c: str(za[f"pick__{c}"].dtype) for c in COND},
          "exact_argmax_ties": {c: K.exact_argmax_ties(P1[c]) for c in COND},
          "pick_shares": {c: {h: int((pick1[c] == j).sum()) for j, h in enumerate(K.A1)} for c in COND},
          "stored_alignment_equal": eq(za["anchor_group"], cl) and eq(za["pair_index"], pair_index)
          and eq(za["parity"], parity)}
    ok["item4"] = bool(allv(i4["probs_equal_stored"]) and allv(i4["pick_equal_stored"])
                       and all(s == "int8" for s in i4["stored_pick_dtype"].values()) and i4["stored_alignment_equal"])
    i4["PASSED"] = ok["item4"]
    out["item4_A1_reader"] = i4
    K.log(f"item 4 {'PASSED' if ok['item4'] else 'FAILED'} (ties {i4['exact_argmax_ties']})", t0)
    if not ok["item4"]:
        stop(out, "item 4 failed")

    # ================================================================ item 5: gate algebra (gates only)
    a1 = K.factor_a1_pick(pick1)
    gates = {"AFF": gA,
             "V4": K.apply_factors(gA, a_v),
             "V2": K.apply_factors(gA, a1),
             "V24": K.apply_factors(gA, a1, a_v)}
    stored_a1 = {c: (za[f"pick__{c}"] == 0).astype(np.float32) for c in COND}
    v_lt = (np.asarray(bd["F0"]["a"][:, 6:8], np.float64).min(axis=1) < 0.021043562795966864).astype(np.float32)
    i5 = {"float32_0_1": {k: all(g[t][c].dtype == np.float32 and np.isin(g[t][c], (0.0, 1.0)).all()
                                 for t in range(4) for c in COND) for k, g in gates.items()},
          "closed_where_AFF_closed": {k: all(not np.any(gates[k][t][c][gA[t][c] == 0]) for t in range(4) for c in COND)
                                      for k in K.CANDIDATES},
          "V24_equals_V4_times_V2": all(np.array_equal(gates["V24"][t][c], gates["V4"][t][c] * gates["V2"][t][c])
                                        for t in range(4) for c in COND),
          "V2_equals_AFF_times_stored_pick_affect": all(np.array_equal(gates["V2"][t][c], gA[t][c] * stored_a1[c])
                                                        for t in range(4) for c in COND),
          "V4_equals_AFF_times_v_below_v75": all(np.array_equal(gates["V4"][t][c], gA[t][c] * v_lt)
                                                 for t in range(4) for c in COND)}
    i5["open_counts"] = {k: K.open_counts(g) for k, g in {**gates, "R1": gR1, "R1_x_a_v": gI}.items()}
    i5["tau0_open_counts"] = {k: i5["open_counts"][k]["tau_0"] for k in ("AFF", "V4", "V2", "V24")}
    ok["item5"] = bool(allv(i5["float32_0_1"]) and allv(i5["closed_where_AFF_closed"]) and i5["V24_equals_V4_times_V2"]
                       and i5["V2_equals_AFF_times_stored_pick_affect"] and i5["V4_equals_AFF_times_v_below_v75"])
    i5["PASSED"] = ok["item5"]
    out["item5_gate_algebra"] = i5
    K.log(f"item 5 {'PASSED' if ok['item5'] else 'FAILED'}; tau0 open counts {i5['tau0_open_counts']}", t0)
    if not ok["item5"]:
        stop(out, "item 5 failed")
    out["regression_items_1_to_5_passed"] = True
    K.log("items 1 to 5 passed; candidate results may now be computed", t0)

    # ================================================================ item 6: development numbers
    dev, fam, arrs = {}, {}, {}
    for k in K.CANDIDATES:
        rec, pn, pc, ex = F3.run_family(bd["B"], Tr, gates[k], taus, parity)
        order = ([("Bprime_A1", pBp1)] if K.READS_CSD[k] else []) + order_A0(pc)
        ev, ar = K.evaluate(pn, pc, order, cl, pair_index, p_ref=pnA)
        dev[k] = {**summary_of(rec, ev, taus, pn, pc), "family": rec, "evaluation": ev, "Delta_k": ev["Delta_k"],
                  "comparator_order": [n for n, _ in order], "reads_csd": K.READS_CSD[k]}
        fam[k], arrs[k] = (pn, pc), ar
        K.log(f"{k}: fused {dev[k]['fused_r1']:.6f}, cf {dev[k]['counterpart_r1']:.6f}, comparator "
              f"{ev['bar_comparator']}, bar {ev['bar_margin']['point']:.6f} [{ev['bar_margin']['ci95'][0]:.6f}, "
              f"{ev['bar_margin']['ci95'][1]:.6f}], gain lo {ev['gain_statistic']['ci95'][0]:.6f}, Delta_k "
              f"{ev['Delta_k']['int']}, D10 {ev['D10_clauses']['clears_bar']}", t0)
    out["item6_dev"] = dev
    out["item7_D10"] = {k: dev[k]["D10_clauses"] for k in K.CANDIDATES}

    # ================================================================ item 8 / 9: carry or kill
    cr = K.carry(dev)
    out["item8_carry"] = cr
    boundaries = []
    for k in K.CANDIDATES:
        if any(dev[k]["evaluation"]["D10_boundary_flags"].values()):
            boundaries.append({"candidate": k, "D10_boundary_flags": dev[k]["evaluation"]["D10_boundary_flags"]})
        if dev[k]["Delta_k"]["at_zero"]:
            boundaries.append({"candidate": k, "Delta_k_at_zero": True})
    for k in cr.get("gap_exactly_24", []):
        boundaries.append({"candidate": k, "tie_gap_exactly_24": True})
    out["boundaries_8"] = boundaries
    K.log(f"carry: E {cr['E']}, M {cr['M']}, tied {cr['tied']}, carried {cr['carried']}, kill {cr['kill']}", t0)

    # ================================================================ §6.1 sensitivity for the carried candidate
    if not cr["kill"]:
        k = cr["carried"]
        pn, pc = fam[k]
        diffs = {"R1_minus_cosine": pn["r1"] - pcos["r1"], "R1_minus_RCA": pn["r1"] - prca["r1"],
                 "R1_minus_B": pn["r1"] - pB["r1"], "R1_minus_Bprime_A0": pn["r1"] - pBp0["r1"]}
        if K.READS_CSD[k]:
            diffs["R1_minus_Bprime_A1"] = pn["r1"] - pBp1["r1"]
        diffs.update({"R1_minus_counterpart": pn["r1"] - pc["r1"], "gain_statistic": pn["gain"] - pc["gain"],
                      "gain_minus_RCA": pn["gain"] - prca["gain"], "AFF_check": pn["r1"] - pnA["r1"]})
        sens = {}
        for name, dv in diffs.items():
            s = K.sensitivity(100.0 * np.asarray(dv, np.float64), cl)
            b = K.point_ci(dv, cl)
            s.update({"seed42_point_bootstrap": b["point"], "seed42_ci95": b["ci95"],
                      "seed42_bootstrap_half_width": (b["ci95"][1] - b["ci95"][0]) / 2.0})
            sens[name] = s
        out["sensitivity_6_1"] = {"candidate": k, "n_checks": len(sens), "checks": sens}
        K.log(f"sensitivity ({k}, {len(sens)} checks): AFF check x {sens['AFF_check']['x']:.6f}", t0)
    else:
        out["sensitivity_6_1"] = {"candidate": None, "note": "kill: no candidate carried, no sensitivity (rule §6.1)"}

    out["ok"] = ok
    out["runtime_s"] = round(time.time() - t0, 1)
    out["written_amsterdam"] = K.now_ams()

    # per-anchor arrays: saved, and their SHA-256 (float64 bytes) recorded for the comparison
    npz = {"anchor_group": cl, "pair_index": pair_index, "parity": parity, "taus": np.asarray(taus), "v": v}
    scorers = {"R1": (pnR1, pcR1), "AFF": (pnA, pcA), "R1_x_a_v": (pnI, pcI), **fam}
    for name, (pn, pc) in scorers.items():
        for m in METRICS:
            npz[f"{name}__fused__{m}"] = np.asarray(pn[m], np.float64)
            npz[f"{name}__cf__{m}"] = np.asarray(pc[m], np.float64)
    for name, p in (("B", pB), ("Bprime_A0", pBp0), ("Bprime_A1", pBp1), ("cosine", pcos), ("RCA", prca)):
        for m in METRICS:
            npz[f"{name}__{m}"] = np.asarray(p[m], np.float64)
    for k in K.CANDIDATES:
        for nm, a_ in arrs[k].items():
            npz[f"{k}__{nm}"] = a_
    for c in COND:
        npz[f"pick_A0__{c}"], npz[f"margin_A0__{c}"], npz[f"P_A0__{c}"] = picks[c], margins[c], P[c]
        npz[f"pick_A1__{c}"], npz[f"P_A1__{c}"] = pick1[c], P1[c]
        for gname, g in {**gates, "R1": gR1, "R1_x_a_v": gI}.items():
            npz[f"gate__{gname}__{c}"] = np.stack([g[t][c] for t in range(4)])
    out["per_anchor_sha256"] = {k_: K.sha_arr(np.asarray(v_, np.float64)) for k_, v_ in npz.items()
                                if np.asarray(v_).shape == (E,) and ("__fused__" in k_ or "__cf__" in k_ or
                                                                     k_.split("__")[0] in ("B", "Bprime_A0",
                                                                                           "Bprime_A1", "cosine",
                                                                                           "RCA"))}
    out["gate_sha256"] = {k_: K.sha_arr(v_) for k_, v_ in npz.items() if k_.startswith("gate__")}
    np.savez(OUTNPZ, **npz)
    out["arrays_npz_sha256"] = K.sha_file(OUTNPZ)
    K.save_json(OUTJSON, out)
    sha = K.sha_file(OUTJSON)
    OUTSHA.write_text(f"{sha}  rd4_phase1.json\n")
    K.log(f"phase 1 written: rd4_phase1.json SHA-256 {sha}", t0)


if __name__ == "__main__":
    main()
