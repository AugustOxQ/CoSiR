"""Phase-1 agreement (round-4 rule §8): compare the re-derivation's out/rd4_phase1.json and out/rd4_phase1_arrays.npz
with the implementation's results/regression_check.json, results/dev_seed42.json, results/carry.json and
results/seed42_arrays.npz (results/sensitivity.json is expected only if a candidate was carried), under the rule's
agreement tolerances:

  every discrete quantity identical (picks, gates, chosen cells, sigma*, the bar comparator, Delta_k, the carry, each
  clause's pass or fail); tau and v75 within 1e-15 absolute or 1e-9 relative; every margin, gain statistic, point and
  bound within 1e-9 percentage points (redundancy values within 1e-9 absolute, round 3 §8); every per-anchor array
  exactly.

Version 2 (after the implementation's files existed): the key layout was read from those four result files only (never
from the implementation's code), and the first version's token heuristic was replaced by the explicit map below
(IMPL_MAP is now this code: every comparison names its implementation key). Each implementation leaf that no comparison
uses, and each of our quantities with no counterpart, is listed with its reason; an implementation leaf without a reason
makes all_agree false. A few comparisons are derived: quantities the implementation records that our JSON lacks, or the
reverse, recomputed from the per-anchor and gate arrays (ours are bound to rd4_phase1.json by its arrays_npz_sha256),
labelled "derived".

Writes out/agreement_phase1.json. Usage: python rd4_compare_phase1.py [--impl-dir DIR] [--out PATH]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rd4_core as K
from rd4_core import COND, DIRS, METRICS, PAIR_NAMES

MINE = K.OUT / "rd4_phase1.json"
MINE_SHA = K.OUT / "rd4_phase1.sha256"
MINE_NPZ = K.OUT / "rd4_phase1_arrays.npz"
REF42 = K.OUT / "rd4_ref42.json"
STORED_A1 = K.T / "20261118_reader_fix_round2/results/cand_R1_A1.npz"
TOL = 1e-9
SCORERS = {"r1": "R1", "imgabst": "R1_x_a_v", "aff": "AFF", "v4": "V4", "v2": "V2", "v24": "V24"}   # impl -> ours
CANDS = K.CANDIDATES


# ---------------------------------------------------------------- comparison primitives

def as_num(x):
    if isinstance(x, (bool, np.bool_)) or x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return None


COMP = {"bprimea1": "Bprime_A1", "bprimea0": "Bprime_A0", "bprime": "Bprime_A0", "counterpart": "counterpart",
        "b": "B"}


def norm_name(x):
    if isinstance(x, str):
        s = "".join(ch for ch in x.lower() if ch.isalnum())
        return COMP.get(s, x)
    return x


def agree(kind, a, b):
    """-> (agree, abs_diff or None)."""
    if kind in ("float", "red"):
        x, y = as_num(a), as_num(b)
        if x is None or y is None:
            return False, None
        return abs(x - y) <= TOL, abs(x - y)
    if kind == "tau":
        x, y = as_num(a), as_num(b)
        if x is None or y is None:
            return False, None
        d = abs(x - y)
        return (d <= 1e-15 or d <= 1e-9 * abs(x)), d
    if kind == "int":
        x, y = as_num(a), as_num(b)
        ok = x is not None and y is not None and x == y and float(b) == int(float(b))
        return ok, (None if x is None or y is None else abs(x - y))
    if kind == "bool":
        return isinstance(b, (bool, np.bool_)) and isinstance(a, (bool, np.bool_)) and bool(a) == bool(b), None
    if kind == "name":
        return norm_name(a) == norm_name(b), None
    if kind == "names":
        return (isinstance(b, (list, tuple)) and isinstance(a, (list, tuple))
                and [norm_name(x) for x in a] == [norm_name(x) for x in b]), None
    if kind == "exact":                                          # identical JSON value (strings, None, lists)
        return json.dumps(K.jsonable(a), sort_keys=True) == json.dumps(K.jsonable(b), sort_keys=True), None
    raise ValueError(kind)


class Book:
    def __init__(self):
        self.rows = []
        self.used = set()

    def add(self, name, kind, mine, impl_ref, impl_val, derived=None):
        ok, d = agree(kind, mine, impl_val)
        r = {"name": name, "kind": kind, "mine": mine, "impl": impl_val, "impl_ref": impl_ref,
             "agree": bool(ok), "abs_diff": d}
        if derived:
            r["derived"] = derived
        self.rows.append(r)
        return ok

    def use(self, ref):
        self.used.add(ref)


def flatten(obj, prefix=()):
    out = {}
    if isinstance(obj, dict):
        if not obj:
            out[prefix] = obj
        for k, v in obj.items():
            out.update(flatten(v, prefix + (str(k),)))
    elif isinstance(obj, list):
        if not obj:
            out[prefix] = obj
        for i, v in enumerate(obj):
            out.update(flatten(v, prefix + (i,)))
    else:
        out[prefix] = obj
    return out


def dotted(path):
    return ".".join(str(p) for p in path)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl-dir", default=None, help="test only: another folder in place of ../results")
    ap.add_argument("--out", default=None, help="test only: another output path")
    args = ap.parse_args()
    res_dir = Path(args.impl_dir) if args.impl_dir else K.IMPL_RESULTS
    out_path = Path(args.out) if args.out else K.OUT / "agreement_phase1.json"
    K.assert_rules()
    K.assert_inputs(["20261118_reader_fix_round2/results/cand_R1_A1.npz"])
    sha = K.sha_file(MINE)
    recorded = MINE_SHA.read_text().split()[0]
    if sha != recorded:
        raise SystemExit(f"rd4_phase1.json SHA-256 {sha} differs from the recorded {recorded}")
    m = json.loads(MINE.read_text())
    if K.sha_file(MINE_NPZ) != m["arrays_npz_sha256"]:
        raise SystemExit("rd4_phase1_arrays.npz differs from the SHA-256 recorded in rd4_phase1.json")
    A = np.load(MINE_NPZ)
    ref42 = json.loads(REF42.read_text())
    kill = bool(m["item8_carry"]["kill"])
    files = {"regression": "regression_check.json", "dev": "dev_seed42.json", "carry": "carry.json",
             "arrays": "seed42_arrays.npz"}
    if not kill:
        files["sens"] = "sensitivity.json"
    res = {"version": 2, "rule_sha256": K.RULE4_SHA, "rd4_phase1_sha256": sha, "written_amsterdam": K.now_ams(),
           "impl_dir": str(res_dir),
           "tolerances": {"discrete": "identical", "tau_v75": "1e-15 abs or 1e-9 rel", "numbers_pp": TOL,
                          "redundancy_abs": TOL, "per_anchor_arrays": "identical (np.array_equal, same shape)"},
           "impl_files": {}}
    for f in files.values():
        p = res_dir / f
        res["impl_files"][f] = {"present": p.exists(), **({"sha256": K.sha_file(p)} if p.exists() else {})}
    missing = [f for f, v in res["impl_files"].items() if not v["present"]]
    if missing:
        res["all_agree"] = False
        res["missing_files"] = missing
        K.save_json(out_path, res)
        raise SystemExit(f"implementation files missing: {missing}")
    reg = json.loads((res_dir / files["regression"]).read_text())
    dev = json.loads((res_dir / files["dev"]).read_text())
    car = json.loads((res_dir / files["carry"]).read_text())
    Z = np.load(res_dir / files["arrays"], allow_pickle=False)
    bk = Book()

    # ---- impl accessors (each marks the leaf it reads as used)
    REG = {c["name"]: c for c in reg["comparisons"]}
    if len(REG) != len(reg["comparisons"]):
        raise SystemExit("duplicate comparison names in regression_check.json")

    def rg(name, *sub):
        """The 'got' of a named regression comparison (sub-indices into a list); an array comparison ('got' a
        description string) becomes its pass flag with max_abs_diff 0."""
        c = REG[name]
        bk.use(f"R:{name}")
        g = c["got"]
        if isinstance(g, str) and g.startswith("array"):
            return bool(c["pass"] and c.get("max_abs_diff", None) == 0)
        if isinstance(g, bool) and c["expected"] is True:          # a check outcome: true only if it also passed
            return bool(g and c["pass"])
        for s in sub:
            g = g[s]
        return g

    def jv(tag, obj, path):
        cur = obj
        for p in path.split("."):
            cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        bk.use(f"{tag}:{path}")
        return cur

    reg_v = lambda p: jv("regression_check.json", reg, p)  # noqa: E731
    dev_v = lambda p: jv("dev_seed42.json", dev, p)  # noqa: E731
    car_v = lambda p: jv("carry.json", car, p)  # noqa: E731

    def za(key):
        bk.use(f"Z:{key}")
        return Z[key]

    # ---- our quantities
    i1, i2, i3, i4, i5 = (m["item1_bundle"], m["item2_R1_AFF"], m["item3_IMGABST_q75"], m["item4_A1_reader"],
                          m["item5_gate_algebra"])
    S = {"R1": i2["summary"]["R1"], "AFF": i2["summary"]["AFF"], "R1_x_a_v": i3["summary"],
         **{k: m["item6_dev"][k] for k in CANDS}}
    cr = m["item8_carry"]
    pci3 = lambda s: [s["point"], s["ci95"][0], s["ci95"][1]]  # noqa: E731
    cells = lambda s, kind: [s[kind]["0"]["cell"], s[kind]["1"]["cell"]]  # noqa: E731
    sig = lambda s: [s["sigma_star"]["0"], s["sigma_star"]["1"]]  # noqa: E731

    def sett(s, kind, cell_id):
        for h in ("0", "1"):
            if s[kind][h]["cell"] == cell_id:
                c = s[kind][h]
                return [c["tau_index"], c["lambda_u"], c["lambda_a"]]
        raise KeyError(cell_id)

    def add_list(name, kinds, mine, ref, impl, derived=None):
        if not isinstance(impl, (list, tuple)) or len(impl) != len(mine):
            bk.add(name, "exact", mine, ref, impl, derived)
            return
        for j, (k_, a_, b_) in enumerate(zip(kinds, mine, impl)):
            bk.add(f"{name}[{j}]", k_, a_, f"{ref}[{j}]", b_, derived)

    F3 = ["float"] * 3

    # ================================================================ regression_check.json: named comparisons
    lb_ok = all(bool(v) for v in ref42["load_bundle_checks"].values())
    bools = {"r3.episodes.anchor": i1["episodes_equal_load_bundle"]["anchor"],
             "r3.episodes.candidates": i1["episodes_equal_load_bundle"]["candidates"],
             "r3.anchor": i1["anchors_equal_episodes_file"] and i1["episodes_equal_load_bundle"]["anchor"],
             "r3.parity": i1["episodes_equal_load_bundle"]["parity"],
             "r3.anchor_paintings": i1["episodes_equal_load_bundle"]["anchor_group"],
             "r3.pair_index": i1["episodes_equal_load_bundle"]["pair_index"],
             "r3.affect_head": i1["affect_head_equals_told_oracle_L"],
             "r3.round1_load_bundle_own_checks": lb_ok, "a1.round1_load_bundle_own_checks": lb_ok,
             "r3.external_per_anchor_seed42_aligned": i1["per_anchor_seed42_anchor_group_and_pair_index_equal"],
             "r3.redundancy.affect_least_redundant_both_directions": i1["affect_smallest_both_directions"],
             "D7.affect_least_redundant_both_directions": i1["affect_smallest_both_directions"],
             "a1.post.csd.img": i1["csd_full_sha256_equal_load_bundle"],
             "a1.post.csd.txt": i1["csd_full_sha256_equal_load_bundle"],
             "a1.v_equals_round1_image_S_C": bool(all(i1["features_A0_equal_rd2_cache"].values())
                                                  and i3["v_equal_bundle"]),
             "a1.Bprime_A1_mean_r1_equals_rule": i1["Bprime_A1_r1_equal_rule"],
             "v75.v75_recomputed_equals_rule": i3["v75_exact"],
             "v75.v75_keep_count_equals_rule": i3["a_v_count_equal_rule"],
             "v75.n_episodes_equals_seed42": m["bundle_record"]["n_episodes"] == K.N_EP_SEED42,
             "external_cosine_rca_loaded": all(i1["cosine_per_anchor_equal_stored"].values()),
             "R1.gates__a": i2["R1"]["gates_equal_stored"]["a"], "R1.gates__b": i2["R1"]["gates_equal_stored"]["b"],
             "R1.bar_v": i2["R1"]["bar_v_equal_stored"],
             "R1.cf_gain_exactly_0": bool(np.all(A["R1__cf__gain"] == 0)),
             "AFF.unit_factor_gates_float32": i2["AFF"]["AFF_gates_float32"],
             "AFF.cf_gain_exactly_0": bool(np.all(A["AFF__cf__gain"] == 0)),
             "cf_gain_exactly_0": bool(np.all(A["R1_x_a_v__cf__gain"] == 0)),
             "v_b_equals_v_a": i3["v_b_equal_v_a"],
             "candidate_gates_float32_0_1": all(i5["float32_0_1"].values()),
             "tau0_open_counts_are_integers": all(isinstance(v, int) for g in i5["tau0_open_counts"].values()
                                                  for v in g.values())}
    for c in COND:
        for d in DIRS:
            bools[f"r3.cos.{c}.{d}"] = i1["cosine_scores_equal_load_bundle"][f"{c}__{d}"]
            bools[f"r3.B.{c}.{d}"] = i1["B_scores_equal_load_bundle"][f"{c}__{d}"]
            bools[f"r3.Bprime_A0.{c}.{d}"] = i1["Bprime_A0_scores_equal_load_bundle"][f"{c}__{d}"]
            bools[f"a1.Bprime_A1.{c}.{d}"] = i1["Bprime_A1_scores_equal_load_bundle"][f"{c}__{d}"]
            bools[f"R1.T__{c}__{d}"] = i2["R1"]["T_equal_stored"][f"{c}__{d}"]
        bools[f"r3.features.{c}"] = i1["features_A0_equal_rd2_cache"][c]
        bools[f"a1.features_A1.{c}"] = i1["features_A1_equal_rd2_cache"][c]
        bools[f"a1.features_A1.{c}.first18_equal_A0"] = i1["features_A1_first18_equal_A0"][c]
        bools[f"R1.margin__{c}"] = i2["R1"]["margin_equal_stored"][c]
        bools[f"R1.pick__{c}"] = i2["R1"]["pick_equal_stored"][c]
        bools[f"P_A1__{c}_equals_probs__{c}"] = i4["probs_equal_stored"][c]
        bools[f"pi_A1__{c}_equals_pick__{c}"] = i4["pick_equal_stored"][c]
    for mm in METRICS:
        bools[f"r3.pB.{mm}"] = i1["B_per_anchor_equal_load_bundle"][mm]
        bools[f"r3.pBprime_A0.{mm}"] = i1["Bprime_A0_per_anchor_equal_load_bundle"][mm]
        bools[f"a1.pBprime_A1.{mm}"] = i1["Bprime_A1_per_anchor_equal_load_bundle"][mm]
        bools[f"R1.fused__{mm}"] = i2["R1"]["per_anchor_equal_stored"][f"fused__{mm}"]
        bools[f"R1.cf__{mm}"] = i2["R1"]["per_anchor_equal_stored"][f"cf__{mm}"]
    for h in K.A0:
        for mo in ("img", "txt"):
            bools[f"r3.post.{h}.{mo}"] = i1["posteriors_equal_load_bundle"][f"{h}__{mo}"]
        for d in DIRS:
            bools[f"r3.redundancy.{h}.{d}_equals_rule"] = i1["redundancy_exact"][h][d]
    for d in DIRS:
        bools[f"r3.stack.{d}"] = i1["grouping_scores_A0_equal_rd2_cache"][d]
    # item 5, per tau index and condition, derived from our gate arrays (bound to rd4_phase1.json)
    za1 = np.load(STORED_A1)
    gm = {k: {c: A[f"gate__{k}__{c}"] for c in COND} for k in ("AFF", "V4", "V2", "V24")}
    v_lt = (np.asarray(A["v"], np.float64) < K.V75).astype(np.float32)
    derived_g = {}
    for t in range(4):
        for c in COND:
            for k in CANDS:
                derived_g[f"{k}_closed_where_AFF_closed.tau{t}.{c}"] = bool(not np.any(gm[k][c][t][gm["AFF"][c][t] == 0]))
            derived_g[f"V24_equals_V4_times_V2.tau{t}.{c}"] = bool(np.array_equal(gm["V24"][c][t],
                                                                                 gm["V4"][c][t] * gm["V2"][c][t]))
            derived_g[f"V2_equals_AFF_times_stored_pick_affect.tau{t}.{c}"] = bool(np.array_equal(
                gm["V2"][c][t], gm["AFF"][c][t] * (za1[f"pick__{c}"] == 0).astype(np.float32)))
            derived_g[f"V4_equals_AFF_times_v_below_v75.tau{t}.{c}"] = bool(np.array_equal(gm["V4"][c][t],
                                                                                         gm["AFF"][c][t] * v_lt))
            bools[f"AFF.unit_factor_gate_equals_gates_aff.tau{t}.{c}"] = \
                i2["AFF"]["candidate_gate_function_with_unit_factors_equals_AFF_gates"]
    for name, v in bools.items():
        bk.add(f"reg.{name}", "bool", bool(v), f"regression_check.json:comparisons[{name}]", rg(name))
    for name, v in derived_g.items():
        bk.add(f"reg.{name}", "bool", v, f"regression_check.json:comparisons[{name}]", rg(name),
               derived="per tau index and condition from our gate arrays (rd4_phase1_arrays.npz)")

    # numbers among the named comparisons
    for h in K.A0:
        for d in DIRS:
            bk.add(f"reg.D7.{h}.{d}", "red", i1["redundancy"][h][d], f"comparisons[D7.{h}.{d}]", rg(f"D7.{h}.{d}"))
            bk.add(f"reg.redundancy_D7.{h}.{d}", "red", i1["redundancy"][h][d], f"redundancy_D7.{h}.{d}",
                   reg_v(f"redundancy_D7.{h}.{d}"))
    bk.add("reg.Bprime_A1_mean_r1", "float", i1["Bprime_A1_r1"], "comparisons[Bprime_A1_mean_r1]",
           rg("Bprime_A1_mean_r1"))
    bk.add("reg.v75_value", "tau", i3["v75_recomputed"], "comparisons[v75_value]", rg("v75_value"))
    bk.add("reg.v75_keep_count", "int", i3["a_v_count"], "comparisons[v75_keep_count]", rg("v75_keep_count"))
    bk.add("reg.v75_recomputed", "tau", i3["v75_recomputed"], "comparisons[v75_recomputed]", rg("v75_recomputed"))
    bk.add("reg.a_v_keep_count", "int", i3["a_v_count"], "comparisons[a_v_keep_count]", rg("a_v_keep_count"))
    bk.add("reg.v75_rule_text_equals_r4_common", "tau", K.V75, "comparisons[v75_rule_text_equals_r4_common]",
           rg("v75_rule_text_equals_r4_common"), derived="the rule's v75 constant as our code holds it")
    bk.add("reg.R1.n_margins", "int", 2 * len(A["margin_A0__a"]), "comparisons[R1.n_margins]", rg("R1.n_margins"),
           derived="2 x our episode count")
    add_list("reg.R1.tau_recomputed_equals_rc_tau", ["tau"] * 4, i2["R1"]["taus_recomputed"],
             "comparisons[R1.tau_recomputed_equals_rc_tau]", rg("R1.tau_recomputed_equals_rc_tau"))
    add_list("reg.R1.stored_extra_taus_equal_rc_tau", ["tau"] * 4, i2["R1"]["taus_rc_tau_json"],
             "comparisons[R1.stored_extra_taus_equal_rc_tau]", rg("R1.stored_extra_taus_equal_rc_tau"))
    # scorers R1, AFF (item 2) and R1 x a_v (item 3)
    for pre, s in (("R1.", S["R1"]), ("AFF.", S["AFF"]), ("", S["R1_x_a_v"])):
        add_list(f"reg.{pre}fused_cells", ["int"] * 2, cells(s, "fused_cells"), f"comparisons[{pre}fused_cells]",
                 rg(f"{pre}fused_cells"))
        add_list(f"reg.{pre}cf_cells", ["int"] * 2, cells(s, "counterpart_cells"), f"comparisons[{pre}cf_cells]",
                 rg(f"{pre}cf_cells"))
        add_list(f"reg.{pre}sigma_star", ["float"] * 2, sig(s), f"comparisons[{pre}sigma_star]", rg(f"{pre}sigma_star"))
        for kind in ("fused_cells", "counterpart_cells"):
            for h in ("0", "1"):
                cid = s[kind][h]["cell"]
                nm = f"{pre}cell_{cid}_is_the_rule_text"
                if nm in REG and f"R:{nm}" not in bk.used:
                    add_list(f"reg.{nm}", ["int", "float", "float"], sett(s, kind, cid), f"comparisons[{nm}]", rg(nm))
        comp_key = f"{pre}comparator" if pre else "bar_comparator"
        bk.add(f"reg.{comp_key}", "name", s["bar_comparator"], f"comparisons[{comp_key}]", rg(comp_key))
        bk.add(f"reg.{pre}fused_r1", "float", s["fused_r1"], f"comparisons[{pre}fused_r1]", rg(f"{pre}fused_r1"))
        bk.add(f"reg.{pre}cf_r1", "float", s["counterpart_r1"], f"comparisons[{pre}cf_r1]", rg(f"{pre}cf_r1"))
        add_list(f"reg.{pre}bar_margin", F3, pci3(s["bar_margin"]), f"comparisons[{pre}bar_margin]",
                 rg(f"{pre}bar_margin"))
        add_list(f"reg.{pre}gain_statistic", F3, pci3(s["gain_statistic"]), f"comparisons[{pre}gain_statistic]",
                 rg(f"{pre}gain_statistic"))
    for nm in ("R1.bar_margin_equals_stored_json", "R1.dev_record.bar_margin"):
        add_list(f"reg.{nm}", F3, pci3(S["R1"]["bar_margin"]), f"comparisons[{nm}]", rg(nm))
    for nm in ("R1.gain_statistic_equals_stored_json", "R1.dev_record.gain_statistic"):
        add_list(f"reg.{nm}", F3, pci3(S["R1"]["gain_statistic"]), f"comparisons[{nm}]", rg(nm))
    for nm in ("R1.comparator_equals_stored_json", "R1.dev_record.bar_comparator"):
        bk.add(f"reg.{nm}", "name", S["R1"]["bar_comparator"], f"comparisons[{nm}]", rg(nm))
    bk.add("reg.R1.dev_record.fused_r1", "float", S["R1"]["fused_r1"], "comparisons[R1.dev_record.fused_r1]",
           rg("R1.dev_record.fused_r1"))
    bk.add("reg.R1.dev_record.cf_r1", "float", S["R1"]["counterpart_r1"], "comparisons[R1.dev_record.cf_r1]",
           rg("R1.dev_record.cf_r1"))
    sa = S["AFF"]
    for pre in ("AFF.", "AFF.dev_record."):
        mv = "margin_vs_counterpart"
        add_list(f"reg.{pre}{mv}", F3, pci3(sa[mv]), f"comparisons[{pre}{mv}]", rg(f"{pre}{mv}"))
        ek = "either_vs_counterpart" if pre == "AFF." else "either_change"
        bk.add(f"reg.{pre}{ek}", "float", sa["either_change_vs_counterpart"]["point"], f"comparisons[{pre}{ek}]",
               rg(f"{pre}{ek}"))
        for p in PAIR_NAMES:
            bk.add(f"reg.{pre}per_pair_bar_margin.{p}", "float", sa["per_pair_bar_margin"][p]["point"],
                   f"comparisons[{pre}per_pair_bar_margin.{p}]", rg(f"{pre}per_pair_bar_margin.{p}"))
    for nm, key in (("fused_r1", "fused_r1"), ("cf_r1", "counterpart_r1")):
        bk.add(f"reg.AFF.dev_record.{nm}", "float", sa[key], f"comparisons[AFF.dev_record.{nm}]",
               rg(f"AFF.dev_record.{nm}"))
    bk.add("reg.AFF.dev_record.bar_comparator", "name", sa["bar_comparator"],
           "comparisons[AFF.dev_record.bar_comparator]", rg("AFF.dev_record.bar_comparator"))
    for nm in ("bar_margin", "gain_statistic"):
        add_list(f"reg.AFF.dev_record.{nm}", F3, pci3(sa[nm]), f"comparisons[AFF.dev_record.{nm}]",
                 rg(f"AFF.dev_record.{nm}"))
    bk.add("reg.AFF.dev_record.d10_clears", "bool", sa["D10_clauses"]["clears_bar"],
           "comparisons[AFF.dev_record.d10_clears]", rg("AFF.dev_record.d10_clears"))
    amr_f, amr_b = i2["AFF"]["AFF_minus_R1_fused"]["ours"], i2["AFF"]["AFF_minus_R1_bar"]["ours"]
    add_list("reg.AFF.aff_minus_r1_fused_r1", F3, amr_f, "comparisons[AFF.aff_minus_r1_fused_r1]",
             rg("AFF.aff_minus_r1_fused_r1"))
    add_list("reg.AFF.aff_minus_r1_bar_margin", F3, amr_b, "comparisons[AFF.aff_minus_r1_bar_margin]",
             rg("AFF.aff_minus_r1_bar_margin"))
    for c in COND:
        bk.add(f"reg.AFF.tau0_open_count.{c}", "int", i2["AFF"]["tau0_open_counts"][c],
               f"comparisons[AFF.tau0_open_count.{c}]", rg(f"AFF.tau0_open_count.{c}"))
    s3 = S["R1_x_a_v"]
    bk.add("reg.either_vs_counterpart", "float", s3["either_change_vs_counterpart"]["point"],
           "comparisons[either_vs_counterpart]", rg("either_vs_counterpart"))
    for p in PAIR_NAMES:
        bk.add(f"reg.per_pair_bar_margin.{p}", "float", s3["per_pair_bar_margin"][p]["point"],
               f"comparisons[per_pair_bar_margin.{p}]", rg(f"per_pair_bar_margin.{p}"))
    # the implementation's rule-constant lists (the brainstorm files' values) against our computed values
    nm = "AFF.rule_constants_equal_bs_04_readers.json"
    mine_l = [sa["fused_r1"], sa["counterpart_r1"], sa["bar_comparator"], pci3(sa["bar_margin"]),
              pci3(sa["margin_vs_counterpart"]), pci3(sa["gain_statistic"]),
              sa["either_change_vs_counterpart"]["point"], [sa["per_pair_bar_margin"][p]["point"] for p in PAIR_NAMES]]
    got = rg(nm)
    for j, (a_, b_) in enumerate(zip(mine_l, got)):
        if isinstance(a_, list):
            add_list(f"reg.{nm}[{j}]", ["float"] * len(a_), a_, f"comparisons[{nm}][{j}]", b_)
        else:
            bk.add(f"reg.{nm}[{j}]", "name" if isinstance(a_, str) else "float", a_, f"comparisons[{nm}][{j}]", b_)
    nm = "AFF.rule_constants_equal_bs_05_aff.json"
    got = rg(nm)
    add_list(f"reg.{nm}[0]", F3, amr_f, f"comparisons[{nm}][0]", got[0])
    add_list(f"reg.{nm}[1]", F3, amr_b, f"comparisons[{nm}][1]", got[1])
    nm = "rule_constants_equal_bs_04_readers.json"
    got = rg(nm)
    mine_l = [s3["fused_r1"], s3["counterpart_r1"], s3["bar_comparator"], pci3(s3["bar_margin"]),
              pci3(s3["gain_statistic"]), s3["either_change_vs_counterpart"]["point"],
              [s3["per_pair_bar_margin"][p]["point"] for p in PAIR_NAMES],
              [sett(s3, "fused_cells", s3["fused_cells"][h]["cell"]) for h in ("0", "1")],
              [sett(s3, "counterpart_cells", s3["counterpart_cells"][h]["cell"]) for h in ("0", "1")]]
    for j, (a_, b_) in enumerate(zip(mine_l, got)):
        if isinstance(a_, list) and a_ and isinstance(a_[0], list):
            for h, (aa, bb) in enumerate(zip(a_, b_)):
                add_list(f"reg.{nm}[{j}][{h}]", ["int", "float", "float"], aa, f"comparisons[{nm}][{j}][{h}]", bb)
        elif isinstance(a_, list):
            add_list(f"reg.{nm}[{j}]", ["float"] * len(a_), a_, f"comparisons[{nm}][{j}]", b_)
        else:
            bk.add(f"reg.{nm}[{j}]", "name" if isinstance(a_, str) else "float", a_, f"comparisons[{nm}][{j}]", b_)
    # item 4 dtypes and ties
    for c in COND:
        bk.add(f"reg.P_A1__{c}_dtype_equals_stored", "exact", str(A[f"P_A1__{c}"].dtype),
               f"comparisons[P_A1__{c}_dtype_equals_stored]", rg(f"P_A1__{c}_dtype_equals_stored"),
               derived="dtype of our P_A1 array")
        bk.add(f"reg.stored_pick__{c}_is_int8", "exact", i4["stored_pick_dtype"][c],
               f"comparisons[stored_pick__{c}_is_int8]", rg(f"stored_pick__{c}_is_int8"))
        bk.add(f"reg.no_exact_argmax_tie__{c}", "int", i4["exact_argmax_ties"][c],
               f"comparisons[no_exact_argmax_tie__{c}]", rg(f"no_exact_argmax_tie__{c}"))
    # every named comparison of the implementation passed, as every item of ours did
    impl_all_pass = all(c["pass"] for c in reg["comparisons"])
    bk.add("reg.every_comparison_pass_flag", "bool", bool(m["regression_items_1_to_5_passed"]),
           "comparisons[*].pass (all)", impl_all_pass)

    # ================================================================ regression_check.json: top-level fields
    ok = m["ok"]
    bk.add("reg.passed", "bool", bool(m["regression_items_1_to_5_passed"]), "passed", reg_v("passed"))
    bk.add("reg.items_passed", "exact", [i for i in range(1, 6) if ok[f"item{i}"]], "items_passed",
           reg_v("items_passed"))
    bk.add("reg.failed", "exact", [], "failed", reg_v("failed"))
    bk.add("reg.stopped_at_item", "exact", None, "stopped_at_item", reg_v("stopped_at_item"))
    for nm, s in (("R1", S["R1"]), ("AFF", S["AFF"]), ("IMGABST_q75", S["R1_x_a_v"])):
        add_list(f"reg.cells.{nm}.fused", ["int"] * 2, cells(s, "fused_cells"), f"cells.{nm}.fused",
                 reg_v(f"cells.{nm}.fused"))
        add_list(f"reg.cells.{nm}.counterpart", ["int"] * 2, cells(s, "counterpart_cells"), f"cells.{nm}.counterpart",
                 reg_v(f"cells.{nm}.counterpart"))
        add_list(f"reg.cells.{nm}.sigma", ["float"] * 2, sig(s), f"cells.{nm}.sigma", reg_v(f"cells.{nm}.sigma"))
    for g in ("AFF", "V4", "V2", "V24"):
        for c in COND:
            bk.add(f"reg.open_tau0_counts.{g}.{c}", "int", i5["tau0_open_counts"][g][c], f"open_tau0_counts.{g}.{c}",
                   reg_v(f"open_tau0_counts.{g}.{c}"))
            bk.add(f"dev.open_tau0_counts.{g}.{c}", "int", i5["tau0_open_counts"][g][c], f"open_tau0_counts.{g}.{c}",
                   dev_v(f"open_tau0_counts.{g}.{c}"))
    for fpath, sha_ in reg["inputs_sha256"].items():          # keys are file paths (they contain dots)
        if fpath in K.INPUT_SHA:
            bk.use(f"regression_check.json:inputs_sha256.{fpath}")
            bk.add(f"reg.inputs_sha256.{fpath}", "exact", K.INPUT_SHA[fpath], f"inputs_sha256.{fpath}", sha_)
    bk.add("reg.provenance.rule_sha256", "exact", K.RULE4_SHA, "provenance.rule_sha256",
           reg_v("provenance.rule_sha256"))
    bk.add("reg.dry_run", "exact", False, "dry_run", reg_v("dry_run"))
    bk.add("reg.provenance.smoke", "exact", False, "provenance.smoke", reg_v("provenance.smoke"))
    bk.add("reg.n_comparisons_equals_list_length", "int", len(reg["comparisons"]), "n_comparisons",
           reg_v("n_comparisons"), derived="length of the implementation's own comparison list")

    # ================================================================ dev_seed42.json
    bk.add("dev.rule_sha256", "exact", K.RULE4_SHA, "rule_sha256", dev_v("rule_sha256"))
    bk.add("dev.provenance.rule_sha256", "exact", K.RULE4_SHA, "provenance.rule_sha256",
           dev_v("provenance.rule_sha256"))
    bk.add("dev.dry_run", "exact", False, "dry_run", dev_v("dry_run"))
    bk.add("dev.provenance.smoke", "exact", False, "provenance.smoke", dev_v("provenance.smoke"))
    bk.add("dev.order", "names", list(CANDS), "order", dev_v("order"))
    for key, nm in [(f"candidates.{k}", k) for k in CANDS] + [("aff", "AFF")]:
        s = S[nm]
        bk.add(f"dev.{key}.name", "name", nm, f"{key}.name", dev_v(f"{key}.name"))
        bk.add(f"dev.{key}.fused_r1", "float", s["fused_r1"], f"{key}.fused_r1", dev_v(f"{key}.fused_r1"))
        bk.add(f"dev.{key}.cf_r1", "float", s["counterpart_r1"], f"{key}.cf_r1", dev_v(f"{key}.cf_r1"))
        for h in ("0", "1"):
            bk.add(f"dev.{key}.cells.fpick.{h}", "int", s["fused_cells"][h]["cell"], f"{key}.cells.fpick.{h}",
                   dev_v(f"{key}.cells.fpick.{h}"))
            bk.add(f"dev.{key}.cells.cpick.{h}", "int", s["counterpart_cells"][h]["cell"], f"{key}.cells.cpick.{h}",
                   dev_v(f"{key}.cells.cpick.{h}"))
            bk.add(f"dev.{key}.sigma.{h}", "float", s["sigma_star"][h], f"{key}.sigma.{h}", dev_v(f"{key}.sigma.{h}"))
            for part, kind in (("fused", "fused_cells"), ("cf", "counterpart_cells")):
                for fld, kd in (("cell", "int"), ("tau_index", "int"), ("tau", "tau"), ("lambda_u", "float"),
                                ("lambda_a", "float")):
                    p = f"{key}.cell_text.{part}.{h}.{fld}"
                    bk.add(f"dev.{p}", kd, s[kind][h][fld], p, dev_v(p))
        bk.add(f"dev.{key}.bar_comparator", "name", s["bar_comparator"], f"{key}.bar_comparator",
               dev_v(f"{key}.bar_comparator"))
        for fld in ("bar_margin", "margin_vs_counterpart", "gain_statistic"):
            bk.add(f"dev.{key}.{fld}.point", "float", s[fld]["point"], f"{key}.{fld}.point",
                   dev_v(f"{key}.{fld}.point"))
            for j in (0, 1):
                bk.add(f"dev.{key}.{fld}.ci95.{j}", "float", s[fld]["ci95"][j], f"{key}.{fld}.ci95.{j}",
                       dev_v(f"{key}.{fld}.ci95.{j}"))
        for fld in ("margin_vs_counterpart", "gain_statistic"):
            bk.add(f"dev.{key}.{fld}.n_clusters", "int", m["bundle_record"]["n_clusters"], f"{key}.{fld}.n_clusters",
                   dev_v(f"{key}.{fld}.n_clusters"), derived="our bundle's anchor-painting count")
        bk.add(f"dev.{key}.either_change", "float", s["either_change_vs_counterpart"]["point"],
               f"{key}.either_change", dev_v(f"{key}.either_change"))
        for cl_, mk in (("c1", "c1_bar_point_ge_0.5"), ("c2", "c2_bar_lower_gt_0"), ("c3", "c3_gain_lower_gt_0"),
                        ("clears", "clears_bar")):
            bk.add(f"dev.{key}.d10.{cl_}", "bool", s["D10_clauses"][mk], f"{key}.d10.{cl_}", dev_v(f"{key}.d10.{cl_}"))
        for p in PAIR_NAMES:
            v = s["per_pair_bar_margin"][p]
            bk.add(f"dev.{key}.per_pair_bar_margin.{p}.point", "float", v["point"],
                   f"{key}.per_pair_bar_margin.{p}.point", dev_v(f"{key}.per_pair_bar_margin.{p}.point"))
            for j in (0, 1):
                bk.add(f"dev.{key}.per_pair_bar_margin.{p}.ci95.{j}", "float", v["ci95"][j],
                       f"{key}.per_pair_bar_margin.{p}.ci95.{j}", dev_v(f"{key}.per_pair_bar_margin.{p}.ci95.{j}"))
        if nm in CANDS:
            dk = s["Delta_k"]
            bk.add(f"dev.{key}.delta_int", "int", dk["int"], f"{key}.delta_int", dev_v(f"{key}.delta_int"))
            bk.add(f"dev.{key}.delta.point", "float", dk["point_pp"], f"{key}.delta.point", dev_v(f"{key}.delta.point"))
            for j in (0, 1):
                bk.add(f"dev.{key}.delta.ci95.{j}", "float", dk["ci95"][j], f"{key}.delta.ci95.{j}",
                       dev_v(f"{key}.delta.ci95.{j}"))
            mine_b = [b["candidate"] for b in m["boundaries_8"] if b.get("candidate") == nm]
        else:
            bk.add("dev.aff.delta_int", "exact", None, "aff.delta_int", dev_v("aff.delta_int"),
                   derived="Delta_k is not defined for AFF (D9); ours has none")
            bk.add("dev.aff.delta", "exact", None, "aff.delta", dev_v("aff.delta"),
                   derived="Delta_k is not defined for AFF (D9); ours has none")
            mine_b = []
        bk.add(f"dev.{key}.boundaries", "exact", mine_b, f"{key}.boundaries", dev_v(f"{key}.boundaries"))
    bk.add("dev.beside_aff.Bprime_A1_mean_r1", "float", i1["Bprime_A1_r1"], "beside_aff.Bprime_A1_mean_r1",
           dev_v("beside_aff.Bprime_A1_mean_r1"))
    bk.add("dev.beside_aff.Bprime_A0_mean_r1", "float", i1["Bprime_A0_r1"], "beside_aff.Bprime_A0_mean_r1",
           dev_v("beside_aff.Bprime_A0_mean_r1"))
    bk.add("dev.beside_aff.B_mean_r1", "float", i1["B_r1"], "beside_aff.B_mean_r1", dev_v("beside_aff.B_mean_r1"))
    bk.add("dev.beside_aff.AFF_fused_r1", "float", sa["fused_r1"], "beside_aff.AFF_fused_r1",
           dev_v("beside_aff.AFF_fused_r1"))
    amb = K.point_ci(np.asarray(A["AFF__fused__r1"], np.float64) - np.asarray(A["Bprime_A1__r1"], np.float64),
                     A["anchor_group"])
    der = "AFF fused R@1 minus B'(A1), paired, painting bootstrap, from our per-anchor arrays"
    bk.add("dev.beside_aff.AFF_minus_Bprime_A1.point", "float", amb["point"], "beside_aff.AFF_minus_Bprime_A1.point",
           dev_v("beside_aff.AFF_minus_Bprime_A1.point"), derived=der)
    for j in (0, 1):
        bk.add(f"dev.beside_aff.AFF_minus_Bprime_A1.ci95.{j}", "float", amb["ci95"][j],
               f"beside_aff.AFF_minus_Bprime_A1.ci95.{j}", dev_v(f"beside_aff.AFF_minus_Bprime_A1.ci95.{j}"),
               derived=der)

    # ================================================================ carry.json
    bk.add("carry.order", "names", cr["order"], "order", car_v("order"))
    bk.add("carry.tie_band_units", "int", cr["tie_band"], "tie_band_units", car_v("tie_band_units"))
    bk.add("carry.E", "names", cr["E"], "carry.E", car_v("carry.E"))
    bk.add("carry.M", "exact" if cr["M"] is None else "int", cr["M"], "carry.M", car_v("carry.M"))
    bk.add("carry.tied", "names", cr["tied"], "carry.tied", car_v("carry.tied"))
    bk.add("carry.carried", "exact" if cr["carried"] is None else "name", cr["carried"], "carry.carried",
           car_v("carry.carried"))
    bk.add("carry.boundaries", "exact", m["boundaries_8"], "carry.boundaries", car_v("carry.boundaries"))
    bk.add("carry.boundary_reported", "exact", None if not m["boundaries_8"] else m["boundaries_8"],
           "boundary_reported", car_v("boundary_reported"),
           derived="our boundary list is empty, so nothing is reported (None)")
    bk.add("carry.kill", "bool", cr["kill"], "kill", car_v("kill"))
    for k in CANDS:
        bk.add(f"carry.delta_int.{k}", "int", cr["Delta_k"][k], f"delta_int.{k}", car_v(f"delta_int.{k}"))
        for cl_, mk in (("c1", "c1_bar_point_ge_0.5"), ("c2", "c2_bar_lower_gt_0"), ("c3", "c3_gain_lower_gt_0"),
                        ("clears", "clears_bar")):
            bk.add(f"carry.d10.{k}.{cl_}", "bool", cr["D10_clauses"][k][mk], f"d10.{k}.{cl_}",
                   car_v(f"d10.{k}.{cl_}"))
    bk.add("carry.dry_run", "exact", False, "dry_run", car_v("dry_run"))
    bk.add("carry.provenance.rule_sha256", "exact", K.RULE4_SHA, "provenance.rule_sha256",
           car_v("provenance.rule_sha256"))
    bk.add("carry.provenance.smoke", "exact", False, "provenance.smoke", car_v("provenance.smoke"))
    # the implementation's own SHA-256 bindings between its files, checked against the files themselves
    for tag, obj, fld, f in (("dev", dev, "regression_check_sha256", files["regression"]),
                             ("dev", dev, "seed42_arrays_sha256", files["arrays"]),
                             ("carry", car, "regression_check_sha256", files["regression"]),
                             ("carry", car, "seed42_arrays_sha256", files["arrays"]),
                             ("carry", car, "dev_seed42_sha256", files["dev"])):
        getter = dev_v if tag == "dev" else car_v
        bk.add(f"{tag}.{fld}", "exact", res["impl_files"][f]["sha256"], fld, getter(fld),
               derived=f"SHA-256 of the implementation's {f} itself (internal binding)")

    # ================================================================ seed42_arrays.npz, exactly
    arr_rows = []

    def arr(name, mine_a, impl_key, derived=None):
        b = za(impl_key) if not impl_key.startswith("=") else None
        mine_a = np.asarray(mine_a)
        b = np.asarray(b) if b is not None else np.asarray(derived_impl[impl_key[1:]])
        ok_ = bool(mine_a.shape == b.shape and np.array_equal(mine_a, b))
        r = {"name": name, "impl_key": impl_key, "shape": list(b.shape), "dtype_mine": str(mine_a.dtype),
             "dtype_impl": str(b.dtype), "equal": ok_}
        if not ok_ and mine_a.shape == b.shape and mine_a.dtype.kind in "fiub":
            r["n_differ"] = int((mine_a != b).sum())
        if derived:
            r["derived"] = derived
        arr_rows.append(r)

    derived_impl = {}
    arr("anchor_group", A["anchor_group"], "cl")
    for nm in ("pair_index", "parity", "taus", "v"):
        arr(nm, A[nm], nm)
    arr("a_v (keep)", (np.asarray(A["v"], np.float64) < K.V75).astype(np.float32), "keep",
        derived="ours = 1[v < v75] from our v (float32)")
    for ik, ok_name in SCORERS.items():
        for part in ("fused", "cf"):
            for mm in METRICS:
                arr(f"{ok_name}__{part}__{mm}", A[f"{ok_name}__{part}__{mm}"], f"{ik}_{part}__{mm}")
        for c in COND:
            arr(f"gate__{ok_name}__{c}", A[f"gate__{ok_name}__{c}"], f"{ik}_gate__{c}")
        s = S[ok_name]
        add_list(f"npz.{ik}_fused_cells", ["int"] * 2, cells(s, "fused_cells"), f"seed42_arrays.npz:{ik}_fused_cells",
                 [int(x) for x in za(f"{ik}_fused_cells")])
        add_list(f"npz.{ik}_cf_cells", ["int"] * 2, cells(s, "counterpart_cells"), f"seed42_arrays.npz:{ik}_cf_cells",
                 [int(x) for x in za(f"{ik}_cf_cells")])
        add_list(f"npz.{ik}_sigma", ["float"] * 2, sig(s), f"seed42_arrays.npz:{ik}_sigma",
                 [float(x) for x in za(f"{ik}_sigma")])
    for ik, ok_name in (("B", "B"), ("Bp0", "Bprime_A0"), ("Bp1", "Bprime_A1"), ("cosine", "cosine"), ("rca", "RCA")):
        for mm in METRICS:
            arr(f"{ok_name}__{mm}", A[f"{ok_name}__{mm}"], f"{ik}__{mm}")
    for c in COND:
        arr(f"pick_A0__{c}", A[f"pick_A0__{c}"], f"pick__{c}")
        arr(f"margin_A0__{c}", A[f"margin_A0__{c}"], f"margin__{c}")
        arr(f"P_A0__{c}", A[f"P_A0__{c}"], f"P__{c}")
        arr(f"P_A1__{c}", A[f"P_A1__{c}"], f"P_A1__{c}")
        arr(f"pick_A1__{c}", A[f"pick_A1__{c}"], f"pick_A1__{c}")
    # our difference arrays against the same differences formed from the implementation's arrays
    comp_arr = {"Bprime_A1": "Bp1__r1", "Bprime_A0": "Bp0__r1", "B": "B__r1"}
    for k in CANDS:
        ik = k.lower()
        cname = S[k]["bar_comparator"]
        cmp_r1 = Z[f"{ik}_cf__r1"] if cname == "counterpart" else Z[comp_arr[cname]]
        derived_impl[f"{k}__bar_v"] = Z[f"{ik}_fused__r1"] - cmp_r1
        derived_impl[f"{k}__gain_v"] = Z[f"{ik}_fused__gain"] - Z[f"{ik}_cf__gain"]
        derived_impl[f"{k}__marg_v"] = Z[f"{ik}_fused__r1"] - Z[f"{ik}_cf__r1"]
        derived_impl[f"{k}__d_vs_AFF"] = Z[f"{ik}_fused__r1"] - Z["aff_fused__r1"]
        for nm in ("bar_v", "gain_v", "marg_v", "d_vs_AFF"):
            arr(f"{k}__{nm}", A[f"{k}__{nm}"], f"={k}__{nm}",
                derived="formed from the implementation's per-anchor arrays (its comparator for bar_v)")
        # integer sums of 4 R@1 behind every mean, from the implementation's arrays
        sums = S[k]["r1_int4_sums"]
        impl_sums = {"fused": Z[f"{ik}_fused__r1"], "counterpart": Z[f"{ik}_cf__r1"], "B": Z["B__r1"],
                     "Bprime_A0": Z["Bp0__r1"], "Bprime_A1": Z["Bp1__r1"]}
        for nm, v in sums.items():
            bk.add(f"npz.int4_sum.{k}.{nm}", "int", v, f"4 x sum of seed42_arrays.npz R@1 ({nm})",
                   int(K.as_int4(impl_sums[nm]).sum()), derived="integer sum of 4 R@1 from the implementation's arrays")
        bk.add(f"npz.Delta_int.{k}", "int", S[k]["Delta_k"]["int"], f"4 x sum({ik}_fused__r1 - aff_fused__r1)",
               int((K.as_int4(Z[f"{ik}_fused__r1"]) - K.as_int4(Z["aff_fused__r1"])).sum()),
               derived="Delta_k from the implementation's arrays")
    # open counts at every tau index, of every gate we wrote, from the implementation's gate arrays
    for ik, ok_name in SCORERS.items():
        for t in range(4):
            for c in COND:
                bk.add(f"npz.open_count.{ok_name}.tau{t}.{c}", "int", i5["open_counts"][ok_name][f"tau_{t}"][c],
                       f"sum of seed42_arrays.npz:{ik}_gate__{c}[{t}]", int(Z[f"{ik}_gate__{c}"][t].sum()),
                       derived="open count from the implementation's gate arrays")
    res["arrays"] = arr_rows

    # ================================================================ unused implementation leaves, with reasons
    def reason(ref):
        if ref.startswith("R:"):
            nm = ref[2:]
            if nm.startswith("r3.T_N1u."):
                return "T_N1u is an intermediate of B and B'; our bundle does not store it (B, B' and their arrays are compared)"
            if nm.startswith("r3.episodes."):
                return ("episode-file arrays against round 1's bundle; ours compares anchors and candidates with the "
                        "episodes file and load_bundle, and the 18 features (which read every support and contrast "
                        "pair) with the rd2 cache, all equal")
            return None
        path = ref.split(":", 1)[1]
        if ref.startswith("Z:"):
            return None
        if path.startswith("inputs_sha256."):
            return "an input we do not read (round 3's implementation modules, by mandate)"
        if path.endswith(".k_top"):
            return "round 2's top-k parameter (13 = no restriction); the rule uses no top-k, our cells carry none"
        if path in ("what", "written_amsterdam", "runtime_s", "provenance.written", "provenance.git_head"):
            return "metadata"
        if path.startswith("comparisons"):
            return "covered through the named comparisons"
        return None

    all_leaves = [f"R:{c['name']}" for c in reg["comparisons"]]
    for tag, obj in (("regression_check.json", reg), ("dev_seed42.json", dev), ("carry.json", car)):
        for p in flatten(obj):
            dp = dotted(p)
            if tag == "regression_check.json" and dp.startswith("comparisons"):
                continue
            all_leaves.append(f"{tag}:{dp}")
    all_leaves += [f"Z:{k}" for k in Z.files]

    def covered(ref):
        if ref in bk.used:
            return True
        tag, _, path = ref.partition(":")
        # a list leaf counts as used when its parent list was read whole
        parts = path.split(".")
        for i in range(len(parts) - 1, 0, -1):
            if f"{tag}:{'.'.join(parts[:i])}" in bk.used:
                return True
        return False

    unused = []
    for ref in all_leaves:
        if not covered(ref):
            unused.append({"impl_leaf": ref, "reason": reason(ref)})
    res["impl_leaves_total"] = len(all_leaves)
    res["impl_leaves_compared"] = len(all_leaves) - len(unused)
    res["impl_leaves_without_counterpart"] = unused
    unexplained = [u for u in unused if u["reason"] is None]
    res["impl_leaves_unexplained"] = unexplained

    # ================================================================ our quantities with no counterpart (reasons)
    res["ours_without_counterpart"] = [
        {"quantity": "item1 references other than round 1's load_bundle (episodes file, step-1 eval, rd2 cache, "
                     "n6_posteriors, told_oracle.json picks)",
         "reason": "the implementation compares against load_bundle only; both sides pass"},
        {"quantity": "item1 csd shape and dtype; B, B'(A0), B'(A1) cross-fit picks",
         "reason": "not recorded by the implementation (the scores they produce are compared)"},
        {"quantity": "either-change intervals (AFF, R1 x a_v, V4, V2, V24)",
         "reason": "the implementation records the either change as a point only (points compared)"},
        {"quantity": "R1 x a_v margin against its counterpart (with interval)",
         "reason": "not recorded in the implementation's item 3; equals its bar margin since its comparator is the "
                   "counterpart (bar margin compared)"},
        {"quantity": "per-pair margin and per-pair gain statistic (all scorers); bs_04 per-pair gains of R1 x a_v",
         "reason": "descriptive; the implementation records per-pair bar margins only (compared)"},
        {"quantity": "Delta_k bootstrap point and episodes up/down",
         "reason": "not recorded; the bootstrap point equals the integer point, and the arrays behind them agree"},
        {"quantity": "family internals: cross-fit criteria, rho, gamma, rho_ctrl, cells at the maximum, control sums",
         "reason": "not recorded by the implementation; the chosen cells, sigma* and assembled arrays they decide are "
                   "compared"},
        {"quantity": "D10 boundary flags per clause", "reason": "the implementation records boundaries as lists "
                                                                "(empty on both sides, compared)"},
        {"quantity": "item 4 pick shares; reader C values; item 3 bs_04 cross-check", "reason":
            "not recorded by the implementation; the pick and probability arrays and the rule constants are compared"},
        {"quantity": "sensitivity (§6.1)", "reason": "kill on both sides: no candidate carried, no sensitivity"},
    ]

    rows = bk.rows
    n_dis = [r for r in rows if not r["agree"]]
    a_dis = [r for r in arr_rows if not r["equal"]]
    res["quantities"] = rows
    res["counts"] = {"scalar_comparisons": len(rows), "scalar_agree": len(rows) - len(n_dis),
                     "scalar_disagree": len(n_dis), "array_comparisons": len(arr_rows),
                     "array_equal": len(arr_rows) - len(a_dis), "array_differ": len(a_dis),
                     "derived_scalar_comparisons": sum(1 for r in rows if r.get("derived")),
                     "impl_leaves_total": len(all_leaves), "impl_leaves_unexplained": len(unexplained)}
    res["disagreements"] = n_dis + a_dis
    res["all_agree"] = bool(not n_dis and not a_dis and not unexplained)
    K.save_json(out_path, res)
    print(json.dumps({"all_agree": res["all_agree"], "counts": res["counts"],
                      "disagreements": [r["name"] for r in res["disagreements"]],
                      "unexplained_impl_leaves": [u["impl_leaf"] for u in unexplained]}, indent=1))


if __name__ == "__main__":
    main()
