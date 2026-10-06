"""Round 3 on seed 42: the regression checks (DECISION_RULE.md §5 items 1 to 4, in order) and then the sensitivity
projection (§6.1). The main session runs it for real; implementers run only --dry.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/run_r3_seed42.py
    ... --dry      the same computations, written to results/smoke/ (*_dry.*); prints only PASS/FAIL per comparison

Order, stopping at the first failed item (exit 1; regression_check.json records the comparisons made so far and
nothing further is written):
  1. r3_bundle.build_bundle(42, False) equals round 1's common.load_bundle() (compare_with_round1: episodes, B, B'(A0),
     posteriors, s_h, the 18 features, per_anchor_seed42.npz's alignment) and D7's six redundancy values equal the
     rule's exactly with affect the least redundant in both directions (the values are written);
  2. R1 = round-1 R-c: T, margins and picks equal cand_Rc_Rb_expected_A0.npz's; tau recomputed with
     numpy.percentile(m, [0, 25, 50, 75]) over the 24,576 margins (condition a first) equals rc_tau.json's; the gates
     equal the stored ones; cells 116/119 and 58/123, sigma* 0/0; fused__*, cf__*, bar_v exact; bar margin and gain
     statistic equal the rule's numbers (and the stored JSON's) at full precision;
  3. only then AFF: every value of the rule's §5 item 3 at full precision (r3_common.AFF_BRAINSTORM, itself checked
     against bs_04_readers.json and bs_05_aff.json), cells 39/119 and 149/10, sigma* 0/0, tau_0 open counts as integers;
  4. D13 for AFF (clauses recorded; a failure stops).
Then §6.1: r3_stats.sensitivity for the seven GO checks and the secondary check on AFF's seed-42 per-episode
differences (cosine and RCA from per_anchor_seed42.npz), in percentage points.
Writes results/regression_check.json (every comparison: item, name, expected, got, pass), results/seed42_arrays.npz
(R1 and AFF per-anchor arrays, bar vectors, gates, picks, margins, probabilities, chosen cells) and
results/sensitivity.json (per check: SE, half_width = 1.96 SE, x = 2.80 SE, seed42_half_width; pp). Never overwritten.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r3_bundle as RB  # noqa: E402
import r3_common as R3  # noqa: E402
import r3_fusion as RF  # noqa: E402
import r3_stats as RS  # noqa: E402

C = R3.C
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402

RC_NPZ = "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz"
RC_JSON = "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json"
RC_TAU = "20261117_reader_fix_csd/results/rc_tau.json"
BS04 = "20261120_r1_levers_brainstorm/results/bs_04_readers.json"
BS05 = "20261120_r1_levers_brainstorm/results/bs_05_aff.json"
EXT42 = "20261030_aspect_baselines/results/per_anchor_seed42.npz"
SEED42_INPUTS = [RC_NPZ, RC_JSON, RC_TAU, BS04, BS05, EXT42]
# the rule's text for the named cells (§5 items 2 and 3): cell -> (tau index, lambda_u, lambda_a)
CELL_TEXT = {116: (2, 0, 2), 119: (2, 0, 16), 58: (1, 0, 0.5), 123: (2, 0.5, 1), 39: (0, 4, 16), 149: (2, 4, 4),
             10: (0, 0.5, 0.5)}
SENS_ORDER = RS.GO_CHECKS + ("secondary",)
N_MARGINS = 24_576


class Stop(Exception):
    """An item failed: nothing further is computed or written."""


def _js(x):
    if isinstance(x, np.ndarray):
        return f"array {x.shape} {x.dtype}"
    return C.jsonable(x)


def _norm(x):
    """Comparable plain-Python form (tuples -> lists, numpy scalars -> Python)."""
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


class Recorder:
    """Every comparison of §5 with its expected and obtained value; prints only PASS/FAIL."""

    def __init__(self):
        self.rows = []

    def add(self, item, name, expected, got, ok=None):
        if ok is None:
            ok = _norm(expected) == _norm(got)
        self.rows.append({"item": item, "name": name, "expected": _js(expected), "got": _js(got), "pass": bool(ok)})
        print(f"CHECK item{item} {name} {'PASS' if ok else 'FAIL'}", flush=True)

    def arr(self, item, name, got, expected):
        """Exact array equality (shape and value); the JSON gets shapes, dtypes and the largest absolute difference."""
        g, e = np.asarray(got), np.asarray(expected)
        ok = bool(g.shape == e.shape and np.array_equal(g, e))
        row = {"item": item, "name": name, "expected": _js(e), "got": _js(g), "pass": ok}
        if g.shape == e.shape and g.size:
            row["max_abs_diff"] = float(np.max(np.abs(g.astype(np.float64) - e.astype(np.float64))))
        self.rows.append(row)
        print(f"CHECK item{item} {name} {'PASS' if ok else 'FAIL'}", flush=True)

    def require(self, item):
        bad = [r["name"] for r in self.rows if r["item"] == item and not r["pass"]]
        print(f"ITEM {item} {'PASS' if not bad else 'FAIL'} "
              f"({sum(r['item'] == item for r in self.rows)} comparisons)", flush=True)
        if bad:
            raise Stop(item)


def _cell_text(cell):
    d = RF.describe(int(cell))
    return (d["tau_index"], d["lambda_u"], d["lambda_a"])


def _ci3(r):
    return [r["point"], *r["ci95"]]


# ---------------------------------------------------------------- the four items

def item1(rec, st):
    b = RB.build_bundle(42, False)
    try:
        res = RB.compare_with_round1(b)
    except RB.BundleMismatch as e:
        res = e.result
    for k, v in res["checks"].items():
        rec.add(1, k, True, v is True)
    red = res["redundancy"]
    for h in R3.A0:
        for d in DIRECTIONS:
            rec.add(1, f"D7.{h}.{d}", R3.REDUNDANCY_42[h][d], red[h][d])
    rec.add(1, "D7.affect_least_redundant_both_directions", True, RB.affect_least_redundant(red))
    st.update(b=b, red=red)


def item2(rec, st, taus):
    b = st["b"]
    with np.load(R3.input_path(RC_NPZ)) as z:
        stored = {k: z[k] for k in z.files}
    sj = json.loads(R3.input_path(RC_JSON).read_text())
    rd = RF.reader(b, readers=b.readers)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            rec.arr(2, f"T__{c}__{d}", rd["T"][c][d], stored[f"T__{c}__{d}"])
        rec.arr(2, f"margin__{c}", rd["m"][c], stored[f"margin__{c}"])
        rec.arr(2, f"pick__{c}", np.asarray(rd["pick"][c], np.int64), stored[f"pick__{c}"].astype(np.int64))
    m_all = np.concatenate([np.asarray(rd["m"]["a"], np.float64), np.asarray(rd["m"]["b"], np.float64)])
    rec.add(2, "n_margins", N_MARGINS, int(m_all.size))
    rec.add(2, "tau_recomputed_equals_rc_tau", list(taus), [float(v) for v in np.percentile(m_all, [0, 25, 50, 75])])
    rec.add(2, "stored_extra_taus_equal_rc_tau", list(taus), [float(v) for v in stored["extra__taus"]])
    g_r1 = RF.gates_r1(rd["m"], taus)
    for c in CONDITIONS:
        rec.arr(2, f"gates__{c}", np.stack([g_r1[t][c] for t in range(len(taus))]), stored[f"extra__gate_{c}"])
    fam = RF.run_family(b, rd["T"], g_r1)
    rec.add(2, "fused_cells", list(R3.RC_CELLS["fused"]), [fam["fpick"][0], fam["fpick"][1]])
    rec.add(2, "cf_cells", list(R3.RC_CELLS["cf"]), [fam["cpick"][0], fam["cpick"][1]])
    rec.add(2, "sigma_star", list(R3.RC_CELLS["sigma"]), [fam["sigma"][0], fam["sigma"][1]])
    for cell in (*R3.RC_CELLS["fused"], *R3.RC_CELLS["cf"]):
        rec.add(2, f"cell_{cell}_is_the_rule_text", list(CELL_TEXT[cell]), list(_cell_text(cell)))
    for m in METRICS:
        rec.arr(2, f"fused__{m}", fam["fused"][m], stored[f"fused__{m}"])
        rec.arr(2, f"cf__{m}", fam["cf"][m], stored[f"cf__{m}"])
    rec.add(2, "cf_gain_exactly_0", True, bool((np.asarray(fam["cf"]["gain"]) == 0).all()))
    bar_v, bar = C.bar_info(fam["fused"], fam["cf"], b.pBp, b.pB, np.asarray(b.cl), np.asarray(b.pair_index))
    rec.arr(2, "bar_v", bar_v, stored["bar_v"])
    gs = C.diff3(fam["fused"], fam["cf"], np.asarray(b.cl))["gain"]
    N = R3.RC_NUMBERS
    rec.add(2, "comparator", N["comparator"], bar["comparator"])
    rec.add(2, "fused_r1", N["fused_r1"], 100 * float(np.mean(fam["fused"]["r1"])))
    rec.add(2, "cf_r1", N["cf_r1"], 100 * float(np.mean(fam["cf"]["r1"])))
    rec.add(2, "bar_margin", list(N["bar"]), _ci3(bar["r1"]))
    rec.add(2, "gain_statistic", list(N["gain_statistic"]), _ci3(gs))
    rec.add(2, "bar_margin_equals_stored_json", _ci3(sj["bar"]["r1"]), _ci3(bar["r1"]))
    rec.add(2, "gain_statistic_equals_stored_json", _ci3(sj["gain_statistic"]), _ci3(gs))
    rec.add(2, "comparator_equals_stored_json", sj["bar"]["comparator"], bar["comparator"])
    st.update(rd=rd, g_r1=g_r1, fam_r1=fam, bar_v_r1=bar_v)


def item3(rec, st, taus):
    b, rd = st["b"], st["rd"]
    cl = np.asarray(b.cl)
    A = R3.AFF_BRAINSTORM
    bs4 = json.loads(R3.input_path(BS04).read_text())["results"]["AFF"]
    bs5 = json.loads(R3.input_path(BS05).read_text())["A0"]["AFF_minus_R1"]
    # the rule's constants are the brainstorm's recorded numbers (provenance of r3_common.AFF_BRAINSTORM)
    rec.add(3, "rule_constants_equal_bs_04_readers.json",
            [A["fused_r1"], A["cf_r1"], A["comparator"], list(A["bar"]), list(A["margin"]), list(A["gain_statistic"]),
             A["either"], [A["per_pair_bar"][p] for p in C.POOLED_ORDER]],
            [bs4["fused_r1"], bs4["cf_r1"], bs4["comparator"], _ci3(bs4["bar"]), _ci3(bs4["margin"]),
             _ci3(bs4["gain"]), bs4["either"]["point"], [bs4["per_pair_bar"][p] for p in C.POOLED_ORDER]])
    rec.add(3, "rule_constants_equal_bs_05_aff.json", [list(A["aff_minus_r1_fused"]), list(A["aff_minus_r1_bar"])],
            [_ci3(bs5["fused_r1"]), _ci3(bs5["bar_margin"])])
    g_aff = RF.gates_aff(rd["m"], rd["pick"], taus)
    fa = RF.run_family(b, rd["T"], g_aff)
    rec.add(3, "fused_cells", list(R3.AFF_CELLS["fused"]), [fa["fpick"][0], fa["fpick"][1]])
    rec.add(3, "cf_cells", list(R3.AFF_CELLS["cf"]), [fa["cpick"][0], fa["cpick"][1]])
    rec.add(3, "sigma_star", list(R3.AFF_CELLS["sigma"]), [fa["sigma"][0], fa["sigma"][1]])
    for cell in (*R3.AFF_CELLS["fused"], *R3.AFF_CELLS["cf"]):
        rec.add(3, f"cell_{cell}_is_the_rule_text", list(CELL_TEXT[cell]), list(_cell_text(cell)))
    rec.add(3, "cf_gain_exactly_0", True, bool((np.asarray(fa["cf"]["gain"]) == 0).all()))
    bar_v, bar = C.bar_info(fa["fused"], fa["cf"], b.pBp, b.pB, cl, np.asarray(b.pair_index))
    d3 = C.diff3(fa["fused"], fa["cf"], cl)
    rec.add(3, "fused_r1", A["fused_r1"], 100 * float(np.mean(fa["fused"]["r1"])))
    rec.add(3, "cf_r1", A["cf_r1"], 100 * float(np.mean(fa["cf"]["r1"])))
    rec.add(3, "comparator", A["comparator"], bar["comparator"])
    rec.add(3, "bar_margin", list(A["bar"]), _ci3(bar["r1"]))
    rec.add(3, "margin_vs_counterpart", list(A["margin"]), _ci3(d3["r1"]))
    rec.add(3, "gain_statistic", list(A["gain_statistic"]), _ci3(d3["gain"]))
    rec.add(3, "either_vs_counterpart", A["either"], d3["either"]["point"])
    for p in C.POOLED_ORDER:
        rec.add(3, f"per_pair_bar_margin.{p}", A["per_pair_bar"][p], bar["per_pair_r1"][p]["point"])
    r1n = st["fam_r1"]["fused"]
    rec.add(3, "aff_minus_r1_fused_r1", list(A["aff_minus_r1_fused"]),
            _ci3(C.point_ci(np.asarray(fa["fused"]["r1"], np.float64) - np.asarray(r1n["r1"], np.float64), cl)))
    rec.add(3, "aff_minus_r1_bar_margin", list(A["aff_minus_r1_bar"]), _ci3(C.point_ci(bar_v - st["bar_v_r1"], cl)))
    for c in CONDITIONS:
        rec.add(3, f"tau0_open_count.{c}", A["open_tau0_counts"][c], RF.open_count(g_aff[0], c))
    st.update(g_aff=g_aff, fam_aff=fa, bar_v_aff=bar_v, bar_aff=bar, d3_aff=d3)


def item4(rec, st):
    cb = C.clears_bar(st["bar_aff"]["r1"], st["d3_aff"]["gain"])
    for k, v in cb.items():
        rec.add(4, f"D13.{k}", True, v)
    st["d13"] = cb


# ---------------------------------------------------------------- §6.1

def sensitivity(st):
    b = st["b"]
    ext = RB.load_external(b)                    # per_anchor_seed42.npz (SHA-256 asserted), alignment asserted
    f = (lambda x: np.asarray(x, np.float64))
    pn, pc, r1n = st["fam_aff"]["fused"], st["fam_aff"]["cf"], st["fam_r1"]["fused"]
    diffs = {"r1_vs_cosine": f(pn["r1"]) - f(ext["cosine"]["r1"]), "r1_vs_rca": f(pn["r1"]) - f(ext["rca"]["r1"]),
             "r1_vs_B": f(pn["r1"]) - f(b.pB["r1"]), "r1_vs_Bprime": f(pn["r1"]) - f(b.pBp["r1"]),
             "r1_vs_counterpart": f(pn["r1"]) - f(pc["r1"]), "gain_statistic": f(pn["gain"]) - f(pc["gain"]),
             "gain_vs_rca": f(pn["gain"]) - f(ext["rca"]["gain"]), "secondary": f(pn["r1"]) - f(r1n["r1"])}
    if tuple(diffs) != SENS_ORDER:
        raise AssertionError("sensitivity checks out of the rule's order")
    out = {k: RS.sensitivity(diffs[k], np.asarray(b.cl)) for k in SENS_ORDER}
    for k, v in out.items():
        if not all(math.isfinite(float(v[q])) for q in ("SE", "half_width", "x", "seed42_half_width")):
            raise AssertionError(f"sensitivity {k}: non-finite value")
    return out


def _arrays(st):
    arr = {"cl": np.asarray(st["b"].cl), "pair_index": np.asarray(st["b"].pair_index),
           "parity": np.asarray(st["b"].parity), "taus": np.asarray(R3.TAUS, np.float64)}
    for who, fam, bar_v, g in (("r1", st["fam_r1"], st["bar_v_r1"], st["g_r1"]),
                               ("aff", st["fam_aff"], st["bar_v_aff"], st["g_aff"])):
        for part in ("fused", "cf"):
            for m in METRICS:
                arr[f"{who}_{part}__{m}"] = np.asarray(fam[part][m], np.float64)
        arr[f"{who}_bar_v"] = np.asarray(bar_v, np.float64)
        for c in CONDITIONS:
            arr[f"{who}_gate__{c}"] = np.stack([g[t][c] for t in range(len(g))])
        arr[f"{who}_fused_cells"] = np.array([fam["fpick"][0], fam["fpick"][1]], np.int64)
        arr[f"{who}_cf_cells"] = np.array([fam["cpick"][0], fam["cpick"][1]], np.int64)
        arr[f"{who}_sigma"] = np.array([fam["sigma"][0], fam["sigma"][1]], np.float64)
    rd = st["rd"]
    for c in CONDITIONS:
        arr[f"pick__{c}"] = np.asarray(rd["pick"][c], np.int64)
        arr[f"margin__{c}"] = np.asarray(rd["m"][c], np.float64)
        arr[f"P__{c}"] = np.asarray(rd["P"][c], np.float64)
    return arr


def run(dry) -> int:
    R3.assert_rule()
    out = R3.res_dir(dry)
    sfx = "_dry" if dry else ""
    P = {"reg": out / f"regression_check{sfx}.json", "arr": out / f"seed42_arrays{sfx}.npz",
         "sens": out / f"sensitivity{sfx}.json"}
    R3.refuse_existing(P.values(), dry)
    inputs = R3.assert_inputs(SEED42_INPUTS)
    taus = R3.assert_taus()
    rec, st, t0 = Recorder(), {}, time.time()
    base = {"what": "rule §5 items 1 to 4 on seed 42 (regression checks)" + ("; dry run, not a result" if dry else ""),
            "dry_run": bool(dry), "inputs_sha256": inputs}
    try:
        item1(rec, st)
        rec.require(1)
        item2(rec, st, taus)
        rec.require(2)                            # no AFF number before items 1 and 2 pass
        item3(rec, st, taus)
        rec.require(3)
        item4(rec, st)
        rec.require(4)
    except Stop as e:
        R3.write_json_once(P["reg"], {**base, "passed": False, "stopped_at_item": int(e.args[0]),
                                      "failed": [r["name"] for r in rec.rows if not r["pass"]],
                                      "n_comparisons": len(rec.rows), "comparisons": rec.rows,
                                      "runtime_s": round(time.time() - t0, 1)}, dry)
        print(f"SEED42{'_DRY' if dry else ''} FAIL at item {e.args[0]}: stop and report to the user (rule §5, §9); "
              f"{P['reg'].name} written, nothing further", flush=True)
        return 1
    reg = R3.write_json_once(P["reg"], {
        **base, "passed": True, "stopped_at_item": None, "failed": [], "n_comparisons": len(rec.rows),
        "comparisons": rec.rows, "redundancy_D7": st["red"], "D13_AFF": st["d13"],
        "cells": {"R1": {"fused": [st["fam_r1"]["fpick"][h] for h in (0, 1)],
                         "counterpart": [st["fam_r1"]["cpick"][h] for h in (0, 1)]},
                  "AFF": {"fused": [st["fam_aff"]["fpick"][h] for h in (0, 1)],
                          "counterpart": [st["fam_aff"]["cpick"][h] for h in (0, 1)]}},
        "runtime_s": round(time.time() - t0, 1)}, dry)
    print(f"REGRESSION PASS: {reg['n_comparisons']} comparisons, items 1 to 4; {P['reg'].name} written", flush=True)
    arr = _arrays(st)
    np.savez_compressed(P["arr"], **arr)
    print(f"{P['arr'].name} written ({len(arr)} arrays)", flush=True)
    sens = sensitivity(st)
    R3.write_json_once(P["sens"], {
        "what": "rule §6.1: projected pooled SE over three seeds from AFF's seed-42 per-episode differences (one-way "
                "decomposition by anchor painting); half_width = 1.96 SE, detectable margin x = 2.80 SE, and the "
                "seed-42 bootstrap half-width beside them; percentage points; used only to read a failed check (§6.7)"
                + ("; dry run, not a result" if dry else ""),
        "units": "percentage points", "z": RS.Z_HALF, "k_detect": RS.K_DETECT, "checks": sens,
        "regression_check_sha256": R3.sha_file(P["reg"]), "seed42_arrays_sha256": R3.sha_file(P["arr"])}, dry)
    if dry:
        print(f"{P['sens'].name} written ({len(sens)} checks; dry run: no value printed)", flush=True)
    else:
        print("sensitivity (rule §6.1; pp): SE, 1.96 SE, x = 2.80 SE, seed-42 bootstrap half-width", flush=True)
        for k, v in sens.items():
            print(f"  {k:20s} {v['SE']:.4f} {v['half_width']:.4f} {v['x']:.4f} {v['seed42_half_width']:.4f}", flush=True)
    print(f"SEED42{'_DRY' if dry else ''} PASS [{time.time() - t0:.0f}s]", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="write to results/smoke/ only; print only PASS/FAIL")
    args = ap.parse_args(argv)
    sys.exit(run(args.dry))


if __name__ == "__main__":
    main()
