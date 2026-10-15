"""Figures and figure data for docs/reports/auto/v2/2026-11-23_idea3_goemotions.md (reader fix, round 5, idea 3).

Reads stored outputs only (nothing under src/ is written) and asserts every decision quantity it reports against its
source before any figure is drawn:
  src/test/20261123_idea3_goemotions/results/dev_seed42.json         rule section 5 items 5 and 6 (development record)
  src/test/20261123_idea3_goemotions/results/carry.json              rule section 5 items 7 and 8 (carry, kill)
  src/test/20261123_idea3_goemotions/results/regression_check.json   rule section 5 items 1 to 4 (361 comparisons)
  src/test/20261123_idea3_goemotions/results/diagnostics_seed42.json rule section 5 measured diagnostics (a) to (d)
  src/test/20261123_idea3_goemotions/results/placement.json          item 3 and the GE head's held-out accuracy
  src/test/20261123_idea3_goemotions/results/seed42_arrays.npz       seed-42 per-anchor arrays, gates, picks, cells
  why_rebuild.json (this folder)                                      the descriptive rebuild of section 6

Step 1 re-derives, from the per-anchor arrays alone, every decision quantity of the development step except the chosen
cells, sigma* and tau' (each candidate's and AFF's fused and counterpart R@1, the four comparator means, the bar
comparator, bar margin, margin, gain statistic and either change with their intervals, the D10 clauses, the per-pair
bar margins, the integer Delta_k with its point and interval, B'_G minus B'(A0), each scorer minus B'(A1), the tau_0
open counts, the empty carry set) and asserts that each equals dev_seed42.json / carry.json exactly (difference 0.0).
The chosen cells, sigma* and tau' cannot be derived from per-anchor arrays: it reads them from seed42_arrays.npz and
checks them against dev_seed42.json. It also re-derives diagnostic (a) (the detection AUCs, from the stored P) and the
either cost per unit of gain of diagnostic (d), and asserts them against diagnostics_seed42.json.
Step 2 computes the descriptive breakdowns of the report's section 6 from the stored arrays (pairs, picks, open
counts, B'_G against B'(A0) per pair, candidate minus AFF) and collects why_rebuild.json's numbers. They are computed
after the kill, decide nothing, and score no new variant beyond the fixed-cell decomposition that why_rebuild.py
labels as such. Intervals: src.eval.aspect_metrics.cluster_bootstrap (5,000 resamples, seed 42, chunk 250, clusters =
anchor paintings), the helper behind round 1's common.point_ci that the implementation used.

Run from the repo root (CPU only), after why_rebuild.py:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-23_idea3_goemotions/build_figures.py
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

FOLDER = ROOT / "src/test/20261123_idea3_goemotions"
RES = FOLDER / "results"
RULE_SHA = "19e59fc7220c05b630f4773a94578aa3858d29853dbf7d438455e1ee973d735e"
OUT = Path(__file__).resolve().parent
DPI = 150
N_EP = 12288
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
PAIR_LABEL = {"emotion__style": "emotion × style", "emotion__genre": "emotion × genre",
              "style__genre": "style × genre"}
SIDE = {("emotion__style", "a"): "emotion", ("emotion__style", "b"): "style",
        ("emotion__genre", "a"): "emotion", ("emotion__genre", "b"): "genre",
        ("style__genre", "a"): "style", ("style__genre", "b"): "genre"}
CANDS = ("G-T", "G-TF")
KEY = {"G-T": "gt", "G-TF": "gtf", "AFF": "aff"}
BAR = 0.5                          # D10 clause 1
NESTED_U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NESTED_A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

# dataviz reference palette, first three categorical slots (validated all-pairs, light mode); grey for AFF and the
# CLIP placement it uses, aqua for the GE placement
C = {"G-T": "#2a78d6", "G-TF": "#eb6834", "AFF": "#8a8984", "CLIP": "#8a8984", "GE": "#1baf7a", "img": "#4a3aa7"}
MK = {"G-T": "o", "G-TF": "s", "AFF": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pp(values, clusters):
    """Mean in percentage points with the painting-clustered 95% interval (round 1's common.point_ci)."""
    r = cluster_bootstrap(np.asarray(values, dtype=np.float64), np.asarray(clusters))
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]]}


def same(x, y):
    assert x == y, (x, y)


def same_ci(x, y):
    same(x["point"], y["point"])
    same(list(x["ci95"]), list(y["ci95"]))


def fmt(x, nd=3):
    return f"{x['point']:+.{nd}f} [{x['ci95'][0]:+.{nd}f}, {x['ci95'][1]:+.{nd}f}]"


def as_int4(x):
    """Round 2's r2_fusion.as_int4: 4 x per-episode R@1 as integers, asserting every value is a multiple of 0.25."""
    q = 4 * np.asarray(x, dtype=np.float64)
    r = np.rint(q)
    assert np.all(q == r), "per-anchor R@1 is not a multiple of 0.25"
    return r.astype(np.int64)


def cell_lambdas(cell):
    """cell = (tau_index*7 + u)*8 + a (k_top 13) -> (tau index, lambda_u, lambda_a)."""
    return cell // 56, NESTED_U[(cell // 8) % 7], NESTED_A[cell % 8]


# ---------------------------------------------------------------- step 1: the decision quantities, re-derived

def comparators(name, pa):
    """D8's condition-free comparators in tie order: B'_G, B'(A0), counterpart, B for a candidate; round 3's
    B'(A0) ("B_prime"), counterpart, B for AFF (its stored reference record)."""
    cf = pa[f"{KEY[name]}_cf"]
    if name == "AFF":
        return [("B_prime", pa["Bp0"]), ("counterpart", cf), ("B", pa["B"])]
    return [("Bprime_G", pa["BpG"]), ("Bprime_A0", pa["Bp0"]), ("counterpart", cf), ("B", pa["B"])]


def bar_comparator(comps):
    means = [float(np.mean(np.asarray(p["r1"], dtype=np.float64))) for _, p in comps]
    best = 0
    for i in range(1, len(comps)):
        if means[i] > means[best]:          # strictly greater: ties go to the earliest
            best = i
    return comps[best][0], comps[best][1], {lab: 100 * m for (lab, _), m in zip(comps, means)}


def record(name, pa, z, cl, pidx):
    k = KEY[name]
    fu, cf = pa[f"{k}_fused"], pa[f"{k}_cf"]
    assert np.all(cf["gain"] == 0.0), f"{name}: counterpart gain is not exactly 0"
    label, comp, means = bar_comparator(comparators(name, pa))
    v = fu["r1"] - comp["r1"]
    bar = pp(v, cl)
    gain = pp(fu["gain"] - cf["gain"], cl)
    either = pp((fu["r1"] + fu["other"]) - (cf["r1"] + cf["other"]), cl)
    c1, c2, c3 = bar["point"] >= BAR, bar["ci95"][0] > 0, gain["ci95"][0] > 0
    rec = {"fused_r1": 100 * float(np.mean(fu["r1"])), "cf_r1": 100 * float(np.mean(cf["r1"])),
           "fpick": [int(c) for c in z[f"{k}_fused_cells"]], "cpick": [int(c) for c in z[f"{k}_cf_cells"]],
           "sigma": [float(s) for s in z[f"{k}_sigma"]], "bar_comparator": label, "comparator_means": means,
           "bar_margin": bar, "margin_vs_counterpart": pp(fu["r1"] - cf["r1"], cl), "gain_statistic": gain,
           "either_change": either["point"], "either_change_ci": either["ci95"],
           "either_cost_per_unit_gain": -either["point"] / gain["point"],
           "d10": {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "clears": bool(c1 and c2 and c3)},
           "per_pair_bar_margin": {p: pp(v[pidx == i], cl[pidx == i]) for i, p in enumerate(PAIRS)},
           "minus_Bprime_A1": pp(fu["r1"] - pa["Bp1"]["r1"], cl)}
    if name != "AFF":
        aff = pa["aff_fused"]["r1"]
        di = int((as_int4(fu["r1"]) - as_int4(aff)).sum())
        rec["delta_int"] = di
        rec["delta"] = {"point": 100.0 * di / (4 * len(aff)), "ci95": pp(fu["r1"] - aff, cl)["ci95"]}
        rec["Bprime_G_minus_Bprime_A0"] = pp(pa["BpG"]["r1"] - pa["Bp0"]["r1"], cl)
    return rec


def check_against_files(rec, dev, carry, z):
    """Every re-derived decision quantity equals the stored record exactly."""
    for name in CANDS + ("AFF",):
        s = dev["aff"] if name == "AFF" else dev["candidates"][name]
        r = rec[name]
        same(r["fused_r1"], s["fused_r1"]), same(r["cf_r1"], s["cf_r1"])
        same_ci(r["bar_margin"], s["bar_margin"])
        same_ci(r["margin_vs_counterpart"], s["margin_vs_counterpart"])
        same_ci(r["gain_statistic"], s["gain_statistic"])
        same(r["either_change"], s["either_change"])
        same(r["bar_comparator"], s["bar_comparator"])
        for p in PAIRS:
            same_ci(r["per_pair_bar_margin"][p], s["per_pair_bar_margin"][p])
        if name == "AFF":
            same(r["fpick"], list(s["cells"]["fused"])), same(r["cpick"], list(s["cells"]["cf"]))
            same(r["sigma"], list(s["sigma"]))
            same_ci(r["minus_Bprime_A1"], s["minus_Bprime_A1"])
            same_ci(r["minus_Bprime_A1"], dev["beside"]["AFF_minus_Bprime_A1"])
            continue
        same(r["fpick"], [s["cells"]["fpick"]["0"], s["cells"]["fpick"]["1"]])
        same(r["cpick"], [s["cells"]["cpick"]["0"], s["cells"]["cpick"]["1"]])
        same(r["sigma"], [s["sigma"]["0"], s["sigma"]["1"]])
        for k, v in r["comparator_means"].items():
            same(v, s["comparator_means"][k])
        same(r["d10"], s["d10"]), same(r["d10"], carry["candidates"][name]["d10"])
        same(r["delta_int"], s["delta_int"]), same(r["delta_int"], carry["candidates"][name]["delta_int"])
        same_ci(r["delta"], s["delta"])
        same_ci(r["Bprime_G_minus_Bprime_A0"], s["Bprime_G_minus_Bprime_A0"])
        same_ci(r["Bprime_G_minus_Bprime_A0"], dev["comparators"]["Bprime_G_minus_Bprime_A0"])
        same_ci(r["minus_Bprime_A1"], s["beside_Bprime_A1"]["candidate_minus"])
        for kind, picks in (("fused", r["fpick"]), ("cf", r["cpick"])):
            for h, cell in enumerate(picks):
                t, lu, la = cell_lambdas(cell)
                ct = s["cell_text"][kind][str(h)]
                same(ct["cell"], cell), same(ct["tau_index"], t), same(ct["lambda_u"], lu), same(ct["lambda_a"], la)
        assert s["boundaries"] == []
    # the carry (rule section 5 items 7 and 8), from the re-derived integers and clauses
    E = [n for n in CANDS if rec[n]["d10"]["clears"] and rec[n]["delta_int"] > 0]
    same(E, carry["E"])
    assert E == [] and carry["kill"] is True and carry["carried"] is None and carry["M"] is None
    assert carry["boundaries"] == [] and carry["tied"] == []
    assert all(rec[n]["delta_int"] != 0 for n in CANDS)                       # no boundary of rule section 8
    for n in CANDS:                                                           # no D10 value within 1e-12
        for v, thr in ((rec[n]["bar_margin"]["point"], BAR), (rec[n]["bar_margin"]["ci95"][0], 0.0),
                       (rec[n]["gain_statistic"]["ci95"][0], 0.0)):
            assert abs(v - thr) > 1e-12
    # tau', the tau_0 open counts and the comparator means
    same([float(x) for x in z["tau_prime"]], dev["tau_prime"])
    same([float(x) for x in z["taus"]], dev["taus_AFF"])
    for name in CANDS + ("AFF",):
        for c in "ab":
            same(int(z[f"{KEY[name]}_gate__{c}"][0].sum()), dev["open_tau0_counts"][name][c])
    same(rec["G-T"]["comparator_means"]["Bprime_G"], dev["comparators"]["Bprime_G_mean_r1"])
    same(rec["G-T"]["comparator_means"]["Bprime_A0"], dev["comparators"]["Bprime_A0_mean_r1"])
    same(rec["G-T"]["comparator_means"]["B"], dev["comparators"]["B_mean_r1"])
    # G-T's gates are AFF's at every tau index and condition (D6)
    for c in "ab":
        assert np.array_equal(z[f"gt_gate__{c}"], z[f"aff_gate__{c}"])


def check_diagnostics(rec, z, pidx, diag):
    """Diagnostic (a) from the stored P, and the either cost per unit of gain of (d), equal the stored values."""
    y = np.concatenate([pidx < 2, np.zeros_like(pidx, dtype=bool)])          # condition a of the two emotion pairs
    auc = {n: float(roc_auc_score(y, np.concatenate([z[f"{k}_P__a"][:, 0], z[f"{k}_P__b"][:, 0]])))
           for n, k in (("G-TF", "gtf"), ("AFF", "aff"))}
    same(auc["G-TF"], diag["a_detection_auc"]["G-TF"])
    same(auc["AFF"], diag["a_detection_auc"]["AFF_recomputed"])
    same(auc["AFF"], 0.7870951145887375)
    for n in CANDS:
        same(rec[n]["either_cost_per_unit_gain"], diag["d_either_cost_per_unit_gain"][n])
    same(rec["AFF"]["either_cost_per_unit_gain"], diag["d_either_cost_per_unit_gain"]["AFF_recomputed"])
    return auc


# ---------------------------------------------------------------- step 2: descriptive breakdowns (after the kill)

def descriptive(pa, z, cl, pidx, par, rec, why):
    n = len(par)
    out = {}
    out["int_sums"] = {lab: int(as_int4(pa[k]["r1"]).sum()) for lab, k in
                       (("AFF", "aff_fused"), ("G-T", "gt_fused"), ("G-TF", "gtf_fused"), ("Bprime_A1", "Bp1"),
                        ("Bprime_A0", "Bp0"), ("Bprime_G", "BpG"), ("B", "B"))}
    for name in CANDS:
        same(out["int_sums"][name] - out["int_sums"]["AFF"], rec[name]["delta_int"])
    # candidate minus AFF, fused, paired per anchor: pooled and per pair; gain, other, either pooled
    aff = pa["aff_fused"]
    pm = {}
    for name in CANDS:
        fu = pa[f"{KEY[name]}_fused"]
        d = fu["r1"] - aff["r1"]
        d4 = as_int4(fu["r1"]) - as_int4(aff["r1"])
        e = {"pooled": pp(d, cl), "better": int((d4 > 0).sum()), "worse": int((d4 < 0).sum())}
        assert abs(e["pooled"]["point"] - rec[name]["delta"]["point"]) < 1e-12
        same(e["pooled"]["ci95"], rec[name]["delta"]["ci95"])
        for i, p in enumerate(PAIRS):
            e[p] = pp(d[pidx == i], cl[pidx == i]) | {"net_rankings": int(d4[pidx == i].sum())}
        e["gain"] = pp(fu["gain"] - aff["gain"], cl)
        e["other"] = pp(fu["other"] - aff["other"], cl)
        e["either"] = pp((fu["r1"] + fu["other"]) - (aff["r1"] + aff["other"]), cl)
        assert abs((e["either"]["point"] + e["gain"]["point"]) / 2 - e["pooled"]["point"]) < 1e-12
        pm[name] = e
    out["minus_AFF"] = pm
    # B'_G against B'(A0), per pair
    dBp = pa["BpG"]["r1"] - pa["Bp0"]["r1"]
    out["BprimeG_minus_BprimeA0"] = {"pooled": pp(dBp, cl), "net_rankings": int((as_int4(pa["BpG"]["r1"])
                                                                                   - as_int4(pa["Bp0"]["r1"])).sum())}
    for i, p in enumerate(PAIRS):
        out["BprimeG_minus_BprimeA0"][p] = pp(dBp[pidx == i], cl[pidx == i])
    out["BprimeG_minus_B"] = pp(pa["BpG"]["r1"] - pa["B"]["r1"], cl)
    out["BprimeA0_minus_B"] = pp(pa["Bp0"]["r1"] - pa["B"]["r1"], cl)
    # G-TF's reader on the GE features: picks and gates against AFF's, per pair and condition
    ar = np.arange(n)
    eff = {nm: np.where(par == 1, int(z[f"{KEY[nm]}_fused_cells"][0]) // 56, int(z[f"{KEY[nm]}_fused_cells"][1]) // 56)
           for nm in ("AFF", "G-TF")}
    rd = {}
    for i, p in enumerate(PAIRS):
        for c in "ab":
            m = pidx == i
            pa_, pg_ = z[f"aff_pick__{c}"][m], z[f"gtf_pick__{c}"][m]
            ga = z[f"aff_gate__{c}"][eff["AFF"], ar][m].astype(bool)
            gg = z[f"gtf_gate__{c}"][eff["G-TF"], ar][m].astype(bool)
            rd[f"{p}|{c}"] = {"affect_pick_share_AFF": 100 * float((pa_ == 0).mean()),
                              "affect_pick_share_GTF": 100 * float((pg_ == 0).mean()),
                              "pick_agreement": 100 * float((pa_ == pg_).mean()),
                              "open_as_scored_AFF": int(ga.sum()), "open_as_scored_GTF": int(gg.sum()),
                              "open_both": int((ga & gg).sum())}
    out["GTF_reader_vs_AFF"] = rd
    out["GTF_pick_agreement_overall"] = 100 * float(np.mean(np.concatenate(
        [z["aff_pick__a"] == z["gtf_pick__a"], z["aff_pick__b"] == z["gtf_pick__b"]])))
    # either cost per unit gain and the fixed-cell decomposition, from why_rebuild.json
    out["fixed_cells"] = why["fixed_cells"]
    out["own_cells_gain_either_vs_AFF"] = {nm: {"gain": pm[nm]["gain"], "either": pm[nm]["either"]} for nm in CANDS}
    out["per_cell_minus_AFF"] = why["per_cell_minus_AFF"]
    out["per_cell_levels"] = why["per_cell_levels"]
    # in-sample family: best fused cell over lambda_u at each (tau index, lambda_a)
    fam = {}
    for nm in ("AFF",) + CANDS:
        r = np.asarray(why["in_sample_family"][nm]["fused_r1"])
        g = np.asarray(why["in_sample_family"][nm]["fused_gain"])
        best = int(r.argmax())
        fam[nm] = {"best_cell": best, "best_cell_lambdas": list(cell_lambdas(best)), "best_fused_r1": float(r[best]),
                   "best_over_lambda_u": {str(t): [float(max(r[(t * 7 + u) * 8 + a] for u in range(7)))
                                                   for a in range(8)] for t in range(4)},
                   "gain_at_best_over_lambda_u": {str(t): [float(g[(t * 7 + max(range(7), key=lambda u: r[(t * 7 + u) * 8 + a])) * 8 + a])
                                                           for a in range(8)] for t in range(4)},
                   "at_own_cells": [float(r[c]) for c in z[f"{KEY[nm]}_fused_cells"]],
                   "best_cf_r1": float(max(why["in_sample_family"][nm]["cf_r1"]))}
    out["in_sample_family"] = fam
    for k in ("affect_score_alone", "redundancy", "pair_lifts", "same_painting", "sharpness", "Bprime_picks"):
        out[k] = why[k]
    return out


def data():
    dev = load(RES / "dev_seed42.json")
    carry = load(RES / "carry.json")
    reg = load(RES / "regression_check.json")
    diag = load(RES / "diagnostics_seed42.json")
    plc = load(RES / "placement.json")
    why = load(OUT / "why_rebuild.json")
    assert sha(FOLDER / "DECISION_RULE.md") == RULE_SHA
    for d in (dev, carry, reg, diag, plc):
        assert d["rule_sha256"] == RULE_SHA and d["provenance"]["smoke"] is False
    assert reg["dry_run"] is False and reg["all_passed"] is True and reg["n_comparisons"] == 361
    assert all(c["pass"] for c in reg["comparisons"]) and len(reg["comparisons"]) == 361
    assert [reg["items"][k]["n_comparisons"] for k in "1234"] == [184, 20, 21, 136]
    assert dev["seed42_arrays_sha256"] == carry["seed42_arrays_sha256"] == sha(RES / "seed42_arrays.npz")
    assert dev["regression_check_sha256"] == carry["regression_check_sha256"] == sha(RES / "regression_check.json")
    assert carry["dev_seed42_sha256"] == sha(RES / "dev_seed42.json")
    assert diag["carry_sha256"] == sha(RES / "carry.json")
    assert why["rebuild_equals_stored_run"] is True
    assert why["provenance"]["inputs_sha256"]["seed42_arrays.npz"] == sha(RES / "seed42_arrays.npz")
    assert plc["ge_head"]["heldout_accuracy"] == 86.17 and plc["ge_head"]["fallback_used"] is False
    assert plc["item3"]["heldout_accuracy"] == {"txt": 35.72, "img": 9.81} and plc["item3"]["passed"] is True
    same(dev["order"], list(CANDS))
    assert all(v is True for v in dev["positive_check_D5"].values())

    z = np.load(RES / "seed42_arrays.npz", allow_pickle=False)
    cl, pidx, par = z["cl"], z["pair_index"], z["parity"]
    assert len(cl) == N_EP and len(np.unique(cl)) == 4602 and np.array_equal(np.bincount(pidx), [4096] * 3)
    names = ("aff_fused", "aff_cf", "gt_fused", "gt_cf", "gtf_fused", "gtf_cf", "B", "Bp0", "Bp1", "BpG",
             "cosine", "rca")
    pa = {k: {m: np.asarray(z[f"{k}__{m}"], dtype=np.float64) for m in ("r1", "gain", "other")} for k in names}

    rec = {name: record(name, pa, z, cl, pidx) for name in CANDS + ("AFF",)}
    check_against_files(rec, dev, carry, z)
    auc = check_diagnostics(rec, z, pidx, diag)
    desc = descriptive(pa, z, cl, pidx, par, rec, why)
    diagnostics = {"placement": {"image_head": plc["item3"]["heldout_accuracy"]["img"],
                                 "caption_CLIP": plc["item3"]["heldout_accuracy"]["txt"],
                                 "caption_GE": plc["ge_head"]["heldout_accuracy"],
                                 "majority": plc["ge_head"]["check_majority_share"],
                                 "uniform": plc["ge_head"]["uniform"], "GE_n_iter": plc["ge_head"]["n_iter"]},
                   "detection_auc": {"AFF_R1": auc["AFF"], "G-TF": auc["G-TF"]},
                   "delta_affect_auc": dict(diag["b_delta_affect_auc"]),
                   "pair_lift": {"CLIP": diag["c_pair_lift"]["CLIP_item4"], "GE": {
                       k: diag["c_pair_lift"]["GE"][k] for k in ("ratio_same_over_diff", "emotionxstyle",
                                                                 "emotionxgenre")},
                       "groups": diag["c_pair_lift"]["group_lift"]},
                   "either_cost_per_unit_gain": {n: rec[n]["either_cost_per_unit_gain"] for n in ("AFF",) + CANDS},
                   "sharper_term": {n: {k: diag["d_sharper_term"][n][k] for k in ("per_pair_condition", "per_direction")}
                                    for n in ("AFF",) + CANDS},
                   "sharper_term_minus_AFF": {n: diag["d_sharper_term"][n]["minus_aff"] for n in CANDS}}

    fig_data = {
        "what": "round 5 figure data. 'rederived': the decision quantities, re-derived from seed42_arrays.npz and "
                "asserted equal to dev_seed42.json and carry.json. 'diagnostics': the rule's measured diagnostics "
                "(descriptive). 'descriptive': breakdowns computed after the kill (seed 42, decides nothing); pp, "
                "95% painting-bootstrap intervals (5,000 resamples, seed 42)",
        "rederived": {"records": rec, "E": [], "kill": True, "tau_prime": dev["tau_prime"],
                      "Bprime_A1_mean_r1": dev["beside"]["Bprime_A1_mean_r1"]},
        "diagnostics": diagnostics,
        "descriptive": desc,
        "inputs_sha256": {f.name: sha(f) for f in (RES / "dev_seed42.json", RES / "carry.json",
                                                   RES / "regression_check.json", RES / "diagnostics_seed42.json",
                                                   RES / "placement.json", RES / "seed42_arrays.npz",
                                                   FOLDER / "DECISION_RULE.md", OUT / "why_rebuild.json")},
    }
    (OUT / "figure_data.json").write_text(json.dumps(fig_data, indent=1, ensure_ascii=False) + "\n")
    return rec, diagnostics, desc


# ---------------------------------------------------------------- figure 1: what changed

def box(ax, x, y, w, h, text, kind, fs=8.6, ls="-"):
    fill = {"same": ("#ecebe8", "#8a8984"), "unchanged": ("#e3e0f4", "#4a3aa7"), "replaced": ("#fde3d7", "#eb6834"),
            "new": ("#d3f1e6", "#12876a")}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fill[0],
                                ec=fill[1], lw=1.6, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK2))


def fig_what_changed():
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(-0.012, 1.012), ax.set_ylim(0.0, 0.95), ax.axis("off")
    # row 1: where the caption posterior comes from
    y1, h1 = 0.74, 0.16
    box(ax, 0.005, y1, 0.17, h1, "every selection caption\n(32,413) through\nGoEmotions once:\n28 probabilities", "new")
    box(ax, 0.205, y1, 0.20, h1, "GE head: logistic regression\n28 probabilities → 41 Leiden\naffect communities "
        "(the CLIP\ncaption head's recipe)", "new")
    box(ax, 0.435, y1, 0.20, h1, "caption side of the affect\ngrouping: CLIP caption head\n(35.7% held out) → GE head\n"
        "(86.2% held out)", "replaced")
    box(ax, 0.665, y1, 0.16, h1, "image side of the affect\ngrouping: CLIP image\nhead (9.8% held out)", "unchanged")
    box(ax, 0.855, y1, 0.14, h1, "image and caption\ngroupings, their\nheads; B", "same")
    arrow(ax, 0.175, y1 + h1 / 2, 0.205, y1 + h1 / 2)
    arrow(ax, 0.405, y1 + h1 / 2, 0.435, y1 + h1 / 2)
    # row 2: what reads the affect agreement
    y2, h2 = 0.43, 0.20
    box(ax, 0.005, y2, 0.30, h2, "affect agreement s_affect = p_img · q_caption\n(every pair and every ranking is "
        "image × caption)\nCLIP placement → GE placement in:\nG-T: the steering term only\nG-TF: the term and the "
        "reader's 6 affect features", "replaced", fs=8.6)
    box(ax, 0.335, y2 + 0.01, 0.20, h2 - 0.02, "AFF's A0 half-readers, frozen;\nG-T: AFF's P, picks, margins,\ngates "
        "at τ_0..τ_3 exactly;\nG-TF: same readers on the GE\nfeatures, τ′ from its margins", "unchanged", fs=8.4)
    box(ax, 0.565, y2 + 0.01, 0.20, h2 - 0.02, "fused score\nz(B) + λ_u·z(B) + λ_a·g·z(T^c),\n224 cells, integer "
        "cross-fits;\nmatched counterpart under\nthe candidate's own gates", "unchanged", fs=8.4)
    box(ax, 0.795, y2 + 0.01, 0.20, h2 - 0.02, "comparators: B, B′(A0),\nthe counterpart, and new\nB′_G (B′ rebuilt with "
        "the\nGE placement); B′(A1)\nbeside, descriptive", "new", fs=8.4)
    arrow(ax, 0.535, y1 - 0.012, 0.20, y2 + h2 + 0.012)
    arrow(ax, 0.745, y1 - 0.012, 0.25, y2 + h2 + 0.012)
    for x0, x1 in ((0.305, 0.335), (0.535, 0.565), (0.765, 0.795)):
        arrow(ax, x0, y2 + h2 / 2, x1, y2 + h2 / 2)
    # row 3: decision
    y3, h3 = 0.07, 0.24
    box(ax, 0.005, y3, 0.30, h3, "seed 42 only (12,288 episodes):\nregression checks, then the\ndevelopment bar "
        "(round 2's D10):\nbar margin ≥ +0.5, its lower bound > 0,\ngain statistic's lower bound > 0", "same")
    box(ax, 0.335, y3, 0.32, h3, "carry (round 4's rule): clears the bar\nAND beats AFF, Δ_k = Σ 4·(candidate − AFF) > 0\n"
        "(integer, paired per anchor);\ntie band 24 rankings; order G-T, G-TF;\nempty carry set = kill", "same")
    box(ax, 0.685, y3, 0.31, h3, "fresh-seed test on seeds 52 to 54,\nnine GO checks including the\npaired check "
        "against AFF\n(pre-registered; not run: the carry\nset was empty, so the rule killed)", "new", ls="--")
    arrow(ax, 0.305, y3 + h3 / 2, 0.335, y3 + h3 / 2)
    arrow(ax, 0.655, y3 + h3 / 2, 0.685, y3 + h3 / 2)
    handles = [Patch(fc="#ecebe8", ec="#8a8984", label="grey: same as rounds 2 to 4"),
               Patch(fc="#e3e0f4", ec="#4a3aa7", label="purple: unchanged from AFF"),
               Patch(fc="#fde3d7", ec="#eb6834", label="orange: replaced (the caption-side affect placement)"),
               Patch(fc="#d3f1e6", ec="#12876a", label="teal: new in round 5 (dashed: planned, not run)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("What round 5 changed: captions are placed in the affect communities by GoEmotions, not by CLIP",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.925, "The communities, the image head, the other groupings, B, the readers, the 224 cells and "
             "the cross-fit rules are AFF's; only the caption posterior of the affect grouping changes",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.06)
    fig.savefig(OUT / "what_changed.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: the development step

def fig_development(rec):
    left = [("AFF", rec["AFF"]["bar_margin"], "AFF vs B′(A0)\n(reference, round 3)"),
            ("G-T", rec["G-T"]["bar_margin"], "G-T vs B′(A0)"),
            ("G-TF", rec["G-TF"]["bar_margin"], "G-TF vs B′(A0)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.22, wspace=0.55)
    ax = axes[0]
    ys = -np.arange(len(left), dtype=float)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvline(BAR, color=INK, lw=1.0, ls="--", zorder=1)
    ax.text(BAR, 0.55, "bar +0.5", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for y, (who, v, lab) in zip(ys, left):
        lo, hi = v["ci95"]
        hollow = who == "AFF"
        ax.plot([lo, hi], [y, y], color=C[who], lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot(v["point"], y, marker=MK[who], ms=8, color=C[who], mfc="white" if hollow else C[who], mew=1.8, zorder=4)
        ax.text(1.0, y + 0.2, fmt(v), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[2] for r in left])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys.min() - 0.6, 0.75)
    ax.set_xlim(-0.1, 1.25)
    ax.set_xlabel("Bar margin, fused R@1 minus the bar comparator (pp)")
    ax.set_title("D10: bar margin against the strongest comparator", fontsize=10, loc="left", color=INK2)

    ax = axes[1]
    ys = -np.arange(len(CANDS), dtype=float)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for y, who in zip(ys, CANDS):
        v = rec[who]["delta"]
        lo, hi = v["ci95"]
        ax.plot([lo, hi], [y, y], color=C[who], lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot(v["point"], y, marker=MK[who], ms=8, color=C[who], zorder=4)
        ax.text(0.0, y + 0.2, f"Δ = {rec[who]['delta_int']:+d} rankings   {fmt(v)}", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{w} minus AFF" for w in CANDS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys.min() - 0.6, 0.75)
    ax.set_xlim(-0.65, 0.15)
    ax.set_xlabel("Δ_k: fused R@1, candidate minus AFF, paired per anchor (pp)")
    ax.set_title("Carry condition: Δ_k > 0 (none)", fontsize=10, loc="left", color=INK2)
    handles = [Line2D([0], [0], color=C[w], lw=2.2, marker=MK[w], ms=7, label=w) for w in CANDS] + \
              [Line2D([0], [0], color=C["AFF"], lw=2.2, marker="D", ms=7, mfc="white", mew=1.8, label="AFF (reference)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Seed 42: neither candidate reached the +0.5 bar, and both lost to AFF",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.875, "95% painting-bootstrap intervals (5,000 resamples). One rule outcome: the carry set E is "
             "empty, so the round was killed (rule §5 item 8).", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "development.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3: the measured diagnostics

def hbars(ax, labels, values, colors, fmt_s, xlim, title, xlabel, refs=(), base=0.0):
    """Horizontal bars drawn from a meaningful baseline (0, chance AUC 0.5 or lift 1.0), never from a cut axis."""
    ys = -np.arange(len(labels), dtype=float)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.barh(ys, [v - base for v in values], left=base, height=0.56, color=colors, zorder=3)
    for y, v in zip(ys, values):
        ax.text(v + (xlim[1] - xlim[0]) * 0.015, y, fmt_s.format(v), va="center", ha="left", fontsize=8.6, color=INK)
    for x, lab in refs:
        ax.axvline(x, color=INK2, lw=1.0, ls="--", zorder=2)
        ax.text(x, 0.62, lab, ha="center", va="bottom", fontsize=7.8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(ys.min() - 0.6, 0.95)
    ax.set_title(title, fontsize=9.6, loc="left", color=INK)
    ax.set_xlabel(xlabel, fontsize=8.6)


def fig_diagnostics(diag):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.86, bottom=0.08, wspace=0.62, hspace=0.62)
    p = diag["placement"]
    hbars(axes[0, 0], ["image head (CLIP)", "caption head (CLIP)", "caption head (GE)"],
          [p["image_head"], p["caption_CLIP"], p["caption_GE"]], [C["img"], C["CLIP"], C["GE"]], "{:.2f}%", (0, 100),
          "(a) placement: held-out accuracy over the 41 communities", "% of 10,000 scorer-train check rows",
          refs=((p["majority"], "majority 6.41"),))
    a, d = diag["detection_auc"], diag["delta_affect_auc"]
    hbars(axes[0, 1], ["reader P(affect): AFF (R1)", "reader P(affect): G-TF", "feature Δ_affect: CLIP", "feature Δ_affect: GE"],
          [a["AFF_R1"], a["G-TF"], d["F"], d["F_G"]], [C["AFF"], C["G-TF"], C["CLIP"], C["GE"]], "{:.3f}", (0.5, 0.9),
          "(b) detection AUC of emotion conditions", "AUC, bars from chance (0.5)", base=0.5)
    lift = diag["pair_lift"]
    hbars(axes[1, 0], ["through the heads: CLIP caption", "through the heads: GE caption", "on the groups (perfect placement)"],
          [lift["CLIP"]["ratio_same_over_diff"], lift["GE"]["ratio_same_over_diff"], lift["groups"]],
          [C["CLIP"], C["GE"], "#c3c2b7"], "{:.3f}×", (0.9, 3.0),
          "(c) emotion pair lift, image × caption agreement", "same-emotion over different-emotion agreement, bars from 1.0",
          refs=((1.0, "no signal"),), base=1.0)
    e = diag["either_cost_per_unit_gain"]
    hbars(axes[1, 1], ["AFF (CLIP placement)", "G-T", "G-TF"], [e["AFF"], e["G-T"], e["G-TF"]],
          [C["AFF"], C["G-T"], C["G-TF"]], "{:.3f}", (0, 0.8),
          "(d) either-rate cost per unit of condition gain", "−(either change) / gain statistic, at the chosen cells")
    fig.suptitle("A much sharper caption placement barely sharpened the image × caption agreement the method reads",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.925, "Seed 42, measured diagnostics of rule §5 (descriptive, decide nothing). (a) from "
             "placement.json; (b) to (d) from diagnostics_seed42.json; the reader AUCs of (b) and (d) re-derived here",
             fontsize=8.5, color=INK2)
    fig.savefig(OUT / "diagnostics.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4: why the sharper placement did not carry

def fig_why_agreement(desc):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.10, wspace=0.28, hspace=0.62)
    L = desc["pair_lifts"]
    # (a) lift decomposition
    ax = axes[0, 0]
    groups = [("caption × caption", L["caption_x_caption_CLIP"]["ratio_same_over_diff"],
               L["caption_x_caption_GE"]["ratio_same_over_diff"]),
              ("image × caption\n(what the method reads)", L["image_x_caption_CLIP"]["ratio_same_over_diff"],
               L["image_x_caption_GE"]["ratio_same_over_diff"])]
    xs = np.arange(len(groups) + 1, dtype=float)
    w = 0.34
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for i, (lab, vc, vg) in enumerate(groups):
        for off, v, col in ((-w / 2, vc, C["CLIP"]), (w / 2, vg, C["GE"])):
            ax.bar(xs[i] + off, v - 1.0, bottom=1.0, width=w - 0.03, color=col, zorder=3)
            ax.text(xs[i] + off, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4, color=INK)
    vi = L["image_x_image"]["ratio_same_over_diff"]
    ax.bar(xs[2], vi - 1.0, bottom=1.0, width=w - 0.03, color=C["img"], zorder=3)
    ax.text(xs[2], vi + 0.03, f"{vi:.2f}", ha="center", va="bottom", fontsize=8.4, color=INK)
    ax.axhline(1.0, color=INK2, lw=1.0, ls="--", zorder=2)
    ax.axhline(desc_group_lift, color=INK, lw=1.0, ls=":", zorder=2)
    ax.text(2.45, desc_group_lift + 0.03, f"on the groups (perfect placement) {desc_group_lift:.2f}", ha="right",
            va="bottom", fontsize=8, color=INK)
    ax.text(-0.45, 1.02, "no signal 1.0", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups] + ["image × image\n(the image head alone)"], fontsize=8.6)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.9, 3.0)
    ax.set_ylabel("same / different-emotion agreement", fontsize=8.6)
    ax.set_title("(a) emotion lift by side (bars from 1.0): the image head carries almost none", fontsize=9.6,
                 loc="left", color=INK)
    # (b) redundancy with B and with cosine
    ax = axes[0, 1]
    R = desc["redundancy"]
    labs, vc, vg = [], [], []
    for what, src_c, src_g in (("with z(B)", R["CLIP_D7"]["affect"], R["GE_D7"]["affect"]),
                               ("with z(cosine)", R["with_cosine"]["CLIP"]["affect"], R["with_cosine"]["GE"]["affect"])):
        for d in ("i2t", "t2i"):
            labs.append(f"{what}\n{'image query' if d == 'i2t' else 'caption query'}")
            vc.append(src_c[d]), vg.append(src_g[d])
    xs = np.arange(len(labs), dtype=float)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for off, vals, col in ((-w / 2, vc, C["CLIP"]), (w / 2, vg, C["GE"])):
        ax.bar(xs + off, vals, width=w - 0.03, color=col, zorder=3)
        for x, v in zip(xs + off, vals):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.2, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=8.2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("mean per-row correlation over 13 candidates", fontsize=8.6)
    ax.set_title("(b) the affect agreement's overlap with B and with CLIP cosine", fontsize=9.6, loc="left", color=INK)
    # (c) the affect score alone: p_A and p_B first, emotion pairs
    A = desc["affect_score_alone"]
    rows = [(p, d) for p in PAIRS[:2] for d in ("i2t", "t2i")]
    for ax, metric, title, ylim in ((axes[1, 0], ("pA_first", "pB_first"),
                                     "(c) the affect score alone: how often p_A or p_B ranks first", (0, 23)),
                                    (axes[1, 1], ("pA_above_pB",),
                                     "(d) the affect score alone: how often p_A scores above p_B (bars from 50%)",
                                     (48, 62))):
        xs = np.arange(len(rows), dtype=float)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        nb = 2 * len(metric)
        ww = 0.8 / nb
        k = 0
        for mt in metric:
            for plc in ("CLIP", "GE"):
                vals = [A[plc][f"{p}|{d}"][mt] for p, d in rows]
                hatch = "//" if mt == "pB_first" else None
                base = 50.0 if mt == "pA_above_pB" else 0.0          # bars from chance (50%) or from 0
                ax.bar(xs - 0.4 + ww * (k + 0.5), [v - base for v in vals], bottom=base, width=ww - 0.02,
                       color=C[plc], hatch=hatch, edgecolor="white", zorder=3)
                for x, v in zip(xs - 0.4 + ww * (k + 0.5), vals):
                    ax.text(x, v + (ylim[1] - ylim[0]) * 0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2,
                            color=INK)
                k += 1
        if metric == ("pA_above_pB",):
            ax.axhline(50, color=INK2, lw=1.0, ls="--", zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{PAIR_LABEL[p]}\n{'image query' if d == 'i2t' else 'caption query'}" for p, d in rows],
                           fontsize=8.2)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(*ylim)
        ax.set_ylabel("% of rankings", fontsize=8.6)
        ax.set_title(title, fontsize=9.6, loc="left", color=INK)
    handles = [Patch(fc=C["CLIP"], label="CLIP caption placement (AFF)"), Patch(fc=C["GE"], label="GE caption placement"),
               Patch(fc=C["img"], label="image head with itself"),
               Patch(fc="white", ec=INK2, hatch="//", label="hatched: p_B (the other aspect's candidate)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Why the sharper caption placement did not sharpen the term: the image side caps it, and it lost "
                 "CLIP's similarity", fontsize=11.5, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.92, "Seed 42, descriptive (after the kill). (a) selection rows, different paintings; (b) to (d) "
             "the 12,288 seed-42 episodes; emotion labels used only to group pairs or rankings", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "why_agreement.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5: where the rankings were lost

def fig_why_cells(desc):
    pc = desc["per_cell_minus_AFF"]
    fx = desc["fixed_cells"]
    cells = [(p, c) for p in PAIRS for c in "ab"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.26, wspace=0.28)
    ax = axes[0]
    xs = np.arange(len(cells), dtype=float)
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    series = [("G-T", [pc["G-T"][f"{p}|{c}"]["net_rankings"] for p, c in cells], C["G-T"], None, "G-T (own cells)"),
              ("G-TF", [pc["G-TF"][f"{p}|{c}"]["net_rankings"] for p, c in cells], C["G-TF"], None, "G-TF (own cells)"),
              ("G-T@AFF", [fx["G-T_at_AFF_cells"]["per_cell_net_vs_AFF"][f"{p}|{c}"] for p, c in cells], C["G-T"], "//",
               "G-T at AFF's cells (39, 119)")]
    ww = 0.26
    for j, (_, vals, col, hatch, _) in enumerate(series):
        ax.bar(xs + (j - 1) * ww, vals, width=ww - 0.03, color=col if hatch is None else "white", edgecolor=col,
               hatch=hatch, lw=1.2, zorder=3)
        for x, v in zip(xs + (j - 1) * ww, vals):
            ax.text(x, v + (2 if v >= 0 else -2), f"{v:+d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.4,
                    color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{PAIR_LABEL[p]}\ncondition {c}\n({SIDE[(p, c)]} side)" for p, c in cells], fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.axvline(1.5, color=GRID, lw=1.0), ax.axvline(3.5, color=GRID, lw=1.0)
    ax.set_ylim(-95, 30)
    ax.set_ylabel("net rankings won (+) or lost (−) against AFF")
    ax.set_title("(a) per pair and condition (sums: G-T −169, G-TF −134, G-T at AFF's cells −146)", fontsize=9.6,
                 loc="left", color=INK)
    handles = [Patch(fc=s[2] if s[3] is None else "white", ec=s[2], hatch=s[3], label=s[4]) for s in series]
    # (b) gain and either against AFF: at own cells and at AFF's cells
    ax = axes[1]
    rows = [("G-T, own cells", desc["own_cells_gain_either_vs_AFF"]["G-T"]["gain"],
             desc["own_cells_gain_either_vs_AFF"]["G-T"]["either"], C["G-T"], False),
            ("G-T at AFF's cells", fx["G-T_at_AFF_cells"]["gain_minus_AFF"], fx["G-T_at_AFF_cells"]["either_minus_AFF"],
             C["G-T"], True),
            ("G-TF, own cells", desc["own_cells_gain_either_vs_AFF"]["G-TF"]["gain"],
             desc["own_cells_gain_either_vs_AFF"]["G-TF"]["either"], C["G-TF"], False),
            ("G-TF at AFF's cells", fx["G-TF_at_AFF_cells"]["gain_minus_AFF"],
             fx["G-TF_at_AFF_cells"]["either_minus_AFF"], C["G-TF"], True),
            ("AFF at G-T's cells", fx["AFF_at_G-T_cells"]["gain_minus_AFF"], fx["AFF_at_G-T_cells"]["either_minus_AFF"],
             C["AFF"], True)]
    ys = -np.arange(len(rows), dtype=float)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for y, (lab, g, e, col, hollow) in zip(ys, rows):
        for off, v, mk in ((0.14, g, "^"), (-0.14, e, "v")):
            ax.plot(v["ci95"], [y + off] * 2, color=col, lw=2.0, solid_capstyle="round", zorder=3)
            ax.plot(v["point"], y + off, marker=mk, ms=7.5, color=col, mfc="white" if hollow else col, mew=1.6, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-2.0, 1.4)
    ax.set_xlabel("minus AFF at its own cells (pp)", fontsize=8.6)
    ax.set_title("(b) condition gain (▲) and either rate (▼) against AFF", fontsize=9.6, loc="left", color=INK)
    handles += [Line2D([0], [0], color=INK2, lw=0, marker="^", ms=7, label="gain minus AFF"),
                Line2D([0], [0], color=INK2, lw=0, marker="v", ms=7, label="either minus AFF"),
                Line2D([0], [0], color=INK2, lw=0, marker="^", ms=7, mfc="white", mew=1.4,
                       label="hollow: assembled at another scorer's cells")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The losses sat where the gate opens (condition a); at AFF's own weights the GE term bought more "
                 "gain but cost more either", fontsize=11.5, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.9, "Seed 42, descriptive (after the kill); net rankings out of 8,192 per (pair, condition); "
             "fixed-cell rows (hollow, hatched) are decompositions, not methods; R@1 = (either + gain) / 2",
             fontsize=8.5, color=INK2)
    fig.savefig(OUT / "why_cells.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 6: the in-sample family

def fig_why_family(desc, rec):
    fam = desc["in_sample_family"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.2, wspace=0.08)
    xs = np.arange(len(NESTED_A), dtype=float)
    for ax, t, title in ((axes[0], "0", "τ_0 (AFF, G-T and G-TF all chose τ_0 on tune half 0)"),
                         (axes[1], "2", "τ_2 (all chose τ_2 on tune half 1)")):
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for nm in ("AFF",) + CANDS:
            ax.plot(xs, fam[nm]["best_over_lambda_u"][t], color=C[nm], lw=2.0, marker=MK[nm], ms=6, zorder=3,
                    mfc="white" if nm == "AFF" else C[nm], mew=1.6, label=nm)
        h = 0 if t == "0" else 1
        for nm in ("AFF",) + CANDS:
            cell = rec[nm]["fpick"][h]
            _, _, la = cell_lambdas(cell)
            a = NESTED_A.index(la)
            ax.plot(xs[a], fam[nm]["best_over_lambda_u"][t][a], marker="o", ms=14, mfc="none", mec=C[nm], mew=1.4,
                    zorder=5)
        ax.axhline(rec["AFF"]["fused_r1"], color=INK2, lw=1.0, ls="--", zorder=2)
        ax.text(0.1, rec["AFF"]["fused_r1"] + 0.015, f"AFF cross-fitted {rec['AFF']['fused_r1']:.3f}", fontsize=8,
                color=INK2, va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a:g}" for a in NESTED_A])
        ax.set_xlabel("λ_a, the weight of the gated term")
        ax.set_title(title, fontsize=9.6, loc="left", color=INK2)
        ax.set_ylim(18.25, 19.4)
    axes[0].set_ylabel("whole-seed fused R@1, best λ_u (pp)")
    handles = [Line2D([0], [0], color=C[n], lw=2.0, marker=MK[n], ms=6, mfc="white" if n == "AFF" else C[n], mew=1.6,
                      label=n) for n in ("AFF",) + CANDS] + \
              [Line2D([0], [0], marker="o", ms=12, mfc="none", mec=INK2, lw=0, label="λ_a of the chosen cell")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The GE-placement families level off from λ_a = 2, about 0.3 pp below AFF's at λ_a 4 to 16",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.875, "Seed 42, in sample (all 12,288 episodes, no cross-fit, so optimistic; descriptive). Each "
             "point: the best of the 7 λ_u values at that λ_a; circles mark the λ_a of each chosen cell (its λ_u may differ); "
             "G-TF's index is τ′", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "why_family.png", dpi=DPI)
    plt.close(fig)


desc_group_lift = 2.7111312041209863          # told_oracle.json arms.L.pairs.groups.lift.lift (asserted in main)


def main():
    rec, diag, desc = data()
    same(diag["pair_lift"]["groups"], desc_group_lift)
    print("re-derived decision quantities equal dev_seed42.json and carry.json (exact)")
    for n in CANDS + ("AFF",):
        r = rec[n]
        print(n, f"fused {r['fused_r1']:.4f} cf {r['cf_r1']:.4f}", r["bar_comparator"], "bar", fmt(r["bar_margin"]),
              "gain", fmt(r["gain_statistic"]), "either", round(r["either_change"], 4), r["d10"],
              "delta", r.get("delta_int"), fmt(r["delta"]) if "delta" in r else "")
    fig_what_changed()
    fig_development(rec)
    fig_diagnostics(diag)
    fig_why_agreement(desc)
    fig_why_cells(desc)
    fig_why_family(desc, rec)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
