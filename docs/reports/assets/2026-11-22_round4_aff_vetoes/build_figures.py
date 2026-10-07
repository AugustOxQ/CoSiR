"""Figures and figure data for docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md (reader fix, round 4).

Reads stored outputs only (nothing under src/ is written) and asserts every decision quantity it plots against its
source before any figure is drawn:
  src/test/20261122_round4_aff_vetoes/results/dev_seed42.json        rule section 5 items 6 and 7 (development record)
  src/test/20261122_round4_aff_vetoes/results/carry.json             rule section 5 items 8 and 9 (carry, kill)
  src/test/20261122_round4_aff_vetoes/results/regression_check.json  rule section 5 items 1 to 5 (262 comparisons)
  src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz      seed-42 per-anchor arrays, gates, picks, v

Step 1 re-derives, from the per-anchor arrays alone, every decision quantity of the development step except the chosen
cells and sigma* (each candidate's and AFF's fused and counterpart R@1, bar comparator, bar margin, margin, gain
statistic and either change with their intervals, the D10 clauses, the per-pair bar margins, the integer Delta_k with
its point and interval, AFF minus B'(A1), the tau_0 open counts, the empty carry set) and asserts that each equals
dev_seed42.json / carry.json exactly (difference 0.0). The chosen cells and sigma* cannot be derived from per-anchor
arrays: it reads them from seed42_arrays.npz and checks them against dev_seed42.json. Step 2 computes the descriptive breakdowns of the report's section 6 (closure shares, per
pair candidate minus AFF, net rankings by closure class, comparator gaps, R1's abstention against AFF's gate). They
are computed after the kill, decide nothing, and score no new variant: every number is a breakdown of arrays the
seed-42 run already stored. Intervals: src.eval.aspect_metrics.cluster_bootstrap (5,000 resamples, seed 42, chunk
250, clusters = anchor paintings), the helper behind round 1's common.point_ci that the implementation used.

Run from the repo root (CPU only):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-22_round4_aff_vetoes/build_figures.py
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
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

FOLDER = ROOT / "src/test/20261122_round4_aff_vetoes"
RES = FOLDER / "results"
RULE_SHA = "cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b"
OUT = Path(__file__).resolve().parent
DPI = 150
N_EP = 12288
PAIRS = ("emotion__style", "emotion__genre", "style__genre")
PAIR_LABEL = {"emotion__style": "emotion × style", "emotion__genre": "emotion × genre",
              "style__genre": "style × genre"}
CANDS = ("V4", "V2", "V24")
KEY = {"V4": "v4", "V2": "v2", "V24": "v24", "AFF": "aff", "R1": "r1", "IMGABST": "imgabst"}
FUSED_CELLS = (39, 119)            # every fused cross-fit's cells (asserted below)
TAU_OF_CELL = {39: 0, 119: 2}       # cell (t*7+u)*8+a: 39 -> tau_0, 119 -> tau_2 (asserted against cell_text)
BAR = 0.5                          # D10 clause 1
TIE_BAND = 24                      # rule section 5 item 8

# dataviz reference palette, first three categorical slots (validated all-pairs, light mode); grey for AFF
C = {"V4": "#2a78d6", "V2": "#eb6834", "V24": "#1baf7a", "AFF": "#8a8984"}
MK = {"V4": "o", "V2": "s", "V24": "^", "AFF": "D"}
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


# ---------------------------------------------------------------- step 1: the decision quantities, re-derived

def comparators(name, pa):
    """D8's condition-free comparators in tie order (V4 and AFF: B'(A0), counterpart, B; V2 and V24: B'(A1) first)."""
    cf = pa[f"{KEY[name]}_cf"]
    base = [("Bprime_A0", pa["Bp0"]), ("counterpart", cf), ("B", pa["B"])]
    return ([("Bprime_A1", pa["Bp1"])] + base) if name in ("V2", "V24") else base


def bar_comparator(comps):
    means = [float(np.mean(np.asarray(p["r1"], dtype=np.float64))) for _, p in comps]
    best = 0
    for i in range(1, len(comps)):
        if means[i] > means[best]:          # strictly greater: ties go to the earliest
            best = i
    return comps[best][0], comps[best][1]


def record(name, pa, z, cl, pidx):
    k = KEY[name]
    fu, cf = pa[f"{k}_fused"], pa[f"{k}_cf"]
    assert np.all(cf["gain"] == 0.0), f"{name}: counterpart gain is not exactly 0"
    label, comp = bar_comparator(comparators(name, pa))
    v = fu["r1"] - comp["r1"]
    bar = pp(v, cl)
    gain = pp(fu["gain"] - cf["gain"], cl)
    either = pp((fu["r1"] + fu["other"]) - (cf["r1"] + cf["other"]), cl)
    c1, c2, c3 = bar["point"] >= BAR, bar["ci95"][0] > 0, gain["ci95"][0] > 0
    rec = {"fused_r1": 100 * float(np.mean(fu["r1"])), "cf_r1": 100 * float(np.mean(cf["r1"])),
           "fpick": [int(c) for c in z[f"{k}_fused_cells"]], "cpick": [int(c) for c in z[f"{k}_cf_cells"]],
           "sigma": [float(s) for s in z[f"{k}_sigma"]], "bar_comparator": label, "bar_margin": bar,
           "margin_vs_counterpart": pp(fu["r1"] - cf["r1"], cl), "gain_statistic": gain,
           "either_change": either["point"], "either_change_ci": either["ci95"],
           "d10": {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "clears": bool(c1 and c2 and c3)},
           "per_pair_bar_margin": {p: pp(v[pidx == i], cl[pidx == i]) for i, p in enumerate(PAIRS)}}
    if name != "AFF":
        aff = pa["aff_fused"]["r1"]
        di = int((as_int4(fu["r1"]) - as_int4(aff)).sum())
        rec["delta_int"] = di
        rec["delta"] = {"point": 100.0 * di / (4 * len(aff)), "ci95": pp(fu["r1"] - aff, cl)["ci95"]}
    return rec


def check_against_files(rec, dev, carry, z):
    """Every re-derived decision quantity equals the stored record exactly."""
    for name in CANDS + ("AFF",):
        s = dev["aff"] if name == "AFF" else dev["candidates"][name]
        r = rec[name]
        same(r["fused_r1"], s["fused_r1"]), same(r["cf_r1"], s["cf_r1"])
        same(r["fpick"], [s["cells"]["fpick"]["0"], s["cells"]["fpick"]["1"]])
        same(r["cpick"], [s["cells"]["cpick"]["0"], s["cells"]["cpick"]["1"]])
        same(r["sigma"], [s["sigma"]["0"], s["sigma"]["1"]])
        same(r["bar_comparator"], s["bar_comparator"])
        same_ci(r["bar_margin"], s["bar_margin"])
        same_ci(r["margin_vs_counterpart"], s["margin_vs_counterpart"])
        same_ci(r["gain_statistic"], s["gain_statistic"])
        same(r["either_change"], s["either_change"])
        same(r["d10"], s["d10"])
        for p in PAIRS:
            same_ci(r["per_pair_bar_margin"][p], s["per_pair_bar_margin"][p])
        for h, cell in enumerate(r["fpick"]):
            same(s["cell_text"]["fused"][str(h)]["cell"], cell)
            same(s["cell_text"]["fused"][str(h)]["tau_index"], TAU_OF_CELL[cell])
        if name != "AFF":
            same(r["delta_int"], s["delta_int"]), same(r["delta_int"], carry["delta_int"][name])
            same_ci(r["delta"], s["delta"])
            same(r["d10"], carry["d10"][name])
        same(tuple(r["fpick"]), FUSED_CELLS)          # every fused cross-fit chose AFF's own cells
    # the carry (rule section 5 items 8 and 9), from the re-derived integers and clauses
    E = [n for n in CANDS if rec[n]["d10"]["clears"] and rec[n]["delta_int"] > 0]
    same(E, carry["carry"]["E"])
    assert E == [] and carry["kill"] is True and carry["carry"]["carried"] is None and carry["carry"]["M"] is None
    assert carry["carry"]["boundaries"] == [] and all(dev["candidates"][n]["boundaries"] == [] for n in CANDS)
    assert all(rec[n]["delta_int"] != 0 for n in CANDS)          # no boundary of rule section 8
    # tau_0 open counts (gate statistics) and AFF's open counts at every tau
    for name in CANDS + ("AFF",):
        for c in "ab":
            same(int(z[f"{KEY[name]}_gate__{c}"][0].sum()), dev["open_tau0_counts"][name][c])


def beside_aff(pa, cl, dev):
    b = dev["beside_aff"]
    out = {"Bprime_A1_mean_r1": 100 * float(np.mean(pa["Bp1"]["r1"])),
           "Bprime_A0_mean_r1": 100 * float(np.mean(pa["Bp0"]["r1"])),
           "B_mean_r1": 100 * float(np.mean(pa["B"]["r1"])),
           "AFF_fused_r1": 100 * float(np.mean(pa["aff_fused"]["r1"])),
           "AFF_minus_Bprime_A1": pp(pa["aff_fused"]["r1"] - pa["Bp1"]["r1"], cl)}
    for k in ("Bprime_A1_mean_r1", "Bprime_A0_mean_r1", "B_mean_r1", "AFF_fused_r1"):
        same(out[k], b[k])
    same_ci(out["AFF_minus_Bprime_A1"], b["AFF_minus_Bprime_A1"])
    same(out["Bprime_A1_mean_r1"], 18.804931640625)                # rule D4
    same(out["Bprime_A0_mean_r1"], 18.436686197916664)             # rule D8
    return out


# ---------------------------------------------------------------- step 2: descriptive breakdowns (after the kill)

def descriptive(pa, z, cl, pidx, par, rec):
    n = len(par)
    ar = np.arange(n)
    # the tau index each episode is scored at: the cell chosen on tune half h scores the episodes of parity 1 - h
    # (round 2's r2_fusion.assemble); fused cells 39 (tau_0, half 0) and 119 (tau_2, half 1) for every scorer here
    eff = np.where(par == 1, TAU_OF_CELL[FUSED_CELLS[0]], TAU_OF_CELL[FUSED_CELLS[1]])

    def gate(name, c, t):
        g = z[f"{KEY[name]}_gate__{c}"]
        return (g[eff, ar] if t == "as_scored" else g[t]).astype(bool)

    out = {}
    # integer sums of 4 x per-episode R@1 behind every mean (49,152 rankings in all)
    out["int_sums"] = {lab: int(as_int4(pa[k]["r1"]).sum()) for lab, k in
                       (("AFF", "aff_fused"), ("V4", "v4_fused"), ("V2", "v2_fused"), ("V24", "v24_fused"),
                        ("Bprime_A1", "Bp1"), ("Bprime_A0", "Bp0"), ("B", "B"))}
    for lab in ("AFF",) + CANDS:
        assert out["int_sums"][lab] / (4 * n) * 100 == rec[lab]["fused_r1"] or \
            abs(out["int_sums"][lab] / (4 * n) * 100 - rec[lab]["fused_r1"]) < 1e-12
    for name in CANDS:
        same(out["int_sums"][name] - out["int_sums"]["AFF"], rec[name]["delta_int"])
    # (a) closure shares: among AFF's open (episode, condition) values, the share each candidate's gate closes
    clo = {}
    for name in CANDS:
        clo[name] = {}
        for t in (0, 1, 2, 3, "as_scored"):
            row = {}
            for i, p in enumerate(PAIRS):
                for c in "ab":
                    m = pidx == i
                    a_open = gate("AFF", c, t) & m
                    closed = a_open & ~gate(name, c, t)
                    assert not np.any(gate(name, c, t) & ~gate("AFF", c, t))     # a veto never opens a gate
                    row[f"{p}|{c}"] = {"aff_open": int(a_open.sum()), "closed": int(closed.sum()),
                                       "share": 100 * float(closed.sum()) / float(a_open.sum())}
            # emotion side (condition a of the two emotion pairs) against every other value
            emo = [row[f"{p}|a"] for p in PAIRS[:2]]
            oth = [row[f"{p}|b"] for p in PAIRS[:2]] + [row[f"{PAIRS[2]}|{c}"] for c in "ab"]
            row["emotion_side"] = {"aff_open": sum(r["aff_open"] for r in emo), "closed": sum(r["closed"] for r in emo)}
            row["other_values"] = {"aff_open": sum(r["aff_open"] for r in oth), "closed": sum(r["closed"] for r in oth)}
            for k in ("emotion_side", "other_values"):
                row[k]["share"] = 100 * row[k]["closed"] / row[k]["aff_open"]
            clo[name][str(t)] = row
    out["closure"] = clo

    # (b) open counts of every stored gate at every tau index (gate statistics)
    out["open_counts"] = {name: {str(t): {c: int(z[f"{KEY[name]}_gate__{c}"][t].sum()) for c in "ab"}
                                 for t in range(4)} for name in ("AFF", "V4", "V2", "V24", "R1", "IMGABST")}
    out["open_counts_as_scored"] = {name: {c: int(gate(name, c, "as_scored").sum()) for c in "ab"}
                                    for name in ("AFF",) + CANDS}

    # (c) candidate minus AFF, fused, paired per anchor: R@1 pooled and per pair; condition gain, other, either pooled
    aff = pa["aff_fused"]
    pm = {}
    for name in CANDS:
        fu = pa[f"{KEY[name]}_fused"]
        d = fu["r1"] - aff["r1"]
        e = {"pooled": pp(d, cl)}
        same_ci(e["pooled"], rec[name]["delta"] | {"point": e["pooled"]["point"]})  # interval = the D9 interval
        assert abs(e["pooled"]["point"] - rec[name]["delta"]["point"]) < 1e-12
        for i, p in enumerate(PAIRS):
            e[p] = pp(d[pidx == i], cl[pidx == i])
            # for V4 the bar comparator is AFF's (B'(A0)), so the per-pair difference is the difference of bar margins
            if name == "V4":
                assert abs(e[p]["point"] - (rec["V4"]["per_pair_bar_margin"][p]["point"]
                                            - rec["AFF"]["per_pair_bar_margin"][p]["point"])) < 1e-12
        e["gain"] = pp(fu["gain"] - aff["gain"], cl)
        e["other"] = pp(fu["other"] - aff["other"], cl)
        e["either"] = pp((fu["r1"] + fu["other"]) - (aff["r1"] + aff["other"]), cl)
        assert abs((e["either"]["point"] + e["gain"]["point"]) / 2 - e["pooled"]["point"]) < 1e-12
        pm[name] = e
    out["minus_AFF"] = pm

    # (d) where Delta_k came from: d = candidate minus AFF is 0 outside the episodes whose as-scored gate changed;
    # classes by the condition(s) closed; net rankings = sum of 4 d (rankings won minus lost out of 4 per episode)
    nr = {}
    for name in CANDS:
        fu = pa[f"{KEY[name]}_fused"]["r1"]
        d4 = as_int4(fu) - as_int4(aff["r1"])
        ca = gate("AFF", "a", "as_scored") & ~gate(name, "a", "as_scored")
        cb = gate("AFF", "b", "as_scored") & ~gate(name, "b", "as_scored")
        changed = ca | cb
        assert np.all(d4[~changed] == 0), f"{name}: a fused value moved where the gate did not change"
        full = changed & ~gate(name, "a", "as_scored") & ~gate(name, "b", "as_scored")
        # where the candidate's gate is shut in both conditions its fused score is (1 + lambda_u) z(B), ranked as B
        assert np.array_equal(fu[full], pa["B"]["r1"][full]), f"{name}: fully closed episodes do not score as B"
        row = {"changed_episodes": int(changed.sum()), "fully_closed_episodes": int(full.sum()),
               "nonzero": int((d4 != 0).sum()), "up": int((d4 > 0).sum()), "down": int((d4 < 0).sum()),
               "net_rankings": int(d4.sum())}
        same(row["net_rankings"], rec[name]["delta_int"])
        for i, p in enumerate(PAIRS):
            m = pidx == i
            row[p] = {cls: {"episodes": int((m & msk).sum()), "net_rankings": int(d4[m & msk].sum())}
                      for cls, msk in (("a_only", ca & ~cb), ("b_only", cb & ~ca), ("both", ca & cb))}
            row[p]["total"] = int(d4[m].sum())
        nr[name] = row
    same(nr["V4"]["changed_episodes"], nr["V4"]["fully_closed_episodes"])   # a_v shuts both conditions at once
    out["net_rankings"] = nr
    v4, v2 = pa["v4_fused"]["r1"], pa["v2_fused"]["r1"]
    out["V4_vs_V2"] = {"episodes_differ": int((v4 != v2).sum()), "same_sum": bool(as_int4(v4).sum() == as_int4(v2).sum())}
    assert out["V4_vs_V2"]["episodes_differ"] > 0 and out["V4_vs_V2"]["same_sum"]

    # (e) on the episodes V4 closed (where V4 scores exactly as B) and on those it kept open: AFF minus B, per pair
    ca = gate("AFF", "a", "as_scored") & ~gate("V4", "a", "as_scored")
    cb = gate("AFF", "b", "as_scored") & ~gate("V4", "b", "as_scored")
    closed = ca | cb
    kept = gate("V4", "a", "as_scored") | gate("V4", "b", "as_scored")
    dB = aff["r1"] - pa["B"]["r1"]
    out["V4_closed_vs_kept_AFF_minus_B"] = {
        p: {"closed": pp(dB[(pidx == i) & closed], cl[(pidx == i) & closed]) | {"episodes": int(((pidx == i) & closed).sum())},
            "kept": pp(dB[(pidx == i) & kept], cl[(pidx == i) & kept]) | {"episodes": int(((pidx == i) & kept).sum())},
            "aff_shut": int(((pidx == i) & ~gate("AFF", "a", "as_scored") & ~gate("AFF", "b", "as_scored")).sum())}
        for i, p in enumerate(PAIRS)}

    # (f) comparator gaps, pooled and per pair (fused R@1 or condition-free R@1, paired per anchor)
    gaps = {}
    for lab, v in (("AFF_minus_B", aff["r1"] - pa["B"]["r1"]), ("Bprime_A0_minus_B", pa["Bp0"]["r1"] - pa["B"]["r1"]),
                   ("Bprime_A1_minus_Bprime_A0", pa["Bp1"]["r1"] - pa["Bp0"]["r1"]),
                   ("AFF_minus_Bprime_A1", aff["r1"] - pa["Bp1"]["r1"])):
        gaps[lab] = {"pooled": pp(v, cl)} | {p: pp(v[pidx == i], cl[pidx == i]) for i, p in enumerate(PAIRS)}
    for i, p in enumerate(PAIRS):                       # AFF's bar margin = (AFF - B) - (B'(A0) - B) per pair
        assert abs(gaps["AFF_minus_B"][p]["point"] - gaps["Bprime_A0_minus_B"][p]["point"]
                   - rec["AFF"]["per_pair_bar_margin"][p]["point"]) < 1e-12
    out["comparator_gaps"] = gaps
    out["floor_arithmetic"] = {
        "fused_needed_vs_BprimeA1": 100 * float(np.mean(pa["Bp1"]["r1"])) + BAR,
        "above_AFF": 100 * float(np.mean(pa["Bp1"]["r1"])) + BAR - rec["AFF"]["fused_r1"],
        "V2_V24_fused_minus_BprimeA0": {n: rec[n]["fused_r1"] - 100 * float(np.mean(pa["Bp0"]["r1"])) for n in ("V2", "V24")},
    }

    # (g) the brainstorm's abstention on R1 (stored regression path R1 x a_v) against V4 on AFF, at tau_2 (the tau
    # index of R1's and R1 x a_v's fused cells 116/119 and 117/119): values closed, and how many AFF's gate already shuts
    for k in ("r1", "imgabst"):
        assert all(TAU_OF_CELL.get(int(c), None) == 2 or int(c) // 56 == 2 for c in z[f"{k}_fused_cells"])
    rr = {}
    for c in "ab":
        r1o, iao, afo, v4o = (z[f"{k}_gate__{c}"][2].astype(bool) for k in ("r1", "imgabst", "aff", "v4"))
        r = {}
        for i, p in enumerate(PAIRS + ("all",)):
            m = (pidx == i) if p != "all" else np.ones(n, bool)
            cr = r1o & ~iao & m
            r[p] = {"R1_open": int((r1o & m).sum()), "abstention_closes_on_R1": int(cr.sum()),
                    "of_which_AFF_gate_already_shut": int((cr & ~afo).sum()),
                    "AFF_open": int((afo & m).sum()), "abstention_closes_on_AFF": int((afo & ~v4o & m).sum())}
        rr[c] = r
    out["R1_abstention_vs_AFF_tau2"] = rr
    # fused R@1 and per-pair bar margins of R1 and of R1 x a_v (stored regression scorers, both against their own
    # counterparts; reproduced here, not new): the brainstorm's +0.708 / +1.221 / -0.598 and the rule's item 3 values
    r1f, r1c = pa["r1_fused"]["r1"], pa["r1_cf"]["r1"]
    iaf, iac = pa["imgabst_fused"]["r1"], pa["imgabst_cf"]["r1"]
    out["R1_and_abstention"] = {
        "R1_fused_r1": 100 * float(np.mean(r1f)), "abstention_fused_r1": 100 * float(np.mean(iaf)),
        "R1_per_pair_bar": {p: 100 * float(np.mean((r1f - r1c)[pidx == i])) for i, p in enumerate(PAIRS)},
        "abstention_per_pair_bar": {p: 100 * float(np.mean((iaf - iac)[pidx == i])) for i, p in enumerate(PAIRS)},
    }
    for p, want in zip(PAIRS, (0.708, 1.221, -0.598)):
        assert round(out["R1_and_abstention"]["R1_per_pair_bar"][p], 3) == want
    for p, want in zip(PAIRS, (0.677490234375, 1.28173828125, -0.262451171875)):   # rule section 5 item 3
        same(out["R1_and_abstention"]["abstention_per_pair_bar"][p], want)
    same(out["R1_and_abstention"]["abstention_fused_r1"], 19.059244791666664)
    same(out["R1_and_abstention"]["R1_fused_r1"], 18.918863932291664)
    return out


def data():
    dev = load(RES / "dev_seed42.json")
    carry = load(RES / "carry.json")
    reg = load(RES / "regression_check.json")
    assert sha(FOLDER / "DECISION_RULE.md") == RULE_SHA
    for d in (dev, carry, reg):
        assert d["provenance"]["rule_sha256"] == RULE_SHA and d["dry_run"] is False and d["provenance"]["smoke"] is False
    assert reg["passed"] is True and reg["n_comparisons"] == 262 and reg["items_passed"] == [1, 2, 3, 4, 5]
    assert all(c["pass"] for c in reg["comparisons"]) and len(reg["comparisons"]) == 262
    assert dev["seed42_arrays_sha256"] == carry["seed42_arrays_sha256"] == sha(RES / "seed42_arrays.npz")
    assert dev["regression_check_sha256"] == carry["regression_check_sha256"] == sha(RES / "regression_check.json")
    assert carry["dev_seed42_sha256"] == sha(RES / "dev_seed42.json")
    same(dev["order"], list(CANDS))

    z = np.load(RES / "seed42_arrays.npz", allow_pickle=False)
    cl, pidx, par = z["cl"], z["pair_index"], z["parity"]
    assert len(cl) == N_EP and len(np.unique(cl)) == 4602 and np.array_equal(np.bincount(pidx), [4096] * 3)
    names = ("aff_fused", "aff_cf", "v4_fused", "v4_cf", "v2_fused", "v2_cf", "v24_fused", "v24_cf", "r1_fused",
             "r1_cf", "imgabst_fused", "imgabst_cf", "B", "Bp0", "Bp1", "cosine", "rca")
    pa = {k: {m: np.asarray(z[f"{k}__{m}"], dtype=np.float64) for m in ("r1", "gain", "other")} for k in names}

    rec = {name: record(name, pa, z, cl, pidx) for name in CANDS + ("AFF",)}
    check_against_files(rec, dev, carry, z)
    bes = beside_aff(pa, cl, dev)
    desc = descriptive(pa, z, cl, pidx, par, rec)

    fig_data = {
        "what": "round 4 figure data. 'rederived': the decision quantities, re-derived from seed42_arrays.npz and "
                "asserted equal to dev_seed42.json and carry.json. 'descriptive': breakdowns computed after the kill "
                "(seed 42, decides nothing, no new variant scored); pp, 95% painting-bootstrap intervals (5,000 "
                "resamples, seed 42)",
        "rederived": {"records": rec, "beside_aff": bes, "E": [], "kill": True},
        "descriptive": desc,
        "inputs_sha256": {f.name: sha(f) for f in (RES / "dev_seed42.json", RES / "carry.json",
                                                   RES / "regression_check.json", RES / "seed42_arrays.npz",
                                                   FOLDER / "DECISION_RULE.md")},
    }
    (OUT / "figure_data.json").write_text(json.dumps(fig_data, indent=1, ensure_ascii=False) + "\n")
    return rec, bes, desc


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
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(-0.012, 1.012), ax.set_ylim(0.0, 0.93), ax.axis("off")
    h = 0.15
    y1 = 0.73
    box(ax, 0.005, y1, 0.17, h, "Seed 42 only\n(12,288 episodes):\nregression checks, then\nthe development step", "same")
    box(ax, 0.205, y1, 0.19, h, "A0 reader (round 1's two\nA0 half-readers): P^c(h),\npick π^c, margin m^c;\nterm T^c = Σ_h P^c(h)·s_h",
        "unchanged")
    box(ax, 0.425, y1, 0.18, h, "image agreements\nS_image, C_image →\nv = min(S, C);\nv₇₅ = seed-42 75th pct.", "new")
    box(ax, 0.635, y1, 0.17, h, "CSD style grouping\n+ round 1's two A1\nhalf-readers →\npick π_A1^c", "new")
    box(ax, 0.835, y1, 0.16, h, "B′(A1): B rebuilt\non A1's groupings;\nthe floor of V2\nand V24", "new")
    arrow(ax, 0.175, y1 + h / 2, 0.205, y1 + h / 2)
    y2 = 0.40
    box(ax, 0.005, y2, 0.40, 0.19,
        "gate (the one change), per condition c, τ_t frozen from R1\n"
        "AFF:  g = 1[m^c ≥ τ_t] · 1[π^c = affect]\n"
        "V4:   AFF's gate · 1[v < v₇₅]\n"
        "V2:   AFF's gate · 1[π_A1^c = affect]\n"
        "V24:  AFF's gate · 1[π_A1^c = affect] · 1[v < v₇₅]", "replaced", fs=9)
    box(ax, 0.435, y2 + 0.01, 0.18, 0.17, "fused score\nz(B) + λ_u·z(B) + λ_a·g·z(T^c),\n224 cells; integer\nmin-margin cross-fit",
        "unchanged")
    box(ax, 0.645, y2 + 0.01, 0.17, 0.17, "matched counterpart\nG_cf under the\ncandidate's own gates,\nmax-R@1 cross-fit", "unchanged")
    box(ax, 0.845, y2 + 0.01, 0.15, 0.17, "comparators: B,\nB′(A0), the counterpart\n(cosine, RCA\nfor context)", "same")
    arrow(ax, 0.30, y1 - 0.012, 0.20, y2 + 0.19 + 0.012)     # pick and margin -> gate
    arrow(ax, 0.515, y1 - 0.012, 0.33, y2 + 0.19 + 0.012)    # v -> gate
    arrow(ax, 0.72, y1 - 0.012, 0.38, y2 + 0.19 + 0.012)     # A1 pick -> gate
    arrow(ax, 0.92, y1 - 0.012, 0.92, y2 + 0.18 + 0.012)     # B'(A1) -> comparators
    for x0, x1 in ((0.405, 0.435), (0.615, 0.645), (0.815, 0.845)):
        arrow(ax, x0, y2 + 0.095, x1, y2 + 0.095)
    y3 = 0.06
    box(ax, 0.005, y3, 0.30, 0.20, "development bar (round 2's): bar margin\n≥ +0.5 against the bar comparator,\n"
        "its lower bound > 0, gain statistic's\nlower bound > 0", "same")
    box(ax, 0.335, y3, 0.32, 0.20, "carry: clears the bar AND beats AFF,\nΔ_k = Σ 4·(candidate − AFF) > 0\n"
        "(integer, paired per anchor);\ntie band 24; order V4, V2, V24", "new")
    box(ax, 0.685, y3, 0.31, 0.20, "fresh-seed test on seeds 52 to 54\nwith a paired GO check against AFF\n"
        "(planned; not run: the carry\nset was empty, so the rule killed)", "new", ls="--")
    arrow(ax, 0.305, y3 + 0.10, 0.335, y3 + 0.10)
    arrow(ax, 0.655, y3 + 0.10, 0.685, y3 + 0.10)
    handles = [Patch(fc="#ecebe8", ec="#8a8984", label="grey: same as rounds 2 and 3"),
               Patch(fc="#e3e0f4", ec="#4a3aa7", label="purple: unchanged from AFF"),
               Patch(fc="#fde3d7", ec="#eb6834", label="orange: replaced (the gate)"),
               Patch(fc="#d3f1e6", ec="#12876a", label="teal: new in round 4 (dashed: planned, not run)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("What round 4 changed: each candidate is AFF with one or two veto factors in the gate",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.925, "A veto can only close AFF's gate, never open it; the reader, term, thresholds, cells and "
             "cross-fit rules are AFF's", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.06)
    fig.savefig(OUT / "what_changed.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: the development step

def fig_development(rec, bes):
    comp_label = {"Bprime_A0": "B′(A0)", "Bprime_A1": "B′(A1)"}
    left = [("AFF", rec["AFF"]["bar_margin"], "AFF vs B′(A0)\n(reference, round 3)"),
            ("V4", rec["V4"]["bar_margin"], f"V4 vs {comp_label[rec['V4']['bar_comparator']]}"),
            ("V2", rec["V2"]["bar_margin"], f"V2 vs {comp_label[rec['V2']['bar_comparator']]}"),
            ("V24", rec["V24"]["bar_margin"], f"V24 vs {comp_label[rec['V24']['bar_comparator']]}"),
            ("AFF", bes["AFF_minus_Bprime_A1"], "AFF vs B′(A1)\n(descriptive)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.2, wspace=0.55)
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
    ax.set_xlim(-0.12, 1.3)
    ax.set_xlabel("Bar margin, fused R@1 minus the bar comparator (pp)")
    ax.set_title("D10: bar margin against each candidate's floor", fontsize=10, loc="left", color=INK2)

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
        ax.text(1.0, y + 0.2, f"Δ = {rec[who]['delta_int']:+d} rankings   {fmt(v)}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{w} minus AFF" for w in CANDS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys.min() - 0.6, 0.75)
    ax.set_xlim(-0.2, 0.2)
    ax.set_xlabel("Δ_k: fused R@1, candidate minus AFF, paired per anchor (pp)")
    ax.set_title("Carry condition: Δ_k > 0 (none)", fontsize=10, loc="left", color=INK2)
    handles = [Line2D([0], [0], color=C[w], lw=2.2, marker=MK[w], ms=7, label=w) for w in CANDS] + \
              [Line2D([0], [0], color=C["AFF"], lw=2.2, marker="D", ms=7, mfc="white", mew=1.8, label="AFF (reference)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Seed 42: V4 cleared the bar but lost to AFF; V2 and V24 fell short of the bar against B′(A1)",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.88, "95% painting-bootstrap intervals (5,000 resamples). One rule outcome: the carry set E is "
             "empty, so the round was killed (rule §5 item 9).", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "development.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3: candidate minus AFF, per pair

def fig_per_pair(desc):
    pm = desc["minus_AFF"]
    cats = ["pooled"] + list(PAIRS)
    labels = ["pooled\n(= Δ_k)"] + [PAIR_LABEL[p] for p in PAIRS]
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.2)
    xs = np.arange(len(cats), dtype=float)
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    offs = {"V4": -0.2, "V2": 0.0, "V24": 0.2}
    for who in CANDS:
        for x, cat in zip(xs, cats):
            v = pm[who][cat]
            ax.plot([x + offs[who]] * 2, v["ci95"], color=C[who], lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot(x + offs[who], v["point"], marker=MK[who], ms=8, color=C[who], zorder=4)
            ax.annotate(f"{v['point']:+.3f}", (x + offs[who], v["ci95"][1]), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.axvline(0.5, color=GRID, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylabel("Fused R@1, candidate minus AFF (pp)")
    handles = [Line2D([0], [0], color=C[w], lw=2.2, marker=MK[w], ms=7, label=w) for w in CANDS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Every veto gained a little on style × genre and lost more on the two emotion pairs",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.9, "Seed 42, paired per anchor, 95% painting-bootstrap intervals; per-pair numbers are "
             "descriptive (4,096 episodes per pair)", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "per_pair_vs_aff.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4: closure shares

def fig_closure(desc):
    clo = desc["closure"]
    groups = [(p, c) for p in PAIRS for c in "ab"]
    side = {("emotion__style", "a"): "emotion", ("emotion__style", "b"): "style",
            ("emotion__genre", "a"): "emotion", ("emotion__genre", "b"): "genre",
            ("style__genre", "a"): "style", ("style__genre", "b"): "genre"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.80, bottom=0.27, wspace=0.05)
    for ax, t, title in ((axes[0], "0", "τ_0 (cell 39, scores the parity-1 half)"),
                         (axes[1], "2", "τ_2 (cell 119, scores the parity-0 half)")):
        xs = np.arange(len(groups), dtype=float)
        w = 0.26
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for j, who in enumerate(CANDS):
            vals = [clo[who][t][f"{p}|{c}"]["share"] for p, c in groups]
            ax.bar(xs + (j - 1) * w, vals, width=w - 0.03, color=C[who], zorder=3, label=who)
            for x, v in zip(xs + (j - 1) * w, vals):
                ax.text(x, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{PAIR_LABEL[p]}\ncondition {c}\n({side[(p, c)]} side)\nAFF open {clo['V4'][t][f'{p}|{c}']['aff_open']:,}"
                            for p, c in groups], fontsize=7.6)
        ax.tick_params(axis="x", length=0)
        ax.axvline(1.5, color=GRID, lw=1.0), ax.axvline(3.5, color=GRID, lw=1.0)
        ax.set_title(title, fontsize=10, loc="left", color=INK2)
        ax.set_ylim(0, 72)
    axes[0].set_ylabel("Share of AFF's open values the veto closes (%)")
    handles = [Patch(fc=C[w], label=w) for w in CANDS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("V2 closed mostly non-emotion values; V4 closed every side at similar rates",
                 fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.885, "Seed 42. Among the (episode, condition) values where AFF's gate is open, the share each "
             "candidate's gate closes; the side shown is what the supports share under that condition (labels used "
             "only to group)", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "closure.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5: what V4 closed

def fig_v4_closed(desc):
    d = desc["V4_closed_vs_kept_AFF_minus_B"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.2)
    xs = np.arange(len(PAIRS), dtype=float)
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for off, k, mfc, lab in ((-0.13, "closed", C["V4"], "episodes V4 closed (V4 scores them exactly as B)"),
                             (0.13, "kept", "white", "episodes V4 kept open")):
        for x, p in zip(xs, PAIRS):
            v = d[p][k]
            ax.plot([x + off] * 2, v["ci95"], color=C["V4"], lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot(x + off, v["point"], marker="o", ms=8, color=C["V4"], mfc=mfc, mew=1.8, zorder=4)
            ax.annotate(f"{v['point']:+.2f}\n(n {v['episodes']:,})", (x + off, v["ci95"][1]), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([PAIR_LABEL[p] for p in PAIRS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(PAIRS) - 0.4)
    ax.set_ylim(-1.8, 3.4)
    ax.set_ylabel("AFF's fused R@1 minus B (pp)")
    handles = [Line2D([0], [0], color=C["V4"], lw=2.2, marker="o", ms=7, label="episodes V4 closed (scored as B there)"),
               Line2D([0], [0], color=C["V4"], lw=2.2, marker="o", ms=7, mfc="white", mew=1.8,
                      label="episodes V4 kept open")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Where V4 closed the gate, AFF's steering had paid in the emotion pairs and cost about the usual "
                 "amount in style × genre", fontsize=11.5, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.89, "Seed 42, as scored (τ_0 on parity 1, τ_2 on parity 0); what AFF's steering bought over B on "
             "each set, with 95% painting-bootstrap intervals", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "v4_closed_vs_kept.png", dpi=DPI)
    plt.close(fig)


def main():
    rec, bes, desc = data()
    print("re-derived decision quantities equal dev_seed42.json and carry.json (exact)")
    for n in CANDS + ("AFF",):
        r = rec[n]
        print(n, f"fused {r['fused_r1']:.4f} cf {r['cf_r1']:.4f}", r["bar_comparator"], "bar", fmt(r["bar_margin"]),
              "gain", fmt(r["gain_statistic"]), "either", round(r["either_change"], 3), r["d10"],
              "delta", r.get("delta_int"), fmt(r["delta"]) if "delta" in r else "")
    fig_what_changed()
    fig_development(rec, bes)
    fig_per_pair(desc)
    fig_closure(desc)
    fig_v4_closed(desc)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
