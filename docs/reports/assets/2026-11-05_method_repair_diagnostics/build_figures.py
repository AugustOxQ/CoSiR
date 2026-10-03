"""Figures and figure data for the method-repair diagnostics report
(docs/reports/auto/v2/2026-11-05_method_repair_diagnostics.md).

Reads the stage's stored records and arrays (src/test/20261105_method_repair_diagnostics/results/):
  pilot_seed42.json, per_anchor_pilot_seed42.npz       H1 pilot (seed-42 selection episodes)
  h3.json, per_anchor_h3.npz, h3_pass1_needs_seed43.json   H3 (fresh label episodes, seed-42 transfer, matched-k)
  decision.json, build_record.json, label_checkpoints.json, history_<run>_seed<seed>.json
and, for context, E1's per_anchor_seed42.npz (cosine, RCA), E3's select_seed42.json, per_anchor_select_seed42.npz,
posthoc_lambda_profile_seed42.npz and history_A3_seed42.json. Seed 43, 44 and 45 episode files are never opened, and
val and held rows are never read (every selection-row array is NaN elsewhere, asserted by E3's EvalContext).

Re-derivations (CPU, deterministic):
  * every summary and paired comparison the report quotes, recomputed with src.eval.aspect_metrics (painting-clustered
    bootstrap, 5,000 resamples, seed 42) and asserted equal to the stored JSON (points and interval bounds to 1e-9);
  * A3's 56-cell uncross-fitted nested profile and the control's 30 sigma values, recomputed from the A3 checkpoint on
    the seed-42 selection episodes (the same look the pilot stored as pooled means, recomputed, not a new look) and
    asserted equal to pilot_seed42.json["profile_A3"]; A3's cross-fitted nested and control arrays and picks are
    recomputed too and asserted bit-equal to the stored arrays;
  * the fresh label episodes are rebuilt with the pre-registered seeds (3042 + pair index) and their SHA-256s asserted
    equal to h3.json; C0's and L5 seed 43's term-only arrays on them (not stored) are recomputed from the checkpoints;
  * A3's seed-42 term-only arrays (not stored in per_anchor_h3.npz) are recomputed and asserted bit-equal to E3's
    post-hoc profile at lambda = inf.
Numbers with no stored counterpart (marked "computed_here" in figure_data.json) are descriptive and decide nothing.

Run from the repository root:
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python \
      docs/reports/assets/2026-11-05_method_repair_diagnostics/build_figures.py
Writes h1_profile.png, h1_tradeoff.png, h3_fit.png, h3_transfer.png, h3_training.png and figure_data.json here.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DIAG = ROOT / "src/test/20261105_method_repair_diagnostics"
RES = DIAG / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIAG))
from common import C0_CKPT, FRESH_LABEL_SEED, LAB_PARTS, N_FRESH, folders, local_rows, rg  # noqa: E402

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import (PaintingValueIndex, build_aspect_episodes, concat_episodes,  # noqa: E402
                                      episodes_sha256, validate_aspect_episodes)
from src.eval.aspect_metrics import METRICS, compare, per_anchor, summarize  # noqa: E402
from src.eval.aspect_nested import (NESTED_A, NESTED_U, ceiling_threshold, control_scores, control_sums,  # noqa: E402
                                    crossfit_nested, fit_reading, h3_reading, joint_decision, margin_reading,
                                    nested_cells, nested_scores, predicted_power, se_from_ci)
from src.eval.aspect_scorers import EvalInputs, agreement_term  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

E1 = ROOT / "src/test/20261030_aspect_baselines/results"
E3 = ROOT / "src/test/20261101_aspect_factor_gonogo/results"
MODELS = ("A3", "A1", "A2", "A4", "A5", "A6", "C0", "SE")
LABEL_RUNS = ("L3", "L5", "LT")
TRANSFER_RUNS = ("L3", "L5", "LT", "MK3")
FRESH_PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))
CONST = {True: 3.258, False: 2.565}          # pre-registered constant-score aspect loss (with / without swap term)
TOL = 1e-9

# Colours: reference categorical palette (dataviz skill, light mode, adjacent order validated), grey for the cosine.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_COS = "#6b6a66"
COL = {"A3": "#2a78d6", "L3": "#eb6834", "L5": "#1baf7a", "LT": "#eda100", "MK3": "#e87ba4"}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORD_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#104281", "#0d366b"]   # lambda_u rows, ordinal
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("red_grey_blue", ["#c43c3b", "#f0efec", "#2a78d6", "#0d366b"])

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False})

CHECKS: list = []
DATA: dict = {"note": ("values in percentage points; CIs are 95% painting-clustered bootstrap intervals (5,000 "
                       "resamples, seed 42) recomputed by build_figures.py. Keys named computed_here have no stored "
                       "counterpart: descriptive, computed for the report, deciding nothing.")}


def check(cond, what):
    assert cond, f"check failed: {what}"
    CHECKS.append(what)


def same(a, b, what):
    """Assert a recomputed {point, ci95} equals a stored one (to 1e-9)."""
    ok = abs(a["point"] - b["point"]) < TOL and all(abs(x - y) < TOL for x, y in zip(a["ci95"], b["ci95"]))
    check(ok, f"{what}: recomputed {a['point']:.6f} {a.get('ci95')} vs stored {b['point']:.6f} {b.get('ci95')}")


def same_val(a, b, what):
    check(abs(a - b) < TOL, f"{what}: recomputed {a} vs stored {b}")


def same_arrays(pa, pb, what):
    check(all(np.array_equal(np.asarray(pa[m]), np.asarray(pb[m])) for m in METRICS), f"{what}: per-anchor arrays")


def brief(s):
    return {"point": s["point"], "ci95": list(s["ci95"])}


def either_ci(arr, cl):
    v = compare({"e": arr}, {"e": np.zeros_like(arr)}, cl, "e")
    return {"point": v["point"], "ci95": v["ci95"]}


def either_of(pa):
    return np.asarray(pa["r1"], np.float64) + np.asarray(pa["other"], np.float64)


def per(z, prefix):
    return {m: np.asarray(z[f"{prefix}__{m}"], dtype=np.float64) for m in METRICS}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha(rev, rel):
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{rel}"], capture_output=True, check=True).stdout
    return hashlib.sha256(blob).hexdigest()


torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
pilot = json.loads((RES / "pilot_seed42.json").read_text())
h3 = json.loads((RES / "h3.json").read_text())
h3p1 = json.loads((RES / "h3_pass1_needs_seed43.json").read_text())
decision = json.loads((RES / "decision.json").read_text())
build = json.loads((RES / "build_record.json").read_text())
labels_ck = json.loads((RES / "label_checkpoints.json").read_text())

# ------------------------------------------------------------------------------------------------ provenance
check(not any(p for p in ROOT.joinpath("src/test").rglob("*") if "seed45" in p.name or "seed_45" in p.name),
      "no seed-45 file exists under src/test")
for f in ("bank_LAB.npz", "partitions_LAB.npz", "bank_MK.npz", "partitions_MK.npz"):
    check(sha(RES / f) == build["sha256"][f], f"{f} SHA-256 equals build_record.json")
for name, s in labels_ck.items():
    check(sha(DIAG / "checkpoints" / f"{name}.pt") == s, f"LAB checkpoint {name} SHA-256 equals label_checkpoints.json")
for f, key in (("pilot_seed42.json", "pilot_seed42.json"), ("h3.json", "h3.json")):
    check(sha(RES / f) == decision["inputs_sha256"][key], f"decision.json input SHA-256 of {f}")
rel = "src/test/20261105_method_repair_diagnostics"
check(git_blob_sha("83e7027", f"{rel}/score_pilot.py") == pilot["provenance"]["script_sha256"],
      "pilot script SHA-256 equals score_pilot.py at commit 83e7027")
check(git_blob_sha("3afe1bd", f"{rel}/score_h3.py") == h3["provenance"]["script_sha256"],
      "H3 script SHA-256 equals score_h3.py at commit 3afe1bd")
check(git_blob_sha("HEAD", f"{rel}/score_pilot.py") == pilot["provenance"]["script_sha256"]
      and git_blob_sha("HEAD", f"{rel}/score_h3.py") == h3["provenance"]["script_sha256"],
      "scorer scripts unchanged at HEAD")
pick_e3 = json.loads((E3 / "picked.json").read_text())
a3_ckpt = rg.checkpoint_path("A3", 42, False)
check(sha(a3_ckpt) == pick_e3["checkpoint_sha256"] == pilot["A3"]["provenance"]["sha256"]
      == h3["provenance"]["a3_checkpoint_sha256"], "A3 checkpoint SHA-256 equals E3's pick, the pilot and H3")
hist = {k: json.loads((RES / f"history_{k}.json").read_text())
        for k in ("L3_seed42", "L5_seed42", "L5_seed43", "LT_seed42", "MK3_seed42")}
hist["A3_E3"] = json.loads((E3 / "history_A3_seed42.json").read_text())
for k, v in hist.items():
    if k in labels_ck:
        check(v["checkpoint_sha256"] == labels_ck[k], f"history {k} checkpoint SHA-256 equals the registry")
check("MK3_seed42" not in labels_ck, "MK3 is not a LAB checkpoint")
DATA["provenance"] = {"pilot_script_commit": "83e7027", "h3_script_commit": "3afe1bd",
                      "a3_checkpoint_sha256": pick_e3["checkpoint_sha256"], "label_checkpoints": labels_ck,
                      "decision": decision, "build_record_banks": build["banks"], "mk_ami": build["mk_ami"],
                      "eligible_values": build["eligible_values"],
                      "train": {k: {"train_s": v["train_s"], "peak_gpu_gib": v["peak_gpu_gib"],
                                    "code_stats": v["code_stats_scorer_train"]} for k, v in hist.items()}}

# ------------------------------------------------------------------------------------------------ seed-42 context
ctx = rg.EvalContext(42, False)
check(ctx.shas == pilot["provenance"]["episodes_sha256"] == h3["provenance"]["seed42_episodes_sha256"],
      "seed-42 episode SHA-256s equal the pilot's and H3's")
cl = ctx.anchor_group
check(len(np.unique(cl)) == pilot["provenance"]["n_clusters"] == 4602, "seed-42 painting clusters = 4,602")
t42 = np.load(E1 / "per_anchor_seed42.npz")
cos_pa, rca = per(t42, "cosine"), per(t42, "rca")
same_arrays(per_anchor(ctx.cos), cos_pa, "cosine recomputed vs E1")
cos_sum, rca_sum = ctx.summary(cos_pa), ctx.summary(rca)
cos_either = either_ci(either_of(cos_pa), cl)
check(round(cos_sum["r1"]["point"], 2) == 12.96 and round(rca_sum["r1"]["point"], 2) == 13.38,
      "cosine 12.96 and RCA 13.38 R@1 (E1, E3 report)")
e3sel = json.loads((E3 / "select_seed42.json").read_text())["runs"]["A3"]["overall"]
g42 = np.load(E3 / "per_anchor_select_seed42.npz")
e3_a3 = per(g42, "A3")
e3_a3_sum = ctx.summary(e3_a3)
for m in METRICS:
    same(e3_a3_sum[m], e3sel[m], f"E3 cross-fitted A3 seed-42 {m}")
check(json.loads((E3 / "select_seed42.json").read_text())["runs"]["A3"]["uniform_lambda_picks"] == {"0": 8.0, "1": 8.0},
      "E3's A3 uniform-weight control picked weight 8 on both halves of seed 42")
DATA["context_seed42"] = {"cosine": {m: brief(cos_sum[m]) for m in ("r1", "gain")}, "cosine_either": cos_either,
                          "rca": {m: brief(rca_sum[m]) for m in ("r1", "gain")},
                          "E3_A3_crossfit": {m: brief(e3_a3_sum[m]) for m in ("r1", "gain", "other")},
                          "E3_A3_crossfit_either": either_ci(either_of(e3_a3), cl)}

# ------------------------------------------------------------------------------------------------ H1 pilot
zp = np.load(RES / "per_anchor_pilot_seed42.npz")
h1 = {}
for name in MODELS:
    e = pilot[name]
    pn, pc = per(zp, f"{name}__nested"), per(zp, f"{name}__control")
    sn, sc = ctx.summary(pn), ctx.summary(pc)
    for m in METRICS:
        same(sn[m], e["nested"][m], f"pilot {name} nested {m}")
        same(sc[m], e["control"][m], f"pilot {name} control {m}")
    en, ec = either_ci(either_of(pn), cl), either_ci(either_of(pc), cl)
    same(en, e["either"]["nested"], f"pilot {name} either nested")
    same(ec, e["either"]["control"], f"pilot {name} either control")
    same(cos_either, e["either"]["cosine"], f"pilot {name} either cosine")
    row = {"nested": {m: brief(sn[m]) for m in ("r1", "gain", "other", "swap")},
           "control": {m: brief(sc[m]) for m in ("r1", "gain")}, "either_nested": en, "either_control": ec,
           "picks": e["picks"]}
    for ref, pb in (("vs_control", pc), ("vs_cosine", cos_pa), ("vs_rca", rca)):
        row[ref] = {}
        for m in ("r1", "gain"):
            r = compare(pn, pb, cl, m)
            same(r, e[ref][m], f"pilot {name} {ref} {m}")
            row[ref][m] = brief(r)
    h1[name] = row
a3e = pilot["A3"]
vr, vg = h1["A3"]["vs_control"]["r1"], h1["A3"]["vs_control"]["gain"]
m_r, se_r, m_g, se_g = vr["point"], se_from_ci(vr["ci95"]), vg["point"], se_from_ci(vg["ci95"])
for k, v in (("m_R", m_r), ("SE_R", se_r), ("m_g", m_g), ("SE_g", se_g)):
    same_val(v, a3e[k], f"pilot A3 {k}")
reading = margin_reading(m_r, se_r, m_g, se_g)
check(reading == a3e["reading"] == "not_promising", "H1 reading = not_promising")
g_star = ceiling_threshold(se_r, se_g)
same_val(g_star, a3e["g_star"], "g*")
same_val(g_star, h3["g_star"], "g* carried into h3.json")
check(abs(g_star - 2 * 2.80 * se_r) < TOL and 2 * 2.80 * se_r > 2.80 * se_g, "g* = 5.6 SE_R (the R@1 branch binds)")
power = {"r1": predicted_power(m_r, se_r), "gain": predicted_power(m_g, se_g)}
power["joint_if_independent"] = power["r1"] * power["gain"]
for k in power:
    same_val(power[k], a3e["predicted_power"][k], f"predicted power {k}")
a3c0 = {m: compare(per(zp, "A3__nested"), per(zp, "C0__nested"), cl, m) for m in ("r1", "gain")}
for m in a3c0:
    same(a3c0[m], pilot["A3_minus_C0_nested"][m], f"A3 minus C0 nested {m}")
# A3 nested vs E3's cross-fitted A3 (old score): computed here
a3_vs_e3 = {m: brief(compare(per(zp, "A3__nested"), e3_a3, cl, m)) for m in ("r1", "gain")}
DATA["h1"] = {"rows": h1, "A3": {"m_R": m_r, "SE_R": se_r, "m_g": m_g, "SE_g": se_g, "reading": reading,
                                 "g_star": g_star, "predicted_power": power,
                                 "A3_minus_C0_nested": {m: brief(a3c0[m]) for m in a3c0}},
              "computed_here": {"A3_nested_minus_E3_crossfit_A3": a3_vs_e3}}

# A3 from its checkpoint: terms, cross-fit, 56-cell profile (deterministic recomputation of the stored look)
ic, tc = ctx.encode(a3_ckpt)
inp = EvalInputs(ctx.img, ctx.txt, ic, tc)
tu = agreement_term(inp, ctx.pooled, uniform=True)
ta = agreement_term(inp, ctx.pooled)
nested, control, picks = crossfit_nested(ctx.cos, tu, ta, ctx.parity)
pn_a3, pc_a3 = per_anchor(nested), per_anchor(control)
same_arrays(pn_a3, per(zp, "A3__nested"), "A3 cross-fitted nested recomputed vs stored")
same_arrays(pc_a3, per(zp, "A3__control"), "A3 cross-fitted control recomputed vs stored")
same_arrays(per(zp, "A3__control"), per(g42, "A3_uniform"), "A3 nested control equals E3's uniform-weight control")
check({str(k): v for k, v in picks.items()} == a3e["picks"], "A3 picks recomputed equal the stored picks")
half0 = ctx.parity == 0
check(all(np.array_equal(pn_a3[m][half0], pc_a3[m][half0]) for m in METRICS),
      "on the half scored by the (8, 0) pick, A3 nested equals its control anchor by anchor")
d1 = {m: float(100 * (pn_a3[m][~half0] - pc_a3[m][~half0]).mean()) for m in ("r1", "gain")}

prof = pilot["profile_A3"]
cells, ctrl = {}, {}
for u, a in nested_cells():
    pa = per_anchor(nested_scores(ctx.cos, tu, ta, u, a))
    v = {"r1": 100 * float(pa["r1"].mean()), "gain": 100 * float(pa["gain"].mean()),
         "either": 100 * float((pa["r1"] + pa["other"]).mean())}
    for m in v:
        same_val(v[m], prof["cells"][f"{u:g},{a:g}"][m], f"profile cell ({u:g}, {a:g}) {m}")
    v["half"] = {h: {"r1": 100 * float(pa["r1"][ctx.parity == h].mean()),
                     "gain": 100 * float(pa["gain"][ctx.parity == h].mean())} for h in (0, 1)}
    cells[(u, a)] = v
for s in control_sums():
    pa = per_anchor(control_scores(ctx.cos, tu, s))
    v = 100 * float(pa["r1"].mean())
    same_val(v, prof["control_r1"][f"{s:g}"], f"profile control sigma {s:g} R@1")
    ctrl[s] = {"r1": v, "half": {h: 100 * float(pa["r1"][ctx.parity == h].mean()) for h in (0, 1)}}
check(len(cells) == 56 and len(ctrl) == 30, "56 profile cells and 30 control sums")
# the cross-fit's choice reproduced from the half means
for h in (0, 1):
    sig = max(control_sums(), key=lambda s: ctrl[s]["half"][h])
    crit = {c: min(cells[c]["half"][h]["r1"] - ctrl[sig]["half"][h], cells[c]["half"][h]["gain"])
            for c in nested_cells()}
    best = max(nested_cells(), key=lambda c: crit[c])
    check(sig == a3e["picks"][str(h)]["sigma"] and list(best) == a3e["picks"][str(h)]["cell"],
          f"half {h}: picks reproduced from half means")
    DATA.setdefault("h1_crossfit_halves", {})[h] = {
        "sigma": sig, "control_r1_half": ctrl[sig]["half"][h], "cell": list(best), "criterion": crit[best],
        "cell_r1_half": cells[best]["half"][h]["r1"], "cell_gain_half": cells[best]["half"][h]["gain"],
        "n_cells_criterion_positive": int(sum(v > 0 for v in crit.values())),
        "top3": sorted(([list(c), crit[c]] for c in crit), key=lambda x: -x[1])[:3]}
best_sigma = max(control_sums(), key=lambda s: ctrl[s]["r1"])
best_ctrl = ctrl[best_sigma]["r1"]
check(best_sigma == 8.0, "the control's best pooled R@1 is at sigma 8")
above = [c for c in cells if cells[c]["r1"] > best_ctrl + TOL]
check(not above, "no nested cell's pooled R@1 exceeds the control's best")
crit_pooled = {c: min(cells[c]["r1"] - best_ctrl, cells[c]["gain"]) for c in cells}
# trade-off per row: change from the row's lambda_a = 0 cell
trade = {}
for u in NESTED_U:
    base = cells[(u, 0.0)]
    trade[u] = [{"lambda_a": a, "d_gain": cells[(u, a)]["gain"] - base["gain"],
                 "d_either": cells[(u, a)]["either"] - base["either"], "d_r1": cells[(u, a)]["r1"] - base["r1"]}
                for a in NESTED_A[1:]]
rises = [(u, t["lambda_a"], t["d_r1"]) for u in trade for t in trade[u] if t["d_r1"] > 0]
rises_high = [(u, a, d) for u, a, d in rises if u >= 1]
max_r1_low = max(cells[(u, a)]["r1"] for u in NESTED_U if u <= 0.5 for a in NESTED_A)
DATA["h1_profile"] = {
    "cells": {f"{u:g},{a:g}": {k: cells[(u, a)][k] for k in ("r1", "gain", "either")} for u, a in cells},
    "control_r1": {f"{s:g}": ctrl[s]["r1"] for s in ctrl}, "control_best_sigma": best_sigma,
    "control_best_r1": best_ctrl, "cells_above_control_best": [list(c) for c in above],
    "best_pooled_min_margin": {"cell": list(max(crit_pooled, key=crit_pooled.get)),
                               "value": max(crit_pooled.values())},
    "cells_with_r1_rise_over_row_start": [[u, a, d] for u, a, d in rises],
    "rises_in_rows_lambda_u_ge_1": [[u, a, d] for u, a, d in rises_high],
    "max_r1_rows_lambda_u_le_0.5": max_r1_low,
    "max_gain_cell": list(max(cells, key=lambda c: cells[c]["gain"])),
    "max_gain": max(v["gain"] for v in cells.values()),
    "half1_nested_minus_control": d1, "trade": {f"{u:g}": trade[u] for u in trade}}

# ------------------------------------------------------------------------------------------------ H3: fresh label episodes
data = ctx.data
st, local_groups = local_rows(data, artelingo_splits(data))
n = len(st)
check(n == h3["provenance"]["n_scorer_train_rows"] == 183694, "183,694 scorer-train rows")
z = np.load(RES / "partitions_LAB.npz")
lab = {k: z[k] for k in z.files if k in LAB_PARTS}
groups = z["local_groups"]
check(np.array_equal(groups, local_groups), "partitions_LAB local groups equal the scorer-train painting groups")
index = PaintingValueIndex(lab, groups)
parts = []
for i, (a, b, third) in enumerate(FRESH_PAIRS):
    ep = build_aspect_episodes(lab, groups, np.arange(n), a, b, N_FRESH, FRESH_LABEL_SEED + i, third=third,
                               min_paintings=30, index=index)
    validate_aspect_episodes(ep, lab, groups, index, third=third)
    check(episodes_sha256(ep) == h3["fresh"]["episodes"][f"{a}__{b}"]["sha256"],
          f"fresh label episodes {a}__{b} rebuilt (seed {FRESH_LABEL_SEED + i}) with the stored SHA-256")
    parts.append(ep)
fresh = concat_episodes(parts)
fcl = groups[fresh.anchor]
check(len(np.unique(fcl)) == h3["fresh"]["n_clusters"] == 9900, "fresh label episodes: 9,900 painting clusters")


def term_fresh(ckpt):
    model, _ = load_factor_checkpoint(ckpt, device="cpu")
    i_c, t_c = encode_rows(model, data.img_features, data.txt_features, rows=st, device="cpu")
    assert np.isfinite(i_c).all() and np.isfinite(t_c).all()
    return per_anchor(agreement_term(EvalInputs(data.img_features[st], data.txt_features[st], i_c, t_c), fresh))


zh = np.load(RES / "per_anchor_h3.npz")
fr = {r: per(zh, f"fresh__{r}") for r in ("A3",) + LABEL_RUNS}
same_arrays(term_fresh(a3_ckpt), fr["A3"], "A3 fresh term-only recomputed vs stored")
fr["C0"] = term_fresh(C0_CKPT)
check(sha(DIAG / "checkpoints" / "L5_seed43.pt") == h3["runs"]["L5"]["seed43"]["checkpoint_sha256"],
      "L5 seed-43 checkpoint SHA-256 equals h3.json")
fr["L5_s43"] = term_fresh(DIAG / "checkpoints" / "L5_seed43.pt")
h3fit = {}
for r in LABEL_RUNS:
    e = h3["runs"][r]
    c = compare(fr[r], fr["A3"], fcl, "gain")
    same(c, e["vs_A3_gain"], f"H3 fit {r} minus A3 gain")
    same(c, h3p1["runs"][r]["vs_A3_gain"], f"H3 pass 1 {r} minus A3 gain unchanged")
    rd = fit_reading(c)
    check(rd == e["reading"], f"H3 fit reading {r} = {rd}")
    xc0 = {m: compare(fr[r], fr["C0"], fcl, m) for m in ("r1", "gain")}
    for m in xc0:
        same(xc0[m], e["X_minus_C0"][m], f"H3 {r} minus C0 {m}")
    hk = json.loads((RES / f"history_{r}_seed42.json").read_text())
    last = np.asarray(hk["history"]["aspect_loss"][-10:], dtype=np.float64)
    const = CONST[hk["config"]["lambda_swap"] > 0]
    same_val(float(last.mean()), e["loss_vs_constant"]["last10_mean"], f"{r} loss over the last 10 logs")
    same_val(float(100 * (const - last.mean()) / const), e["loss_vs_constant"]["pct_below_constant"],
             f"{r} loss percent below constant")
    same_val(float(hk["history"]["tau"][0]), e["tau"]["first"], f"{r} tau first")
    same_val(float(hk["history"]["tau"][-1]), e["tau"]["last"], f"{r} tau last")
    h3fit[r] = {"vs_A3_gain": brief(c), "reading_seed42": rd, "resolved": e["resolved"],
                "X_minus_C0": {m: brief(xc0[m]) for m in xc0}, "loss_vs_constant": e["loss_vs_constant"],
                "tau": e["tau"]}
avg = {m: 0.5 * (fr["L5"][m] + fr["L5_s43"][m]) for m in METRICS}
c2 = compare(avg, fr["A3"], fcl, "gain")
same(c2, h3["runs"]["L5"]["seed43"]["vs_A3_gain_two_seed_mean"], "L5 two-seed mean minus A3 gain")
check(("fits" if c2["ci95"][0] > 0 else "no_fit") == h3["runs"]["L5"]["resolved"] == "no_fit", "L5 resolved no_fit")
check(h3p1["needs_seed43"] == ["L5"] and h3p1["h3_reading"] is None, "pass 1 recorded L5 as needing seed 43")


def tree_diff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        return [d for k in sorted(set(x) | set(y))
                for d in ([f"{path}/{k}"] if k not in x or k not in y else tree_diff(x[k], y[k], f"{path}/{k}"))]
    return [] if x == y else [path]


check(sorted(tree_diff(h3p1, h3)) == ["/fit/L5", "/h3_reading", "/needs_seed43", "/runs/L5/resolved",
                                      "/runs/L5/seed43"],
      "pass 1 and final h3.json differ only in L5's seed-43 entry, its resolution, the H3 reading and needs_seed43")
h3fit["L5"]["two_seed_mean_vs_A3_gain"] = brief(c2)
h3fit["L5"]["computed_here"] = {"seed43_vs_A3_gain": brief(compare(fr["L5_s43"], fr["A3"], fcl, "gain"))}
fresh_sum = {k: {m: brief(v) for m, v in summarize(fr[k], fcl).items() if m in ("r1", "gain", "other")}
             for k in fr}
DATA["h3_fit"] = {"runs": h3fit, "computed_here": {"fresh_term_only_summaries": fresh_sum}}

# ------------------------------------------------------------------------------------------------ H3: seed-42 transfer
pt_a3 = per_anchor(ta)                                   # A3 term-only (rank-equivalent to lambda = inf)
ph = np.load(E3 / "posthoc_lambda_profile_seed42.npz")
check(all(np.array_equal(pt_a3[m], ph[f"A3__agreement__inf__{m}"]) for m in ("r1", "gain", "other")),
      "A3 term-only seed 42 recomputed equals E3's post-hoc lambda = inf arrays")
tr = {"A3": {"term_only": ctx.summary(pt_a3), "term_only_either": either_ci(either_of(pt_a3), cl),
             "nested": ctx.summary(per(zp, "A3__nested"))}}
check(round(tr["A3"]["term_only"]["gain"]["point"], 2) == 0.99 and
      round(tr["A3"]["term_only_either"]["point"], 2) == 21.04, "A3 term-only seed 42 = 0.99 / 21.04 (E3 report)")
term = {"A3": pt_a3}
for r in TRANSFER_RUNS:
    e = h3["runs"][r]["transfer_seed42"]
    pt, pn = per(zh, f"transfer__{r}__termonly"), per(zh, f"transfer__{r}__nested")
    term[r] = pt
    st_, sn_ = ctx.summary(pt), ctx.summary(pn)
    for m in METRICS:
        same(st_[m], e["term_only"][m], f"transfer {r} term-only {m}")
        same(sn_[m], e["nested"][m], f"transfer {r} nested {m}")
    ei = either_ci(either_of(pt), cl)
    same(ei, e["term_only_either"], f"transfer {r} term-only either")
    same_val(sn_["gain"]["point"], e["nested_gain_point"], f"transfer {r} nested gain point")
    if r in h3p1["runs"] and "transfer_seed42" in h3p1["runs"][r]:
        same(st_["gain"], h3p1["runs"][r]["transfer_seed42"]["term_only"]["gain"], f"pass 1 transfer {r} unchanged")
    tr[r] = {"term_only": st_, "term_only_either": ei, "nested": sn_, "picks": e["picks"],
             "nested_either": either_ci(either_of(pn), cl)}
mk = compare(term["MK3"], pt_a3, cl, "gain")
same(mk, h3["matched_k"]["MK3_minus_A3_term_only_gain"], "matched-k MK3 minus A3 term-only gain")
same(mk, h3p1["matched_k"]["MK3_minus_A3_term_only_gain"], "pass 1 matched-k unchanged")
lever = bool(mk["ci95"][0] > 0)
check(lever is h3["matched_k"]["granularity_lever"] is False, "matched-k: not a granularity lever")
fit = {r: h3["runs"][r]["resolved"] for r in LABEL_RUNS}
check(fit == h3["fit"] == {"L3": "fits", "L5": "no_fit", "LT": "fits"}, "fit readings L3 fits, L5 no fit, LT fits")
gains = {r: tr[r]["nested"]["gain"]["point"] for r in LABEL_RUNS if fit[r] == "fits"}
best_run = max(gains, key=gains.get)
check(best_run == h3["best_fitting_run"] == "LT", "best fitting run = LT")
same_val(gains[best_run], h3["best_nested_gain"], "best nested gain")
h3r = h3_reading(fit, gains[best_run], g_star)
check(h3r == h3["h3_reading"] == "ceiling_too_low", "H3 reading = ceiling_too_low")
dec = joint_decision(reading, h3r)
check(dec == decision["decision"] == "branch_3" and decision["h1"] == reading and decision["h3"] == h3r,
      "joint decision = branch_3")
check(decision["h2_bank"] == ("MK" if lever else "AIC"), "H2 bank field consistent with the matched-k reading")
# descriptive pairs computed here
pairs_here = {r: {"term_only_gain_minus_A3": brief(compare(term[r], pt_a3, cl, "gain")),
                  "term_only_either_minus_cosine": brief(compare({"e": either_of(term[r])}, {"e": either_of(cos_pa)},
                                                                 cl, "e"))}
              for r in ("L3", "L5", "LT")}
pairs_here["A3"] = {"term_only_either_minus_cosine": brief(compare({"e": either_of(pt_a3)}, {"e": either_of(cos_pa)},
                                                                    cl, "e"))}
pairs_here["MK3"] = {"term_only_either_minus_cosine": brief(compare({"e": either_of(term["MK3"])},
                                                                     {"e": either_of(cos_pa)}, cl, "e"))}
DATA["h3_transfer"] = {r: {"term_only": {m: brief(tr[r]["term_only"][m]) for m in ("r1", "gain", "other")},
                           "term_only_either": tr[r]["term_only_either"],
                           "nested": {m: brief(tr[r]["nested"][m]) for m in ("r1", "gain", "other")},
                           **({"picks": tr[r]["picks"], "nested_either": tr[r]["nested_either"]} if r != "A3" else {})}
                       for r in tr}
DATA["h3_decision"] = {"fit": fit, "best_fitting_run": best_run, "best_nested_gain": gains[best_run],
                       "g_star": g_star, "h3_reading": h3r, "matched_k": brief(mk), "granularity_lever": lever,
                       "decision": dec}
DATA["h3_transfer_computed_here"] = pairs_here

# training curves (aspect loss minus constant, tau)
curves = {}
for k, v in hist.items():
    hh = v["history"]
    const = CONST[v["config"]["lambda_swap"] > 0]
    loss = np.asarray(hh["aspect_loss"], np.float64)
    curves[k] = {"step": hh["step"], "loss_minus_const": (loss - const).tolist(), "tau": hh["tau"],
                 "last10_mean": float(loss[-10:].mean()), "pct_below_const": float(100 * (const - loss[-10:].mean())
                                                                                   / const)}
check(round(curves["A3_E3"]["last10_mean"], 3) == 3.222, "E3's A3 last-10 aspect loss 3.222 (E3 report)")
DATA["training_curves"] = curves

# ------------------------------------------------------------------------------------------------ figures
U, A = list(NESTED_U), list(NESTED_A)


def fmt(v):
    return f"{v:g}"


# (a) A3 56-cell profile
fig, axs = plt.subplots(2, 2, figsize=(15, 11.5), layout="constrained")
axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
specs = [("r1", "(i) R@1 (%)", SEQ, None), ("gain", "(ii) condition gain (points)", DIV, "div"),
         ("either", "(iii) either rate (%)", SEQ, None)]
for ax, (key, title, cmap, kind) in zip(axes[:3], specs):
    M = np.array([[cells[(u, a)][key] for a in A] for u in U])
    norm = TwoSlopeNorm(vmin=min(-0.15, M.min()), vcenter=0.0, vmax=M.max()) if kind else None
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="upper")
    for i in range(len(U)):
        for j in range(len(A)):
            rgba = im.cmap(im.norm(M[i, j]))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            bold = key == "r1" and abs(M[i, j] - best_ctrl) < 0.005
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if lum < 0.5 else INK, fontweight="bold" if bold else "normal")
    ax.set_xticks(range(len(A)), [fmt(a) for a in A])
    ax.set_yticks(range(len(U)), [fmt(u) for u in U])
    ax.set_xlabel("λ_a (weight on the agreement-weighted term T_a)")
    ax.set_ylabel("λ_u (weight on the uniform term T_u)")
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    if key == "r1":
        cb.ax.axhline(best_ctrl, color=INK, lw=2)
        cb.ax.text(-0.15, best_ctrl, f"control best {best_ctrl:.2f} ", transform=cb.ax.get_yaxis_transform(),
                   va="center", ha="right", fontsize=8.5, color=INK)
    for (cell, style, colour) in (((4.0, 0.25), "-", "#eb6834"), ((8.0, 0.0), "--", INK)):
        i, j = U.index(cell[0]), A.index(cell[1])
        ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, lw=2.6, ls=style, ec=colour))
ax = axes[3]
sig = sorted(ctrl)
ax.plot(sig, [ctrl[s]["r1"] for s in sig], color=COL["A3"], marker="o", ms=5, lw=2)
ax.axhline(cos_sum["r1"]["point"], color=C_COS, ls=":", lw=1.5)
ax.text(32, cos_sum["r1"]["point"] + 0.08, "cosine 12.96 (σ = 0)", ha="right", fontsize=9, color=INK2)
ax.scatter([best_sigma], [best_ctrl], s=160, facecolor="none", edgecolor=INK, lw=2, zorder=5)
ax.annotate(f"best σ = 8: R@1 {best_ctrl:.2f}", (best_sigma, best_ctrl), xytext=(0.6, 15.6), fontsize=9.5,
            arrowprops={"arrowstyle": "->", "color": INK2})
ax.set_xscale("symlog", linthresh=0.25)
ax.set_xticks([0, 0.25, 0.5, 1, 2, 4, 8, 16, 32], ["0", "0.25", "0.5", "1", "2", "4", "8", "16", "32"])
ax.set_xlabel("σ = λ_u + λ_a, the control's weight on T_u (30 values)")
ax.set_ylabel("R@1 (%)")
ax.set_title("(iv) nested uniform control z(cos) + σ·z(T_u)", loc="left")
ax.grid(color=GRID, lw=0.6)
handles = [plt.Line2D([], [], ls="-", lw=2.6, color="#eb6834"), plt.Line2D([], [], ls="--", lw=2.6, color=INK)]
fig.legend(handles, ["pick tuned on the even half, scores the odd half: (λ_u, λ_a) = (4, 0.25)",
                     "pick tuned on the odd half, scores the even half: (8, 0), identical to the control at σ = 8"],
           loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=2, fontsize=10)
fig.suptitle("A3 under the nested score, uncross-fitted, on the seed-42 selection episodes (12,288 pooled)",
             fontsize=13)
fig.savefig(HERE / "h1_profile.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# (b) trade-off per lambda_u row
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
for k, u in enumerate(U):
    g = [cells[(u, a)]["gain"] for a in A]
    r = [cells[(u, a)]["r1"] for a in A]
    e = [cells[(u, a)]["either"] for a in A]
    for ax, y in ((axes[0], r), (axes[1], e)):
        ax.plot(g, y, color=ORD_RAMP[k], lw=2, marker="osD^vPX"[k], ms=5, zorder=3, label=f"λ_u = {fmt(u)}")
        ax.scatter([g[0]], [y[0]], s=70, facecolor="white", edgecolor=ORD_RAMP[k], lw=2, zorder=4)
axes[0].legend(loc="lower left", fontsize=9, ncol=2, title="row (open circle: λ_a = 0)", title_fontsize=9)
axes[0].axhline(best_ctrl, color=INK, lw=1.5, ls="--")
axes[0].text(1.06, best_ctrl + 0.1, f"best control R@1 {best_ctrl:.2f} (σ 8)", ha="right", fontsize=9, color=INK)
axes[0].axhline(cos_sum["r1"]["point"], color=C_COS, lw=1.2, ls=":")
axes[0].text(1.06, cos_sum["r1"]["point"] + 0.1, "cosine 12.96", ha="right", fontsize=9, color=INK2)
xs = np.linspace(-0.2, 1.1, 10)
axes[1].plot(xs, 2 * best_ctrl - xs, color=INK, lw=1.5, ls="--")
axes[1].text(0.25, 2 * best_ctrl - 0.25 + 0.3, "R@1 = best control R@1 (either = 33.09 − gain)", ha="left",
             fontsize=9, color=INK)
axes[1].axhline(cos_either["point"], color=C_COS, lw=1.2, ls=":")
axes[1].text(1.06, cos_either["point"] + 0.25, "cosine 25.92", ha="right", fontsize=9, color=INK2)
axes[0].set_ylabel("R@1 (%)")
axes[1].set_ylabel("either rate (%) = R@1 + other-aspect rate")
for ax, t in zip(axes, ("(i) R@1 against condition gain", "(ii) either rate against condition gain")):
    ax.set_xlabel("condition gain (points)")
    ax.set_title(t)
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlim(-0.12, 1.12)
fig.text(0.5, -0.03, "Each line is one λ_u row; the open circle is λ_a = 0 and the points follow λ_a = 0.25, 0.5, 1, "
         "2, 4, 8, 16. A cell beats the control's R@1 only above the dashed line.", ha="center", fontsize=9.5,
         color=INK2)
fig.tight_layout()
fig.savefig(HERE / "h1_tradeoff.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# (c) H3 fit forest plot
rows = [("L3", h3fit["L3"]["vs_A3_gain"], "fits", COL["L3"]),
        ("L5, seed 42", h3fit["L5"]["vs_A3_gain"], "inconclusive", COL["L5"]),
        ("L5, mean of seeds 42 and 43", h3fit["L5"]["two_seed_mean_vs_A3_gain"], "no fit (resolved)", COL["L5"]),
        ("LT", h3fit["LT"]["vs_A3_gain"], "fits", COL["LT"])]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
for i, (lbl, c, rd, colour) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.errorbar(c["point"], y, xerr=[[c["point"] - c["ci95"][0]], [c["ci95"][1] - c["point"]]], fmt="o", ms=8,
                color=colour, mec=INK, mew=0.6, capsize=4, lw=2, ls="none",
                mfc="white" if "mean" in lbl else colour)
    ax.text(2.25, y, f"{c['point']:+.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]  {rd}", va="center", fontsize=9,
            color=INK)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.set_xlim(-0.7, 4.3)
ax.set_xlabel("term-only condition gain, LAB run minus A3 (points)")
ax.set_title("(i) fit: paired against A3 (decides)")
ax.grid(axis="x", color=GRID, lw=0.6)
ax = axes[1]
labels = ["L3", "L5", "LT"]
for i, r in enumerate(labels):
    for k, (m, mk_, off) in enumerate((("r1", "s", -0.13), ("gain", "o", 0.13))):
        c = h3fit[r]["X_minus_C0"][m]
        y = len(labels) - 1 - i + off
        ax.errorbar(c["point"], y, xerr=[[c["point"] - c["ci95"][0]], [c["ci95"][1] - c["point"]]], fmt=mk_, ms=7,
                    color=COL[r], mec=INK, mew=0.6, capsize=3, lw=1.8, ls="none")
ax.plot([], [], "s", color=MUTED, label="R@1")
ax.plot([], [], "o", color=MUTED, label="condition gain")
ax.legend(loc="upper left", fontsize=9)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(labels)), labels[::-1])
ax.set_xlabel("term-only, LAB run minus C0 (points)")
ax.set_title("(ii) against C0, no aspect training (descriptive)")
ax.grid(axis="x", color=GRID, lw=0.6)
fig.suptitle("H3 fit on 12,288 fresh label episodes over scorer-train rows (9,900 painting clusters)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "h3_fit.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# (d) seed-42 transfer
order = ["A3", "L3", "L5", "LT", "MK3"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
panels = [("term_only", "gain", "(i) term-only condition gain", "points"),
          ("term_only_either", None, "(ii) term-only either rate", "%"),
          ("nested", "gain", "(iii) cross-fitted nested condition gain", "points")]
for ax, (key, m, title, unit) in zip(axes, panels):
    names = (["cosine"] if key != "nested" else []) + order
    for i, nm in enumerate(names):
        if nm == "cosine":
            v = cos_either if key == "term_only_either" else {"point": 0.0, "ci95": [0.0, 0.0]}
            colour = C_COS
        else:
            v = tr[nm][key] if m is None else tr[nm][key][m]
            colour = COL[nm]
        ax.bar(i, v["point"], width=0.68, color=colour, edgecolor="white", lw=2, zorder=2)
        ax.errorbar(i, v["point"], yerr=[[v["point"] - v["ci95"][0]], [v["ci95"][1] - v["point"]]], color=INK,
                    capsize=3, lw=1.2, zorder=3)
        top = max(v["ci95"][1], v["point"])
        ax.text(i, top + {"term_only": 0.04, "term_only_either": 0.3, "nested": 0.008}[key], f"{v['point']:.2f}",
                ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(f"{'condition gain' if m else 'either rate'} ({unit})")
    ax.set_title(title)
    ax.axhline(0, color=INK, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylim(-0.4, 2.65)
axes[1].axhline(cos_either["point"], color=C_COS, ls=":", lw=1.4)
axes[1].set_ylim(15, 27.5)
axes[2].axhline(g_star, color=INK, ls="--", lw=1.4)
axes[2].text(4.45, g_star + 0.01, f"g* = {g_star:.3f}", ha="right", va="bottom", fontsize=9, color=INK)
axes[2].set_ylim(-0.15, 0.32)
fig.suptitle("Seed-42 selection episodes (12,288 pooled, 4,602 paintings): A3 and the four diagnostic runs",
             fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "h3_transfer.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# (e) training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
lines = [("A3_E3", "A3 (E3, bank AIC)", C_COS, "--", 2.6), ("L3_seed42", "L3", COL["L3"], "-", 1.8),
         ("L5_seed42", "L5, seed 42", COL["L5"], "-", 1.8), ("L5_seed43", "L5, seed 43", COL["L5"], ":", 1.6),
         ("LT_seed42", "LT (τ fixed)", COL["LT"], "-", 1.8), ("MK3_seed42", "MK3", COL["MK3"], "-", 1.8)]
for key, lbl, colour, ls, lw in lines:
    cv = curves[key]
    axes[0].plot(cv["step"], cv["loss_minus_const"], color=colour, ls=ls, lw=lw, label=lbl)
    axes[1].plot(cv["step"], cv["tau"], color=colour, ls=ls, lw=lw, label=lbl)
axes[0].axhline(0, color=INK, lw=1, ls="--")
axes[0].set_ylabel("aspect loss − constant-score value (3.258)")
axes[0].set_title("(i) aspect loss against its constant-score value")
axes[1].set_ylabel("aspect temperature τ")
axes[1].set_title("(ii) learned temperature τ")
for ax in axes:
    ax.set_xlabel("training step (logged every 50 steps)")
    ax.grid(color=GRID, lw=0.6)
axes[0].set_ylim(-0.5, 0.45)
axes[1].legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "h3_training.png", dpi=170, bbox_inches="tight")
plt.close(fig)

DATA["n_checks"] = len(CHECKS)
DATA["checks"] = CHECKS
(HERE / "figure_data.json").write_text(json.dumps(DATA, indent=1, default=float))
print(f"{len(CHECKS)} checks passed; wrote 5 figures and figure_data.json to {HERE.relative_to(ROOT)}")
