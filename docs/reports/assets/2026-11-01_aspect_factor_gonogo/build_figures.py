"""Figures and figure data for the E3 report (docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md).

Reads only stored arrays and records:
  src/test/20261101_aspect_factor_gonogo/results/  per_anchor_select_seed42.npz, per_anchor_gonogo_seed43.npz,
                                                    select_seed42.json, gonogo.json, history_<run>_seed42.json
  src/test/20261030_aspect_baselines/results/       per_anchor_seed{42,43}.npz, baselines_seed{42,43}.json
  src/test/20261102_mllm_probe/results/             per_anchor.npz, probe.json, episodes_seed44.npz (v1, pre-fix prompt)
  src/test/20261102_mllm_probe/results/v2/          per_anchor.npz, probe.json (v2, fixed prompt; used if present)
  src/test/20261101_aspect_factor_gonogo/results/  train_fit_diagnostic.json (post-hoc check; summaries only, copied)

Every summary and paired comparison is recomputed here with src.eval.aspect_metrics (painting-clustered bootstrap,
5,000 resamples, seed 42) and asserted equal to the stored record wherever one exists, so the figures and the report
tables rest on a re-derivation, not on copied log lines.

Run from the repository root:
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python \
      docs/reports/assets/2026-11-01_aspect_factor_gonogo/build_figures.py
Writes r1_vs_gain.png, per_pair.png, aspect_loss.png, method_diagram.png and figure_data.json next to this file.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.eval.aspect_metrics import METRICS, compare, summarize  # noqa: E402

E3 = ROOT / "src/test/20261101_aspect_factor_gonogo/results"
E1 = ROOT / "src/test/20261030_aspect_baselines/results"
MLLM = ROOT / "src/test/20261102_mllm_probe/results"
PAIRS = ["emotion__style", "emotion__genre", "style__genre"]
PAIR_LABEL = {"emotion__style": "emotion x style", "emotion__genre": "emotion x genre", "style__genre": "style x genre"}
RAW = ["diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip"]
TOL = 1e-9

# colour families (report legend): grey backbone, blue raw-feature baselines, purple earlier factor recipes,
# orange uniform-weight controls, teal method A runs, red the in-context MLLM
C_COS, C_RAW, C_OLD, C_UNI, C_A, C_MLLM = "#555555", "#1f77b4", "#7b4fa0", "#e07b00", "#14897f", "#c0392b"


def per(z, name):
    return {m: np.asarray(z[f"{name}__{m}"], dtype=np.float64) for m in METRICS}


def sub(p, mask):
    return {m: v[mask] for m, v in p.items()}


def same(a, b, what):
    """Assert a recomputed {point, ci95} equals a stored one."""
    ok = abs(a["point"] - b["point"]) < TOL and all(abs(x - y) < TOL for x, y in zip(a["ci95"], b["ci95"]))
    assert ok, f"mismatch for {what}: recomputed {a}, stored {b}"
    CHECKS.append(what)


def brief(s):
    return {"point": s["point"], "ci95": list(s["ci95"]), "n_clusters": s["n_clusters"]}


CHECKS: list = []
DATA: dict = {"note": "all values in percentage points; CIs are 95% painting-clustered bootstrap intervals "
                      "(5,000 resamples, seed 42) recomputed by build_figures.py"}

# ----------------------------------------------------------------------------------------------------- seed 42
t42, g42 = np.load(E1 / "per_anchor_seed42.npz"), np.load(E3 / "per_anchor_select_seed42.npz")
assert (t42["anchor_group"] == g42["anchor_group"]).all() and (t42["pair_index"] == g42["pair_index"]).all()
cl42 = t42["anchor_group"]
b42 = json.load(open(E1 / "baselines_seed42.json"))["scorers"]
sel = json.load(open(E3 / "select_seed42.json"))["runs"]
RUNS = ["A1", "A2", "A3", "A4", "A5", "A6", "H1", "S1"]
m42 = {n: per(t42, n) for n in ["cosine", *RAW, "value_prototype", "SE", "C0", "R3", "SE_uniform"]}
for r in RUNS:
    m42[r], m42[f"{r}_uniform"] = per(g42, r), per(g42, f"{r}_uniform")
s42 = {n: summarize(p, cl42) for n, p in m42.items()}
for n in ["cosine", *RAW, "value_prototype", "SE", "C0", "R3", "SE_uniform"]:
    for m in ("r1", "gain"):
        same(s42[n][m], b42[n]["overall"][m], f"seed42 {n} {m} vs baselines_seed42.json")
for r in RUNS:
    for m in ("r1", "gain"):
        same(s42[r][m], sel[r]["overall"][m], f"seed42 {r} {m} vs select_seed42.json")
        same(s42[f"{r}_uniform"][m], sel[r]["uniform_overall"][m], f"seed42 {r}_uniform {m} vs select_seed42.json")
DATA["seed42"] = {n: {m: brief(s[m]) for m in ("r1", "gain", "other", "swap")} for n, s in s42.items()}

# ----------------------------------------------------------------------------------------------------- seed 43
t43, g43 = np.load(E1 / "per_anchor_seed43.npz"), np.load(E3 / "per_anchor_gonogo_seed43.npz")
assert (t43["anchor_group"] == g43["anchor_group"]).all() and (t43["pair_index"] == g43["pair_index"]).all()
cl43, pi43 = t43["anchor_group"], t43["pair_index"]
b43 = json.load(open(E1 / "baselines_seed43.json"))["scorers"]
gj = json.load(open(E3 / "gonogo.json"))
assert gj["go"]["picked"] == "A3" and gj["go"]["comparators"]["go_baseline"]["name"] == "rca"
m43 = {n: per(t43, n) for n in ["cosine", *RAW, "value_prototype", "SE", "C0", "R3", "SE_uniform"]}
m43["A3 (picked)"], m43["A3_uniform"] = per(g43, "picked"), per(g43, "picked_uniform")
for r in ("A1", "H1", "S1"):
    m43[r] = per(g43, r)
m43["H1_uniform"] = per(g43, "H1_uniform")
s43 = {n: summarize(p, cl43) for n, p in m43.items()}
for n in ["cosine", *RAW, "value_prototype", "SE", "C0", "R3", "SE_uniform"]:
    for m in ("r1", "gain"):
        same(s43[n][m], b43[n]["overall"][m], f"seed43 {n} {m} vs baselines_seed43.json")
for m in METRICS:
    same(s43["A3 (picked)"][m], gj["go"]["picked_overall"][m], f"seed43 picked {m} vs gonogo.json")
    same(s43["A3_uniform"][m], gj["go"]["comparators"]["uniform_control"]["overall"][m],
         f"seed43 A3_uniform {m} vs gonogo.json")
    same(s43["A1"][m], gj["ablation"]["rows"]["A1"]["all_pairs"][m], f"seed43 A1 {m} vs gonogo.json")
    same(s43["S1"][m], gj["ablation"]["rows"]["S1"]["all_pairs"][m], f"seed43 S1 {m} vs gonogo.json")
    same(s43["H1"][m], gj["k8"]["H1_all_pairs"][m], f"seed43 H1 {m} vs gonogo.json")
DATA["seed43"] = {n: {m: brief(s[m]) for m in ("r1", "gain", "other", "swap")} for n, s in s43.items()}

# the three GO comparisons, the descriptive picked-vs-baseline context, K8 and the ablation rows
pk = m43["A3 (picked)"]
go = {}
for key, other in (("backbone_only", m43["cosine"]), ("go_baseline", m43["rca"]),
                   ("uniform_control", m43["A3_uniform"])):
    go[key] = {m: compare(pk, other, cl43, m) for m in ("r1", "gain")}
    for m in ("r1", "gain"):
        same(go[key][m], gj["go"]["comparators"][key][m], f"GO {key} {m} vs gonogo.json")
    go[key]["beats"] = bool(go[key]["r1"]["ci95"][0] > 0 and go[key]["gain"]["ci95"][0] > 0)
    assert go[key]["beats"] == gj["go"]["comparators"][key]["beats"]
go["GO"] = all(go[k]["beats"] for k in ("backbone_only", "go_baseline", "uniform_control"))
assert go["GO"] == gj["go"]["GO"] is False
go["strong_go"] = bool(go["GO"] and go["backbone_only"]["r1"]["point"] >= 4.0 and s43["A3 (picked)"]["gain"]["point"] >= 4.0)
assert go["strong_go"] == gj["go"]["strong_go"]
context = {}
for n in [*RAW, "SE_uniform"]:
    context[n] = {m: compare(pk, m43[n], cl43, m) for m in ("r1", "gain")}
    for m in ("r1", "gain"):
        same(context[n][m], gj["baseline_context"]["picked_vs"][n][m], f"picked vs {n} {m} vs gonogo.json")
genre = np.isin(pi43, [1, 2])
emo = np.isin(pi43, [0, 1])
k8 = compare(sub(m43["H1"], genre), sub(m43["H1_uniform"], genre), cl43[genre], "gain")
same(k8, gj["k8"]["gain_vs_uniform"], "K8 gain vs gonogo.json")
k8_r1 = compare(sub(m43["H1"], genre), sub(m43["H1_uniform"], genre), cl43[genre], "r1")
same(k8_r1, gj["k8"]["r1_vs_uniform"], "K8 r1 vs gonogo.json")
k8_pp = {}
for i, p in enumerate(PAIRS[1:], start=1):
    mk = pi43 == i
    k8_pp[p] = compare(sub(m43["H1"], mk), sub(m43["H1_uniform"], mk), cl43[mk], "gain")
    same(k8_pp[p], gj["k8"]["per_pair_gain_vs_uniform"][p], f"K8 per-pair {p} vs gonogo.json")
abl = {}
for key, a, b, mask in (("S1_vs_A1_emotion_pairs", "S1", "A1", emo),
                        ("S1_vs_picked_emotion_pairs", "S1", "A3 (picked)", emo),
                        ("H1_vs_A1_genre_pairs", "H1", "A1", genre)):
    abl[key] = {m: compare(sub(m43[a], mask), sub(m43[b], mask), cl43[mask], m) for m in ("r1", "gain")}
    for m in ("r1", "gain"):
        same(abl[key][m], gj["ablation"][key][m], f"ablation {key} {m} vs gonogo.json")
emo_rows = {}
for n, rowname in (("A3 (picked)", "picked (A3)"), ("A1", "A1"), ("S1", "S1"), ("SE", "SE"), ("C0", "C0"),
                   ("R3", "R3")):
    emo_rows[n] = summarize(sub(m43[n], emo), cl43[emo])
    for m in ("r1", "gain"):
        same(emo_rows[n][m], gj["ablation"]["rows"][rowname]["emotion_pairs"][m], f"emotion-pair row {n} {m}")
DATA["go_comparisons"] = {k: ({m: brief(v[m]) for m in ("r1", "gain")} | {"beats": v["beats"]})
                          for k, v in go.items() if isinstance(v, dict)}
DATA["GO"], DATA["strong_go"] = go["GO"], go["strong_go"]
DATA["picked_vs_baselines"] = {n: {m: brief(v[m]) for m in ("r1", "gain")} for n, v in context.items()}
DATA["k8"] = {"gain_vs_uniform": brief(k8), "r1_vs_uniform": brief(k8_r1),
              "per_pair_gain_vs_uniform": {p: brief(v) for p, v in k8_pp.items()},
              "H1_genre_pairs": {m: brief(v) for m, v in summarize(sub(m43["H1"], genre), cl43[genre]).items()},
              "H1_uniform_genre_pairs": {m: brief(v) for m, v in
                                         summarize(sub(m43["H1_uniform"], genre), cl43[genre]).items()},
              "A1_genre_pairs": {m: brief(v) for m, v in summarize(sub(m43["A1"], genre), cl43[genre]).items()},
              "holds": bool(k8["ci95"][0] > 0)}
for m in METRICS:
    same(DATA["k8"]["H1_genre_pairs"][m], gj["k8"]["H1_genre_pairs"][m], f"K8 H1 genre pairs {m}")
    same(DATA["k8"]["H1_uniform_genre_pairs"][m], gj["k8"]["H1_uniform_genre_pairs"][m], f"K8 H1 uniform genre {m}")
assert DATA["k8"]["holds"] == gj["k8"]["K8"] is False
DATA["ablation"] = {k: {m: brief(v[m]) for m in ("r1", "gain")} for k, v in abl.items()}
DATA["emotion_pair_rows"] = {n: {m: brief(v[m]) for m in ("r1", "gain", "other", "swap")} for n, v in emo_rows.items()}

# per-pair summaries on seed 43 for figure (b)
BARS = ["cosine", "rca", "wang", "SE", "A3 (picked)", "A3_uniform", "A1", "H1", "S1"]
pp43 = {n: {p: summarize(sub(m43[n], pi43 == i), cl43[pi43 == i]) for i, p in enumerate(PAIRS)} for n in BARS}
for p in PAIRS:
    for m in ("r1", "gain"):
        same(pp43["A3 (picked)"][p][m], gj["go"]["picked_per_pair"][p][m], f"picked per-pair {p} {m}")
        for n in ("cosine", "rca", "wang", "SE"):
            same(pp43[n][p][m], b43[n]["per_pair"][p][m], f"seed43 {n} per-pair {p} {m}")
DATA["seed43_per_pair"] = {n: {p: {m: brief(v[m]) for m in ("r1", "gain", "other")} for p, v in d.items()}
                           for n, d in pp43.items()}

# the "either aspect candidate first" rate R@1 + other, a derived mechanism number (no bootstrap)
DATA["either_aspect_first_rate"] = {
    "seed42": {n: s42[n]["r1"]["point"] + s42[n]["other"]["point"] for n in s42},
    "seed43": {n: s43[n]["r1"]["point"] + s43[n]["other"]["point"] for n in s43}}

# ----------------------------------------------------------------------------------------------------- MLLM
ep44 = np.load(MLLM / "episodes_seed44.npz", allow_pickle=True)
assert list(ep44["pair_order"]) == PAIRS


def mllm_block(folder, tag):
    rec = json.load(open(folder / "probe.json"))
    z = np.load(folder / "per_anchor.npz")
    cl = z["anchor_group"]
    n = rec["n_per_pair"]
    assert len(cl) == 3 * n
    mm, cc = per(z, "mllm"), per(z, "cosine")
    out = {"overall": {"mllm": summarize(mm, cl), "cosine": summarize(cc, cl)},
           "compare": {m: compare(mm, cc, cl, m) for m in ("r1", "gain")},
           "per_pair": {}}
    for m in ("r1", "gain"):
        same(out["overall"]["mllm"][m], rec["pooled"]["mllm"][m], f"MLLM {tag} pooled {m}")
        same(out["overall"]["cosine"][m], rec["pooled"]["cosine"][m], f"MLLM {tag} cosine pooled {m}")
        same(out["compare"][m], rec["compare_pooled"][m], f"MLLM {tag} compare {m}")
    for i, p in enumerate(PAIRS):
        sl = slice(i * n, (i + 1) * n)
        out["per_pair"][p] = {"mllm": summarize(sub(mm, sl), cl[sl]), "cosine": summarize(sub(cc, sl), cl[sl])}
        for m in ("r1", "gain"):
            same(out["per_pair"][p]["mllm"][m], rec["pairs"][p]["mllm"][m], f"MLLM {tag} {p} {m}")
    works = bool(out["compare"]["r1"]["ci95"][0] > 0 and out["compare"]["gain"]["ci95"][0] > 0)
    assert works == rec["mllm_works"], f"MLLM {tag}: works flag differs from probe.json"
    out["works"] = works
    out["episodes_sha256"] = rec["episodes_sha256"]
    return out


mllm = {"v1 (pre-fix prompt)": mllm_block(MLLM, "v1")}
V2 = MLLM / "v2"
if (V2 / "probe.json").exists() and (V2 / "per_anchor.npz").exists():
    mllm["v2 (fixed prompt)"] = mllm_block(V2, "v2")
    assert mllm["v2 (fixed prompt)"]["episodes_sha256"] == mllm["v1 (pre-fix prompt)"]["episodes_sha256"], \
        "v1 and v2 should score the same seed-44 episodes"
DATA["mllm"] = {k: {"overall": {s: {m: brief(v["overall"][s][m]) for m in ("r1", "gain", "other", "swap")}
                                for s in ("mllm", "cosine")},
                    "compare": {m: brief(v["compare"][m]) for m in ("r1", "gain")},
                    "per_pair": {p: {s: {m: brief(v["per_pair"][p][s][m]) for m in ("r1", "gain")}
                                     for s in ("mllm", "cosine")} for p in PAIRS},
                    "works": v["works"]} for k, v in mllm.items()}
DATA["mllm_v2_present"] = "v2 (fixed prompt)" in mllm

# ----------------------------------------------------------------------------------------------------- training
hist = {r: json.load(open(E3 / f"history_{r}_seed42.json")) for r in RUNS}
DATA["training"] = {}
for r, h in hist.items():
    al = np.asarray(h["history"]["aspect_loss"])
    chance = math.log(13) + h["config"]["lambda_swap"] * math.log(2)
    DATA["training"][r] = {"aspect_loss_step1": float(al[0]), "aspect_loss_last10_mean": float(al[-10:].mean()),
                           "constant_score_value": chance,
                           "relative_gap_last10": float((chance - al[-10:].mean()) / chance),
                           "tau_start": h["history"]["tau"][0], "tau_end": h["history"]["tau"][-1],
                           "log_tau_rise": math.log(h["history"]["tau"][-1] / h["history"]["tau"][0]),
                           "log_tau_rise_over_max_adam_move": math.log(h["history"]["tau"][-1] / h["history"]["tau"][0])
                           / (h["config"]["lr"] * (h["history"]["step"][-1] - h["history"]["step"][0])),
                           "log_tau_rise_over_max_adam_move_steps_50_to_1050": math.log(
                               h["history"]["tau"][h["history"]["step"].index(1050)]
                               / h["history"]["tau"][h["history"]["step"].index(50)]) / (h["config"]["lr"] * 1000),
                           "aspect_loss_window_means": {
                               f"{a}-{b}": float(al[(st_ >= a) & (st_ <= b)].mean())
                               for st_ in [np.asarray(h["history"]["step"])]
                               for a, b in ((50, 500), (550, 1000), (1050, 1500), (1550, 2000))},
                           "lambda_aspect": h["config"]["lambda_aspect"], "lambda_swap": h["config"]["lambda_swap"],
                           "num_factors": h["config"]["num_factors"], "aspect_beta": h["config"]["aspect_beta"]}
# post-hoc training-fit diagnostic (commits 938d1e5 and 8a8e999; not pre-registered, outside the decision map). Its JSON
# stores summaries only, so the headline numbers are read from it, not recomputed. Since 8a8e999 its "SE" row is E1's
# SE checkpoint (SHA-256 93add21b...) and "S" is the factor-learning grid's style cell, a separate reference row.
FIT = ROOT / "src/test/20261101_aspect_factor_gonogo/results/train_fit_diagnostic.json"
if FIT.exists():
    fit = json.load(open(FIT))
    assert fit["note"].startswith("POST-HOC")
    assert fit["checkpoint_sha256"]["SE"].startswith("93add21b") and fit["checkpoint_sha256"]["S"].startswith("33d35943")
    DATA["train_fit_diagnostic"] = {"note": fit["note"], "n_per_pair": fit["n_per_pair"],
                                    "cosine": {k: {m: fit["cosine"][k]["pooled"][m] for m in ("r1", "gain")}
                                               for k in ("fresh", "bank")},
                                    "models": {}}
    for name, rec in fit["models"].items():
        DATA["train_fit_diagnostic"]["models"][name] = {
            f"{k}_{sc}": {m: rec[k][sc]["pooled"][m] for m in ("r1", "gain", "other", "swap")}
            for k in ("fresh", "bank") for sc in ("agreement", "uniform", "fixed_beta")}
DATA["checks_passed"] = len(CHECKS)


# ----------------------------------------------------------------------------------------------------- plotting
def label_points(ax, items, fontsize=6.5):
    """Greedy label placement: try offsets around each point, keep the first that stays inside the axes and overlaps
    no earlier label and no marker."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    taken = []
    for x, y, _, _ in items:                                       # markers are obstacles too
        px, py = ax.transData.transform((x, y))
        taken.append(matplotlib.transforms.Bbox([[px - 4, py - 4], [px + 4, py + 4]]))
    offsets = [(5, 4), (5, -5), (-5, 4), (-5, -5), (8, 12), (8, -13), (-8, 12), (-8, -13), (0, 18), (0, -20),
               (16, 22), (-16, 22), (16, -24), (-16, -24), (24, 4), (-24, 4), (0, 30), (0, -32),
               (4, 44), (4, -46), (-4, 44), (-4, -46), (4, 60), (4, -62), (-4, 60), (-4, -62), (4, 76), (4, -78),
               (-4, 76), (-4, -78), (4, 92), (4, -94)]
    for x, y, text, color in items:
        best = None
        for dx, dy in offsets:
            kw = {"xytext": (dx, dy), "textcoords": "offset points", "fontsize": fontsize, "color": color,
                  "ha": "left" if dx >= 0 else "right", "va": "bottom" if dy >= 0 else "top"}
            ann = ax.annotate(text, (x, y), **kw)                          # placement test without a leader line
            bb = ann.get_window_extent(renderer).expanded(1.04, 1.1)
            inside = frame.contains(bb.x0, bb.y0) and frame.contains(bb.x1, bb.y1)
            if inside and not any(bb.overlaps(t) for t in taken):
                if abs(dx) + abs(dy) >= 15:                                # long offset: redraw with a leader line
                    ann.remove()
                    ax.annotate(text, (x, y), arrowprops={"arrowstyle": "-", "lw": 0.4, "color": color}, **kw)
                best = bb
                break
            ann.remove()
        if best is None:
            ann = ax.annotate(text, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=fontsize,
                              color=color)
            best = ann.get_window_extent(renderer)
        taken.append(best)


def merged(summ, names, label_names):
    """Group methods whose (R@1, gain) points coincide exactly and label them together."""
    groups = {}
    for n in names:
        key = (summ[n]["r1"]["point"], summ[n]["gain"]["point"])
        groups.setdefault(key, []).append(n)
    out = []
    for (x, y), ns in groups.items():
        ns = sorted(ns, key=lambda n: (n != "cosine", n))
        text = label_names.get(ns[0], ns[0]) if len(ns) == 1 else \
            label_names.get(ns[0], ns[0]) + " = " + ", ".join(ns[1:]) + " (lambda = 0)"
        out.append((x, y, text, family(ns[0])[0]))
    return out


def errpt(ax, s, color, marker="o", ms=4.5, zorder=3, mfc=None):
    r, g = s["r1"], s["gain"]
    ax.errorbar(r["point"], g["point"], xerr=[[r["point"] - r["ci95"][0]], [r["ci95"][1] - r["point"]]],
                yerr=[[g["point"] - g["ci95"][0]], [g["ci95"][1] - g["point"]]], fmt=marker, color=color, ms=ms,
                lw=0.8, capsize=0, zorder=zorder, mfc=mfc if mfc is not None else color)


def family(name):
    if name == "cosine":
        return C_COS, "o", 5
    if name.endswith("_uniform"):
        return C_UNI, "D", 4
    if name in RAW or name == "value_prototype":
        return C_RAW, "o", 4
    if name in ("SE", "C0", "R3"):
        return C_OLD, "s", 4.5
    if name.startswith("A3"):
        return C_A, "*", 11
    if name == "H1":
        return C_A, "^", 6
    if name == "S1":
        return C_A, "v", 6
    return C_A, "o", 5


def scatter_seed(ax_l, ax_r, summ, title, label_names):
    near = [n for n in summ if not n.endswith("_uniform")]
    unis = [n for n in summ if n.endswith("_uniform")]
    for ax, names in ((ax_l, near), (ax_r, unis)):
        for n in names:
            c, mk, ms = family(n)
            errpt(ax, summ[n], c, mk, ms, zorder=4 if n.startswith("A3") else 3)
        ax.axhline(0, color="k", lw=0.5, zorder=1)
    cos = summ["cosine"]["r1"]["point"]
    ax_l.axvline(cos, color=C_COS, lw=0.6, ls=":", zorder=1)
    xs_l = [summ[n]["r1"]["point"] for n in near]
    xs_r = [summ[n]["r1"]["ci95"][k] for n in unis for k in (0, 1)]
    ax_l.set_xlim(min(xs_l) - 0.35, max(xs_l) + 0.45)
    ax_r.set_xlim(min(xs_r) - 0.1, max(xs_r) + 0.12)
    ax_l.set_title(title, fontsize=9, loc="left")
    ax_l.set_ylabel("condition gain (points)")
    for ax in (ax_l, ax_r):
        ax.tick_params(labelsize=7)
    ax_r.tick_params(labelleft=False)
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    label_points(ax_l, merged(summ, near, label_names))
    label_points(ax_r, [(summ[n]["r1"]["point"], summ[n]["gain"]["point"], n.replace("_uniform", " unif."),
                         family(n)[0]) for n in unis], fontsize=6)


# Figure (a): R@1 against condition gain
fig = plt.figure(figsize=(13, 12.5))
gs = fig.add_gridspec(3, 2, width_ratios=[3.4, 1], height_ratios=[1, 1, 0.8], wspace=0.04, hspace=0.3,
                      top=0.95, bottom=0.05)
ax42l, ax42r = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax43l, ax43r = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
axm = fig.add_subplot(gs[2, 0])
names42 = {"cosine": "cosine (backbone only)", "rca": "rca (GO bar)"}
names43 = {"cosine": "cosine (backbone only)", "rca": "rca (GO bar)", "A3 (picked)": "A3 (picked)"}
for ax_l, ax_r in ((ax42l, ax42r), (ax43l, ax43r)):
    ax_l.sharey(ax_r)
scatter_seed(ax42l, ax42r, s42, "(i) seed-42 selection episodes (the pick): 12,288 episodes, 4,602 anchor paintings",
             names42)
scatter_seed(ax43l, ax43r, s43, "(ii) seed-43 test episodes (the GO test): 12,288 episodes, 4,575 anchor paintings",
             names43)
ax42r.set_title("uniform-weight controls", fontsize=8)
ax43r.set_title("uniform-weight controls", fontsize=8)
ax43l.set_xlabel("R@1 (%)")
ax43r.set_xlabel("R@1 (%)")
# (iii) MLLM probe, seed 44
items = []
cos44 = mllm["v1 (pre-fix prompt)"]["overall"]["cosine"]
errpt(axm, cos44, C_COS, "o", 5)
items.append((cos44["r1"]["point"], cos44["gain"]["point"], "cosine (backbone only)", C_COS))
for k, v in mllm.items():
    mfc = "white" if k.startswith("v1") else C_MLLM
    errpt(axm, v["overall"]["mllm"], C_MLLM, "P", 8, mfc=mfc)
    items.append((v["overall"]["mllm"]["r1"]["point"], v["overall"]["mllm"]["gain"]["point"],
                  f"Qwen3-VL-2B in-context, {k}", C_MLLM))
axm.axhline(0, color="k", lw=0.5)
axm.set_xlabel("R@1 (%)")
axm.set_ylabel("condition gain (points)")
axm.tick_params(labelsize=7)
axm.set_title("(iii) seed-44 MLLM probe episodes: 900 episodes (300 per pair), 802 anchor paintings"
              + ("" if "v2 (fixed prompt)" in mllm else "; v2 (fixed prompt) pending"), fontsize=9, loc="left")
axm.set_xlim(11.5, 16.5)
label_points(axm, items, fontsize=7)
axleg = fig.add_subplot(gs[2, 1])
axleg.axis("off")
handles = [plt.Line2D([], [], color=C_COS, marker="o", ls="", label="backbone only (CLIP cosine)"),
           plt.Line2D([], [], color=C_RAW, marker="o", ls="", label="raw-feature baselines (E1)"),
           plt.Line2D([], [], color=C_OLD, marker="s", ls="", label="earlier factor recipes SE, C0, R3"),
           plt.Line2D([], [], color=C_UNI, marker="D", ls="", label="uniform-weight controls"),
           plt.Line2D([], [], color=C_A, marker="o", ls="", label="method A runs (A1 to A6)"),
           plt.Line2D([], [], color=C_A, marker="*", ms=11, ls="", label="picked run A3"),
           plt.Line2D([], [], color=C_A, marker="^", ls="", label="H1 (no caption partition)"),
           plt.Line2D([], [], color=C_A, marker="v", ls="", label="S1 (no affect partition)"),
           plt.Line2D([], [], color=C_MLLM, marker="P", ms=8, ls="", label="in-context MLLM")]
axleg.legend(handles=handles, loc="center", fontsize=8, frameon=False)
fig.suptitle("E3: R@1 against condition gain, with 95% painting-clustered bootstrap intervals", fontsize=11, y=0.985)
fig.savefig(HERE / "r1_vs_gain.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figure (b): per-aspect-pair bars
PAIR_COL = {"emotion__style": "#4c72b0", "emotion__genre": "#dd8452", "style__genre": "#55a868"}
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1.1], wspace=0.18, hspace=0.12)
w = 0.27
mllm_cols = ["cosine"] + list(mllm.keys())
for row, metric in enumerate(("r1", "gain")):
    ax = fig.add_subplot(gs[row, 0])
    for j, p in enumerate(PAIRS):
        vals = [pp43[n][p][metric] for n in BARS]
        pt = np.array([v["point"] for v in vals])
        lo = pt - np.array([v["ci95"][0] for v in vals])
        hi = np.array([v["ci95"][1] for v in vals]) - pt
        ax.bar(np.arange(len(BARS)) + (j - 1) * w, pt, w, yerr=[lo, hi], capsize=1.5, color=PAIR_COL[p],
               label=PAIR_LABEL[p], error_kw={"lw": 0.7})
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("R@1 (%)" if metric == "r1" else "condition gain (points)")
    if metric == "r1":
        ax.set_ylim(8, 21)
        ax.set_title("(i) seed-43 test episodes, 4,096 per pair", fontsize=9, loc="left")
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xticks(np.arange(len(BARS)), [{"rca": "rca (GO bar)", "A3_uniform": "A3 uniform",
                                              "cosine": "cosine"}.get(n, n) for n in BARS], rotation=25, ha="right",
                      fontsize=8)
    ax.tick_params(labelsize=7.5)
    axm = fig.add_subplot(gs[row, 1])
    for j, p in enumerate(PAIRS):
        vals = [mllm["v1 (pre-fix prompt)"]["per_pair"][p]["cosine"][metric]] + \
               [mllm[k]["per_pair"][p]["mllm"][metric] for k in mllm]
        pt = np.array([v["point"] for v in vals])
        lo = pt - np.array([v["ci95"][0] for v in vals])
        hi = np.array([v["ci95"][1] for v in vals]) - pt
        axm.bar(np.arange(len(vals)) + (j - 1) * w, pt, w, yerr=[lo, hi], capsize=1.5, color=PAIR_COL[p],
                error_kw={"lw": 0.7})
    axm.axhline(0, color="k", lw=0.5)
    axm.tick_params(labelsize=7.5)
    if metric == "r1":
        axm.set_ylim(8, 21)
        axm.set_title("(ii) seed-44 MLLM probe, 300 per pair" + ("" if len(mllm) > 1 else " (v2 pending)"),
                      fontsize=9, loc="left")
        axm.tick_params(labelbottom=False)
    else:
        axm.set_xticks(np.arange(len(mllm_cols)), ["cosine"] + [f"MLLM {k.split()[0]}" for k in mllm],
                       rotation=25, ha="right", fontsize=8)
fig.suptitle("E3: per-aspect-pair R@1 and condition gain, with 95% painting-clustered bootstrap intervals",
             fontsize=11, y=0.97)
fig.savefig(HERE / "per_pair.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figure: training aspect loss against its constant-score value
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
run_col = {"A1": "#1b9e77", "A2": "#d95f02", "A3": "#14897f", "A4": "#e7298a", "A5": "#66a61e", "A6": "#e6ab02",
           "H1": "#7570b3", "S1": "#a6761d"}
for r, h in hist.items():
    st = np.asarray(h["history"]["step"])
    al = np.asarray(h["history"]["aspect_loss"])
    chance = DATA["training"][r]["constant_score_value"]
    lw = 2.0 if r == "A3" else 0.9
    axes[0].plot(st, al - chance, color=run_col[r], lw=lw, label=r)
    axes[1].plot(st, h["history"]["tau"], color=run_col[r], lw=lw, label=r)
axes[0].axhline(0, color="k", lw=0.8, ls="--")
axes[0].text(2000, 0.02, "every candidate scored the same (ln 13 + lambda_swap ln 2)", ha="right", va="bottom",
             fontsize=7)
axes[0].set_ylabel("aspect loss minus constant-score value")
axes[0].set_xlabel("training step")
axes[1].set_ylabel("learned temperature tau")
axes[1].set_xlabel("training step")
axes[0].legend(fontsize=7, ncol=4)
for ax in axes:
    ax.tick_params(labelsize=7.5)
fig.suptitle("E3 training: the pseudo-aspect episode loss on 32 training episodes per logged step (every 50 steps)",
             fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "aspect_loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figure: what method A changes relative to C0 and SE
GREY, ORANGE, TEAL, PURPLE = "#bdbdbd", "#f4a259", "#5fbfb3", "#b39ddb"
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.6)
ax.axis("off")


def box(x, y, w_, h_, text, color, fs=8, style="round,pad=0.06", lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle=style, fc=color, ec="#333333", lw=lw, ls=ls))
    ax.text(x + w_ / 2, y + h_ / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#333333"))


box(0.1, 3.6, 2.2, 1.2, "frozen CLIP ViT-B/32\nimage and caption\nfeatures (183,694\nscorer-train rows)", GREY, fs=7.5)
box(2.75, 3.6, 2.6, 1.2, "two small encoders:\nimage, caption -> 32\nnon-negative factors\n(codes); L = 64 in A2", PURPLE,
    fs=7.5)
box(5.85, 4.35, 3.6, 0.85, "InfoNCE pair agreement, decorrelation\nand C0's other terms (same recipe as C0)", GREY,
    fs=7.5)
box(5.85, 3.05, 3.6, 1.05, "pseudo-aspect episode loss (new):\nranking cross-entropy through the\nagreement rule, both "
    "directions, + swap term", TEAL, fs=7.5)
box(5.85, 1.85, 3.6, 0.95, "SE's value-condition episode loss\n(naive rule on value episodes):\nnot used in method A",
    ORANGE, fs=7.5, ls="--")
box(9.95, 3.05, 2.95, 1.05, "pseudo-aspect episode banks\n(E2, new, no labels): AIC (A1 to A6),\nAI (H1), IC (S1); "
    "65,536 each", TEAL, fs=7.5)
box(2.75, 0.2, 6.7, 1.3, "test time (identical for SE, C0, R3 and method A in E1 and E3):\n"
    "agreement rule w = ReLU(mean over S of a_I*a_T - mean over C of a_I*a_T), L1-normalised;\n"
    "score = z(cos) + lambda z(sum_l w_l q_l c_l), lambda cross-fitted on anchor parity;\n"
    "uniform-weight control: the same score with w = 1/L", GREY, fs=7.5)
arrow(2.3, 4.2, 2.75, 4.2)
arrow(5.35, 4.45, 5.85, 4.75)
arrow(5.35, 3.95, 5.85, 3.6)
arrow(9.95, 3.55, 9.45, 3.55)
arrow(4.05, 3.6, 4.05, 1.5)
handles = [FancyBboxPatch((0, 0), 1, 1, fc=GREY, ec="#333333", label="same as C0"),
           FancyBboxPatch((0, 0), 1, 1, fc=PURPLE, ec="#333333", label="unchanged architecture"),
           FancyBboxPatch((0, 0), 1, 1, fc=TEAL, ec="#333333", label="new in method A"),
           FancyBboxPatch((0, 0), 1, 1, fc=ORANGE, ec="#333333", ls="--", label="replaced (SE's value episodes)")]
ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.0))
ax.set_title("Method A (E3) against C0 and SE: what changed", fontsize=10)
fig.savefig(HERE / "method_diagram.png", dpi=150, bbox_inches="tight")
plt.close(fig)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


json.dump(to_jsonable(DATA), open(HERE / "figure_data.json", "w"), indent=1)
print(f"{len(CHECKS)} re-derivation checks passed against the stored records")
print(f"GO {DATA['GO']}  strong GO {DATA['strong_go']}  K8 holds {DATA['k8']['holds']}  "
      f"MLLM versions {list(mllm)}  works {[v['works'] for v in mllm.values()]}")
