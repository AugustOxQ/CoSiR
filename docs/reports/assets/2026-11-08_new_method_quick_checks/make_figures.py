"""Figures for the 2026-11-08 new-method quick checks report. Reads only the JSON results; writes PNGs next to itself."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/project/CoSiR/src/test/20261108_new_method_quick_checks/results/"
R = json.load(open(RES + "checks_seed42.json"))
A = json.load(open(RES + "addendum1.json"))
T = json.load(open(RES + "test_seeds.json"))

INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
GREY, BLUE, ORANGE, TEAL, PURPLE = "#8a8a8a", "#2a6fb0", "#d9822b", "#1b9e8a", "#7b5ea7"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white"})

def pt(d): return d["point"]
def err(d): return [[d["point"] - d["ci95"][0]], [d["ci95"][1] - d["point"]]]

# ---------------- 1. D0 ----------------
d0 = R["d0"]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
rows = [("Cosine", R["cosine"], GREY), ("Told", d0["told"], BLUE),
        ("Inferred hard", d0["hard"], ORANGE), ("Inferred soft", d0["soft"], TEAL)]
mets = [("R@1", lambda r: r["summary"]["r1"]), ("Condition gain", lambda r: r["summary"]["gain"]),
        ("Either rate", lambda r: r["either"])]
w = 0.2
for i, (name, r, c) in enumerate(rows):
    xs = np.arange(3) + (i - 1.5) * w
    ds = [f(r) for _, f in mets]
    a.bar(xs, [pt(d) for d in ds], w * 0.9, color=c, label=name, hatch=["", "//", "xx", ".."][i],
          edgecolor="white", linewidth=0.5, zorder=2)
    a.errorbar(xs, [pt(d) for d in ds], yerr=np.array([err(d) for d in ds]).reshape(3, 2).T,
               fmt="none", ecolor=INK, capsize=2, lw=1, zorder=3)
    for x, d in zip(xs, ds):
        a.text(x, d["ci95"][1] + 0.6, f"{pt(d):.1f}", ha="center", fontsize=7.5, color=INK)
a.set_xticks(range(3)); a.set_xticklabels([m for m, _ in mets])
a.set_ylabel("pp"); a.set_title("(a) Pooled, seed 42 (95% CI)"); a.legend(frameon=False, ncol=2, loc="upper left")
a.set_ylim(0, 50)

pairs = [("emotion__style", "emotion x style"), ("emotion__genre", "emotion x genre"), ("style__genre", "style x genre")]
share = [100 * pt(d0["hard"]["per_pair"][p]["gain"]) / pt(d0["told"]["per_pair"][p]["gain"]) for p, _ in pairs]
pooled = 100 * pt(d0["hard"]["summary"]["gain"]) / pt(d0["told"]["summary"]["gain"])
acc = [d0["hard_pick_accuracy_per_pair"][p] for p, _ in pairs]
labels = [n for _, n in pairs] + ["Pooled"]
vals = share + [pooled]
cols = [ORANGE] * 3 + [PURPLE]
b.bar(range(4), vals, 0.6, color=cols, zorder=2)
b.axhline(50, ls="--", color=INK, lw=1)
b.text(-0.35, 51.5, "decision threshold (pooled)", ha="left", fontsize=8.5)
for i, v in enumerate(vals):
    b.text(i, v + 2.2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    if i < 3:
        b.text(i, v / 2, f"pick acc.\n{acc[i]:.0f}%", ha="center", va="center", color="white", fontsize=8.5)
    else:
        b.text(i, v / 2, f"pick acc.\n{d0['hard_pick_accuracy']['pooled']:.0f}%\n(pooled)", ha="center", va="center", color="white", fontsize=8.5)
b.set_xticks(range(4)); b.set_xticklabels(labels, fontsize=9)
b.set_ylabel("Inferred-hard gain as % of Told gain"); b.set_ylim(0, 100)
b.set_title("(b) Share of Told's gain recovered by hard inference")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "d0.png"), dpi=160); plt.close(fig)

# ---------------- 2. N1 trade-off ----------------
f3 = R["factors"]["A3"]; mA = A["dev"]["models"]["A3"]
def pr(summary, either): return (either, summary["gain"], summary["r1"])
pts = [
    ("Cosine", R["cosine"]["either"], R["cosine"]["summary"]["gain"], "free", GREY, "o", (8, -12)),
    ("RCA (GO bar)", R["rca"]["either"], R["rca"]["summary"]["gain"], "free", GREY, "s", (8, 6)),
    ("A3 term-only uncentered uniform", f3["term_only"]["uniform"]["either"], f3["term_only"]["uniform"]["summary"]["gain"], "free", BLUE, "^", (-8, -14)),
    ("A3 term-only N1 uniform (centered)", f3["term_only"]["N1_uniform"]["either"], f3["term_only"]["N1_uniform"]["summary"]["gain"], "free", BLUE, "v", (-8, 8)),
    ("A3 declared control (nested uniform)", f3["control"]["either"], f3["control"]["summary"]["gain"], "free", "#08306b", "D", (-8, -16)),
    ("A3 matched control (centered)", mA["matched_either"], mA["matched"]["gain"], "free", "#08306b", "P", (8, 10)),
    ("A3 term-only current rule (agree)", f3["term_only"]["agree"]["either"], f3["term_only"]["agree"]["summary"]["gain"], "read", ORANGE, "o", (8, 4)),
    ("A3 term-only N1", f3["term_only"]["N1"]["either"], f3["term_only"]["N1"]["summary"]["gain"], "read", ORANGE, "s", (8, 4)),
    ("A3 nested current rule", f3["nested"]["agree"]["either"], f3["nested"]["agree"]["summary"]["gain"], "read", "#a04a00", "^", (8, -14)),
    ("A3 nested N1", f3["nested"]["N1"]["either"], f3["nested"]["N1"]["summary"]["gain"], "read", "#a04a00", "D", (8, 6)),
]
OFFS = {'Cosine': (-45, -12), 'RCA (GO bar)': (0, 28), 'A3 term-only uncentered uniform': (-60, -58), 'A3 term-only N1 uniform (centered)': (30, 40), 'A3 declared control (nested uniform)': (-110, -34), 'A3 matched control (centered)': (30, -45), 'A3 term-only current rule (agree)': (10, 18), 'A3 term-only N1': (14, 10), 'A3 nested current rule': (-10, -82), 'A3 nested N1': (-60, 36)}
fig, ax = plt.subplots(figsize=(11, 6.4))
xr = np.linspace(10, 45, 50)
for lvl, yl in ((13, 1.75), (16.55, 1.75), (17.94, 1.75)):
    ax.plot(xr, 2 * lvl - xr, ls=":", color=MUTED, lw=1)
    ax.text(2 * lvl - yl + 0.3, yl, f"R@1 = {lvl:g}", fontsize=8.5, color=MUTED, ha="left", va="center", clip_on=True)
for name, ex, gn, kind, c, m, off in pts:
    ax.errorbar(ex["point"], gn["point"], xerr=err(ex), yerr=err(gn), fmt="none", ecolor=c, alpha=0.5, lw=1, zorder=2)
    ax.scatter(ex["point"], gn["point"], s=70, c=c, marker=m, edgecolor="white", linewidth=1, zorder=3)
    o = OFFS[name]
    ax.annotate(name, (ex["point"], gn["point"]), textcoords="offset points", xytext=o, fontsize=8,
                ha="left" if o[0] >= 0 else "right", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlabel("Either rate (pp)"); ax.set_ylabel("Condition gain (pp)")
ax.set_title("Either rate vs condition gain, A3, seed 42 (95% CI)")
ax.set_xlim(17, 41); ax.set_ylim(-0.7, 2.0)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Condition-free scorers (blue, grey)"),
                   Line2D([], [], marker="o", ls="", color=ORANGE, label="Condition-reading scorers (orange, brown)"),
                   Line2D([], [], ls=":", color=MUTED, label="Iso-R@1: R@1 = (either + gain) / 2")],
          frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "n1_tradeoff.png"), dpi=160); plt.close(fig)

# ---------------- 3. N2 ----------------
ks = [2, 3, 5]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
d = [R["n2"][f"both_in_top{k}"] for k in ks]
a.bar(range(3), [pt(x) for x in d], 0.55, color=PURPLE, zorder=2)
a.errorbar(range(3), [pt(x) for x in d], yerr=np.array([err(x) for x in d]).reshape(3, 2).T, fmt="none", ecolor=INK, capsize=3)
for i, x in enumerate(d): a.text(i, x["ci95"][1] + 0.7, f"{pt(x):.1f}", ha="center", fontsize=9)
a.set_xticks(range(3)); a.set_xticklabels([f"k = {k}" for k in ks]); a.set_ylabel("Share of rankings (pp)")
a.set_title("(a) Both aspect candidates in control's top k")
terms = [("agree", "Current rule (agree)", ORANGE, "o", -0.07), ("N1", "N1", TEAL, "s", 0.07)]
for name, lab, c, m, dx in terms:
    for ax_, key, sub in ((b, "r1", "R@1 change"),):
        pass
    xs = np.arange(3) + dx
    r1 = [R["n2"][f"{k}-{name}"]["vs_control"]["r1"] for k in ks]
    b.errorbar(xs, [pt(x) for x in r1], yerr=np.array([err(x) for x in r1]).reshape(3, 2).T, fmt=m, color=c,
               mfc=c, ms=7, capsize=3, label=f"{lab}: R@1 change")
    gn = [R["n2"][f"{k}-{name}"]["vs_control"]["gain"] for k in ks]
    b.errorbar(xs + 0.0, [pt(x) for x in gn], yerr=np.array([err(x) for x in gn]).reshape(3, 2).T, fmt=m, color=c,
               mfc="white", ms=7, capsize=3, label=f"{lab}: condition gain")
b.axhline(0, color=INK, lw=1)
b.set_xticks(range(3)); b.set_xticklabels([f"k = {k}" for k in ks])
b.set_ylabel("pp vs unreordered control"); b.set_title("(b) Reorder terms vs control (filled R@1, open gain)")
b.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "n2.png"), dpi=160); plt.close(fig)

# ---------------- 4. forest ----------------
for c in ("cosine", "rca", "control"):
    for m in ("r1", "gain"):
        assert A["test"]["pooled"]["vs"][c][m] == T["pooled"]["vs"][c][m], (c, m)
comps = [("control", "Declared control"), ("matched", "Matched control"), ("rca", "RCA"), ("cosine", "Cosine")]
seeds = ["45", "47", "48"]
fig, axs = plt.subplots(1, 2, figsize=(11, 7.2), sharey=True)
ylab, ypos, y = [], [], 0
layout = []
for c, cl in comps:
    for s in seeds + ["pooled"]:
        layout.append((c, s, y)); ylab.append(f"{cl}: " + ("pooled" if s == "pooled" else f"seed {s}")); ypos.append(y); y += 1
    y += 0.7
for ax_, m, title in zip(axs, ("r1", "gain"), ("R@1 difference (pp)", "Condition gain difference (pp)")):
    for c, s, yy in layout:
        d = (A["test"]["pooled"]["vs"] if s == "pooled" else A["test"]["per_seed"][s]["vs"])[c][m]
        pooled = s == "pooled"
        col = ORANGE if pooled else BLUE
        ax_.errorbar(d["point"], yy, xerr=err(d), fmt="D" if pooled else "o", color=col, mfc=col,
                     ms=8 if pooled else 5, lw=2.6 if pooled else 1.2, capsize=3)
    ax_.axvline(0, color=INK, lw=1)
    ax_.set_xlabel(title); ax_.grid(axis="y", visible=False)
axs[0].set_yticks(ypos); axs[0].set_yticklabels(ylab, fontsize=8.5); axs[0].invert_yaxis()
fig.suptitle("N1-nested-A3 minus comparator, test seeds (95% CI; diamonds = pooled)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "test_forest.png"), dpi=160); plt.close(fig)

# ---------------- 5. N6 trade-off ----------------
N = json.load(open(RES + "n6_seed42.json"))
G = json.load(open(RES + "n6c_gate.json"))
def e_g(d): return d["either"], d["summary"]["gain"]
pts6 = [
    ("Cosine", *e_g(G["cosine"]), GREY, "o", (-40, 14)),
    ("RCA", *e_g(G["rca"]), GREY, "s", (-10, 36)),
    ("T6u term (label-free head)", *e_g(N["term_only"]["T6u"]), BLUE, "^", (-70, -40)),
    ("N6 control (nested)", *e_g(N["control"]), "#08306b", "D", (-110, 40)),
    ("C1 (centered, cosine)", *e_g(G["control"]), "#08306b", "P", (10, -45)),
    ("C2 (N6c, condition removed)", *e_g(G["matched"]), "#08306b", "X", (30, 30)),
    ("T6 term (hard reader)", *e_g(N["term_only"]["T6"]), ORANGE, "o", (14, 8)),
    ("T6soft term", *e_g(N["term_only"]["T6soft"]), ORANGE, "s", (14, 8)),
    ("N6-nested", *e_g(N["nested"]), "#a04a00", "D", (-80, 40)),
    ("N6c (N6 term, centered A3 base)", *e_g(G["config"]), "#a04a00", "^", (35, 55)),
]
fig, ax = plt.subplots(figsize=(11, 6.8))
xr = np.linspace(10, 45, 50)
for lvl in (13, 17.07, 18.34, 18.49):
    ax.plot(xr, 2 * lvl - xr, ls=":", color=MUTED, lw=1)
for lvl, yl in ((13, 4.1), (17.07, 4.1), (18.34, 4.6), (18.49, 3.0)):
    ax.text(2 * lvl - yl + 0.3, yl, f"R@1 = {lvl:g}", fontsize=8.5, color=MUTED, ha="left", va="center", clip_on=True)
for name, ex, gn, c, m, o in pts6:
    ax.errorbar(ex["point"], gn["point"], xerr=err(ex), yerr=err(gn), fmt="none", ecolor=c, alpha=0.5, lw=1, zorder=2)
    ax.scatter(ex["point"], gn["point"], s=70, c=c, marker=m, edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(name, (ex["point"], gn["point"]), textcoords="offset points", xytext=o, fontsize=8,
                ha="left" if o[0] >= 0 else "right", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlabel("Either rate (pp)"); ax.set_ylabel("Condition gain (pp)")
ax.set_title("Either rate vs condition gain, N6 variants, seed 42 (95% CI)")
ax.set_xlim(22, 40); ax.set_ylim(-0.8, 5.0)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Condition-free scorers (blue, grey)"),
                   Line2D([], [], marker="o", ls="", color=ORANGE, label="Condition-reading scorers (orange, brown)"),
                   Line2D([], [], ls=":", color=MUTED, label="Iso-R@1: R@1 = (either + gain) / 2")],
          frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "n6_tradeoff.png"), dpi=160); plt.close(fig)
print("ok")
