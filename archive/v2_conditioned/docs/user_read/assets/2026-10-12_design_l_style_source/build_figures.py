"""Figures for docs/user_read/2026-10-12_design_l_style_source.md. Numbers are read from the stored result files of
src/test/20261126_design_l_style_source/results/ (descriptive_ceiling.json, descriptive_gate.json, gate_verdict.json);
nothing is recomputed. Run: PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python <this file>"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ONLY = set(sys.argv[1:])  # e.g. `fig2` rebuilds only figure 2; no argument rebuilds both
OUT = Path(__file__).resolve().parent
RES = Path("/project/CoSiR/src/test/20261126_design_l_style_source/results")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
AFF, R1, CTL, REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
dc = json.load(open(RES / "descriptive_ceiling.json"))
dg = json.load(open(RES / "descriptive_gate.json"))
gv = json.load(open(RES / "gate_verdict.json"))
S = dg["scorers"]
for v, rc in dg["random_controls"].items():
    for k, s in rc["scorers"].items():
        S[f"{k}_{rc['grouping']}"] = s


def sg(name):
    m = S[name]["per_pair"]["style__genre"]["margin"]
    return m["point"], m["ci95"][0], m["ci95"][1]


# Figure 1: style x genre, fused minus matched counterpart
t = dc["told_T"]["per_pair"]["style__genre"]["margin"]
rows = [("told true styles\n(the ceiling)", t["point"], t["ci95"][0], t["ci95"][1], INK),
        ("extended AFF on V3", *sg("XAFF_V3"), AFF), ("R1 on V3", *sg("R1_V3"), R1),
        ("R1 on V2", *sg("R1_V2"), R1), ("extended AFF on raw CSD\n(reference)", *sg("XAFF_R0"), REF)]
fig, ax = plt.subplots(figsize=(8.4, 4.9))
for i, (n, m, lo, hi, c) in enumerate(rows):
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=5, lw=2, ms=8)
    ax.text(hi + 0.12, i, f"{m:+.2f}", va="center", fontsize=11, color=INK)
zs = [sg(f"{k}_Z{i}")[0] for k in ("R1", "XAFF") for i in range(1, 7)]
ax.plot(zs, [len(rows)] * len(zs), "o", color=CTL, mfc="none", ms=7)
ax.text(max(zs) + 0.25, len(rows), f"{min(zs):+.2f} to {max(zs):+.2f}", va="center", fontsize=11, color=INK)
labels = [r[0] for r in rows] + ["random groupings\n(12 scorers)"]
ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
ax.axvline(0, color=INK, lw=1.2)
ax.set_xlim(-1.0, 5.3)
ax.set_xlabel("Fused minus matched counterpart, style × genre, R@1 points (95% interval)")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("A told true-style grouping gains +3.9 on style × genre;\nthe label-free groupings gain at most +0.4", fontsize=11.5, loc="left", color=INK)
fig.text(0.01, 0.01, "Figure 1. Seed 42, 4,096 style × genre episodes. The ceiling is told the style; the other scorers use a reader.",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
if not ONLY or "fig1" in ONLY:
    fig.savefig(OUT / "fig1_ceiling_vs_candidates.png", dpi=170)
plt.close(fig)

# Figure 2: the pass corner is empty (S1 against S3 emotion x genre)
fig, ax = plt.subplots(figsize=(8.4, 5.2))
for name, r in list(gv["candidates"].items()) + list(gv["references"].items()):
    kind, x = name.split("_", 1)
    if x[0] not in "VZ":
        continue
    px = 25 * r["S3"]["emotion__genre"]["sum4d"] / 4096
    py = 25 * r["S1"]["sum4d"] / 12288
    col = R1 if kind == "R1" else AFF
    if x[0] == "V":
        ax.plot(px, py, "o", color=col, ms=8)
        if name in ("R1_V2", "R1_V3", "XAFF_V3"):
            ax.annotate(name.replace("XAFF", "X-AFF").replace("_", " "), (px, py), textcoords="offset points", xytext=(6, 5), fontsize=9.5, color=INK2)
    else:
        ax.plot(px, py, "o", color=col, mfc="none", ms=7)
ax.axhline(0.5, color=INK, lw=1.1, ls="--"); ax.axvline(-0.3, color=INK, lw=1.1, ls="--")
ax.text(-1.58, 0.52, "S1 bar +0.5", fontsize=9.5, color=INK2)
ax.text(-0.28, -0.62, "S3 bar −0.3", fontsize=9.5, color=INK2)
ax.fill_between([-0.3, 0.2], 0.5, 0.65, color=GRID, alpha=0.8)
ax.text(-0.27, 0.57, "pass corner: empty", fontsize=10, color=INK)
ax.set_xlim(-1.75, 0.2); ax.set_ylim(-0.65, 0.65)
ax.set_xlabel("Emotion × genre, scorer minus AFF, R@1 points (S3 needs at least −0.3)")
ax.set_ylabel("Pooled margin over the strongest\ncondition-free scorer, R@1 points (S1 needs +0.5)")
ax.xaxis.grid(True, color=GRID); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color="w", mfc=R1, ms=8, label="R1, label-free grouping"),
                   Line2D([], [], marker="o", color="w", mfc=AFF, ms=8, label="extended AFF, label-free grouping"),
                   Line2D([], [], marker="o", color="w", mfc="none", mec=INK2, ms=7, label="random grouping, same sizes")],
          loc="lower right", bbox_to_anchor=(0.74, 0.0), frameon=False, fontsize=9.5)
ax.set_title("No scorer reaches the pass corner; on emotion × genre the random groupings\nlose less than the real ones (we did not trace why)", fontsize=11.5, loc="left", color=INK)
fig.text(0.01, 0.01, "Figure 2. Seed 42, 12 candidate scorers and their 12 random controls. Style × genre (S2) is not drawn.",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
if not ONLY or "fig2" in ONLY:
    fig.savefig(OUT / "fig2_pass_corner.png", dpi=170)
plt.close(fig)
