"""Figures for docs/user_read/2026-10-10_artelingo_held_test.md. Numbers are typed from
docs/reports/auto/v2/2026-11-25_artelingo_held_test.md (Results brief and Analysis tables); nothing is recomputed.
Run: PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python <this file>"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
AFF, CTL, COS, RCA = "#2a78d6", "#1baf7a", "#555555", "#7b5ea7"
B, B0 = "#8a8984", "#b3b1ab"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

# Figure 1: pooled R@1 (full report, Results brief)
rows = [("cosine", 13.25, COS), ("RCA", 13.54, RCA), ("B", 18.81, B), ("B′(A0)", 18.84, B0),
        ("matched\ncontrol", 18.80, CTL), ("AFF", 19.44, AFF)]
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, (n, v, c) in enumerate(rows):
    ax.bar(i, v, color=c, width=0.7)
    ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=11, color=INK)
ax.set_xticks(range(len(rows)), [r[0] for r in rows])
ax.set_ylabel("Pooled R@1 (%), 36,864 held episodes")
ax.set_ylim(0, 23)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("AFF is far above cosine and RCA, and about 0.6 points above\nB, B′(A0) and its matched control",
             fontsize=11.5, loc="left", color=INK)
fig.text(0.01, 0.01, "Figure 1. R@1 on the held paintings, pooled over three seeds and three aspect pairs.",
         fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "fig1_pooled_r1.png", dpi=170)
plt.close(fig)

# Figure 2: per-pair margin over B'(A0), AFF minus it (full report, Analysis table 1 and Results brief)
pairs = [("emotion × style\n(held)", 0.59, 0.35, 0.83, AFF), ("emotion × genre\n(held)", 1.64, 1.40, 1.90, AFF),
         ("style × genre\n(held)", -0.43, -0.65, -0.21, AFF), ("style × genre\n(round 3)", -0.580, -0.793, -0.363, B)]
fig, ax = plt.subplots(figsize=(8, 4.4))
for i, (n, m, lo, hi, c) in enumerate(pairs):
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=5, lw=2, ms=8)
    ax.text(*((hi + 0.06, i) if m > 0 else (lo - 0.06, i)), f"{m:+.2f}", va="center",
            ha="left" if m > 0 else "right", fontsize=11, color=INK)
ax.axvline(0, color=INK, lw=1.3)
ax.set_yticks(range(len(pairs)), [p[0] for p in pairs]); ax.invert_yaxis()
ax.set_xlim(-1.25, 2.25)
ax.set_xlabel("AFF minus B′(A0), R@1 points (95% interval)")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.text(0.03, 0.97, "← AFF worse", transform=ax.transAxes, fontsize=10, color=INK2, va="top")
ax.set_title("AFF wins two aspect pairs and loses style × genre, as in round 3", fontsize=11.5, loc="left", color=INK)
fig.text(0.01, 0.01, "Figure 2. Margin over the condition-free scorer B′(A0), 12,288 held episodes per pair.",
         fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "fig2_per_pair.png", dpi=170)
