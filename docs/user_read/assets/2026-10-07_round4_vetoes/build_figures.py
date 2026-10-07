"""Figures for docs/user_read/2026-10-07_round4_vetoes.md, the user-read briefing on round 4 of the CoSiR v2 reader-fix
line (three vetoes on affect steering's gate). Full report: docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md.

Both charts read only the full report's figure data; nothing is recomputed except sums of stored integer counts.
  docs/reports/assets/2026-11-22_round4_aff_vetoes/figure_data.json
    rederived.records.{V4,V2,V24}.delta, delta_int     each veto minus affect steering (AFF), seed 42
    descriptive.closure.{V4,V2}.as_scored               share of AFF's open values each veto shuts, as scored
The script asserts that the stored values match the numbers the full report prints.

Run from the repo root:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/user_read/assets/2026-10-07_round4_vetoes/build_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[4]
FD = ROOT / "docs/reports/assets/2026-11-22_round4_aff_vetoes/figure_data.json"
OUT = Path(__file__).resolve().parent
DPI = 150

# Colours: the dataviz reference palette, categorical slots 1 to 3 (validated, light mode), in the same order as the
# full report's figures (V4 blue, V2 orange, V24 aqua). Every mark is also labelled directly.
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
C_V4 = "#2a78d6"   # image-agreement veto
C_V2 = "#eb6834"   # CSD-informed veto
C_V24 = "#1baf7a"  # both vetoes

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.labelsize": 13,
    "xtick.color": INK2,
    "ytick.color": INK,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
WHITE_BG = dict(fc="white", ec="none", pad=1.5)


def close(a, b, tol=6e-4):
    assert abs(a - b) < tol, (a, b)


def sg(v, d=3):
    """Signed number with a true minus sign."""
    return f"{v:+.{d}f}".replace("-", "−")


def title(fig, text, y=0.975):
    fig.text(0.02, y, text, fontsize=14.5, fontweight="bold", va="top", color=INK)


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------------------------- data
def data():
    fd = json.loads(FD.read_text())
    rec = fd["rederived"]["records"]
    assert fd["rederived"]["E"] == [] and fd["rederived"]["kill"] is True
    d = {"delta": {}, "share": {}}
    # each veto minus AFF (full report Table 3 and Figure 2, right)
    printed = {"V4": (-18, -0.037, -0.105, +0.033), "V2": (-18, -0.037, -0.115, +0.039),
               "V24": (-34, -0.069, -0.165, +0.026)}
    for k, (n, p, lo, hi) in printed.items():
        r = rec[k]
        assert r["delta_int"] == n
        close(r["delta"]["point"], p)
        close(r["delta"]["ci95"][0], lo)
        close(r["delta"]["ci95"][1], hi)
        close(100 * n / 49152, r["delta"]["point"], 1e-12)
        d["delta"][k] = (n, r["delta"]["point"], r["delta"]["ci95"][0], r["delta"]["ci95"][1])
    # bar margins and D10 outcome the briefing's table prints
    for k, bm, clears in (("AFF", 0.700, True), ("V4", 0.663, True), ("V2", 0.295, False), ("V24", 0.262, False)):
        close(rec[k]["bar_margin"]["point"], bm)
        assert rec[k]["d10"]["clears"] is clears
    close(rec["V24"]["bar_margin"]["ci95"][0], -0.016)

    # closure shares as scored, summed by side from the per (pair, condition) counts
    sides = {
        "emotion": ["emotion__style|a", "emotion__genre|a"],
        "other_emotion_pairs": ["emotion__style|b", "emotion__genre|b"],
        "sxg": ["style__genre|a", "style__genre|b"],
    }
    for k in ("V4", "V2", "V24"):
        cs = fd["descriptive"]["closure"][k]["as_scored"]
        d["share"][k] = {}
        for s, keys in sides.items():
            opened = sum(cs[x]["aff_open"] for x in keys)
            closed = sum(cs[x]["closed"] for x in keys)
            d["share"][k][s] = (100 * closed / opened, closed, opened)
        assert cs["emotion_side"]["closed"] == d["share"][k]["emotion"][1]
        assert cs["emotion_side"]["aff_open"] == d["share"][k]["emotion"][2]
    # the numbers the full report prints (Summary, Table 5)
    close(d["share"]["V4"]["emotion"][0], 12.2, 0.05)
    close(d["share"]["V4"]["sxg"][0], 19.6, 0.05)
    close(d["share"]["V2"]["emotion"][0], 6.2, 0.05)
    close(fd["descriptive"]["closure"]["V2"]["as_scored"]["other_values"]["share"], 29.6, 0.05)
    # V24 closes exactly what either veto closes (its gate is V4's times V2's), so it is left out of Figure 2
    return d


# ----------------------------------------------------------------------------------------------- figure 1
def fig1(d):
    rows = [("V4", "image-agreement veto", C_V4), ("V2", "CSD-informed veto", C_V2), ("V24", "both vetoes", C_V24)]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.subplots_adjust(left=0.31, right=0.97, top=0.78, bottom=0.27)
    title(fig, "No veto beat affect steering on the\ndevelopment episodes")
    ax.axvspan(0, 0.12, color="#f1f6fc", zorder=0)
    ax.axvline(0, color=INK, lw=1.4, zorder=1)
    ax.text(0.045, -0.42, "beats affect\nsteering →", fontsize=11.5, color=INK2, va="bottom")
    for i, (k, lab, c) in enumerate(rows):
        y = 2 - i
        n, p, lo, hi = d["delta"][k]
        ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot(p, y, "o", ms=11, color=c, mec="white", mew=1.8, zorder=4)
        ax.text(lo - 0.006, y + 0.2, f"{sg(p)}  ({sg(n, 0)} of 49,152 rankings)", fontsize=11.5, color=INK,
                va="bottom", bbox=WHITE_BG)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 2.7)
    ax.set_xlim(-0.19, 0.12)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("R@1, veto minus affect steering (points)")
    fig.legend(handles=[Line2D([], [], color=INK2, marker="o", lw=0, ms=9, label="difference in R@1"),
                        Line2D([], [], color=INK2, lw=2.4, label="95% interval"),
                        Line2D([], [], color=INK, lw=1.4, label="affect steering (0)")],
               loc="lower center", ncol=3, frameon=False, fontsize=11.5, bbox_to_anchor=(0.55, 0.0))
    save(fig, "fig1_vs_affect_steering.png")


# ----------------------------------------------------------------------------------------------- figure 2
def fig2(d):
    groups = [("emotion", "emotion side\n(steering paid here)"),
              ("other_emotion_pairs", "other side of the\ntwo emotion pairs"),
              ("sxg", "style × genre\n(both sides)")]
    vetoes = [("V4", "image-agreement veto", C_V4), ("V2", "CSD-informed veto", C_V2)]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.15)
    title(fig, "The image veto switched off every side at similar rates;\nthe CSD veto spared the emotion side")
    w = 0.36
    for j, (k, lab, c) in enumerate(vetoes):
        for i, (s, _) in enumerate(groups):
            x = i + (j - 0.5) * (w + 0.03)
            v = d["share"][k][s][0]
            ax.bar(x, v, width=w, color=c, zorder=2)
            ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 50)
    ax.set_ylabel("cases switched off (% of\nthose affect steering steered)")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(color=c, label=lab) for _, lab, c in vetoes], loc="upper left", ncol=2,
               frameon=False, fontsize=12, bbox_to_anchor=(0.11, 0.86))
    save(fig, "fig2_where_switched_off.png")


if __name__ == "__main__":
    dd = data()
    for k in ("V4", "V2", "V24"):
        print(k, {s: f"{v[0]:.1f}% ({v[1]}/{v[2]})" for s, v in dd["share"][k].items()})
    fig1(dd)
    fig2(dd)
