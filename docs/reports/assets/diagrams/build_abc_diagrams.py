#!/usr/bin/env python3
"""Generate the ABC bridge diagram (Experiment 12) and the ABCD positive-control
diagram (Experiment 14) for the 2026-09-02 weekly report/slides.

Uses the same palette as docs/reports/assets/build_2026-08-26_weekly_slides.py
so the figures drop cleanly into the existing deck style.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent

BLACK = "#1A1A1A"
GRAY = "#595959"
BORDER = "#BFBFBF"
BLUE = "#1F77B4"     # image-only edges
ORANGE = "#FF7F0E"   # text-only edges
RED = "#C03030"      # the "no real edge" / false-transitivity gap
NODE_FILL = "#F5F5F5"

plt.rcParams["font.family"] = "DejaVu Sans"


def node(ax, xy, label, r=0.5, fill=NODE_FILL, edge=BLACK, fontsize=20, fontweight="bold"):
    c = Circle(xy, r, facecolor=fill, edgecolor=edge, linewidth=2.0, zorder=3)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=BLACK, zorder=4)
    return c


def edge(ax, p, q, color, style="-", lw=2.6, label=None, label_dy=0.28, label_dx=0.0,
         fontsize=13, curve=0.0):
    conn = f"arc3,rad={curve}" if curve else "arc3,rad=0"
    arrow = FancyArrowPatch(p, q, connectionstyle=conn, arrowstyle="-",
                             linestyle=style, linewidth=lw, color=color, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx, my = (p[0] + q[0]) / 2 + label_dx, (p[1] + q[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=color, fontweight="bold")


def new_axes(figsize=(11, 5.4), xlim=(-1, 11), ylim=(-1, 6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def build_exp12_bridge():
    """A = bridge node, B = image-only neighbor, C = text-only neighbor.
    B and C are never directly connected, yet the trained embedding pulls
    them together — the 'false transitivity' pull Experiment 12 measures."""
    fig, ax = new_axes(figsize=(10.5, 5.2), xlim=(-1, 10.5), ylim=(-0.8, 5.2))

    A = (5.0, 3.3)
    B = (1.4, 0.9)
    C = (8.6, 0.9)

    edge(ax, A, B, BLUE, label="image-only edge\n(real, mutual kNN)", label_dx=-0.55, label_dy=0.55, fontsize=12.5)
    edge(ax, A, C, ORANGE, label="text-only edge\n(real, mutual kNN)", label_dx=0.55, label_dy=0.55, fontsize=12.5)
    edge(ax, B, C, RED, style=(0, (6, 4)), lw=2.2, curve=-0.18,
         label="no edge of any kind —\nyet the embedding pulls B and C together\n(“false transitivity”, mean pull +1.98, mean/SEM +102.1)",
         label_dy=-1.05, fontsize=12.2)

    node(ax, A, "A", fill="#DCE9F5")
    node(ax, B, "B")
    node(ax, C, "C")

    ax.text(A[0], A[1] + 0.75, "bridge node\n(disagrees across modalities)", ha="center", va="bottom",
            fontsize=11.5, color=GRAY, style="italic")
    ax.text(B[0], B[1] - 0.75, "B: A's image-only\nneighbor", ha="center", va="top", fontsize=11, color=GRAY)
    ax.text(C[0], C[1] - 0.75, "C: A's text-only\nneighbor", ha="center", va="top", fontsize=11, color=GRAY)

    ax.text(5.0, 5.0, "Experiment 12 — the bridge pair (A, B, C)", ha="center", va="top",
            fontsize=15, fontweight="bold", color=BLACK)

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "exp12_bridge_abc.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def build_exp14_positive_control():
    """Two panels: genuinely-unconnected C-D (the control Experiment 12 lacked)
    vs. closed-triangle C-D (a real image-only edge) — Experiment 14's fix."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=200)

    for ax, kind in zip(axes, ["unconnected", "closed"]):
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1.3, 7.4)
        ax.set_aspect("equal")
        ax.axis("off")

        A = (4.0, 3.6)
        C = (1.2, 0.5)
        D = (6.8, 0.5)

        edge(ax, A, C, ORANGE, label="text-only", label_dx=-0.5, label_dy=0.45, fontsize=12)
        edge(ax, A, D, ORANGE, label="text-only", label_dx=0.5, label_dy=0.45, fontsize=12)

        if kind == "unconnected":
            edge(ax, C, D, GRAY, style=(0, (6, 4)), lw=2.0, curve=-0.15,
                 label="no edge of any kind\n(genuinely-unconnected control)", label_dy=-0.85, fontsize=12)
            title = "Genuinely-unconnected control\npull mean = +2.4248 ± 0.0096"
        else:
            edge(ax, C, D, BLUE, lw=2.8, curve=-0.15,
                 label="real image-only edge\n(closed triangle)", label_dy=-0.85, fontsize=12)
            title = "Closed-triangle positive control\npull mean = +3.1772 ± 0.0034"

        node(ax, A, "A", fill="#DCE9F5")
        node(ax, C, "C")
        node(ax, D, "D")
        ax.text(A[0], A[1] + 0.72, "hub node\n(≥ 2 text-only neighbors)", ha="center", va="bottom",
                fontsize=10.8, color=GRAY, style="italic")
        ax.text(4.0, 6.9, title, ha="center", va="top", fontsize=13.5, fontweight="bold", color=BLACK)

    fig.suptitle("Experiment 14 — closed-triangle vs. genuinely-unconnected (C, D) pairs   •   ratio 1.31×",
                  fontsize=15.5, fontweight="bold", y=1.04)
    fig.tight_layout(pad=0.4)
    out = OUT_DIR / "exp14_positive_control_abcd.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    build_exp12_bridge()
    build_exp14_positive_control()
