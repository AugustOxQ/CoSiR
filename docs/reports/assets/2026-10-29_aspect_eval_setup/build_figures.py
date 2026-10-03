"""Figures for docs/reports/auto/v2/2026-10-29_aspect_eval_setup.md (matplotlib, PNG).
Run: /root/miniconda3/envs/CoSiR/bin/python build_figures.py
Numbers are those of src/test/20261029_aspect_eval_setup/results/cub_third_aspect.json (task 5 run)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
GREY, ORANGE, TEAL, PURPLE = "#bdbdbd", "#f0a04b", "#2a9d8f", "#9b7fc4"


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def episode_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.2, 3.6, 2.0, 1.0, "anchor\nemotion e*, style s*", PURPLE)
    box(ax, 2.6, 3.6, 3.3, 1.0, "P_emo: 4 cross-item pairs\nshare an emotion, not e*", ORANGE)
    box(ax, 6.2, 3.6, 3.5, 1.0, "P_style: 4 cross-item pairs\nshare a style, not s*", ORANGE)
    box(ax, 0.2, 1.9, 2.8, 1.0, "p_emo: shares e*,\ndiffers on style", TEAL)
    box(ax, 3.3, 1.9, 2.8, 1.0, "p_style: shares s*,\ndiffers on emotion", TEAL)
    box(ax, 6.4, 1.9, 3.3, 1.0, "11 negatives: share neither\n(and lack the third aspect's value)", GREY)
    ax.text(0.2, 1.35, "Emotion condition: supports P_emo, contrasts P_style, target p_emo.", fontsize=9.5)
    ax.text(0.2, 0.95, "Style condition: supports P_style, contrasts P_emo, target p_style. Same 13 candidates, same anchor.", fontsize=9.5)
    ax.text(0.2, 0.55, "Condition gain = R@1 minus other-aspect rate; a scorer that ignores the condition scores exactly 0.", fontsize=9.5)
    ax.text(0.2, 0.15, "All 30 rows come from 30 distinct paintings. Colour legend: purple = the query, orange = condition examples, "
            "teal = targets, grey = negatives.", fontsize=8.5, color="#333")
    ax.set_title("An aspect episode and its roles", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "aspect_episode_roles.png", dpi=160); plt.close(fig)


def cub_margins():
    groups = ["has_shape", "has_wing_pattern", "has_breast_pattern", "has_wing_color"]
    img = [0.616, 0.444, 0.551, 0.430]
    cap = [0.596, 0.413, 0.507, 0.470]
    maj = [0.572, 0.359, 0.499, 0.276]
    margin = [0.024, 0.055, 0.008, 0.154]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    x = range(len(groups)); w = 0.26
    a.bar([i - w for i in x], img, w, label="image probe", color=TEAL)
    a.bar(list(x), cap, w, label="caption probe", color=ORANGE)
    a.bar([i + w for i in x], maj, w, label="majority rate (baseline)", color=GREY)
    a.set_xticks(list(x)); a.set_xticklabels(groups, rotation=15, fontsize=8); a.set_ylabel("accuracy on 30 dev species")
    a.legend(fontsize=8); a.set_title("(a) Probe accuracy against the majority rate", fontsize=10)
    cols = [GREY, GREY, GREY, PURPLE]
    b.bar(groups, margin, color=cols, hatch="//", edgecolor="#222")
    for i, m in enumerate(margin):
        b.text(i, m + 0.004, f"{m:.3f}", ha="center", fontsize=8)
    b.set_xticks(range(4)); b.set_xticklabels(groups, rotation=15, fontsize=8)
    b.set_ylabel("min(image, caption) minus majority"); b.set_title("(b) Selection margin", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "cub_third_aspect.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    episode_diagram(); cub_margins()
