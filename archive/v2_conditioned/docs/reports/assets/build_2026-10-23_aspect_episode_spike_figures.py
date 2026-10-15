"""Figure for docs/reports/auto/v2/2026-10-23_aspect_episode_spike.md.

Reads (gitignored, local only):
    src/test/20261023_aspect_episode_spike/results/aspect_results.json   (run_aspect.py)
The label-supervised ceiling numbers are the printed output of
src/test/20261023_aspect_episode_spike/aspect_ceiling.py (probes fit on 60,000 scorer-train rows, diagnostic only).
Writes docs/reports/assets/2026-10-23_aspect_episode_spike/aspect_episodes.png.

Run from the repository root:
    python docs/reports/assets/build_2026-10-23_aspect_episode_spike_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "src/test/20261023_aspect_episode_spike/results/aspect_results.json"
OUT = ROOT / "docs/reports/assets/2026-10-23_aspect_episode_spike"

SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#a3a29d"
CHANCE = 100.0 / 13

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

# Printed by aspect_ceiling.py (selection episodes, R@1 %)
CEIL = {"emotion": {"i2t": 23.58, "t2i": 24.29, "same": 36.62},   # same = caption-caption
        "style": {"i2t": 21.41, "t2i": 23.07, "same": 49.80}}     # same = image-image
CEIL_X = {k: (v["i2t"] + v["t2i"]) / 2 for k, v in CEIL.items()}
CEIL_POOLED = float(np.mean(list(CEIL_X.values())))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.load(open(RES))["main"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 4.9), dpi=200, gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) pooled R@1 per scorer
    items = [("CLIP only", m["clip"]["r1_pooled"], REFERENCE), ("SE agree", m["SE_agree_cv"]["r1_pooled"], SERIES_2),
             ("C0 agree", m["C0_agree_cv"]["r1_pooled"], "#c9c8c3"), ("R3 agree", m["R3_agree_cv"]["r1_pooled"], "#c9c8c3"),
             ("Raw CLIP agree", m["raw_agree_cv"]["r1_pooled"], "#c9c8c3"),
             ("Value prototype", m["proto_cv"]["r1_pooled"], SERIES_3),
             ("Names (privileged)", m["names_cv"]["r1_pooled"], "#0f7f58"),
             ("Label ceiling\n(cross-modal)", CEIL_POOLED, SERIES_1)]
    x = np.arange(len(items))
    for xi, (n, v, col) in zip(x, items):
        ax_a.bar(xi, v, 0.7, color=col, zorder=2, edgecolor=SURFACE)
        ax_a.text(xi, v + 0.4, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax_a.axhline(CHANCE, color=MUTED, linestyle=":", linewidth=1.0, zorder=1, label=f"chance: {CHANCE:.2f}")
    ax_a.set_xticks(x, [i[0] for i in items], fontsize=7.5, rotation=20, ha="right")
    ax_a.set_ylim(0, 28)
    ax_a.set_ylabel("R@1 (%), mean of i2t and t2i, both aspects")
    ax_a.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_a.set_title("(a) No factor model selects the aspect", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")

    # (b) per aspect
    aspects = [("emotion", "Emotion", "caption-caption"), ("style", "Art style", "image-image")]
    series = [("CLIP only", REFERENCE, lambda k: m["clip"][f"r1_{k}"]),
              ("SE agree", SERIES_2, lambda k: m["SE_agree_cv"][f"r1_{k}"]),
              ("Cross-modal ceiling", SERIES_1, lambda k: CEIL_X[k]),
              ("Same-modality ceiling", "#7fb0ee", lambda k: CEIL[k]["same"])]
    width = 0.2
    x = np.arange(len(aspects))
    for i, (n, col, f) in enumerate(series):
        vals = [f(a[0]) for a in aspects]
        xs = x + (i - 1.5) * width
        ax_b.bar(xs, vals, width * 0.92, color=col, label=n, zorder=2, edgecolor=SURFACE)
        for xi, v in zip(xs, vals):
            ax_b.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax_b.axhline(CHANCE, color=MUTED, linestyle=":", linewidth=1.0, zorder=1, label=f"chance: {CHANCE:.2f}")
    ax_b.set_xticks(x, [f"{a[1]}\n(same: {a[2]})" for a in aspects], fontsize=8.5)
    ax_b.set_ylim(0, 58)
    ax_b.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_b.set_title("(b) Where each aspect lives", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Aspect episodes (4,096 anchors, 13 candidates), selection rows; cv rows cross-fitted by anchor parity",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "aspect_episodes.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "aspect_episodes.png")


if __name__ == "__main__":
    main()
