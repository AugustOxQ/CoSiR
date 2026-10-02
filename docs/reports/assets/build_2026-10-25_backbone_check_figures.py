"""Figure for docs/reports/auto/v2/2026-10-25_backbone_check.md.

Reads (gitignored, local only): src/test/20261025_backbone_check/results/backbone_check.json
Writes docs/reports/assets/2026-10-25_backbone_check/backbone_check.png.

Run from the repository root:
    python docs/reports/assets/build_2026-10-25_backbone_check_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "src/test/20261025_backbone_check/results/backbone_check.json"
OUT = ROOT / "docs/reports/assets/2026-10-25_backbone_check"

SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#a3a29d"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
MODELS = [("clip", "CLIP B/32"), ("siglip2", "SigLIP 2"), ("pe", "PE-Core"), ("qwen", "Qwen3-VL-Emb")]
EMO_MAJORITY = 31.8  # majority class of the emotion probe, from the aspect-episode spike report


def bars(ax, groups, labels, colors, ylim, fmt="{:.1f}"):
    n = len(groups)
    w = 0.8 / n
    x = np.arange(len(MODELS))
    for i, (g, lab, col) in enumerate(zip(groups, labels, colors)):
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, g, w * 0.92, color=col, label=lab, zorder=2, edgecolor=SURFACE)
        for xi, v in zip(xs, g):
            ax.text(xi, v + 0.6, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [m[1] for m in MODELS], fontsize=8.5)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(RES))
    a = {k: d[k]["artelingo"] for k, _ in MODELS}
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 4.9), dpi=200)

    bars(ax_a, [[a[k]["probe_emotion_img"] for k, _ in MODELS], [a[k]["probe_style_txt"] for k, _ in MODELS],
                [a[k]["probe_emotion_txt"] for k, _ in MODELS], [a[k]["probe_style_img"] for k, _ in MODELS]],
         ["Emotion from images (weak)", "Style from captions (weak)", "Emotion from captions (strong)",
          "Style from images (strong)"], ["#7fb0ee", "#f4a97f", SERIES_1, SERIES_2], (0, 85))
    ax_a.axhline(EMO_MAJORITY, color=MUTED, linestyle=":", linewidth=1.0, zorder=1,
                 label=f"emotion majority class: {EMO_MAJORITY}")
    ax_a.set_ylabel("Probe accuracy (%)")
    ax_a.set_title("(a) Weak-side probes stay put, strong-side probes move", fontsize=11, color=INK, loc="left",
                   fontweight="bold")
    ax_a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)

    bars(ax_b, [[a[k]["ceil_emotion_cross_mean"] for k, _ in MODELS], [a[k]["ceil_style_cross_mean"] for k, _ in MODELS]],
         ["Emotion cross-modal ceiling", "Style cross-modal ceiling"], [SERIES_1, SERIES_2], (0, 34), "{:.2f}")
    x = np.arange(len(MODELS))
    ax_b.plot(x, [a[k]["r1_pooled"] for k, _ in MODELS], "D", color=INK, markersize=6, zorder=3,
              label="Backbone-only aspect R@1 (pooled)")
    for xi, k in zip(x, [m[0] for m in MODELS]):
        ax_b.text(xi + 0.08, a[k]["r1_pooled"] - 1.6, f"{a[k]['r1_pooled']:.2f}", fontsize=7, color=INK)
    ax_b.axhline(100 / 13, color=MUTED, linestyle=":", linewidth=1.0, zorder=1, label="chance: 7.69")
    ax_b.set_ylabel("R@1 (%), mean of i2t and t2i")
    ax_b.set_title("(b) Cross-modal ceilings per backbone", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("ArtELingo selection rows: four frozen backbones, label probes fit on 60,000 scorer-train rows",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "backbone_check.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "backbone_check.png")


if __name__ == "__main__":
    main()
