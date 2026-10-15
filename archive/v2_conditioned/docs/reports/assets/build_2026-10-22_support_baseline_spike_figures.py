"""Figure for docs/reports/auto/v2/2026-10-22_support_baseline_spike.md.

Reads (gitignored, local only):
    src/test/20261022_support_baseline_spike/results/spike_results.json   (run_spike.py)
    src/test/20261022_support_baseline_spike/results/spike_ranks.npz      (run_spike.py)
    src/test/20261018_affect_factor_learning/results/selection_ranks.npz  (SE, C0, CLIP only ranks)
and writes docs/reports/assets/2026-10-22_support_baseline_spike/support_baselines.png.

Run from the repository root:
    python docs/reports/assets/build_2026-10-22_support_baseline_spike_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SPIKE = ROOT / "src/test/20261022_support_baseline_spike/results"
SEL_RANKS = ROOT / "src/test/20261018_affect_factor_learning/results/selection_ranks.npz"
OUT = ROOT / "docs/reports/assets/2026-10-22_support_baseline_spike"

SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#a3a29d"
CHANCE = 100.0 / 13

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

STORED = {"CLIP only": "clip_only__-__0.3", "C0": "naive__C0__0.3", "SE": "naive__SE__0.3"}
JSON_NAME = {"CLIP only": "CLIP_only", "C0": "C0", "SE": "SE"}
LABELS = ("emotion", "art_style")


def r1_from_ranks(ranks, stem: str, lab: str, d: str) -> float:
    return 100.0 * float((ranks[f"{stem}__{lab}__{d}"] == 1).mean())


def cells() -> dict:
    """{scorer: {(label, direction): R@1 %}} for every scorer drawn."""
    res = json.load(open(SPIKE / "spike_results.json"))["r1"]
    sel = np.load(SEL_RANKS)
    out = {}
    for name, stem in STORED.items():
        out[name] = {(lab, d): r1_from_ranks(sel, stem, lab, d) for lab in LABELS for d in ("i2t", "t2i")}
        for lab in LABELS:                       # the stored ranks must agree with the spike's own copy
            for d in ("i2t", "t2i"):
                assert abs(out[name][(lab, d)] - res[JSON_NAME[name]][lab][d]) < 1e-6, (name, lab, d)
    for name, key in (("Prototype, condition only", "proto__laminf"), ("Prototype + query", "proto_cv"),
                      ("Logistic probe, both modalities", "probe_x_cv")):
        out[name] = {(lab, d): res[key][lab][d] for lab in LABELS for d in ("i2t", "t2i")}
    return out


def scope_value(c: dict, scope: str) -> float:
    labs = LABELS if scope == "pooled" else (scope,)
    return float(np.mean([c[(lab, d)] for lab in labs for d in ("i2t", "t2i")]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = cells()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 4.9), dpi=200, gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) pooled / emotion / style
    names_a = ["CLIP only", "C0", "SE", "Prototype, condition only", "Prototype + query",
               "Logistic probe, both modalities"]
    colors_a = [REFERENCE, "#c9c8c3", SERIES_2, SERIES_3, "#0f7f58", SERIES_1]
    scopes = ("pooled", "emotion", "art_style")
    scope_names = ("Pooled", "Emotion", "Art style")
    width = 0.13
    x = np.arange(len(scopes))
    for i, (n, col) in enumerate(zip(names_a, colors_a)):
        vals = [scope_value(c[n], s) for s in scopes]
        xs = x + (i - (len(names_a) - 1) / 2) * width
        ax_a.bar(xs, vals, width * 0.92, color=col, label=n, zorder=2, edgecolor=SURFACE, linewidth=0.8)
        for xi, v in zip(xs, vals):
            ax_a.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
    ax_a.axhline(CHANCE, color=MUTED, linestyle=":", linewidth=1.0, zorder=1, label=f"chance: {CHANCE:.1f}")
    ax_a.set_xticks(x, scope_names, fontsize=9)
    ax_a.set_ylim(0, 38)
    ax_a.set_ylabel("R@1 (%), mean of i2t and t2i")
    ax_a.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_a.set_title("(a) Support-set baselines against SE", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax_a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)

    # (b) four direction x label cells
    cell_keys = [("emotion", "i2t"), ("emotion", "t2i"), ("art_style", "i2t"), ("art_style", "t2i")]
    cell_names = ["Emotion\ni2t (captions ranked)", "Emotion\nt2i (images ranked)",
                  "Style\ni2t (captions ranked)", "Style\nt2i (images ranked)"]
    names_b = ["CLIP only", "SE", "Prototype + query"]
    colors_b = [REFERENCE, SERIES_2, "#0f7f58"]
    width = 0.26
    x = np.arange(len(cell_keys))
    for i, (n, col) in enumerate(zip(names_b, colors_b)):
        vals = [c[n][k] for k in cell_keys]
        xs = x + (i - 1) * width
        ax_b.bar(xs, vals, width * 0.92, color=col, label=n, zorder=2, edgecolor=SURFACE, linewidth=0.8)
        for xi, v in zip(xs, vals):
            ax_b.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax_b.axhline(CHANCE, color=MUTED, linestyle=":", linewidth=1.0, zorder=1, label=f"chance: {CHANCE:.1f}")
    ax_b.set_xticks(x, cell_names, fontsize=7.5)
    ax_b.set_ylim(0, 54)
    ax_b.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_b.set_title("(b) Where each scorer wins, per direction", fontsize=11, color=INK, loc="left",
                   fontweight="bold")
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Selection episodes (4,096 emotion + 4,096 art-style, 13 candidates), cross-fitted lambda for "
                 "prototype and probe", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "support_baselines.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "support_baselines.png")


if __name__ == "__main__":
    main()
