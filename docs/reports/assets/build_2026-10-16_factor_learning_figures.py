"""Figures for docs/reports/auto/v2/2026-10-16_candidate_a_factor_learning_selection.md.

Reads the grid's stored selection results (gitignored, local only):
    src/test/20261016_factor_learning_grid/results/selection_results.json
and writes four PNGs to docs/reports/assets/2026-10-16_factor_learning/.

Run from the repository root:
    python docs/reports/assets/build_2026-10-16_factor_learning_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "src/test/20261016_factor_learning_grid/results/selection_results.json"
OUT = ROOT / "docs/reports/assets/2026-10-16_factor_learning"

SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots (light surface)
INK, MUTED, GRID, SURFACE, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#a3a29d"
CELLS = ("C0", "A", "S", "AS")
MODELS = ("R3", "C0", "A", "S", "AS")
CELL_NAMES = {"R3": "R3\n(original)", "C0": "C0\n(control)", "A": "A\npainting agr.", "S": "S\nstyle episodes",
              "AS": "AS\nboth"}
SCOPE_NAMES = {"pooled": "Pooled (4,096 episodes)", "emotion": "Emotion episodes", "art_style": "Art-style episodes"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def point_ci(block: dict) -> tuple[float, float, float]:
    return block["point"], block["ci95"][0], block["ci95"][1]


def naive_figure(r: dict) -> None:
    """Naive R@1 at beta 0.3 per cell and label type, with the C0 / R3 / CLIP-only / chance lines."""
    head = r["headline_beta0.3"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), dpi=200, sharey=True)
    x = np.arange(len(CELLS))
    for ax, scope in zip(axes, ("pooled", "emotion", "art_style")):
        stats = [point_ci(head[c]["naive"][scope]["mean"]) for c in CELLS]
        pts = np.array([s[0] for s in stats])
        err = np.array([[p - lo for p, lo, _ in stats], [hi - p for p, _, hi in stats]])
        colors = [REFERENCE if c == "C0" else SERIES_1 for c in CELLS]
        ax.bar(x, pts, 0.6, color=colors, zorder=2, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(x, pts, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=3, zorder=3)
        for xi, p in zip(x, pts):                       # values inside the bars: no clash with the lines
            ax.text(xi, 1.0, f"{p:.1f}", ha="center", va="bottom", fontsize=9, color=SURFACE, fontweight="bold")
        refs = ((head["C0"]["naive"][scope]["mean"]["point"], "-", "C0 (matched control)"),
                (head["R3"]["naive"][scope]["mean"]["point"], "--", "original R3 (current system)"),
                (head["clip_only"][scope]["mean"]["point"], "-.", "CLIP only"),
                (r["chance_r1"], ":", "chance"))
        for value, style, name in refs:
            ax.axhline(value, color=MUTED, linestyle=style, linewidth=1.0, zorder=1, label=f"{name}: {value:.1f}")
        ax.set_xticks(x, [CELL_NAMES[c] for c in CELLS], fontsize=8.5)
        ax.set_title(SCOPE_NAMES[scope], fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(0, 44)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    axes[0].set_ylabel("Naive R@1 (%) at β 0.3, mean of i2t and t2i")
    fig.suptitle("Naive-rule R@1 of the four factor models on the selection episodes (seed 42, 95% bootstrap CIs)",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "naive_r1.png", facecolor=SURFACE)
    plt.close(fig)


def criterion_figure(r: dict) -> None:
    """D (pooled) and D_emotion per cell vs C0, with the 0 and -1.0 reference lines; art style as context."""
    cells = ("A", "S", "AS")
    series = ((r["d"], SERIES_1, "D: pooled (criterion, lower bound must be > 0)"),
              (r["d_emotion"], SERIES_2, "D_emotion (guard, lower bound must be > −1.0)"),
              ({c: r["vs_c0"][c]["art_style"]["mean"] for c in cells}, SERIES_3, "art style (context, not gating)"))
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=200)
    y = np.arange(len(cells))[::-1].astype(float)
    offsets = (0.22, 0.0, -0.22)
    lows = []
    highs = []
    for (blocks, color, name), off in zip(series, offsets):
        stats = [point_ci(blocks[c]) for c in cells]
        pts = np.array([s[0] for s in stats])
        lo, hi = np.array([s[1] for s in stats]), np.array([s[2] for s in stats])
        lows.append(lo.min())
        highs.append(hi.max())
        ax.errorbar(pts, y + off, xerr=[pts - lo, hi - pts], fmt="o", color=color, ecolor=color, elinewidth=1.6,
                    capsize=3, markersize=6, label=name, zorder=3)
        for yi, p, h in zip(y + off, pts, hi):
            ax.text(h + 0.08, yi, f"{p:+.2f}".replace("-", "−"), va="center", fontsize=8, color=INK)
    ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
    ax.axvline(-1.0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(0.05, len(cells) - 0.45, "0 (criterion)", color=INK, fontsize=8, va="bottom", ha="left")
    ax.text(-1.05, len(cells) - 0.45, "−1.0 (emotion guard)", color=MUTED, fontsize=8, va="bottom", ha="right")
    gate_note = {c: f"{r['gates'][c]['n_passed']}/9 gates" for c in cells}
    ax.set_yticks(y, [f"{c} − C0\n({gate_note[c]})" for c in cells], fontsize=10)
    left, right = min(min(lows), -1.0) - 0.4, max(max(highs), 0.0) + 0.9
    ax.set_xlim(left, right)
    ax.set_ylim(-0.6, len(cells) - 0.2)
    ax.set_xlabel("Paired difference in naive R@1 at β 0.3 (points), mean of i2t and t2i, 95% bootstrap CI")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / "criterion_d.png", facecolor=SURFACE)
    plt.close(fig)


def oracle_figure(r: dict) -> None:
    """Naive at beta 0.3 vs the label oracle at beta 0 per model (pooled), with R3's oracle and the null."""
    head = r["headline_beta0.3"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), dpi=200, sharey=True)
    x = np.arange(len(MODELS))
    width = 0.38
    for ax, scope in zip(axes, ("pooled", "emotion", "art_style")):
        for offset, field, color, name in ((-width / 2 - 0.01, "naive", SERIES_1, "naive rule, β 0.3"),
                                           (width / 2 + 0.01, "oracle_0", SERIES_2, "label oracle, β 0")):
            stats = [point_ci(head[m][field][scope]["mean"]) for m in MODELS]
            pts = np.array([s[0] for s in stats])
            err = np.array([[p - lo for p, lo, _ in stats], [hi - p for p, _, hi in stats]])
            ax.bar(x + offset, pts, width, color=color, label=name, zorder=2, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x + offset, pts, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=3)
            for xi, p in zip(x + offset, pts):           # values inside the bars: no clash with the lines
                ax.text(xi, 1.0, f"{p:.1f}", ha="center", va="bottom", fontsize=7, color=SURFACE,
                        fontweight="bold", rotation=90)
        null = np.mean([head[m]["oracle_null_0"][scope]["mean"]["point"] for m in MODELS])
        ax.axhline(head["R3"]["oracle_0"][scope]["mean"]["point"], color=MUTED, linestyle="--", linewidth=1.0,
                   zorder=1, label=f"R3 label oracle: {head['R3']['oracle_0'][scope]['mean']['point']:.1f}")
        ax.axhline(null, color=MUTED, linestyle=":", linewidth=1.0, zorder=1,
                   label=f"oracle null (mean of models): {null:.1f}")
        ax.set_xticks(x, [CELL_NAMES[m] for m in MODELS], fontsize=8.5)
        ax.set_title(SCOPE_NAMES[scope], fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(0, 40)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    axes[0].set_ylabel("R@1 (%), mean of i2t and t2i")
    fig.suptitle("Does any factor model carry more label information? Naive rule vs cross-validated label oracle",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "label_oracle.png", facecolor=SURFACE)
    plt.close(fig)


def training_figure(r: dict) -> None:
    """Condition loss and the learned tau over training for the two cells with the condition loss."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), dpi=200)
    for cell, color in (("S", SERIES_1), ("AS", SERIES_2)):
        h = r["training"][cell]["history"]
        axes[0].plot(h["step"], h["condition_loss"], color=color, linewidth=1.4, marker="o", markersize=2.5,
                     label=cell)
        axes[1].plot(h["step"], h["tau"], color=color, linewidth=1.4, marker="o", markersize=2.5, label=cell)
    axes[0].axhline(np.log(4), color=MUTED, linestyle=":", linewidth=1.0,
                    label="uniform scores: log 4 = 1.386")
    axes[0].set_ylabel("condition episode loss (one batch)")
    axes[1].set_ylabel("learned temperature τ")
    for ax, title in zip(axes, ("Condition loss over training", "Learned τ over training")):
        ax.set_xlabel("training step")
        ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
    axes[0].set_ylim(0.2, 1.75)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "condition_training.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads(RESULTS.read_text())
    naive_figure(r)
    criterion_figure(r)
    oracle_figure(r)
    training_figure(r)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))} to {OUT}")


if __name__ == "__main__":
    main()
