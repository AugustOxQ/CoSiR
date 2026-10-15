"""Figures for docs/reports/auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md.

Reads the probe's stored results (gitignored, local only):
    src/test/20261015_factor_headroom_probe/results/probe_results.json
and writes three PNGs to docs/reports/assets/2026-10-15_factor_headroom_probe/.

Run from the repository root:
    python docs/reports/assets/build_2026-10-15_factor_headroom_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "src/test/20261015_factor_headroom_probe/results/probe_results.json"
OUT = ROOT / "docs/reports/assets/2026-10-15_factor_headroom_probe"

SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"        # validated categorical slots 1-2 (light surface)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
CODES = ("R3", "clip512", "pca32", "pca128", "labelprobe")
CODE_NAMES = {"R3": "R3\n(current)", "clip512": "raw CLIP\n512-d", "pca32": "CLIP PCA\n32-d",
              "pca128": "CLIP PCA\n128-d", "labelprobe": "label probe\n(ceiling)"}
SCOPE_NAMES = {"emotion": "Emotion episodes", "art_style": "Art-style episodes"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def point_ci(block: dict) -> tuple[float, float, float]:
    return block["point"], block["ci95"][0], block["ci95"][1]


def headroom_figure(r: dict) -> None:
    head = r["headline"]
    baseline = r["r1"]["naive|R3|0.3"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=200, sharey=True)
    x = np.arange(len(CODES))
    width = 0.36
    for ax, scope in zip(axes, ("emotion", "art_style")):
        for offset, scorer, color, name in ((-width / 2 - 0.01, "naive", SERIES_1, "naive rule"),
                                            (width / 2 + 0.01, "oracle", SERIES_2, "label oracle")):
            stats = [point_ci(head[c][scorer]["r1"][scope]["mean"]) for c in CODES]
            pts = np.array([s[0] for s in stats])
            err = np.array([[p - lo for p, lo, _ in stats], [hi - p for p, _, hi in stats]])
            ax.bar(x + offset, pts, width, color=color, label=name, zorder=2, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x + offset, pts, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=3)
            if scorer == "oracle":
                for xi, p in zip(x + offset, pts):
                    ax.text(xi, p + 1.6, f"{p:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        refs = ((baseline[scope]["mean"], "-", "baseline: naive on R3, β 0.3"),
                (head["clip_only"]["r1"][scope]["mean"]["point"], "--", "CLIP only"),
                (r["chance_r1"], ":", "chance"))
        for value, style, name in refs:
            ax.axhline(value, color=MUTED, linestyle=style, linewidth=1.1, zorder=1, label=f"{name} ({value:.1f})")
        ax.set_xticks(x, [CODE_NAMES[c] for c in CODES], fontsize=8.5)
        ax.set_title(SCOPE_NAMES[scope], fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(0, 60)
    axes[0].set_ylabel("R@1 (%), mean of i2t and t2i")
    for ax in axes:                                     # each panel's reference values differ
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("How much emotion and style information a code on frozen CLIP can carry (selection rows, best β)",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "headroom_oracle.png", facecolor=SURFACE)
    plt.close(fig)


def alignment_figure(r: dict) -> None:
    parts = r["ami"]["partitions"]
    order = ("clip_image", "community", "clip_caption", "caption_residual", "random64")
    names = {"clip_image": "CLIP image k-means (64)", "community": "Block 1 communities (19)",
             "clip_caption": "CLIP caption k-means (64)", "caption_residual": "caption residual k-means (64)",
             "random64": "random 64-way (null)"}
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=200)
    y = np.arange(len(order))[::-1]
    h = 0.36
    for offset, label, color, name in ((h / 2 + 0.01, "emotion", SERIES_1, "emotion"),
                                       (-h / 2 - 0.01, "art_style", SERIES_2, "art style")):
        vals = [parts[p][label]["ami"] for p in order]
        ax.barh(y + offset, vals, h, color=color, label=name, zorder=2, edgecolor=SURFACE, linewidth=1)
        for yi, v in zip(y + offset, vals):
            ax.text(max(v, 0) + 0.004, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [names[p] for p in order], fontsize=9)
    ax.set_xlabel("Adjusted mutual information with the human label (scorer-train rows)")
    ax.set_xlim(0, 0.36)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Which self-generated condition sources line up with the human labels", fontsize=11, color=INK,
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "source_alignment.png", facecolor=SURFACE)
    plt.close(fig)


def probe_figure(r: dict) -> None:
    probes = r["codes"]["labelprobe"]["probes"]
    order = ("image->art_style", "text->art_style", "image->emotion", "text->emotion")
    names = {"image->art_style": "image → art style", "text->art_style": "caption → art style",
             "image->emotion": "image → emotion", "text->emotion": "caption → emotion"}
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=200)
    x = np.arange(len(order))
    w = 0.36
    for offset, field, color, name in ((-w / 2 - 0.01, "selection_majority_accuracy", SERIES_1, "majority class"),
                                       (w / 2 + 0.01, "selection_top1_accuracy", SERIES_2, "linear probe on CLIP")):
        vals = [100 * probes[p][field] for p in order]
        ax.bar(x + offset, vals, w, color=color, label=name, zorder=2, edgecolor=SURFACE, linewidth=1)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [names[p] for p in order], fontsize=9)
    ax.set_ylabel("Top-1 accuracy on selection rows (%)")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.set_title("Style is read from the image, emotion from the caption", fontsize=11, color=INK, loc="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "probe_accuracy.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads(RESULTS.read_text())
    if r["mode"] != "run":
        raise ValueError("expected the full --run results, not smoke")
    headroom_figure(r)
    alignment_figure(r)
    probe_figure(r)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))} to {OUT}")


if __name__ == "__main__":
    main()
