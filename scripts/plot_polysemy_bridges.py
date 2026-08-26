"""Create the four Experiment 12 polysemy-bridge report figures from saved raw arrays.

Usage
-----
python scripts/plot_polysemy_bridges.py \
    --raw res/CoSiR_condition_freeze_ablation/redcaps_150k/polysemy_bridges_raw.npz \
    --summary res/CoSiR_condition_freeze_ablation/redcaps_150k/polysemy_bridges_summary.json \
    --out-dir docs/reports/assets/polysemy_bridges
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ("neither", "img_only_only", "txt_only_only", "bridge")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_node_label_counts(summary: dict, out_path: Path) -> None:
    counts = [summary["label_counts"][label] for label in LABELS]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(LABELS, counts, color=("#7f8c8d", "#4c78a8", "#f58518", "#54a24b"))
    ax.set_yscale("log")
    ax.set_ylabel("nodes (log scale)")
    ax.set_title("Cross-modal polysemy node labels")
    ax.tick_params(axis="x", rotation=15)
    for bar, count in zip(bars, counts):
        ax.annotate(f"{count:,}\n{count / total:.1%}",
                    (bar.get_x() + bar.get_width() / 2, count),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    _save(fig, out_path)


def plot_pull_distribution(raw: np.lib.npyio.NpzFile, summary: dict, out_path: Path) -> None:
    pull = raw["dist_bc_baseline"] - raw["dist_bc"]
    pull_summary = summary["pull_summary"]
    mean = pull_summary["mean"]
    frac = pull_summary["frac_pulled_closer"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(pull, bins=45, color="#4c78a8", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="no pull")
    ax.axvline(mean, color="#e45756", linewidth=2,
               label=f"mean pull = {mean:+.4f}")
    ax.set_xlabel("pull = baseline distance − bridge-pair distance")
    ax.set_ylabel("sampled bridge pairs")
    ax.set_title("Buddy-init pull for bridge-derived pairs")
    ax.legend(title=f"fraction pulled closer = {frac:.1%}")
    _save(fig, out_path)


def plot_jaccard_vs_pull(raw: np.lib.npyio.NpzFile, summary: dict, out_path: Path) -> None:
    jaccard = raw["jaccard"]
    pull = raw["dist_bc_baseline"] - raw["dist_bc"]
    slope, intercept = np.polyfit(jaccard, pull, deg=1)
    x_line = np.linspace(jaccard.min(), jaccard.max(), 100)
    corr = summary["grading_corr_jaccard_vs_pull"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(jaccard, pull, s=7, alpha=0.22, color="#4c78a8", edgecolors="none")
    ax.plot(x_line, slope * x_line + intercept, color="#e45756", linewidth=2,
            label="least-squares fit")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("shared-neighbor Jaccard")
    ax.set_ylabel("pull = baseline distance − bridge-pair distance")
    ax.set_title("Shared-neighbor overlap weakly grades bridge-pair pull")
    ax.legend()
    ax.text(0.98, 0.04, f"Spearman ρ = {corr['rho']:+.3f}\np = {corr['p']:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9})
    _save(fig, out_path)


def plot_retrieval_by_label(raw: np.lib.npyio.NpzFile, summary: dict, out_path: Path) -> None:
    if "label_kept" not in raw or "delta_rank" not in raw:
        raise ValueError("raw .npz lacks label_kept/delta_rank; rerun analysis with --per-sample-npz")
    labels = raw["label_kept"].astype(str)
    delta_rank = raw["delta_rank"]
    medians = [float(np.median(np.abs(delta_rank[labels == label]))) for label in LABELS]
    retrieval = summary["retrieval_correlation"]
    corr = retrieval["corr_is_polysemic_vs_abs_delta_rank"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(LABELS, medians, color=("#7f8c8d", "#4c78a8", "#f58518", "#54a24b"))
    ax.margins(y=0.18)
    ax.set_ylabel("median |Δ retrieval rank|")
    ax.set_title("Retrieval-rank change by polysemy label")
    ax.tick_params(axis="x", rotation=15)
    for bar, label, median in zip(bars, LABELS, medians):
        n = int((labels == label).sum())
        ax.annotate(f"n={n:,}\n{median:.1f}",
                    (bar.get_x() + bar.get_width() / 2, median),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.text(0.98, 0.96, f"corr(is_polysemic, |Δrank|)\nρ = {corr['rho']:+.3f}, p = {corr['p']:.2g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9})
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help=".npz written by analyze_polysemy_bridges.py --save-raw")
    parser.add_argument("--summary", required=True, help="summary JSON written by analyze_polysemy_bridges.py --out")
    parser.add_argument("--out-dir", required=True, help="directory for the four report PNGs")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    raw = np.load(args.raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_node_label_counts(summary, out_dir / "node_label_counts.png")
    plot_pull_distribution(raw, summary, out_dir / "pull_distribution.png")
    plot_jaccard_vs_pull(raw, summary, out_dir / "jaccard_vs_pull.png")
    plot_retrieval_by_label(raw, summary, out_dir / "retrieval_by_label.png")
    print(f"Wrote 4 figures to {out_dir}")


if __name__ == "__main__":
    main()
