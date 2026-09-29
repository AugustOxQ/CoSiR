#!/usr/bin/env python3
"""Plot Attention-h1's independently clustered embedding space before and after training."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.manifold import TSNE


SNAPSHOT_PATH = Path(
    "/project/CoSiR-buddy_prototype_conditioning/src/test/"
    "20260923_artelingo_buddy_analysis/attention_h1_embedding_snapshot.npz"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "diagrams/attention_h1_embedding_before_after_tsne.png"
)

RANDOM_SEED = 42
MAX_TSNE_POINTS = 6_000

BLACK = "#1A1A1A"
GRAY = "#595959"
LIGHT_GRAY = "#E8E8E8"
BORDER = "#BFBFBF"
WHITE = "#FFFFFF"


def categorical_colormap() -> ListedColormap:
    """Combine the three tab20 palettes for up to 60 community colors."""
    colors = np.vstack([
        plt.get_cmap("tab20")(np.arange(20)),
        plt.get_cmap("tab20b")(np.arange(20)),
        plt.get_cmap("tab20c")(np.arange(20)),
    ])
    return ListedColormap(colors, name="attention_h1_tab60")


def load_panels() -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    """Read each embedding and its own Leiden partition from the fixed snapshot."""
    panels = []
    with np.load(SNAPSHOT_PATH, allow_pickle=True) as data:
        for split, display_split in (("train", "Train"), ("heldout", "Held-out")):
            for stage, display_stage in (
                ("pre", "pre-training (epoch 0"),
                ("post", "post-training (200 epochs"),
            ):
                embedding = data[f"{split}_embedding_{stage}"]
                labels = data[f"{split}_community_{stage}"]
                if embedding.ndim != 2 or embedding.shape[1] != 32:
                    raise ValueError(f"Expected a 32-D {split}/{stage} embedding.")
                if labels.ndim != 1 or len(labels) != len(embedding):
                    raise ValueError(f"Community labels do not match {split}/{stage} rows.")
                if len(labels) == 0 or not np.isfinite(embedding).all():
                    raise ValueError(f"Empty or non-finite {split}/{stage} embedding.")
                n_communities = int(labels.max()) + 1
                if not np.array_equal(np.unique(labels), np.arange(n_communities)):
                    raise ValueError(f"Expected contiguous zero-based {split}/{stage} Leiden IDs.")
                title = f"{display_split}, {display_stage}, {n_communities} communities)"
                panels.append((title, embedding, labels, n_communities))
    return panels


def subsample(
    embedding: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one representative per community, then fill to the panel cap."""
    if len(embedding) <= MAX_TSNE_POINTS:
        return embedding, labels

    # Each panel is sampled independently with the same seeded RNG. Reserve one
    # member of each community so even tiny Leiden communities appear in the plot.
    representatives = np.array([
        rng.choice(np.flatnonzero(labels == community))
        for community in range(int(labels.max()) + 1)
    ])
    available = np.ones(len(embedding), dtype=bool)
    available[representatives] = False
    remainder = rng.choice(
        np.flatnonzero(available), size=MAX_TSNE_POINTS - len(representatives),
        replace=False,
    )
    indices = rng.permutation(np.concatenate((representatives, remainder)))
    return embedding[indices], labels[indices]


def tsne_2d(embedding: np.ndarray) -> np.ndarray:
    """Run a deterministic 2-D t-SNE separately for each panel."""
    return TSNE(
        n_components=2,
        perplexity=min(30, max(5, (len(embedding) - 1) // 3)),
        init="pca",
        random_state=RANDOM_SEED,
    ).fit_transform(embedding)


def style_scatter_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(WHITE)
    axis.grid(False)
    axis.tick_params(
        axis="both", which="both", length=2.5, color=BORDER,
        labelbottom=False, labelleft=False,
    )
    for spine in axis.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.7)


def main() -> None:
    panels = load_panels()
    cmap = categorical_colormap()
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=WHITE)

    for axis, (title, embedding, labels, n_communities) in zip(axes.flat, panels):
        # Train and held-out panels both use the 6,000-point cap for comparable
        # visual density and manageable local CPU runtime. Their Leiden IDs are
        # independent, as are their t-SNE coordinate systems.
        sampled_embedding, sampled_labels = subsample(embedding, labels, rng)
        projection = tsne_2d(sampled_embedding)
        axis.scatter(
            projection[:, 0], projection[:, 1], c=sampled_labels, cmap=cmap,
            vmin=0, vmax=cmap.N - 1, s=2.5, alpha=0.62, linewidths=0,
            rasterized=True,
        )
        style_scatter_axis(axis)
        axis.set_title(title, color=BLACK, fontsize=11, fontweight="bold", pad=10)
        note = f"{n_communities} communities, n={len(sampled_embedding):,}"
        if len(sampled_embedding) < len(embedding):
            note += f" (subsampled from {len(embedding):,})"
        axis.text(
            0.5, -0.12, note, transform=axis.transAxes, ha="center",
            va="top", color=GRAY, fontsize=9,
        )
        print(f"{title}: {len(sampled_embedding):,} points from {len(embedding):,}")

    fig.suptitle(
        "Attention-h1 (our fusion method): embedding space before vs. after training",
        color=BLACK, fontsize=16, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.925, bottom=0.075,
        hspace=0.31, wspace=0.14,
    )
    fig.savefig(OUTPUT_PATH, dpi=300, facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
