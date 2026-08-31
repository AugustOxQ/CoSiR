"""
Reproduce the eigenvector-localization collapse and the rank-normalization fix on
real Impressions features, side by side. Generates evidence figures for the report.

Run:
    python src/test/20260609_conditional_buddy/make_collapse_comparison.py

Outputs (docs/reports/assets/):
    collapse_zscore.png      — original z-score normalization (collapsed)
    fixed_rank.png           — per-dim rank normalization (fixed)
    collapse_comparison.png  — both panels side by side
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import FeatureManager
from src.conditional_buddy.compute_buddies import _l2_normalize
from src.conditional_buddy.buddy_graph import (
    ensure_min_degree,
    mix_distances,
    mutual_knn,
    rank_normalise_sparse,
    sparse_cosine_distance,
    union_graph,
)
from src.conditional_buddy.embedding_methods import normalise_embedding, spectral_embedding
from src.conditional_buddy.visualize import buddy_vs_random_distance

STORAGE = "/data/SSD2/pre_extract/impressions/features"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "reports", "assets"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K, ALPHA, SEED = 30, 0.5, 42


def _panel(ax, emb, B, title, max_edges=2000):
    stats = buddy_vs_random_distance(emb, B, seed=SEED)
    coo = B.tocoo()
    mask = coo.row < coo.col
    rows, cols = coo.row[mask], coo.col[mask]
    if len(rows) > max_edges:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(rows), size=max_edges, replace=False)
        rows, cols = rows[sel], cols[sel]
    for r, c in zip(rows, cols):
        ax.plot([emb[r, 0], emb[c, 0]], [emb[r, 1], emb[c, 1]],
                color="tab:blue", alpha=0.08, linewidth=0.5, zorder=1)
    ax.scatter(emb[:, 0], emb[:, 1], s=4, color="tab:red", alpha=0.5, zorder=2)
    ax.set_title(f"{title}\nbuddy/random dist ratio = {stats['ratio']:.3f}")
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    return stats


def main():
    os.makedirs(OUT, exist_ok=True)
    fm = FeatureManager(STORAGE)
    d = fm.load_all_to_ram(["img_features", "txt_features"])
    img = _l2_normalize(d["img_features"].numpy())
    txt = _l2_normalize(d["txt_features"].numpy())
    print(f"Loaded {img.shape[0]:,} samples")

    A_img = mutual_knn(img, K, DEVICE)
    A_txt = mutual_knn(txt, K, DEVICE)
    E, _ = ensure_min_degree(union_graph(A_img, A_txt), img, txt, DEVICE)
    B = A_img.multiply(A_txt)
    B.data[:] = 1.0
    B = B.tocsr()

    D_mixed = mix_distances(
        rank_normalise_sparse(sparse_cosine_distance(img, E)),
        rank_normalise_sparse(sparse_cosine_distance(txt, E)),
        ALPHA,
    )
    raw = spectral_embedding(D_mixed, 2, seed=SEED)  # compute once, normalise two ways

    emb_zscore = normalise_embedding(raw, method="zscore")
    emb_rank = normalise_embedding(raw, method="rank")

    # Individual panels
    for emb, title, fname in [
        (emb_zscore, "Before: z-score normalization (collapsed)", "collapse_zscore.png"),
        (emb_rank, "After: per-dim rank normalization (fixed)", "fixed_rank.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 7))
        s = _panel(ax, emb, B, title)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=120)
        plt.close(fig)
        print(f"{fname}: buddy={s['buddy_mean']:.4f} random={s['random_mean']:.4f} ratio={s['ratio']:.3f}")

    # Side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    _panel(axes[0], emb_zscore, B, "Before: z-score (collapsed)")
    _panel(axes[1], emb_rank, B, "After: per-dim rank (fixed)")
    fig.suptitle("Conditional buddies 2D init — Impressions (12,123 samples)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "collapse_comparison.png"), dpi=120)
    plt.close(fig)
    print(f"Saved figures → {OUT}")


if __name__ == "__main__":
    main()
