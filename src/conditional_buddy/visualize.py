"""
2D sanity-check visualization for the conditional-buddies init.

Renders the 2D embedding with mutual-buddy edges overdrawn and reports whether
buddy pairs end up closer than random pairs — a quick check that the embedding
preserves neighbourhood structure before committing to the full N_DIM run.
"""

from typing import Dict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def buddy_vs_random_distance(emb: np.ndarray, B: csr_matrix, seed: int = 42) -> Dict[str, float]:
    """
    Mean Euclidean distance of buddy pairs vs. random pairs in the embedding.

    B: a binary adjacency (e.g. the intersection or union buddy graph). Buddy pairs
    should be markedly closer than random pairs if the embedding is meaningful.
    """
    coo = B.tocoo()
    mask = coo.row < coo.col  # upper triangle, each undirected pair once
    rows, cols = coo.row[mask], coo.col[mask]
    if len(rows) == 0:
        return {"buddy_mean": float("nan"), "random_mean": float("nan"), "ratio": float("nan")}

    buddy_d = np.linalg.norm(emb[rows] - emb[cols], axis=1)

    rng = np.random.default_rng(seed)
    n = len(rows)
    a = rng.integers(0, emb.shape[0], size=n)
    b = rng.integers(0, emb.shape[0], size=n)
    random_d = np.linalg.norm(emb[a] - emb[b], axis=1)

    buddy_mean = float(buddy_d.mean())
    random_mean = float(random_d.mean())
    return {
        "buddy_mean": buddy_mean,
        "random_mean": random_mean,
        "ratio": buddy_mean / (random_mean + 1e-12),
    }


def plot_2d_buddies(
    emb2d: np.ndarray,
    B: csr_matrix,
    out_path: str,
    max_edges: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Scatter the 2D embedding with a sample of buddy edges drawn, save to out_path.

    Returns the buddy-vs-random distance report (also printed).
    """
    stats = buddy_vs_random_distance(emb2d, B, seed=seed)
    print(
        f"[buddies] 2D check — buddy mean dist={stats['buddy_mean']:.4f}, "
        f"random mean dist={stats['random_mean']:.4f}, "
        f"ratio={stats['ratio']:.3f} (lower is better; <1 means buddies are closer)"
    )

    coo = B.tocoo()
    mask = coo.row < coo.col
    rows, cols = coo.row[mask], coo.col[mask]

    fig, ax = plt.subplots(figsize=(9, 9))
    if len(rows) > max_edges:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(rows), size=max_edges, replace=False)
        rows, cols = rows[sel], cols[sel]
    for r, c in zip(rows, cols):
        ax.plot(
            [emb2d[r, 0], emb2d[c, 0]],
            [emb2d[r, 1], emb2d[c, 1]],
            color="tab:blue",
            alpha=0.08,
            linewidth=0.5,
            zorder=1,
        )
    ax.scatter(emb2d[:, 0], emb2d[:, 1], s=4, color="tab:red", alpha=0.5, zorder=2)
    ax.set_title(
        f"Conditional buddies 2D init\n"
        f"buddy/random dist ratio = {stats['ratio']:.3f}"
    )
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[buddies] Saved 2D visualization → {out_path}")
    return stats
