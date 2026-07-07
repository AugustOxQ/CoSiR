"""
Deeper study of the conditional-buddies init on Impressions:

  Part A — dimensionality sweep (n_dim in {2,4,8,16,32}) at K=30, alpha=0.5.
           For each dim: UMAP-to-2D plot coloured by caption type + metrics.
  Part B — hyperparameter sweep (K x alpha) at n_dim=16: metric heatmaps.

Metrics
  buddy_ratio       mean dist(intersection-buddy pairs) / mean dist(random pairs)
                    in the final normalised embedding. Lower = buddies closer.
  knn_preservation  fraction of each node's union-graph neighbours that fall in
                    its top-k embedding neighbours (k=15). Higher = better local
                    structure retention.
  participation     participation ratio of the RAW spectral embedding's first
                    n_dim columns ((sum eig)^2 / sum eig^2) = effective #dims used.

Key efficiency trick: spectral eigenvectors are nested, so we solve n_components=32
ONCE and slice the first n_dim columns for every dim in the sweep.

Run (cuml needs the conda libstdc++ first on PATH):
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
        python src/test/20260609_conditional_buddy/dim_hparam_study.py
"""

import json
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
ANNOT = "/project/Impressions/metadata/impressions_train.json"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "reports", "assets"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DIMS = [2, 4, 8, 16, 32]
TYPE_MAP = {"caption": 0, "description": 1, "impression": 2, "aesthetic": 3}
TYPE_NAMES = ["caption", "description", "impression", "aesthetic"]


# ── Metrics ──────────────────────────────────────────────────────────────────


def knn_preservation(emb: np.ndarray, E, k: int = 15, device: str = "cuda") -> float:
    """Mean fraction of each node's graph neighbours found in its top-k embedding NN."""
    N = emb.shape[0]
    x = torch.from_numpy(emb).float().to(device)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-9)
    out = np.empty((N, k), dtype=np.int64)
    bs = 2048
    for s in range(0, N, bs):
        e = min(s + bs, N)
        sims = x[s:e] @ x.t()
        sims[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -1e9
        out[s:e] = sims.topk(k, dim=1).indices.cpu().numpy()

    E = E.tocsr()
    fracs = []
    for i in range(N):
        nbrs = set(E.indices[E.indptr[i]:E.indptr[i + 1]].tolist())
        if not nbrs:
            continue
        emb_nn = set(out[i].tolist())
        denom = min(len(nbrs), k)
        fracs.append(len(nbrs & emb_nn) / denom)
    return float(np.mean(fracs))


def participation_ratio(raw: np.ndarray) -> float:
    c = np.cov(raw.T)
    w = np.linalg.eigvalsh(c)
    w = np.clip(w, 0, None)
    return float((w.sum() ** 2) / (np.square(w).sum() + 1e-12))


# ── Pipeline pieces ──────────────────────────────────────────────────────────


def build_graph(img, txt, K):
    A_img = mutual_knn(img, K, DEVICE)
    A_txt = mutual_knn(txt, K, DEVICE)
    E, _ = ensure_min_degree(union_graph(A_img, A_txt), img, txt, DEVICE)
    B = A_img.multiply(A_txt)
    B.data[:] = 1.0
    return E, B.tocsr()


def load_types(sample_ids):
    ann = json.load(open(ANNOT))
    return np.array([TYPE_MAP[ann[i]["caption_type"]] for i in sample_ids])


def umap_2d(emb):
    from cuml.manifold import UMAP

    if emb.shape[1] == 2:
        return emb  # already 2-D; show the embedding directly
    u = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
    return np.asarray(u.fit_transform(emb.astype(np.float32)))


def scatter_by_type(ax, xy, types, title):
    for t in range(4):
        m = types == t
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.5, label=TYPE_NAMES[t])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    sweep_dir = os.path.join(OUT, "dim_sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    fm = FeatureManager(STORAGE)
    d = fm.load_all_to_ram(["img_features", "txt_features"])
    img = _l2_normalize(d["img_features"].numpy())
    txt = _l2_normalize(d["txt_features"].numpy())
    sample_ids = [int(s) for s in d["sample_ids"].tolist()]
    types = load_types(sample_ids)
    print(f"Loaded {img.shape[0]:,} samples; type counts: "
          f"{ {TYPE_NAMES[t]: int((types==t).sum()) for t in range(4)} }")

    # ---- Part A: dimensionality sweep (K=30, alpha=0.5) ----
    E, B = build_graph(img, txt, K=30)
    print(f"Graph: avg degree(E)={E.nnz / E.shape[0]:.2f}")
    D_mixed = mix_distances(
        rank_normalise_sparse(sparse_cosine_distance(img, E)),
        rank_normalise_sparse(sparse_cosine_distance(txt, E)),
        0.5,
    )
    raw32 = spectral_embedding(D_mixed, max(DIMS), seed=SEED)  # solve once, slice

    rows = []
    fig, axes = plt.subplots(1, len(DIMS), figsize=(5 * len(DIMS), 5))
    for ax, n_dim in zip(axes, DIMS):
        raw = raw32[:, :n_dim]
        emb = normalise_embedding(raw, method="rank")
        ratio = buddy_vs_random_distance(emb, B, seed=SEED)["ratio"]
        pres = knn_preservation(emb, E, k=15, device=DEVICE)
        pr = participation_ratio(raw)
        rows.append({"n_dim": n_dim, "buddy_ratio": ratio, "knn_preservation": pres,
                     "participation": pr})
        print(f"  n_dim={n_dim:2d}: buddy_ratio={ratio:.3f}  knn_pres={pres:.3f}  participation={pr:.2f}")
        xy = umap_2d(emb)
        scatter_by_type(ax, xy, types,
                        f"n_dim={n_dim}\nratio={ratio:.3f}, knn_pres={pres:.3f}")
    axes[-1].legend(markerscale=3, fontsize=8, loc="upper right")
    fig.suptitle("Conditional buddies — UMAP by caption type across n_dim (Impressions, K=30, α=0.5)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(sweep_dir, "umap_by_dim.png"), dpi=120)
    plt.close(fig)

    # metrics-vs-dim figure
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(arr["n_dim"], arr["buddy_ratio"], "o-"); ax[0].set_title("buddy/random ratio (lower=better)")
    ax[1].plot(arr["n_dim"], arr["knn_preservation"], "o-", color="tab:green"); ax[1].set_title("kNN preservation (higher=better)")
    ax[2].plot(arr["n_dim"], arr["participation"], "o-", color="tab:purple"); ax[2].set_title("participation ratio (raw spectral)")
    for a in ax:
        a.set_xlabel("n_dim"); a.set_xticks(DIMS); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dim_sweep_metrics.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "dim_sweep_stats.csv"), "w") as f:
        f.write("n_dim,buddy_ratio,knn_preservation,participation\n")
        for r in rows:
            f.write(f"{r['n_dim']},{r['buddy_ratio']:.4f},{r['knn_preservation']:.4f},{r['participation']:.4f}\n")

    # ---- Part B: hyperparameter sweep at n_dim=16 ----
    Ks = [15, 30, 50]
    alphas = [0.0, 0.5, 1.0]
    ratio_mat = np.zeros((len(Ks), len(alphas)))
    pres_mat = np.zeros((len(Ks), len(alphas)))
    hp_rows = []
    for ki, K in enumerate(Ks):
        Ek, Bk = build_graph(img, txt, K=K)
        Di = rank_normalise_sparse(sparse_cosine_distance(img, Ek))
        Dt = rank_normalise_sparse(sparse_cosine_distance(txt, Ek))
        for ai, alpha in enumerate(alphas):
            Dm = mix_distances(Di, Dt, alpha)
            emb = normalise_embedding(spectral_embedding(Dm, 16, seed=SEED), method="rank")
            ratio = buddy_vs_random_distance(emb, Bk, seed=SEED)["ratio"]
            pres = knn_preservation(emb, Ek, k=15, device=DEVICE)
            ratio_mat[ki, ai] = ratio
            pres_mat[ki, ai] = pres
            hp_rows.append({"K": K, "alpha": alpha, "buddy_ratio": ratio, "knn_preservation": pres})
            print(f"  K={K:2d} alpha={alpha:.1f}: buddy_ratio={ratio:.3f}  knn_pres={pres:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, mat, title, cmap in [
        (ax[0], ratio_mat, "buddy/random ratio (lower=better)", "viridis_r"),
        (ax[1], pres_mat, "kNN preservation (higher=better)", "viridis"),
    ]:
        im = a.imshow(mat, cmap=cmap, aspect="auto")
        a.set_xticks(range(len(alphas))); a.set_xticklabels([f"α={x}" for x in alphas])
        a.set_yticks(range(len(Ks))); a.set_yticklabels([f"K={x}" for x in Ks])
        a.set_title(title)
        for ki in range(len(Ks)):
            for ai in range(len(alphas)):
                a.text(ai, ki, f"{mat[ki, ai]:.3f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Conditional buddies — K × α sweep at n_dim=16 (Impressions)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hparam_heatmaps.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "hparam_sweep_stats.csv"), "w") as f:
        f.write("K,alpha,buddy_ratio,knn_preservation\n")
        for r in hp_rows:
            f.write(f"{r['K']},{r['alpha']},{r['buddy_ratio']:.4f},{r['knn_preservation']:.4f}\n")

    print(f"\nSaved figures + CSVs → {OUT}")


if __name__ == "__main__":
    main()
