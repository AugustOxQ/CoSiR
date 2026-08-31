"""
Phase 1 (offline) RedCaps buddy analysis:
  (1) subreddit lift (same-subreddit enrichment over buddy edges, B vs E),
  (2) held-out DINOv2 NN test (run extract_dino.py first), and
  (3) init-structure exploration of the [N,16] buddy spectral init:
      subreddit silhouette, per-dim mutual information (coarse->fine hierarchy),
      KMeans agreement with subreddit at low vs high K, and a 2D PCA scatter.

Writes figures + stats.json under docs/reports/assets/redcaps_buddy/.
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import redcaps_buddy as rb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.conditional_buddy import compute_buddy_init

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "docs", "reports", "assets", "redcaps_buddy"))
DINO = os.path.join(os.path.dirname(__file__), "dino_feats.npy")
os.makedirs(ASSETS, exist_ok=True)
SEED = 42
# Structure init is computed on a representative random subsample: SpectralEmbedding
# (arpack shift-invert) factorizes an N×N Laplacian and scales badly past ~50K.
# Validation probes (lift, DINO) still use the FULL 150K graph; only the
# init-structure exploration uses this subsample (silhouette/MI/PCA are valid on it).
STRUCT_N = 12000
STRUCT_K = 60


def structure_probes(y, emb, sub_names, stats):
    """Explore the structure of the [n,16] buddy init w.r.t. subreddit labels."""
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(SEED)
    n_sub = len(sub_names)
    n = len(y)

    # --- silhouette on a subsample (full n is O(n^2)) ---
    idx = rng.choice(n, size=min(8000, n), replace=False)
    sil_all = float(silhouette_score(emb[idx], y[idx]))
    # restrict to top-30 subreddits (cleaner, less long-tail noise)
    top = np.argsort(-np.bincount(y, minlength=n_sub))[:30]
    topmask = np.isin(y[idx], top)
    sil_top = float(silhouette_score(emb[idx][topmask], y[idx][topmask])) \
        if topmask.sum() > 100 else float("nan")

    # --- per-dim mutual information with subreddit (coarse->fine hierarchy) ---
    midx = rng.choice(n, size=min(12000, n), replace=False)
    mi = mutual_info_classif(emb[midx], y[midx], discrete_features=False,
                             random_state=SEED)
    mi = mi.tolist()

    # --- KMeans agreement with subreddit at low vs high K ---
    aris = {}
    for k in (20, 100, 350):
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=3,
                             batch_size=4096).fit_predict(emb)
        aris[str(k)] = float(adjusted_rand_score(y, km))

    stats["structure"] = {
        "struct_subsample_n": int(n),
        "silhouette_all_subreddits": sil_all,
        "silhouette_top30": sil_top,
        "per_dim_mutual_info": mi,
        "kmeans_ari_vs_subreddit": aris,
    }
    print(f"  silhouette: all={sil_all:.3f} top30={sil_top:.3f}")
    print(f"  KMeans ARI vs subreddit: {aris}")
    print(f"  per-dim MI (sorted desc): "
          f"{', '.join(f'{m:.3f}' for m in sorted(mi, reverse=True))}")
    return top, mi


def fig_structure(y, emb, sub_names, top, mi):
    """2D PCA scatter colored by top subreddits + per-dim MI bar."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(SEED)
    n = len(y)
    idx = rng.choice(n, size=min(20000, n), replace=False)
    p2 = PCA(n_components=2, random_state=SEED).fit_transform(emb[idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    top12 = top[:12]
    cmap = plt.get_cmap("tab20")
    yi = y[idx]
    other = ~np.isin(yi, top12)
    ax.scatter(p2[other, 0], p2[other, 1], s=2, c="lightgray", alpha=0.3, label="other")
    for j, s in enumerate(top12):
        m = yi == s
        ax.scatter(p2[m, 0], p2[m, 1], s=4, color=cmap(j % 20),
                   label=sub_names[s], alpha=0.7)
    ax.set_title("Buddy init (16-d) → PCA 2-D, colored by top-12 subreddits")
    ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best")

    ax = axes[1]
    ax.bar(range(len(mi)), mi, color="steelblue")
    ax.set_xlabel("init dimension (spectral order, coarse→fine)")
    ax.set_ylabel("mutual information with subreddit")
    ax.set_title("Per-dimension subreddit information")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "init_structure.png"), dpi=130)
    plt.close(fig)


def fig_lift_and_dino(stats):
    graphs = stats["graphs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # subreddit lift
    ax = axes[0]
    names = ["B", "E"]
    lifts = [graphs[g]["subreddit_lift"]["overall_lift"] for g in names]
    ax.bar(names, lifts, color=["#4c72b0", "#dd8452"])
    ax.axhline(1.0, ls="--", c="k", lw=1, label="chance")
    ax.set_ylabel("same-subreddit lift (obs / expected)")
    ax.set_title("Subreddit lift over buddy edges")
    for i, v in enumerate(lifts):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax.legend()
    # DINO
    ax = axes[1]
    if "dino" in graphs["B"]:
        x = np.arange(2)
        buddy = [graphs[g]["dino"]["buddy"] for g in names]
        rand = [graphs[g]["dino"]["random"] for g in names]
        ax.bar(x - 0.2, buddy, 0.4, label="buddy", color="#4c72b0")
        ax.bar(x + 0.2, rand, 0.4, label="random", color="lightgray")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("mean held-out DINOv2 cosine distance")
        ax.set_title("Held-out DINOv2: buddy vs random (lower = closer)")
        for i in range(2):
            ax.text(i - 0.2, buddy[i], f"{buddy[i]:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + 0.2, rand[i], f"{rand[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "DINOv2 not extracted\n(run extract_dino.py)",
                ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "lift_and_dino.png"), dpi=130)
    plt.close(fig)


def main():
    data = rb.load_data()
    print(f"Loaded {data.n:,} samples, {len(data.sub_names)} subreddits")
    G = rb.build_graphs(data, K=30)

    heldout = np.load(DINO) if os.path.exists(DINO) else None
    if heldout is not None:
        print(f"held-out DINO loaded: {heldout.shape}")

    stats = {"dataset": "redcaps_150k", "n": data.n,
             "n_subreddits": len(data.sub_names), "K": 30, "graphs": {}}

    for name in ("B", "E"):
        e = rb.edges(G[name])
        gs = rb.graph_stats(data, e)
        lift = rb.subreddit_lift(data, e)
        g = {**gs, "subreddit_lift": lift}
        if heldout is not None:
            g["dino"] = rb.heldout_distance_test(heldout, data, e)
        stats["graphs"][name] = g
        print(f"[{name}] edges={gs['n_edges']:,} avgdeg={gs['avg_degree']:.2f} "
              f"isolated={gs['isolated_nodes']} | sub-lift={lift['overall_lift']:.2f} "
              f"(obs {lift['obs_same_frac']:.3f} vs exp {lift['exp_same_frac']:.4f})"
              + (f" | DINO buddy={g['dino']['buddy']:.2f} rand={g['dino']['random']:.2f}"
                 if heldout is not None else ""))
        print(f"    top enriched subreddits: "
              f"{', '.join(f'{n}×{l:.1f}' for n, l, _ in lift['top_enriched'][:8])}")

    # Write validation results FIRST, so the (slow, best-effort) structure init
    # below can never cost us the lift/DINO numbers.
    statspath = os.path.join(ASSETS, "stats.json")
    with open(statspath, "w") as f:
        json.dump(stats, f, indent=2)
    fig_lift_and_dino(stats)
    print(f"Wrote validation {statspath} + lift_and_dino.png", flush=True)

    # --- structure of the init space (smaller, DENSER subsample) ---
    # A denser graph (higher K) on fewer nodes stays (nearly) connected, so the
    # arpack spectral solver isn't crippled by a cluster of near-zero eigenvalues
    # (one per connected component). Best-effort: never block the validation result.
    try:
        rng = np.random.default_rng(SEED)
        sidx = np.sort(rng.choice(data.n, size=min(STRUCT_N, data.n), replace=False))
        print(f"Computing buddy init [{len(sidx)},16] K={STRUCT_K} for structure "
              f"probes (subsample of {data.n}) …", flush=True)
        emb = compute_buddy_init(data.img[sidx], data.txt[sidx], n_dim=16,
                                 method="spectral", K=STRUCT_K, alpha=0.5,
                                 device="cuda", normalize_method="rank", seed=SEED)
        y_s = data.sub_id[sidx]
        top, mi = structure_probes(y_s, emb, data.sub_names, stats)
        fig_structure(y_s, emb, data.sub_names, top, mi)
        with open(statspath, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote structure probes to {statspath} + init_structure.png", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[structure probes FAILED — validation results kept] {exc!r}", flush=True)


if __name__ == "__main__":
    main()
