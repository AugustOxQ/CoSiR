"""
Experiment 9 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md S4):
what predicts buddy-signal strength across RedCaps subreddits? Correlates per-subreddit
buddy-edge lift (redcaps_buddy.subreddit_lift, full breakdown) against three properties:
sample count, caption diversity (1 - mean pairwise CLIP-text cosine similarity within the
subreddit), and visual homogeneity (mean pairwise CLIP-image cosine similarity within the
subreddit) — both computed via a closed-form identity (no O(n^2) pairwise loop).

Usage
-----
  python analyze_subreddit_correlates.py            # full run against cached RedCaps data
  python analyze_subreddit_correlates.py --selftest # offline arithmetic check

Requires: numpy, matplotlib (Agg backend). Run from anywhere; sys.path is fixed up below.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

MIN_SUBREDDIT_SIZE = 20  # below this, pairwise-similarity estimates are too noisy to trust


def mean_pairwise_cosine_sim(X: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity over unit-norm row vectors, via the closed-form
    identity ||sum(X)||^2 = n + 2*sum_{i<j}(x_i . x_j) (exact for unit-norm rows) —
    O(n*d) instead of O(n^2*d). Returns nan for n < 2.
    """
    n = X.shape[0]
    if n < 2:
        return float("nan")
    S = X.sum(axis=0)
    total = float(S @ S) - n  # = 2 * sum_{i<j} x_i . x_j
    return total / (n * (n - 1))


def subreddit_properties(data, sub_id_filter=None):
    """
    Per-subreddit (size, caption_diversity, visual_homogeneity), keyed by subreddit name.
    caption_diversity = 1 - mean_pairwise_cosine_sim(txt rows); visual_homogeneity =
    mean_pairwise_cosine_sim(img rows). Skips subreddits with < MIN_SUBREDDIT_SIZE samples.
    """
    props = {}
    n_sub = len(data.sub_names)
    for s in range(n_sub):
        idx = np.where(data.sub_id == s)[0]
        if len(idx) < MIN_SUBREDDIT_SIZE:
            continue
        diversity = 1.0 - mean_pairwise_cosine_sim(data.txt[idx])
        homogeneity = mean_pairwise_cosine_sim(data.img[idx])
        props[data.sub_names[s]] = {
            "size": len(idx), "caption_diversity": diversity, "visual_homogeneity": homogeneity,
        }
    return props


def correlate(lift_by_sub: dict, props_by_sub: dict):
    """Pearson r between per-subreddit lift and each of the three properties, over
    subreddits present in both dicts. Returns {property_name: (r, n)}."""
    names = [n for n in lift_by_sub if n in props_by_sub]
    lifts = np.array([lift_by_sub[n] for n in names])
    out = {}
    for prop in ("size", "caption_diversity", "visual_homogeneity"):
        vals = np.array([props_by_sub[n][prop] for n in names])
        mask = ~np.isnan(lifts) & ~np.isnan(vals)
        if mask.sum() < 3:
            out[prop] = (float("nan"), int(mask.sum()))
            continue
        r = float(np.corrcoef(lifts[mask], vals[mask])[0, 1])
        out[prop] = (r, int(mask.sum()))
    return out


def run():
    import redcaps_buddy as rb

    print("Loading RedCaps data + buddy graph...")
    data = rb.load_data()
    # Reuses the same buddy-graph construction path as the rest of the RedCaps buddy
    # analysis (K=30, alpha=0.5 — the project-wide default, configs/train/default.yaml).
    from src.conditional_buddy.buddy_graph import mutual_knn, union_graph
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()
    A_img = mutual_knn(data.img, 30, device, 1024, use_half=use_half)
    A_txt = mutual_knn(data.txt, 30, device, 1024, use_half=use_half)
    E = union_graph(A_img, A_txt)
    e = np.stack(E.nonzero(), axis=1)
    e = e[e[:, 0] < e[:, 1]]  # upper triangle only, matches subreddit_lift's expected input

    print(f"Computing full per-subreddit lift ({len(data.sub_names)} subreddits)...")
    lift_result = rb.subreddit_lift(data, e, top_k=None)
    lift_by_sub = {name: lift for name, lift, _deg in lift_result["top_enriched"]}
    print(f"  overall_lift={lift_result['overall_lift']:.2f}x over "
          f"{len(lift_result['top_enriched'])} qualifying subreddits")

    print("Computing per-subreddit properties (size, caption diversity, visual homogeneity)...")
    props_by_sub = subreddit_properties(data)

    corr = correlate(lift_by_sub, props_by_sub)
    print("\nCorrelation(subreddit lift, property):")
    for prop, (r, n) in corr.items():
        print(f"  {prop:>20}: r={r:+.3f}  (n={n} subreddits)")

    _write_figure(lift_by_sub, props_by_sub)
    return lift_by_sub, props_by_sub, corr


def _write_figure(lift_by_sub, props_by_sub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in lift_by_sub if n in props_by_sub]
    lifts = [lift_by_sub[n] for n in names]
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "docs", "reports", "assets", "redcaps_subreddit_correlates")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, prop in zip(axes, ("size", "caption_diversity", "visual_homogeneity")):
        vals = [props_by_sub[n][prop] for n in names]
        ax.scatter(vals, lifts, s=14, alpha=0.6)
        ax.set_xlabel(prop)
        ax.set_ylabel("subreddit lift")
        if prop == "size":
            ax.set_xscale("log")
    fig.tight_layout()
    path = os.path.join(out_dir, "lift_vs_properties.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def _selftest():
    # Orthogonal unit vectors -> mean pairwise cosine similarity = 0.
    orth = np.eye(5, dtype=np.float32)
    assert abs(mean_pairwise_cosine_sim(orth) - 0.0) < 1e-6, mean_pairwise_cosine_sim(orth)

    # Identical unit vectors -> mean pairwise cosine similarity = 1.
    same = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (6, 1))
    assert abs(mean_pairwise_cosine_sim(same) - 1.0) < 1e-6, mean_pairwise_cosine_sim(same)

    # Single row -> undefined (nan), not a crash.
    assert np.isnan(mean_pairwise_cosine_sim(np.array([[1.0, 0.0]], dtype=np.float32)))

    # Cross-check the closed form against a brute-force O(n^2) loop on random unit vectors.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    fast = mean_pairwise_cosine_sim(X)
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
    brute = np.mean([X[i] @ X[j] for i, j in pairs])
    assert abs(fast - brute) < 1e-4, (fast, brute)

    print("SELFTEST OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        run()
