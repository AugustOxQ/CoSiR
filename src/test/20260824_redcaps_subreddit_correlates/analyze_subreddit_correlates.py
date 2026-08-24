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
from scipy.stats import spearmanr

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


def subreddit_properties(data):
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


def correlate(metric_by_sub: dict, props_by_sub: dict):
    """Pearson r AND Spearman rho between a per-subreddit metric (lift, or the
    enrichment z-score — anything keyed by subreddit name) and each of the three
    properties, over subreddits present in both dicts. Pearson only detects linear
    association; Spearman (rank correlation) also catches a monotone-but-nonlinear
    relationship, which matters here because lift vs. size is exactly that (see the
    report's log-x scatter panel). Returns
    {property_name: {"pearson": (r, n), "spearman": (rho, n)}}.
    """
    names = [n for n in metric_by_sub if n in props_by_sub]
    vals_metric = np.array([metric_by_sub[n] for n in names])
    out = {}
    for prop in ("size", "caption_diversity", "visual_homogeneity"):
        vals = np.array([props_by_sub[n][prop] for n in names])
        mask = ~np.isnan(vals_metric) & ~np.isnan(vals)
        n = int(mask.sum())
        if n < 3:
            out[prop] = {"pearson": (float("nan"), n), "spearman": (float("nan"), n)}
            continue
        r = float(np.corrcoef(vals_metric[mask], vals[mask])[0, 1])
        rho, _p = spearmanr(vals_metric[mask], vals[mask])
        out[prop] = {"pearson": (r, n), "spearman": (float(rho), n)}
    return out


def _print_correlation(label: str, corr: dict):
    print(f"\nCorrelation(subreddit {label}, property) — Pearson (linear) and Spearman (monotone rank):")
    for prop, stats in corr.items():
        r, n_p = stats["pearson"]
        rho, n_s = stats["spearman"]
        print(f"  {prop:>20}: pearson r={r:+.3f}  spearman rho={rho:+.3f}  (n={n_p} subreddits)")


# Scale -> (storage_dir, annotation_path). "150k" uses redcaps_buddy's own defaults
# (None -> don't override). 300k/500k point at the UNIFORM-RANDOM diverse subsamples
# built by build_subsample.py (all 350 subreddits present) -- NOT Experiment 1's
# same-named training feature stores, which are raw prefixes of a subreddit-grouped
# file and only span 15/28 subreddits respectively (see docs/reports/
# 2026-08-24_redcaps_subreddit_signal_correlates.md's scale-extension section).
SCALES = {
    "150k": (None, None),
    "300k": ("/data/SSD2/pre_extract/redcaps_300k_diverse/features",
             "/data/PDD/redcaps/redcaps_plus/redcaps_300k_diverse.json"),
    "500k": ("/data/SSD2/pre_extract/redcaps_500k_diverse/features",
             "/data/PDD/redcaps/redcaps_plus/redcaps_500k_diverse.json"),
}


def run(scale: str = "150k"):
    import redcaps_buddy as rb

    storage_dir, annotation_path = SCALES[scale]
    print(f"Loading RedCaps data ({scale}) + buddy graph...")
    if storage_dir is None:
        data = rb.load_data()
    else:
        data = rb.load_data(storage_dir=storage_dir, annotation_path=annotation_path)
    # Reuses the same buddy-graph construction path as the rest of the RedCaps buddy
    # analysis (K=30 — the project-wide default, configs/train/default.yaml). `alpha`
    # is not a parameter of this path: union_graph combines two unweighted mutual-kNN
    # adjacency matrices by set union, no distance weighting involved (alpha only
    # matters for a different, weighted-distance graph construction elsewhere).
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
    # Keep deg_s (edge-endpoint degree) instead of discarding it: it's needed below to
    # recover each subreddit's purity (obs_s / deg_s) and to check whether the size
    # effect on lift is a content effect or a structural artifact of the lift metric's
    # own per-subreddit normalization (see the report's "mechanism" section).
    lift_by_sub = {name: lift for name, lift, _deg in lift_result["top_enriched"]}
    deg_by_sub = {name: deg for name, _lift, deg in lift_result["top_enriched"]}
    print(f"  overall_lift={lift_result['overall_lift']:.2f}x over "
          f"{len(lift_result['top_enriched'])} qualifying subreddits")

    print("Computing per-subreddit enrichment z-score (significance, complementing lift's "
          "effect-size-only ratio)...")
    z_result = rb.subreddit_enrichment_zscore(data, e, top_k=None)
    z_by_sub = {name: z for name, z, _m in z_result["top_enriched"]}
    m_by_sub = {name: m for name, _z, m in z_result["top_enriched"]}
    print(f"  z_overall={z_result['z_overall']:.1f} over "
          f"{len(z_result['top_enriched'])} subreddits passing the z reliability filter "
          f"(vs. {len(lift_by_sub)} passing lift's filter — the two nulls have different "
          f"minimum-count requirements, see subreddit_enrichment_zscore's docstring)")

    print("Computing per-subreddit properties (size, caption diversity, visual homogeneity)...")
    props_by_sub = subreddit_properties(data)

    corr = correlate(lift_by_sub, props_by_sub)
    _print_correlation("lift", corr)

    z_corr = correlate(z_by_sub, props_by_sub)
    _print_correlation("z-score", z_corr)

    # --- Does z suffer the same size confound lift does? ---
    # lift is structurally pulled toward 1/size (see the purity check below); z, being a
    # count-based significance statistic, should if anything correlate POSITIVELY with
    # size (bigger subreddits -> more edges -> more statistical power to detect a fixed
    # true effect) rather than negatively -- checking this directly rather than assuming it.
    names_lz = [n for n in z_by_sub if n in props_by_sub]
    z_common = np.array([z_by_sub[n] for n in names_lz])
    size_common_z = np.array([props_by_sub[n]["size"] for n in names_lz], dtype=np.float64)
    rho_size_z, _ = spearmanr(size_common_z, z_common)
    r_logsize_z = float(np.corrcoef(np.log(size_common_z), z_common)[0, 1])
    print(f"\nz-score vs. size (sanity check against lift's known 1/size confound):")
    print(f"  spearman size vs z : {rho_size_z:+.3f}  "
          f"({'positive as expected (more power at larger N)' if rho_size_z > 0 else 'UNEXPECTED: negative, investigate before trusting z'})")
    print(f"  pearson  log(size) vs z : {r_logsize_z:+.3f}")

    # --- Purity check: is the size effect content-driven or a normalization artifact? ---
    # subreddit_lift computes, per subreddit s: p_s = deg_s / total_endpoints,
    # exp_s = deg_s * p_s, lift_s = obs_s / exp_s. So purity_s := obs_s / deg_s (the
    # fraction of subreddit s's own edge endpoints that land on a same-subreddit edge)
    # is recoverable without obs_s directly: purity_s = lift_s * exp_s / deg_s
    # = lift_s * p_s = lift_s * deg_s / total_endpoints.
    total_endpoints = 2 * len(e)  # each edge contributes 2 endpoints; e is upper-triangle-only
    purity_by_sub = {name: lift_by_sub[name] * deg_by_sub[name] / total_endpoints
                      for name in lift_by_sub}

    names_common = [n for n in lift_by_sub if n in props_by_sub]
    sizes = np.array([props_by_sub[n]["size"] for n in names_common], dtype=np.float64)
    degs = np.array([deg_by_sub[n] for n in names_common], dtype=np.float64)
    purities = np.array([purity_by_sub[n] for n in names_common], dtype=np.float64)
    lifts_common = np.array([lift_by_sub[n] for n in names_common], dtype=np.float64)

    rho_deg_size, _ = spearmanr(degs, sizes)
    rho_size_purity, _ = spearmanr(sizes, purities)
    r_logsize_purity = float(np.corrcoef(np.log(sizes), purities)[0, 1])
    r_logsize_lift = float(np.corrcoef(np.log(sizes), lifts_common)[0, 1])
    r_logsize_loglift = float(np.corrcoef(np.log(sizes), np.log(lifts_common))[0, 1])
    print("\nPurity check (does the size effect on lift reflect purity, or lift's own "
          "1/size normalization?):")
    print(f"  spearman deg_s      vs size          : {rho_deg_size:+.3f}  "
          f"(deg_s tracks size almost exactly)")
    print(f"  spearman size       vs purity         : {rho_size_purity:+.3f}  "
          f"(purity rises with size — opposite of the lift trend)")
    print(f"  pearson  log(size)  vs purity         : {r_logsize_purity:+.3f}")
    print(f"  pearson  log(size)  vs lift            : {r_logsize_lift:+.3f}")
    print(f"  pearson  log(size)  vs log(lift)       : {r_logsize_loglift:+.3f}")

    # --- Full per-subreddit table (name, lift, z, size, deg_s/M_s, purity), sorted by lift desc ---
    print(f"\nFull per-subreddit table ({len(names_common)} subreddits, sorted by lift desc):")
    header = f"{'subreddit':<24}{'lift':>10}{'z':>10}{'size':>8}{'deg_s':>8}{'M_s':>8}{'purity':>9}"
    print(header)
    for n in sorted(names_common, key=lambda x: -lift_by_sub[x]):
        z_str = f"{z_by_sub[n]:>10.1f}" if n in z_by_sub else f"{'--':>10}"
        m_str = f"{m_by_sub[n]:>8}" if n in m_by_sub else f"{'--':>8}"
        print(f"{n:<24}{lift_by_sub[n]:>10.2f}{z_str}{props_by_sub[n]['size']:>8}"
              f"{deg_by_sub[n]:>8}{m_str}{purity_by_sub[n]:>9.4f}")

    _write_figure(lift_by_sub, z_by_sub, props_by_sub, scale)
    return lift_by_sub, z_by_sub, props_by_sub, corr, z_corr, purity_by_sub


def _write_figure(lift_by_sub, z_by_sub, props_by_sub, scale: str = "150k"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "docs", "reports", "assets", "redcaps_subreddit_correlates")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (metric_by_sub, metric_label) in enumerate(
        [(lift_by_sub, "subreddit lift"), (z_by_sub, "enrichment z-score")]
    ):
        names = [n for n in metric_by_sub if n in props_by_sub]
        vals_metric = [metric_by_sub[n] for n in names]
        for col, prop in enumerate(("size", "caption_diversity", "visual_homogeneity")):
            ax = axes[row, col]
            vals = [props_by_sub[n][prop] for n in names]
            ax.scatter(vals, vals_metric, s=14, alpha=0.6)
            ax.set_xlabel(prop)
            ax.set_ylabel(metric_label)
            if prop == "size":
                ax.set_xscale("log")
    fig.suptitle(f"RedCaps subreddit signal-strength correlates — {scale} scale")
    fig.tight_layout()
    fname = "lift_vs_properties.png" if scale == "150k" else f"lift_vs_properties_{scale}.png"
    path = os.path.join(out_dir, fname)
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

    # --- subreddit_properties: synthetic Data-like object, 3 subreddits, known geometry ---
    import types
    rng = np.random.default_rng(0)
    n_per_sub = 25  # >= MIN_SUBREDDIT_SIZE=20
    sub_names = ["identical_img", "random_img", "identical_txt"]
    sub_id = np.repeat(np.arange(len(sub_names)), n_per_sub)

    def _rand_unit(n, d):
        x = rng.normal(size=(n, d)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    d = 8
    identical_img_row = np.eye(1, d, dtype=np.float32)  # [1, 0, ..., 0]
    identical_txt_row = np.eye(1, d, k=1, dtype=np.float32)  # [0, 1, 0, ..., 0]
    img_rows, txt_rows = [], []
    # sub 0: identical image rows (visual_homogeneity -> 1), random captions.
    img_rows.append(np.tile(identical_img_row, (n_per_sub, 1)))
    txt_rows.append(_rand_unit(n_per_sub, d))
    # sub 1: random image rows (visual_homogeneity -> ~0), random captions.
    img_rows.append(_rand_unit(n_per_sub, d))
    txt_rows.append(_rand_unit(n_per_sub, d))
    # sub 2: random image rows, identical captions (caption_diversity -> 0).
    img_rows.append(_rand_unit(n_per_sub, d))
    txt_rows.append(np.tile(identical_txt_row, (n_per_sub, 1)))

    fake_data = types.SimpleNamespace(
        sub_id=sub_id, sub_names=sub_names,
        img=np.concatenate(img_rows, axis=0), txt=np.concatenate(txt_rows, axis=0),
    )
    props = subreddit_properties(fake_data)
    assert set(props.keys()) == set(sub_names)
    for name in sub_names:
        assert props[name]["size"] == n_per_sub
    assert abs(props["identical_img"]["visual_homogeneity"] - 1.0) < 1e-5
    assert abs(props["identical_img"]["visual_homogeneity"]
               - props["random_img"]["visual_homogeneity"]) > 0.5, "identical vs random img rows should differ sharply"
    assert abs(props["identical_txt"]["caption_diversity"] - 0.0) < 1e-5
    print("SELFTEST subreddit_properties OK")

    # --- correlate(): Pearson vs. Spearman on a perfect-but-nonlinear relationship ---
    # lift = 1000 / size is a strictly monotone decreasing function of size, so Spearman
    # must recover rho = -1.0 exactly (rank-invariant to the nonlinearity), while Pearson
    # on the raw (non-log) values is pulled well short of -1 by the curvature — the same
    # gap the real RedCaps size-vs-lift relationship shows.
    sizes = np.array([10, 20, 40, 80, 160, 320, 640, 1280], dtype=np.float64)
    lifts = 1000.0 / sizes
    names = [f"s{i}" for i in range(len(sizes))]
    lift_by_sub = dict(zip(names, lifts))
    props_by_sub = {
        n: {"size": s, "caption_diversity": float("nan"), "visual_homogeneity": float("nan")}
        for n, s in zip(names, sizes)
    }
    corr = correlate(lift_by_sub, props_by_sub)
    rho, n_s = corr["size"]["spearman"]
    r, n_p = corr["size"]["pearson"]
    assert n_s == n_p == len(sizes)
    assert abs(rho - (-1.0)) < 1e-9, f"expected spearman rho=-1.0 exactly, got {rho}"
    assert r > -0.9, f"expected pearson to visibly underestimate the monotone relationship, got r={r}"
    # caption_diversity/visual_homogeneity are all-nan here -> correlate must report nan, not crash.
    assert np.isnan(corr["caption_diversity"]["pearson"][0])
    assert np.isnan(corr["caption_diversity"]["spearman"][0])
    print(f"SELFTEST correlate OK (spearman={rho:+.3f}, pearson={r:+.3f} on a "
          f"perfect monotone-nonlinear relationship)")

    print("SELFTEST OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scale", choices=list(SCALES.keys()), default="150k")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        run(scale=args.scale)
