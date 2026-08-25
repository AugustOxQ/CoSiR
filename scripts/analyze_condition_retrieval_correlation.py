"""
Post-hoc drift/shift-vs-retrieval-rank correlation diagnostic (Experiment 11.2, spec
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md S4).

Experiment 11.1 found that letting the per-sample condition table keep training after
buddy-init hurts i2t retrieval relative to freezing it right after init (frozen beats
trained, mean Delta=+4.67 R1, mean/SEM=+32.1, 3/3 seeds). This script asks *why*, at the
per-sample level, using only artifacts 11.1 already saved: for each of a run's training
samples, does how far its condition moved from its (frozen-arm-preserved) init value, or
how much conditioning shifts its combine-side embedding, predict how much worse (or
better) that exact sample ranks under its own trained condition vs. its own frozen
condition -- ranked against the FULL training population's projected "other side"
embeddings (not a small closed gallery), so the rank numbers sit at a realistic
retrieval-task scale. No oracle search over all conditions, no condition_predictor --
each sample uses only its own real, assigned condition, matching the population 11.1's
geometry diagnostic already measures drift/shift over.

One mode:
  --pair FROZEN_DIR TRAINED_DIR   analyze one same-seed frozen/trained pair, write
                                  condition_geometry/retrieval_correlation_vs_frozen.json
                                  inside TRAINED_DIR.
  --selftest                     offline arithmetic check of the pure math helpers.

Usage:
  python scripts/analyze_condition_retrieval_correlation.py --pair <frozen_run_dir> <trained_run_dir>
  python scripts/analyze_condition_retrieval_correlation.py --selftest
"""
import argparse
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr


def condition_drift(cond_current: np.ndarray, cond_init: np.ndarray) -> np.ndarray:
    """Per-sample L2 distance between the current condition table and its buddy-init
    value. Matches train_cosir.py's own `drift_from_init` convention
    (`(embeddings - z_init).norm(dim=1)`), just per-sample instead of pre-averaged.
    Both arrays: [N, D], row-aligned."""
    return np.linalg.norm(cond_current - cond_init, axis=1)


def rank_of_true_match(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    true_idx: np.ndarray,
    chunk: int = 200,
) -> np.ndarray:
    """1-indexed rank of each query's true match within `gallery_emb`, by descending dot
    product (callers pass pre-normalized rows for cosine ranking). `true_idx[i]` is the
    gallery row index that is query i's correct match -- for this diagnostic's 1:1
    paired feature store, that's the query's own row index in the shared population
    order, not necessarily its position within `query_emb` (which may be a subsample).
    Chunked over the query dimension to bound peak memory to chunk x len(gallery_emb)."""
    n_query = query_emb.shape[0]
    ranks = np.empty(n_query, dtype=np.int64)
    for s in range(0, n_query, chunk):
        e = min(s + chunk, n_query)
        sims = query_emb[s:e] @ gallery_emb.T  # [chunk, n_gallery]
        true_scores = sims[np.arange(e - s), true_idx[s:e]]
        ranks[s:e] = 1 + (sims > true_scores[:, None]).sum(axis=1)
    return ranks


def spearman_correlate(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Spearman rank correlation between two per-sample arrays. Returns rho=0/p=1 if
    either array has ~zero variance (scipy raises/warns on constant input) -- mirrors
    analyze_condition_geometry.py's correlate_shift guard for the same reason."""
    if np.std(x) < 1e-8 or np.std(y) < 1e-8 or len(x) < 2:
        return {"rho": 0.0, "p": 1.0}
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p)}


def rank_extremes(
    delta_rank: np.ndarray,
    sample_ids: List[int],
    drift: np.ndarray,
    shift: np.ndarray,
    k: int = 20,
) -> Dict[str, List[Dict]]:
    """Top-k most-degraded (delta_rank most positive: trained ranks this sample worse
    than frozen) and most-improved (most negative) samples, for qualitative inspection.
    All four arrays must be row-aligned (same per-sample order)."""
    order = np.argsort(delta_rank)
    most_improved = [
        {
            "sample_id": int(sample_ids[i]),
            "delta_rank": int(delta_rank[i]),
            "condition_drift": float(drift[i]),
            "embedding_shift": float(shift[i]),
        }
        for i in order[:k]
    ]
    most_degraded = [
        {
            "sample_id": int(sample_ids[i]),
            "delta_rank": int(delta_rank[i]),
            "condition_drift": float(drift[i]),
            "embedding_shift": float(shift[i]),
        }
        for i in order[::-1][:k]
    ]
    return {"most_degraded": most_degraded, "most_improved": most_improved}


def _selftest():
    # condition_drift: simple known-distance vectors.
    cur = np.array([[1.0, 0.0], [3.0, 4.0]])
    init = np.array([[0.0, 0.0], [0.0, 0.0]])
    d = condition_drift(cur, init)
    assert np.allclose(d, [1.0, 5.0]), d

    # rank_of_true_match: perfect match -> rank 1 for every query.
    gallery = np.eye(5)
    ranks = rank_of_true_match(gallery.copy(), gallery, np.arange(5))
    assert list(ranks) == [1, 1, 1, 1, 1], ranks

    # rank_of_true_match: a query identical to a DIFFERENT gallery row than its true
    # match gets demoted behind that row. True match (row 0) has sim=0 to this query;
    # row 2 (sim=1) is the only entry that beats it -> rank = 2.
    gallery4 = np.eye(4)
    query = np.array([[0.0, 0.0, 1.0, 0.0]])
    ranks2 = rank_of_true_match(query, gallery4, np.array([0]))
    assert ranks2[0] == 2, ranks2

    # rank_of_true_match: chunking doesn't change the result.
    rng = np.random.default_rng(0)
    g = rng.normal(size=(37, 6))
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    q_idx = rng.choice(37, size=15, replace=False)
    q = g[q_idx] + rng.normal(scale=0.01, size=(15, 6))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r_chunk1 = rank_of_true_match(q, g, q_idx, chunk=1)
    r_chunk37 = rank_of_true_match(q, g, q_idx, chunk=37)
    assert np.array_equal(r_chunk1, r_chunk37), (r_chunk1, r_chunk37)

    # spearman_correlate: monotonic nonlinear transform -> rho ~= 1.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x ** 3
    c = spearman_correlate(x, y)
    assert abs(c["rho"] - 1.0) < 1e-6, c

    # spearman_correlate: zero-variance input -> guarded degenerate result.
    xc = np.array([1.0, 1.0, 1.0])
    yc = np.array([5.0, 3.0, 9.0])
    cc = spearman_correlate(xc, yc)
    assert cc == {"rho": 0.0, "p": 1.0}, cc

    # rank_extremes: correct extremes, ids, and paired drift/shift values.
    delta_rank = np.array([5, -3, 0, 20, -10])
    sample_ids = [100, 101, 102, 103, 104]
    drift = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    shift = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ex = rank_extremes(delta_rank, sample_ids, drift, shift, k=2)
    assert [r["sample_id"] for r in ex["most_degraded"]] == [103, 100], ex
    assert [r["sample_id"] for r in ex["most_improved"]] == [104, 101], ex
    assert ex["most_degraded"][0]["condition_drift"] == 0.4, ex

    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
