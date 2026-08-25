"""
Post-hoc condition-embedding geometry diagnostic (Experiment 11.1, spec
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md S4).

Retrieval numbers (test_oracle/*_R1) can miss a real difference between the frozen and
trained arms of Experiment 11.1 -- this script inspects the actual embedding geometry
instead: how much does conditioning shift the combine-side embedding, how does that shift
distribution compare across epochs/arms, which samples are moved the most/least, and --
via a condition-vs-text cross grid -- whether conditions are interchangeable/null for a
given text (low diversity across conditions) or one condition dominates and collapses
every text to nearly the same output (low diversity across texts). Retrieval numbers alone
cannot distinguish those two failure modes from each other or from a healthy grid.

Two modes:
  --exp-dir PATH   analyze one run's condition_viz/ snapshots, write
                   condition_geometry/summary.json + a plot inside that run's directory.
  --compare A B    load two already-produced summary.json files (e.g. a frozen run and a
                   trained run, same seed) and print a paired diff.
  --selftest       offline arithmetic check of the pure math helpers, no data needed.

Usage:
  python scripts/analyze_condition_geometry.py --exp-dir <run_dir>
  python scripts/analyze_condition_geometry.py --compare <frozen_run_dir> <trained_run_dir>
  python scripts/analyze_condition_geometry.py --selftest
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


def compute_shift(comb_emb: torch.Tensor, unconditioned_emb: torch.Tensor) -> np.ndarray:
    """Per-sample 1 - cos(conditioned, unconditioned). Both [N, D]; neither need be
    pre-normalized (this normalizes both)."""
    cond_n = F.normalize(comb_emb, dim=-1)
    uncond_n = F.normalize(unconditioned_emb, dim=-1)
    return (1.0 - (cond_n * uncond_n).sum(dim=-1)).cpu().numpy()


def effective_dims(x: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Number of PCA components needed to explain >= variance_threshold of variance.
    x: [N, D]. Falls back to D if N <= D (PCA undefined)."""
    n, d = x.shape
    if n <= d:
        return d
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    var = s ** 2
    ratio = var / var.sum()
    cumsum = np.cumsum(ratio)
    return int(np.argmax(cumsum >= variance_threshold) + 1)


def pairwise_sim_spread(x: np.ndarray, n_sample: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean/std of pairwise cosine similarity over a random subsample (full N^2 is wasteful
    at N~120k). x rows need not be pre-normalized."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    xs = x[idx]
    xs = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    sims = xs @ xs.T
    iu = np.triu_indices(len(idx), k=1)
    off = sims[iu]
    return {"mean": float(off.mean()), "std": float(off.std())}


def rank_most_least_changed(shift: np.ndarray, sample_ids: List[int], k: int = 20) -> Dict[str, List[Dict]]:
    """Top-k / bottom-k samples by shift magnitude, paired with their sample id."""
    order = np.argsort(shift)  # ascending: least-changed first
    least = [{"sample_id": int(sample_ids[i]), "shift": float(shift[i])} for i in order[:k]]
    most = [{"sample_id": int(sample_ids[i]), "shift": float(shift[i])} for i in order[::-1][:k]]
    return {"most_changed": most, "least_changed": least}


def correlate_shift(shift: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    """Pearson r between per-sample shift and another per-sample scalar (condition norm,
    buddy-graph degree, ...). Returns r=0/p=1 if either array has ~zero variance."""
    if shift.std() < 1e-8 or other.std() < 1e-8 or len(shift) < 2:
        return {"r": 0.0, "p": 1.0}
    r, p = pearsonr(shift, other)
    return {"r": float(r), "p": float(p)}


def grid_diversity(comb_grid: torch.Tensor) -> Dict[str, np.ndarray]:
    """comb_grid[i, j] = combine(text_i, condition_j), shape [n_text, n_cond, D]. Splits
    diversity two ways to distinguish two collapse failure modes retrieval numbers can't
    tell apart:

      row_diversity[i] = 1 - mean pairwise cosine sim among {comb_grid[i, j] for all j}
                         (low => for text i, varying the condition barely changes the
                         output => conditions are interchangeable/null for this text)
      col_diversity[j] = 1 - mean pairwise cosine sim among {comb_grid[i, j] for all i}
                         (low => for condition j, varying the text barely changes the
                         output => condition j dominates/collapses the combination)

    Returns {"row_diversity": np.ndarray [n_text], "col_diversity": np.ndarray [n_cond]}.
    NaN for a row/column of length < 2 (nothing to compare pairwise).
    """
    n_text, n_cond, _ = comb_grid.shape
    g = F.normalize(comb_grid, dim=-1)

    row_diversity = np.full(n_text, np.nan)
    iu_cond = torch.triu_indices(n_cond, n_cond, offset=1)
    if iu_cond.shape[1] > 0:
        for i in range(n_text):
            sims = g[i] @ g[i].T
            row_diversity[i] = float(1.0 - sims[iu_cond[0], iu_cond[1]].mean())

    col_diversity = np.full(n_cond, np.nan)
    iu_text = torch.triu_indices(n_text, n_text, offset=1)
    if iu_text.shape[1] > 0:
        for j in range(n_cond):
            col = g[:, j]
            sims = col @ col.T
            col_diversity[j] = float(1.0 - sims[iu_text[0], iu_text[1]].mean())

    return {"row_diversity": row_diversity, "col_diversity": col_diversity}


def _selftest():
    torch.manual_seed(0)
    # compute_shift: identical vectors -> shift 0; orthogonal -> shift 1.
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    s = compute_shift(a, b)
    assert abs(s[0] - 0.0) < 1e-6, s
    assert abs(s[1] - 1.0) < 1e-6, s

    # effective_dims: a rank-1 signal embedded in D=5 needs 1 component for 95% variance.
    rng = np.random.default_rng(0)
    direction = rng.normal(size=5)
    x = np.outer(rng.normal(size=500), direction) + rng.normal(scale=1e-4, size=(500, 5))
    assert effective_dims(x) == 1, effective_dims(x)

    # pairwise_sim_spread: identical rows -> mean sim == 1, std == 0.
    same = np.tile(rng.normal(size=(1, 8)), (100, 1))
    spread = pairwise_sim_spread(same, n_sample=50)
    assert abs(spread["mean"] - 1.0) < 1e-5, spread
    assert spread["std"] < 1e-5, spread

    # rank_most_least_changed: correct extremes and ids.
    shift = np.array([0.1, 0.9, 0.5, 0.0, 1.0])
    ids = [10, 11, 12, 13, 14]
    ranks = rank_most_least_changed(shift, ids, k=2)
    assert [r["sample_id"] for r in ranks["most_changed"]] == [14, 11], ranks
    assert [r["sample_id"] for r in ranks["least_changed"]] == [13, 10], ranks

    # correlate_shift: perfectly correlated inputs -> r ~= 1.
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    c = correlate_shift(x1, x2)
    assert abs(c["r"] - 1.0) < 1e-6, c

    # grid_diversity: three constructed scenarios covering both failure modes + healthy.
    n_text, n_cond, d = 6, 5, 8
    base_text = F.normalize(torch.randn(n_text, d), dim=-1)
    base_cond = F.normalize(torch.randn(n_cond, d), dim=-1)

    # Case A: output depends only on text, ignores condition entirely -> conditions "null".
    # row_diversity ~ 0 (no variation across j, for each i); col_diversity high (varies across i).
    grid_null_cond = base_text.unsqueeze(1).expand(n_text, n_cond, d).clone()
    ga = grid_diversity(grid_null_cond)
    assert np.nanmax(ga["row_diversity"]) < 1e-5, ga["row_diversity"]
    assert np.nanmean(ga["col_diversity"]) > 0.1, ga["col_diversity"]

    # Case B: output depends only on condition, ignores text entirely -> condition dominates.
    # col_diversity ~ 0 (no variation across i, for each j); row_diversity high (varies across j).
    grid_dominant_cond = base_cond.unsqueeze(0).expand(n_text, n_cond, d).clone()
    gb = grid_diversity(grid_dominant_cond)
    assert np.nanmax(gb["col_diversity"]) < 1e-5, gb["col_diversity"]
    assert np.nanmean(gb["row_diversity"]) > 0.1, gb["row_diversity"]

    # Case C: healthy -- independent random grid, both diversities nontrivially positive.
    grid_healthy = F.normalize(torch.randn(n_text, n_cond, d), dim=-1)
    gc = grid_diversity(grid_healthy)
    assert np.nanmean(gc["row_diversity"]) > 0.3, gc["row_diversity"]
    assert np.nanmean(gc["col_diversity"]) > 0.3, gc["col_diversity"]

    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
