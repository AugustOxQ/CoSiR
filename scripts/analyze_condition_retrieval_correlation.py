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

import json
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analyze_condition_geometry import (
    compute_shift,
    _load_run_config,
    _load_train_features,
    _rebuild_combiner,
    _compute_comb_emb,
)
from src.metrics.regularizer import reorder_features_to_z
from src.model.combiner import OtherProjMLP


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


def _load_epoch_snapshot(exp_dir: str, index: int) -> dict:
    """index=-1 for the final epoch, 0 for the first (used to sanity-check the frozen
    arm actually never moved, rather than assuming it)."""
    epoch_files = sorted((Path(exp_dir) / "condition_viz").glob("epoch_*.pt"))
    if not epoch_files:
        raise FileNotFoundError(f"no condition_viz/epoch_*.pt snapshots under {exp_dir}")
    return torch.load(epoch_files[index], map_location="cpu")


def _load_final_epoch_snapshot(exp_dir: str) -> dict:
    return _load_epoch_snapshot(exp_dir, -1)


def _rebuild_other_proj(snapshot: dict) -> nn.Module:
    cfg = snapshot["other_proj_config"]
    if cfg["type"] == "Linear":
        proj = nn.Linear(cfg["feature_dim"], cfg["feature_dim"])
    elif cfg["type"] == "OtherProjMLP":
        proj = OtherProjMLP(
            feature_dim=cfg["feature_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_blocks=cfg["num_blocks"],
        )
    else:
        raise ValueError(f"unknown other_proj type {cfg['type']!r}")
    proj.load_state_dict(snapshot["other_proj_state_dict"])
    proj.eval()
    return proj


def _project_other(other_proj: nn.Module, feat: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Chunked forward through other_proj, L2-normalized output."""
    n = feat.shape[0]
    out = None
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            c = other_proj(feat[s:e])
            if out is None:
                out = torch.empty((n,) + c.shape[1:], dtype=c.dtype)
            out[s:e] = c
    return F.normalize(out, dim=-1)


def analyze_pair(
    frozen_dir: str,
    trained_dir: str,
    n_query_sample: int = 3000,
    seed: int = 0,
    k_extremes: int = 20,
    rank_chunk: int = 200,
) -> dict:
    """Correlate per-sample condition drift and embedding shift (trained arm, against
    the frozen arm's final condition as the exact buddy-init value -- the frozen arm
    never moves, by construction) against per-sample i2t retrieval-rank change (trained
    rank - frozen rank), using each sample's own actual condition ranked against the
    FULL train population's projected "other side" embeddings. Writes
    condition_geometry/retrieval_correlation_vs_frozen.json under `trained_dir`."""
    frozen_snap = _load_final_epoch_snapshot(frozen_dir)
    trained_snap = _load_final_epoch_snapshot(trained_dir)

    combine_side = frozen_snap.get("combine_side", "txt")
    assert combine_side == trained_snap.get("combine_side", "txt"), (
        "frozen/trained arms must share combine_side for a paired comparison"
    )

    frozen_ids = list(frozen_snap["sample_ids"])
    trained_ids = list(trained_snap["sample_ids"])
    assert set(frozen_ids) == set(trained_ids), (
        "frozen/trained arms must cover the exact same sample_ids (shared results_dir "
        "template) for this per-sample correlation to be meaningful"
    )

    frozen_conditions = frozen_snap["label_embeddings_all"]
    trained_conditions = trained_snap["label_embeddings_all"]
    if trained_ids != frozen_ids:
        trained_conditions = reorder_features_to_z(trained_conditions, trained_ids, frozen_ids)
    sample_ids = frozen_ids  # canonical order from here on

    frozen_first_snap = _load_epoch_snapshot(frozen_dir, 0)
    frozen_first_ids = list(frozen_first_snap["sample_ids"])
    frozen_first_conditions = frozen_first_snap["label_embeddings_all"]
    if frozen_first_ids != frozen_ids:
        frozen_first_conditions = reorder_features_to_z(frozen_first_conditions, frozen_first_ids, frozen_ids)

    init_condition = frozen_conditions.numpy()
    trained_condition_np = trained_conditions.numpy()
    frozen_self_drift = condition_drift(init_condition, frozen_first_conditions.numpy())
    assert frozen_self_drift.max() < 1e-4, (
        f"frozen arm's condition table moved between its first and final saved epoch by "
        f"up to {frozen_self_drift.max():.6f}; expected ~0 since em_interval freezes it for "
        f"the whole run -- if this fires, the frozen arm's final condition is NOT a safe "
        f"proxy for buddy-init, and this diagnostic's premise doesn't hold for this run"
    )
    drift_trained = condition_drift(trained_condition_np, init_condition)

    frozen_cfg = _load_run_config(frozen_dir)
    trained_cfg = _load_run_config(trained_dir)
    assert frozen_cfg["featuremanager"]["storage_dir"] == trained_cfg["featuremanager"]["storage_dir"], (
        "frozen/trained arms must share the same FeatureManager storage_dir (shared "
        "results_dir template) for this per-sample correlation to use one feature store"
    )
    img_t, txt_t, feat_sample_ids = _load_train_features(trained_dir, trained_cfg)
    combine_feat = reorder_features_to_z(
        img_t if combine_side == "img" else txt_t, feat_sample_ids, sample_ids
    )
    other_feat = reorder_features_to_z(
        txt_t if combine_side == "img" else img_t, feat_sample_ids, sample_ids
    )

    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    query_idx = np.sort(rng.choice(n, size=min(n_query_sample, n), replace=False))

    def _own_condition_rank(snap: dict, conditions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        combiner = _rebuild_combiner(snap)
        other_proj = _rebuild_other_proj(snap)
        gallery = _project_other(other_proj, other_feat).numpy()
        query_combine_feat = combine_feat[query_idx]
        query_conditions = conditions[query_idx]
        comb_emb = _compute_comb_emb(combiner, query_combine_feat, query_conditions)
        shift = compute_shift(comb_emb, query_combine_feat)
        query_norm = F.normalize(comb_emb, dim=-1).numpy()
        ranks = rank_of_true_match(query_norm, gallery, query_idx, chunk=rank_chunk)
        return ranks, shift

    rank_frozen, _ = _own_condition_rank(frozen_snap, frozen_conditions)
    rank_trained, shift_trained = _own_condition_rank(trained_snap, trained_conditions)

    delta_rank = (rank_trained - rank_frozen).astype(np.int64)
    drift_query = drift_trained[query_idx]
    query_sample_ids = [sample_ids[i] for i in query_idx]

    corr_drift = spearman_correlate(delta_rank.astype(float), drift_query)
    corr_shift = spearman_correlate(delta_rank.astype(float), shift_trained)
    extremes = rank_extremes(delta_rank, query_sample_ids, drift_query, shift_trained, k=k_extremes)

    result = {
        "frozen_dir": str(frozen_dir),
        "trained_dir": str(trained_dir),
        "n_query_sample": int(len(query_idx)),
        "n_population": int(n),
        "combine_side": combine_side,
        "delta_rank_mean": float(delta_rank.mean()),
        "delta_rank_median": float(np.median(delta_rank)),
        "corr_delta_rank_vs_condition_drift": corr_drift,
        "corr_delta_rank_vs_embedding_shift": corr_shift,
        "extremes": extremes,
    }

    out_path = Path(trained_dir) / "condition_geometry" / "retrieval_correlation_vs_frozen.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*78}\nDrift/shift vs retrieval-rank correlation\n"
          f"  frozen:  {frozen_dir}\n  trained: {trained_dir}\n{'='*78}")
    print(f"  n_query_sample={len(query_idx)} / n_population={n}  combine_side={combine_side}")
    print(f"  delta_rank (trained-frozen): mean={delta_rank.mean():+.1f}  median={np.median(delta_rank):+.1f}")
    print(f"  corr(delta_rank, condition_drift): rho={corr_drift['rho']:+.3f} p={corr_drift['p']:.4f}")
    print(f"  corr(delta_rank, embedding_shift):  rho={corr_shift['rho']:+.3f} p={corr_shift['p']:.4f}")
    print(f"  Wrote {out_path}")
    return result


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, default=None, metavar=("FROZEN_DIR", "TRAINED_DIR"))
    ap.add_argument("--n-query-sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-extremes", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.pair:
        analyze_pair(
            args.pair[0], args.pair[1],
            n_query_sample=args.n_query_sample, seed=args.seed, k_extremes=args.k_extremes,
        )
        return
    ap.print_help()


if __name__ == "__main__":
    main()
