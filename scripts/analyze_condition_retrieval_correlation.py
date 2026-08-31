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

IMPORTANT scoping, two ways this differs from 11.1's headline metric:
  (a) population -- everything here is measured *in-sample*, on the 150k rows the run
      trained on, not on the held-out test set 11.1's `test_oracle/i2t_R1` uses;
  (b) metric -- every rank here uses each sample's OWN actual assigned condition, not an
      oracle max over all conditions the way `test_oracle` does.
Neither mismatch is fixable by re-running: per-sample conditions only exist for training
samples. So a result here constrains, but cannot resolve, 11.1's held-out oracle finding.

The naive cross-arm `delta_rank` (trained arm's rank minus frozen arm's rank) compares two
*independently trained models* -- separate combiners, separate other_proj, separate
condition tables -- so most of its variance is whole-model divergence, not the condition
table. `delta_rank_swap` is the condition-only counterfactual that isolates the condition's
own effect: hold the TRAINED arm's combiner/other_proj/gallery fixed and swap only the
condition table (trained -> frozen/buddy-init).

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
    Chunked over the query dimension to bound peak memory to chunk x len(gallery_emb).

    Tie handling is deliberately *optimistic*: a gallery row scoring exactly equal to the
    true match does not push the true match down (`sims > true_score`, strict). With
    float32 dot products over a 150k gallery exact ties are rare, but the same optimistic
    convention on both arms is what makes their difference well-defined -- do not change
    this without re-deriving every number downstream of it."""
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


def extreme_group_percentiles(
    delta_rank: np.ndarray,
    drift: np.ndarray,
    shift: np.ndarray,
    k: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Where the k most-degraded / most-improved samples sit *within the query sample's
    own* condition_drift and embedding_shift distributions, as median percentile (0-100).

    This is the quantitative version of "are the extremes extreme on either axis?" --
    the per-row absolute drift/shift values in `rank_extremes` cannot answer that on
    their own, because their scale varies run to run. All three arrays row-aligned."""
    from scipy.stats import rankdata

    n = len(delta_rank)
    d_pct = 100.0 * (rankdata(drift) - 0.5) / n
    s_pct = 100.0 * (rankdata(shift) - 0.5) / n
    order = np.argsort(delta_rank)
    improved, degraded = order[:k], order[::-1][:k]
    return {
        "most_degraded": {
            "median_condition_drift_pct": float(np.median(d_pct[degraded])),
            "median_embedding_shift_pct": float(np.median(s_pct[degraded])),
        },
        "most_improved": {
            "median_condition_drift_pct": float(np.median(d_pct[improved])),
            "median_embedding_shift_pct": float(np.median(s_pct[improved])),
        },
    }


def _find_shared_init(*run_dirs: str):
    """Locate the real shared buddy-init condition table the arms were seeded from.

    11.1's runs all point at one `<results_root>/<dataset>/template_embeddings/` holding
    `embeddings.npy` + `sample_ids.npy`. That directory sits next to the experiment dirs
    (one level up from a run dir); the extra `parent.parent` probe is a cheap tolerance
    for a differently-nested layout. Returns (embeddings [N, D], sample_ids list) or None
    if no such file is reachable -- the caller falls back to the weaker first-vs-final
    stationarity check in that case rather than failing, since this script may be run in
    environments where only the experiment dirs were copied."""
    for d in run_dirs:
        base = Path(d)
        for cand in (base.parent, base.parent.parent):
            emb_p = cand / "template_embeddings" / "embeddings.npy"
            ids_p = cand / "template_embeddings" / "sample_ids.npy"
            if emb_p.exists() and ids_p.exists():
                return np.load(emb_p), [int(i) for i in np.load(ids_p)]
    return None


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


def build_per_sample_dump(
    sample_ids: List[int],
    delta_rank: np.ndarray,
    delta_rank_swap: np.ndarray,
    condition_drift: np.ndarray,
    embedding_shift: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pack the per-sample arrays analyze_pair already computes internally into one dict
    keyed by sample_id, for persistence (analyze_pair only ever aggregated these into
    summary statistics before; Experiment 12 needs the raw per-sample values)."""
    n = len(sample_ids)
    for name, arr in (
        ("delta_rank", delta_rank), ("delta_rank_swap", delta_rank_swap),
        ("condition_drift", condition_drift), ("embedding_shift", embedding_shift),
    ):
        assert len(arr) == n, f"length mismatch: sample_ids has {n} entries, {name} has {len(arr)}"
    return {
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "delta_rank": np.asarray(delta_rank),
        "delta_rank_swap": np.asarray(delta_rank_swap),
        "condition_drift": np.asarray(condition_drift),
        "embedding_shift": np.asarray(embedding_shift),
    }


def analyze_pair(
    frozen_dir: str,
    trained_dir: str,
    n_query_sample: int = 3000,
    seed: int = 0,
    k_extremes: int = 20,
    rank_chunk: int = 200,
    dump_per_sample: bool = False,
) -> dict:
    """Correlate per-sample condition drift and embedding shift (trained arm, against
    the frozen arm's final condition as the exact buddy-init value -- the frozen arm
    never moves, by construction) against per-sample i2t retrieval-rank change (trained
    rank - frozen rank), using each sample's own actual condition ranked against the
    FULL train population's projected "other side" embeddings.

    Reports three families of correlation: the signed cross-arm `delta_rank` ones (kept
    for continuity -- they are dominated by whole-model divergence between the two
    independently-trained arms), the unsigned `|delta_rank|` ones (how far a sample's rank
    moves, regardless of direction), and the condition-only counterfactual
    `delta_rank_swap` (the trained arm's combiner/other_proj/gallery held fixed, only the
    condition table swapped trained -> frozen), which is the estimator that actually
    isolates the condition's own effect. Also reports each arm's in-sample own-condition
    rank quality -- see the module docstring for why that is NOT 11.1's held-out oracle
    metric. Writes condition_geometry/retrieval_correlation_vs_frozen.json under
    `trained_dir`."""
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

    # The check above only establishes that the frozen arm's table is *stationary*. The
    # load-bearing half of the premise is that this stationary table IS the buddy-init the
    # trained arm started from -- assert that directly against the real shared init file
    # when it is reachable on disk (it is, for 11.1's runs). If it isn't, fall back to the
    # stationarity check alone rather than failing: this script is also meant to run in
    # environments where only the experiment dirs were copied.
    init_source = "frozen_arm_final_epoch (stationarity check only)"
    shared_init = _find_shared_init(frozen_dir, trained_dir)
    if shared_init is not None:
        real_init, real_init_ids = shared_init
        if set(real_init_ids) == set(sample_ids):
            real_init_aligned = reorder_features_to_z(
                torch.from_numpy(real_init).float(), real_init_ids, sample_ids
            ).numpy()
            init_proxy_gap = float(condition_drift(init_condition, real_init_aligned).max())
            assert init_proxy_gap < 1e-4, (
                f"the frozen arm's condition table differs from the shared buddy-init file "
                f"by up to {init_proxy_gap:.6f}; expected ~0. The frozen arm's table is NOT "
                f"the init the trained arm started from, so `condition_drift` below would "
                f"not be drift-from-init and this diagnostic's premise doesn't hold"
            )
            init_source = "shared template_embeddings/embeddings.npy (verified identical)"
        else:
            init_source = "frozen_arm_final_epoch (shared init found but sample_ids differ)"

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

    def _arm_pipeline(snap: dict) -> Tuple[object, np.ndarray]:
        """Rebuild one arm's combiner + other_proj and project the FULL population's
        other-side features into that arm's gallery. Separated from `_rank_with` so the
        trained arm's model/gallery can be reused across two different condition tables
        (the condition-only counterfactual) without re-projecting 150k rows twice."""
        combiner = _rebuild_combiner(snap)
        other_proj = _rebuild_other_proj(snap)
        gallery = _project_other(other_proj, other_feat).numpy()
        return combiner, gallery

    def _rank_with(combiner, gallery: np.ndarray, conditions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Rank each sampled query's true match inside `gallery`, conditioning on
        `conditions` (canonical order; subsampled here). Returns (ranks, shift)."""
        query_combine_feat = combine_feat[query_idx]
        comb_emb = _compute_comb_emb(combiner, query_combine_feat, conditions[query_idx])
        shift = compute_shift(comb_emb, query_combine_feat)
        query_norm = F.normalize(comb_emb, dim=-1).numpy()
        ranks = rank_of_true_match(query_norm, gallery, query_idx, chunk=rank_chunk)
        return ranks, shift

    frozen_combiner, frozen_gallery = _arm_pipeline(frozen_snap)
    rank_frozen, shift_frozen = _rank_with(frozen_combiner, frozen_gallery, frozen_conditions)
    del frozen_combiner, frozen_gallery

    trained_combiner, trained_gallery = _arm_pipeline(trained_snap)
    rank_trained, shift_trained = _rank_with(trained_combiner, trained_gallery, trained_conditions)

    # Condition-only counterfactual: hold the TRAINED arm's combiner/other_proj/gallery
    # completely fixed and swap ONLY the condition table (trained -> frozen == buddy-init).
    # `delta_rank` below compares two *independently trained models* (each with its own
    # combiner + other_proj + conditions), so most of its variance is whole-model
    # divergence rather than the condition table's own doing. This swap isolates the
    # condition's own contribution to rank inside a single fixed model.
    rank_trained_frozen_cond, _ = _rank_with(trained_combiner, trained_gallery, frozen_conditions)
    del trained_combiner, trained_gallery

    delta_rank = (rank_trained - rank_frozen).astype(np.int64)
    delta_rank_swap = (rank_trained_frozen_cond - rank_trained).astype(np.int64)
    drift_query = drift_trained[query_idx]
    query_sample_ids = [sample_ids[i] for i in query_idx]

    corr_drift = spearman_correlate(delta_rank.astype(float), drift_query)
    corr_shift = spearman_correlate(delta_rank.astype(float), shift_trained)
    corr_delta_shift = spearman_correlate(delta_rank.astype(float), shift_trained - shift_frozen)
    corr_abs_drift = spearman_correlate(np.abs(delta_rank).astype(float), drift_query)
    corr_abs_shift = spearman_correlate(np.abs(delta_rank).astype(float), shift_trained)
    corr_swap_drift = spearman_correlate(delta_rank_swap.astype(float), drift_query)
    extremes = rank_extremes(delta_rank, query_sample_ids, drift_query, shift_trained, k=k_extremes)
    extremes_pct = extreme_group_percentiles(delta_rank, drift_query, shift_trained, k=k_extremes)

    result = {
        "frozen_dir": str(frozen_dir),
        "trained_dir": str(trained_dir),
        "n_query_sample": int(len(query_idx)),
        "n_population": int(n),
        "combine_side": combine_side,
        "init_source": init_source,
        "delta_rank_mean": float(delta_rank.mean()),
        "delta_rank_median": float(np.median(delta_rank)),
        "delta_rank_std": float(delta_rank.std()),
        "frac_delta_rank_zero": float((delta_rank == 0).mean()),
        "frac_queries_improved": float((delta_rank < 0).mean()),
        "delta_rank_swap_mean": float(delta_rank_swap.mean()),
        "delta_rank_swap_median": float(np.median(delta_rank_swap)),
        # In-sample, own-condition retrieval quality of each arm on THIS query subsample.
        # NOT comparable to 11.1's held-out `test_oracle/i2t_R1`: different population
        # (training rows vs. held-out test set) and different metric (each sample's own
        # actual condition vs. an oracle max over all conditions).
        "rank_frozen_mean": float(rank_frozen.mean()),
        "rank_trained_mean": float(rank_trained.mean()),
        "rank_frozen_median": float(np.median(rank_frozen)),
        "rank_trained_median": float(np.median(rank_trained)),
        "r1_frozen": float((rank_frozen == 1).mean()),
        "r1_trained": float((rank_trained == 1).mean()),
        "corr_delta_rank_vs_condition_drift": corr_drift,
        "corr_delta_rank_vs_embedding_shift": corr_shift,
        "corr_delta_rank_vs_delta_embedding_shift": corr_delta_shift,
        "corr_abs_delta_rank_vs_condition_drift": corr_abs_drift,
        "corr_abs_delta_rank_vs_embedding_shift": corr_abs_shift,
        "corr_delta_rank_swap_vs_condition_drift": corr_swap_drift,
        "extremes": extremes,
        "extremes_percentiles": extremes_pct,
    }

    out_path = Path(trained_dir) / "condition_geometry" / "retrieval_correlation_vs_frozen.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if dump_per_sample:
        dump = build_per_sample_dump(
            sample_ids=query_sample_ids,
            delta_rank=delta_rank,
            delta_rank_swap=delta_rank_swap,
            condition_drift=drift_query,
            embedding_shift=shift_trained,
        )
        dump_path = Path(trained_dir) / "condition_geometry" / "per_sample_retrieval_correlation.npz"
        np.savez(dump_path, **dump)
        print(f"  Wrote per-sample dump: {dump_path}")

    print(f"\n{'='*78}\nDrift/shift vs retrieval-rank correlation\n"
          f"  frozen:  {frozen_dir}\n  trained: {trained_dir}\n{'='*78}")
    print(f"  n_query_sample={len(query_idx)} / n_population={n}  combine_side={combine_side}")
    print(f"  init_source: {init_source}")
    print(f"  in-sample own-condition rank: frozen mean={rank_frozen.mean():.1f} R1={((rank_frozen == 1).mean()):.4f}  "
          f"trained mean={rank_trained.mean():.1f} R1={((rank_trained == 1).mean()):.4f}")
    print(f"  delta_rank (trained-frozen): mean={delta_rank.mean():+.1f}  median={np.median(delta_rank):+.1f}  "
          f"std={delta_rank.std():.1f}  frac_improved={(delta_rank < 0).mean():.3f}  frac_zero={(delta_rank == 0).mean():.3f}")
    print(f"  delta_rank_swap (trained model, frozen cond - trained cond): "
          f"mean={delta_rank_swap.mean():+.1f}  median={np.median(delta_rank_swap):+.1f}")
    print(f"  corr(delta_rank, condition_drift):        rho={corr_drift['rho']:+.3f} p={corr_drift['p']:.3e}")
    print(f"  corr(delta_rank, embedding_shift):        rho={corr_shift['rho']:+.3f} p={corr_shift['p']:.3e}")
    print(f"  corr(delta_rank, delta_embedding_shift):  rho={corr_delta_shift['rho']:+.3f} p={corr_delta_shift['p']:.3e}")
    print(f"  corr(|delta_rank|, condition_drift):      rho={corr_abs_drift['rho']:+.3f} p={corr_abs_drift['p']:.3e}")
    print(f"  corr(|delta_rank|, embedding_shift):      rho={corr_abs_shift['rho']:+.3f} p={corr_abs_shift['p']:.3e}")
    print(f"  corr(delta_rank_swap, condition_drift):   rho={corr_swap_drift['rho']:+.3f} p={corr_swap_drift['p']:.3e}")
    print(f"  extremes percentile (median, within query sample): "
          f"degraded drift={extremes_pct['most_degraded']['median_condition_drift_pct']:.1f} "
          f"shift={extremes_pct['most_degraded']['median_embedding_shift_pct']:.1f} | "
          f"improved drift={extremes_pct['most_improved']['median_condition_drift_pct']:.1f} "
          f"shift={extremes_pct['most_improved']['median_embedding_shift_pct']:.1f}")
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

    # extreme_group_percentiles: with drift ordered the same way as delta_rank, the
    # single most-degraded sample sits at the top percentile and the most-improved at the
    # bottom; shift here is ordered *opposite* to delta_rank, so the two flip.
    dr = np.array([5.0, -3.0, 0.0, 20.0, -10.0])
    drift_p = np.array([0.2, 0.1, 0.15, 0.5, 0.05])   # co-ordered with dr
    shift_p = np.array([0.4, 0.5, 0.45, 0.1, 0.6])    # anti-ordered with dr
    pct = extreme_group_percentiles(dr, drift_p, shift_p, k=1)
    assert abs(pct["most_degraded"]["median_condition_drift_pct"] - 90.0) < 1e-6, pct
    assert abs(pct["most_degraded"]["median_embedding_shift_pct"] - 10.0) < 1e-6, pct
    assert abs(pct["most_improved"]["median_condition_drift_pct"] - 10.0) < 1e-6, pct
    assert abs(pct["most_improved"]["median_embedding_shift_pct"] - 90.0) < 1e-6, pct

    # _find_shared_init: returns None when no template_embeddings/ is reachable.
    assert _find_shared_init("/nonexistent/results/root/some_run_dir") is None

    # build_per_sample_dump: packs aligned arrays, keyed by sample_id.
    dump = build_per_sample_dump(
        sample_ids=[10, 11, 12],
        delta_rank=np.array([1, -2, 0]),
        delta_rank_swap=np.array([2, -1, 0]),
        condition_drift=np.array([0.1, 0.2, 0.3]),
        embedding_shift=np.array([0.01, 0.02, 0.03]),
    )
    assert list(dump["sample_ids"]) == [10, 11, 12], dump
    assert list(dump["delta_rank"]) == [1, -2, 0], dump
    assert list(dump["embedding_shift"]) == [0.01, 0.02, 0.03], dump

    try:
        build_per_sample_dump(
            sample_ids=[10, 11],
            delta_rank=np.array([1, -2, 0]),
            delta_rank_swap=np.array([2, -1, 0]),
            condition_drift=np.array([0.1, 0.2, 0.3]),
            embedding_shift=np.array([0.01, 0.02, 0.03]),
        )
        raise AssertionError("expected a length-mismatch error")
    except AssertionError as e:
        assert "length" in str(e), e

    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, default=None, metavar=("FROZEN_DIR", "TRAINED_DIR"))
    ap.add_argument("--n-query-sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-extremes", type=int, default=20)
    ap.add_argument("--rank-chunk", type=int, default=200,
                    help="query-dimension chunk size for rank_of_true_match (memory knob only; "
                         "results are chunk-invariant)")
    ap.add_argument("--dump-per-sample", action="store_true",
                    help="also write condition_geometry/per_sample_retrieval_correlation.npz "
                         "with the raw per-sample delta_rank/condition_drift/embedding_shift "
                         "arrays (Experiment 12)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.pair:
        analyze_pair(
            args.pair[0], args.pair[1],
            n_query_sample=args.n_query_sample, seed=args.seed, k_extremes=args.k_extremes,
            rank_chunk=max(1, args.rank_chunk), dump_per_sample=args.dump_per_sample,
        )
        return
    ap.print_help()


if __name__ == "__main__":
    main()
