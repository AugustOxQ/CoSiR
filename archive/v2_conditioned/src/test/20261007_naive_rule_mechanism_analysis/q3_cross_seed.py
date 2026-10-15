"""Q3a: score seed-42 mined episodes with an independently fit factor model.

The second model sees exactly Task 7's training items and content graph. Its
factor columns need not line up with seed 42: conditions and candidate scores
are both rebuilt from the seed-43 codes, while the episode records stay fixed.
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261005_condition_ranking_evaluation"))

from run_ranking_eval import EXPECTED_SAMPLES, episode_arrays, score_pool  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.train_factors import FactorTrainingConfig, train_factors  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def _l1_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    sums = np.abs(weights).sum(axis=1, keepdims=True)
    return np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)


def _weights(img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list) -> dict[str, np.ndarray]:
    supports, contrasts, _, _ = episode_arrays(episodes)
    pair_codes = 0.5 * (img_codes + txt_codes)
    gap = pair_codes[supports].mean(axis=1) - pair_codes[contrasts].mean(axis=1)
    n, factors = gap.shape
    return {
        "naive": _l1_weights(np.maximum(gap, 0)),
        "uniform": np.full((n, factors), 1 / factors, dtype=np.float32),
        "clip_only": np.zeros((n, factors), dtype=np.float32),
    }


def _rank_summary(scores: np.ndarray) -> dict:
    """Count all tied candidates at half-rank, including the positive itself."""
    positive = scores[:, :1]
    greater = (scores[:, 1:] > positive).sum(axis=1)
    tied = (scores[:, 1:] == positive).sum(axis=1)
    ranks = 1 + greater + 0.5 * tied
    return {
        "r1": float(np.mean(ranks <= 1)),
        "r3": float(np.mean(ranks <= 3)),
        "tie_episodes": int(np.count_nonzero(tied)),
        "all_tied_episodes": int(np.count_nonzero(tied == scores.shape[1] - 1)),
    }


def _ranking(features_i: np.ndarray, features_t: np.ndarray,
             codes_i: np.ndarray, codes_t: np.ndarray, episodes: list,
             weights: dict[str, np.ndarray], betas: tuple[float, ...]) -> dict:
    _, _, candidates, _ = episode_arrays(episodes)
    anchors = np.fromiter((ep.anchor_idx for ep in episodes), dtype=np.int64)
    result = {}
    for name, w in weights.items():
        result[name] = {}
        for direction, af, cf, ac, cc in (
            ("i2t", features_i, features_t, codes_i, codes_t),
            ("t2i", features_t, features_i, codes_t, codes_i),
        ):
            # The cosine and factor terms can be reused across the beta grid.
            _, cosine, factor = score_pool(
                af[anchors], cf[candidates], ac[anchors], cc[candidates], w, 0.0
            )
            result[name][direction] = {
                str(beta): _rank_summary(float(beta) * cosine + factor) for beta in betas
            }
    return result


def _select_betas(ranking: dict, betas: tuple[float, ...]) -> dict[str, float]:
    """Use train mean bidirectional R@1; first grid entry wins an exact tie."""
    return {
        name: float(max(betas, key=lambda beta: (
            ranking[name]["i2t"][str(beta)]["r1"]
            + ranking[name]["t2i"][str(beta)]["r1"]
        )))
        for name in ranking
    }


def _alignment(codes42_i: np.ndarray, codes42_t: np.ndarray,
               codes43_i: np.ndarray, codes43_t: np.ndarray,
               held_items: np.ndarray) -> dict:
    left = (0.5 * (codes42_i[held_items] + codes42_t[held_items])).astype(np.float64)
    right = (0.5 * (codes43_i[held_items] + codes43_t[held_items])).astype(np.float64)
    left -= left.mean(axis=0)
    right -= right.mean(axis=0)
    denom = np.sqrt(np.sum(left * left, axis=0)[:, None]
                    * np.sum(right * right, axis=0)[None, :])
    corr = np.divide(left.T @ right, denom, out=np.zeros_like(denom), where=denom > 0)
    matches = np.abs(corr).argmax(axis=1)
    maxima = np.abs(corr[np.arange(corr.shape[0]), matches])
    row, col = linear_sum_assignment(-np.abs(corr))
    one_to_one = np.abs(corr[row, col])
    return {
        "per_seed42_factor_max_abs_pearson": maxima.tolist(),
        "matched_seed43_factor": matches.tolist(),
        "mean_max_abs_pearson": float(maxima.mean()),
        "median_max_abs_pearson": float(np.median(maxima)),
        "min_max_abs_pearson": float(maxima.min()),
        "unique_max_matches": len(set(matches.tolist())),
        "one_to_one_seed43_factor": col.tolist(),
        "one_to_one_abs_pearson": one_to_one.tolist(),
        "one_to_one_mean_abs_pearson": float(one_to_one.mean()),
        "one_to_one_median_abs_pearson": float(np.median(one_to_one)),
        "one_to_one_min_abs_pearson": float(one_to_one.min()),
    }


def _paired_bootstrap(features_i: np.ndarray, features_t: np.ndarray,
                      codes_i: np.ndarray, codes_t: np.ndarray,
                      episodes: list, weights: dict[str, np.ndarray],
                      selected: dict[str, float]) -> dict:
    """Paired held-episode naive-minus-uniform CIs at train-selected betas."""
    _, _, candidates, _ = episode_arrays(episodes)
    anchors = np.fromiter((ep.anchor_idx for ep in episodes), dtype=np.int64)
    draw = np.random.default_rng(42).integers(0, len(episodes), size=(5000, len(episodes)))
    result = {}
    for direction, af, cf, ac, cc in (
        ("i2t", features_i, features_t, codes_i, codes_t),
        ("t2i", features_t, features_i, codes_t, codes_i),
    ):
        success = {}
        for name in ("naive", "uniform"):
            scores, _, _ = score_pool(af[anchors], cf[candidates], ac[anchors],
                                      cc[candidates], weights[name], selected[name])
            positive = scores[:, :1]
            ranks = 1 + (scores[:, 1:] > positive).sum(axis=1) + \
                .5 * (scores[:, 1:] == positive).sum(axis=1)
            success[name] = np.column_stack((ranks <= 1, ranks <= 3)).astype(float)
        diff = success["naive"] - success["uniform"]
        boot = diff[draw].mean(axis=1)
        result[direction] = {
            "point": diff.mean(axis=0).tolist(),
            "ci95": np.quantile(boot, [.025, .975], axis=0).T.tolist(),
            "resamples": 5000,
        }
    return result


def _swaps(features_i: np.ndarray, features_t: np.ndarray,
           codes_i: np.ndarray, codes_t: np.ndarray, episodes: list,
           pairs: list[tuple[int, int]], weights: dict[str, np.ndarray],
           selected: dict[str, float]) -> dict:
    result = {}
    for name, w in weights.items():
        result[name] = {}
        beta = selected[name]
        for direction, af, cf, ac, cc in (
            ("i2t", features_i, features_t, codes_i, codes_t),
            ("t2i", features_t, features_i, codes_t, codes_i),
        ):
            reversals = 0
            for a, b in pairs:
                ep_a, ep_b = episodes[a], episodes[b]
                pool = list(dict.fromkeys([
                    ep_a.positive_idx, ep_b.positive_idx,
                    *ep_a.hard_negative_idxs, *ep_a.condition_distractor_idxs,
                    *ep_a.anchor_distractor_idxs, *ep_b.hard_negative_idxs,
                    *ep_b.condition_distractor_idxs, *ep_b.anchor_distractor_idxs,
                ]))
                scores, _, _ = score_pool(
                    np.broadcast_to(af[ep_a.anchor_idx], (2, af.shape[1])),
                    np.broadcast_to(cf[pool], (2, len(pool), cf.shape[1])),
                    np.broadcast_to(ac[ep_a.anchor_idx], (2, ac.shape[1])),
                    np.broadcast_to(cc[pool], (2, len(pool), cc.shape[1])),
                    w[[a, b]], beta,
                )
                reversals += int(scores[0, 0] - scores[0, 1] > 1e-8
                                 and scores[1, 1] - scores[1, 0] > 1e-8)
            result[name][direction] = {
                "appropriate_reversals": reversals,
                "reversal_rate": reversals / len(pairs) if pairs else None,
            }
    return result


def analyze_cross_seed(
    img_features: np.ndarray, txt_features: np.ndarray,
    img_codes42: np.ndarray, txt_codes42: np.ndarray,
    train_items: np.ndarray, held_items: np.ndarray,
    train_episodes: list, held_episodes: list,
    swap_pairs: list[tuple[int, int]], beta_grid: tuple[float, ...],
) -> dict:
    """Fit seed 43 and evaluate fixed seed-42 episodes in both directions.

    `train_episodes`, `held_episodes`, and `swap_pairs` contain global row IDs
    and must be the exact Task 7 seed-42 objects. No remine occurs here.
    """
    if not beta_grid or any(beta < 0 for beta in beta_grid):
        raise ValueError("beta_grid must be nonempty and nonnegative")
    betas = tuple(float(beta) for beta in beta_grid)
    train_items = np.asarray(train_items, dtype=np.int64)
    held_items = np.asarray(held_items, dtype=np.int64)
    if img_features.shape != txt_features.shape or img_codes42.shape != txt_codes42.shape:
        raise ValueError("Image/text feature and code shapes must match")
    if img_features.shape[0] != img_codes42.shape[0]:
        raise ValueError("Features and seed-42 codes must index the same rows")
    all_items = np.concatenate((train_items, held_items))
    if (len(all_items) != len(img_features)
            or not np.array_equal(np.sort(all_items), np.arange(len(img_features)))):
        raise ValueError("Train and held items must partition every feature row exactly once")

    config = FactorTrainingConfig(lambda_usage_balance=0.1, seed=43, epochs=2000)
    use_cache = len(img_features) == EXPECTED_SAMPLES
    if use_cache:
        CACHE.mkdir(exist_ok=True)
    path_i, path_t, path_meta = (CACHE / "factor43_img.npy", CACHE / "factor43_txt.npy",
                                 CACHE / "factor43_meta.json")
    if use_cache and path_i.exists() and path_t.exists() and path_meta.exists():
        codes_i, codes_t = np.load(path_i), np.load(path_t)
        graph_edges = json.loads(path_meta.read_text())["graph_edges"]
        if codes_i.shape != img_codes42.shape or codes_t.shape != txt_codes42.shape:
            raise ValueError("Cached seed-43 codes have wrong shape")
    else:
        # Task 7's graph uses raw CLIP features and the same GraphConfig defaults.
        graph = build_content_graph(img_features[train_items], txt_features[train_items], GraphConfig())
        graph_edges = int(graph.nnz // 2)
        with redirect_stdout(io.StringIO()):
            model, train_i, train_t = train_factors(
                img_features[train_items], txt_features[train_items], graph, config
            )
        codes_i = np.empty_like(img_codes42)
        codes_t = np.empty_like(txt_codes42)
        codes_i[train_items], codes_t[train_items] = train_i, train_t
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for start in range(0, len(held_items), 8192):
                ids = held_items[start:start + 8192]
                img = torch.as_tensor(img_features[ids], dtype=torch.float32, device=device)
                txt = torch.as_tensor(txt_features[ids], dtype=torch.float32, device=device)
                codes_i[ids] = model.encode_image(img).cpu().numpy()
                codes_t[ids] = model.encode_text(txt).cpu().numpy()
        if use_cache:
            np.save(path_i, codes_i)
            np.save(path_t, codes_t)
            path_meta.write_text(json.dumps({"graph_edges": graph_edges,
                                             "train_items": len(train_items), "seed": 43}))
    if use_cache and graph_edges != 2463679:
        raise AssertionError(f"Seed-43 graph differs from Task 7: {graph_edges}")
    if not np.isfinite(codes_i).all() or not np.isfinite(codes_t).all():
        raise ValueError("Seed-43 encoding contains non-finite values")

    train_weights = _weights(codes_i, codes_t, train_episodes)
    held_weights = _weights(codes_i, codes_t, held_episodes)
    while True:
        train_ranking = _ranking(img_features, txt_features, codes_i, codes_t,
                                 train_episodes, train_weights, betas)
        selected = _select_betas(train_ranking, betas)
        if not any(selected[name] == betas[-1] for name in ("naive", "uniform")):
            break
        if betas[-1] >= 3000:
            break
        betas = (*betas, betas[-1] * 3)
    held_ranking = _ranking(img_features, txt_features, codes_i, codes_t,
                            held_episodes, held_weights, betas)
    selected_ranking = {
        name: {direction: held_ranking[name][direction][str(selected[name])]
               for direction in ("i2t", "t2i")}
        for name in held_weights
    }
    return {
        "factor_seed": 43,
        "factor_epochs": config.epochs,
        "graph_edges": graph_edges,
        "train_items": int(len(train_items)),
        "held_items": int(len(held_items)),
        "train_episodes": len(train_episodes),
        "held_episodes": len(held_episodes),
        "beta_selection": "maximize mean bidirectional train R@1; first grid entry breaks ties",
        "selected_beta": selected,
        "held_selected_ranking": selected_ranking,
        "held_beta_zero_ranking": {
            name: {direction: held_ranking[name][direction][str(0.0)]
                   for direction in ("i2t", "t2i")}
            for name in held_weights
        } if 0.0 in betas else None,
        "held_beta_grid_ranking": held_ranking,
        "swap_pairs": len(swap_pairs),
        "selected_swap": _swaps(img_features, txt_features, codes_i, codes_t,
                                held_episodes, swap_pairs, held_weights, selected),
        "alignment": _alignment(img_codes42, txt_codes42, codes_i, codes_t, held_items),
        "naive_minus_uniform_selected_bootstrap": _paired_bootstrap(
            img_features, txt_features, codes_i, codes_t,
            held_episodes, held_weights, selected),
    }
