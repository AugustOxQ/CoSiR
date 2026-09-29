"""Task 9 scale-fair mechanism analysis on ArtELingo.

Run from the repository root with the CoSiR environment. Cached data lives
under this script's ignored cache directory. All held episodes are verified
against Task 8's ordered-record SHA-256 before any analysis is reported.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "20261005_condition_ranking_evaluation"))

from run_ranking_eval import (  # noqa: E402
    EVAL_EPISODES, EXPECTED_SAMPLES, SEED, TRAIN_EPISODES, EpisodeMiningConfig,
    choose_swap_pairs, episode_arrays, load_real_features, make_weights,
    mine_episodes, prepare_item_disjoint_codes, score_pool, split_items,
    train_recovery_head, validate_roles,
)
from src.train.episodes import Episode  # noqa: E402

EXPECTED_SHA = "cb50ab7f026678d317dddde274d779674ec5e039921612c999cd19751b74410f"
BASE_GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CACHE = HERE / "cache"
RESULTS_PATH = CACHE / "results.json"


def l1_normalize(weights: np.ndarray) -> np.ndarray:
    """Normalize each nonnegative row; leave literal zeros unchanged."""
    values = np.asarray(weights, dtype=np.float64)
    norms = np.abs(values).sum(axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def midrank(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive rank = 1 + strict superiors + half of tied distractors."""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must have positive in column zero and distractors")
    greater = (values[:, 1:] > values[:, :1]).sum(axis=1)
    tied = (values[:, 1:] == values[:, :1]).sum(axis=1)
    return 1.0 + greater + 0.5 * tied, tied


def rank_summary(scores: np.ndarray) -> dict:
    ranks, tied = midrank(scores)
    return {
        "r1": float(np.mean(ranks <= 1)), "r3": float(np.mean(ranks <= 3)),
        "tie_episodes": int(np.count_nonzero(tied)),
        "tied_distractors": int(tied.sum()),
        "all_tied_episodes": int(np.count_nonzero(tied == scores.shape[1] - 1)),
    }


def role_outrank(scores: np.ndarray) -> dict[str, float]:
    """Per role, fraction with at least one candidate strictly above positive."""
    if scores.ndim != 2 or scores.shape[1] != 13:
        raise ValueError("Expected one positive and three four-item role blocks")
    return {
        "hard_negative": float(np.mean(np.any(scores[:, 1:5] > scores[:, :1], axis=1))),
        "condition_only": float(np.mean(np.any(scores[:, 5:9] > scores[:, :1], axis=1))),
        "anchor_only": float(np.mean(np.any(scores[:, 9:13] > scores[:, :1], axis=1))),
    }


def bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int = SEED,
                         resamples: int = 5000) -> dict:
    """Paired episode bootstrap for R@1 and R@3 in one direction."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Paired success arrays must have equal (N, 2) shape")
    difference = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(a), size=(resamples, len(a)))
    draws = difference[indices].mean(axis=1)
    return {
        "point": difference.mean(axis=0).tolist(),
        "ci95": np.quantile(draws, [0.025, 0.975], axis=0).T.tolist(),
        "resamples": resamples,
    }


def successes(scores: np.ndarray) -> np.ndarray:
    ranks, _ = midrank(scores)
    return np.column_stack((ranks <= 1, ranks <= 3))


def components(img_features: np.ndarray, txt_features: np.ndarray,
               img_codes: np.ndarray, txt_codes: np.ndarray,
               episodes: list[Episode], weights: np.ndarray) -> dict:
    """Reuse Task 7's score_pool, retaining cosine and factor for beta sweeps."""
    _, _, candidates, _ = episode_arrays(episodes)
    anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
    out = {}
    for direction, af, cf, ac, cc in (
        ("i2t", img_features, txt_features, img_codes, txt_codes),
        ("t2i", txt_features, img_features, txt_codes, img_codes),
    ):
        _, cosine, factor = score_pool(
            af[anchors], cf[candidates], ac[anchors], cc[candidates], weights, 0.0
        )
        out[direction] = (cosine, factor)
    return out


def score_at(comp: dict, direction: str, beta: float) -> np.ndarray:
    cosine, factor = comp[direction]
    return beta * cosine + factor


def select_beta(comp: dict, *, clip_only: bool = False) -> tuple[float, dict]:
    """Choose beta by mean train i2t/t2i R@1, then lower beta on ties."""
    grid = list(BASE_GRID)
    curve = {}
    while True:
        for beta in grid:
            key = str(beta)
            if key not in curve:
                curve[key] = {d: rank_summary(score_at(comp, d, beta))
                              for d in ("i2t", "t2i")}
        utilities = np.asarray([
            0.5 * (curve[str(b)]["i2t"]["r1"] + curve[str(b)]["t2i"]["r1"])
            for b in grid
        ])
        best = int(np.flatnonzero(utilities >= utilities.max() - 1e-12)[0])
        if best != len(grid) - 1 or clip_only or grid[-1] >= 3000:
            return float(grid[best]), curve
        grid.extend((grid[-1] * 3, grid[-1] * 10))


def weights_for(model, img_codes: np.ndarray, txt_codes: np.ndarray,
                episodes: list[Episode]) -> dict[str, np.ndarray]:
    variants, _, _ = make_weights(model, img_codes, txt_codes, episodes)
    variants["shuffled"] = variants.pop("shuffled_trained")
    variants["clip_only"] = np.zeros_like(variants["uniform"])
    return {name: l1_normalize(w) for name, w in variants.items()}


def ablation_weights(img_codes: np.ndarray, txt_codes: np.ndarray,
                     episodes: list[Episode]) -> dict[str, np.ndarray]:
    supports, contrasts, _, targets = episode_arrays(episodes)
    anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
    pair = 0.5 * (img_codes + txt_codes)
    support_mean = pair[supports].mean(axis=1)
    naive = np.maximum(support_mean - pair[contrasts].mean(axis=1), 0)
    n, width = naive.shape
    minus_target = naive.copy()
    minus_target[np.arange(n), targets] = 0
    result = {
        "naive": naive, "target_only": np.eye(width)[targets],
        "naive_minus_target": minus_target,
        "support_mean_only": np.maximum(support_mean, 0),
        "anchor_profile": np.maximum(pair[anchors], 0),
    }
    order = np.argsort(-naive, axis=1, kind="stable")
    for k in (1, 3, 5):
        top = np.zeros_like(naive)
        rows = np.arange(n)[:, None]
        top[rows, order[:, :k]] = naive[rows, order[:, :k]]
        result[f"top_{k}"] = top
    return {name: l1_normalize(w) for name, w in result.items()}


def target_cluster_ablations(naive: np.ndarray, targets: np.ndarray,
                             correlation: np.ndarray, threshold: float = 0.95) -> dict:
    """Remove/retain target dimensions and their direct correlated copies."""
    mask = np.abs(correlation[targets]) >= threshold
    mask[np.arange(len(targets)), targets] = True
    return {
        "minus_cluster": np.where(mask, 0, naive),
        "cluster_only": np.where(mask, naive, 0),
        "cluster_sizes": mask.sum(axis=1),
    }


def swap_reversal(img_features: np.ndarray, txt_features: np.ndarray,
                  img_codes: np.ndarray, txt_codes: np.ndarray,
                  episodes: list[Episode], weights: np.ndarray,
                  pairs: list[tuple[int, int]], beta: float) -> dict:
    """Task 7's fixed-anchor two-condition positive reversal criterion."""
    out = {}
    for direction, af, cf, ac, cc in (
        ("i2t", img_features, txt_features, img_codes, txt_codes),
        ("t2i", txt_features, img_features, txt_codes, img_codes),
    ):
        count = 0
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
                weights[[a, b]], beta,
            )
            count += int(scores[0, 0] - scores[0, 1] > 1e-8
                         and scores[1, 1] - scores[1, 0] > 1e-8)
        out[direction] = {"count": count, "total": len(pairs),
                          "rate": count / len(pairs)}
    return out


def mine_and_verify(img_codes: np.ndarray, txt_codes: np.ndarray,
                    train_items: np.ndarray, held_items: np.ndarray) -> tuple[list, list, str]:
    """Use Task 8's unchanged mining order, remap, and episode hash."""
    cached = CACHE / "episodes.json"
    if cached.exists():
        payload = json.loads(cached.read_text())
        train = [Episode(**record) for record in payload["train"]]
        held = [Episode(**record) for record in payload["held"]]
    else:
        config = EpisodeMiningConfig(seed=SEED)
        with threadpool_limits(limits=1, user_api="blas"):
            train = mine_episodes(img_codes[train_items], txt_codes[train_items],
                                  config, TRAIN_EPISODES)
            held = mine_episodes(img_codes[held_items], txt_codes[held_items],
                                 config, EVAL_EPISODES)
        for episodes, items in ((train, train_items), (held, held_items)):
            for ep in episodes:
                ep.anchor_idx = int(items[ep.anchor_idx])
                ep.support_idxs = items[ep.support_idxs].astype(int).tolist()
                ep.contrast_idxs = items[ep.contrast_idxs].astype(int).tolist()
                ep.positive_idx = int(items[ep.positive_idx])
                ep.hard_negative_idxs = items[ep.hard_negative_idxs].astype(int).tolist()
                ep.condition_distractor_idxs = items[ep.condition_distractor_idxs].astype(int).tolist()
                ep.anchor_distractor_idxs = items[ep.anchor_distractor_idxs].astype(int).tolist()
        cached.write_text(json.dumps({"train": [vars(ep) for ep in train],
                                      "held": [vars(ep) for ep in held]}))
    for episodes, items in ((train, train_items), (held, held_items)):
        validate_roles(episodes, items)
        episode_arrays(episodes)
    sha = hashlib.sha256(json.dumps([vars(ep) for ep in held], sort_keys=True).encode()).hexdigest()
    if sha != EXPECTED_SHA:
        raise AssertionError(f"Task 8 held-episode SHA mismatch: {sha} != {EXPECTED_SHA}")
    return train, held, sha


def load_or_fit() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    CACHE.mkdir(exist_ok=True)
    img_features, txt_features = load_real_features()
    train_items, held_items = split_items(EXPECTED_SAMPLES)
    img_file, txt_file = CACHE / "factor42_img.npy", CACHE / "factor42_txt.npy"
    meta_file = CACHE / "factor42_meta.json"
    if img_file.exists() and txt_file.exists() and meta_file.exists():
        img_codes, txt_codes = np.load(img_file), np.load(txt_file)
        metadata = json.loads(meta_file.read_text())
    else:
        img_codes, txt_codes, metadata = prepare_item_disjoint_codes(
            img_features, txt_features, train_items
        )
        np.save(img_file, img_codes)
        np.save(txt_file, txt_codes)
        meta_file.write_text(json.dumps(metadata))
    if img_codes.shape != (EXPECTED_SAMPLES, 32) or txt_codes.shape != img_codes.shape:
        raise ValueError("Cached factor-code shape is invalid")
    return img_features, txt_features, img_codes, txt_codes, train_items, held_items, metadata


def load_or_train_head(img_codes: np.ndarray, txt_codes: np.ndarray, train: list):
    cached = CACHE / "recovery42.pt"
    if cached.exists():
        from src.model.condition import ConditionEncoder
        model = ConditionEncoder()
        model.load_state_dict(torch.load(cached, map_location="cpu", weights_only=True))
        return model.eval()
    model = train_recovery_head(img_codes, txt_codes, train)
    torch.save(model.state_dict(), cached)
    return model


def analyze_base(img_features, txt_features, img_codes, txt_codes,
                 train_items, train, held, pairs, model) -> tuple[dict, dict]:
    train_weights = weights_for(model, img_codes, txt_codes, train)
    held_weights = weights_for(model, img_codes, txt_codes, held)
    train_comp = {name: components(img_features, txt_features, img_codes, txt_codes,
                                   train, weights) for name, weights in train_weights.items()}
    held_comp = {name: components(img_features, txt_features, img_codes, txt_codes,
                                  held, weights) for name, weights in held_weights.items()}
    selected = {}
    train_curves = {}
    held_curves = {}
    swaps = {}
    for name in train_weights:
        beta, curve = select_beta(train_comp[name], clip_only=name == "clip_only")
        selected[name], train_curves[name] = beta, curve
        held_curves[name] = {str(b): {d: rank_summary(score_at(held_comp[name], d, b))
                                     for d in ("i2t", "t2i")}
                             for b in (0.0, beta)}
        swaps[name] = swap_reversal(img_features, txt_features, img_codes, txt_codes,
                                    held, held_weights[name], pairs, beta)
    comparisons = {}
    for comparator in ("uniform", "clip_only"):
        for setting, beta_naive, beta_other in (
            ("zero", 0.0, 0.0),
            ("selected", selected["naive"], selected[comparator]),
        ):
            comparisons[f"naive_minus_{comparator}_{setting}"] = {
                d: bootstrap_difference(
                    successes(score_at(held_comp["naive"], d, beta_naive)),
                    successes(score_at(held_comp[comparator], d, beta_other)),
                ) for d in ("i2t", "t2i")
            }
    ab_train = ablation_weights(img_codes, txt_codes, train)
    ab_held = ablation_weights(img_codes, txt_codes, held)
    pair_codes = 0.5 * (img_codes + txt_codes)
    correlation = np.corrcoef(pair_codes[train_items], rowvar=False)
    support, contrast, _, targets = episode_arrays(held)
    naive_raw = np.maximum(pair_codes[support].mean(axis=1) -
                           pair_codes[contrast].mean(axis=1), 0)
    cluster = target_cluster_ablations(naive_raw, targets, correlation, 0.95)
    ab_held["naive_minus_target_cluster_095"] = l1_normalize(cluster["minus_cluster"])
    ab_held["naive_target_cluster_only_095"] = l1_normalize(cluster["cluster_only"])
    ablations = {}
    role_breakdown = {}
    for name, weight in ab_held.items():
        comp = components(img_features, txt_features, img_codes, txt_codes, held, weight)
        ablations[name] = {str(b): {d: rank_summary(score_at(comp, d, b))
                                    for d in ("i2t", "t2i")}
                           for b in (0.0, selected["naive"])}
        role_breakdown[name] = {str(b): {d: role_outrank(score_at(comp, d, b))
                                         for d in ("i2t", "t2i")}
                                for b in (0.0, selected["naive"])}
    for name, comp in held_comp.items():
        role_breakdown[name] = {str(b): {d: role_outrank(score_at(comp, d, b))
                                     for d in ("i2t", "t2i")}
                                for b in (0.0, selected["naive"])}
    result = {
        "beta_grid_initial": BASE_GRID, "beta_selection": "mean train i2t/t2i R@1; lower beta on ties",
        "selected_beta": selected, "train_curves": train_curves,
        "held_ranking": held_curves, "bootstrap": comparisons, "swap_reversal": swaps,
        "ablations": ablations, "per_role": role_breakdown,
        "zero_weight_rows": {name: int(np.count_nonzero(weight.sum(axis=1) == 0))
                             for name, weight in held_weights.items()},
        "factor_correlation": {
            "source": "train-item pair codes", "threshold": 0.95,
            "pairs_abs_ge_090": int(np.count_nonzero(np.triu(np.abs(correlation) >= .9, 1))),
            "pairs_abs_ge_095": int(np.count_nonzero(np.triu(np.abs(correlation) >= .95, 1))),
            "pairs_abs_ge_098": int(np.count_nonzero(np.triu(np.abs(correlation) >= .98, 1))),
            "target_cluster_size_median": float(np.median(cluster["cluster_sizes"])),
            "target_cluster_size_min": int(cluster["cluster_sizes"].min()),
            "target_cluster_size_max": int(cluster["cluster_sizes"].max()),
            "minus_cluster_zero_rows": int(np.count_nonzero(
                ab_held["naive_minus_target_cluster_095"].sum(axis=1) == 0)),
        },
    }
    return result, {"weights": held_weights, "ablation_weights": ab_held,
                    "train_ablation_weights": ab_train}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("base", "q3a", "q3b", "q4", "all"), default="all")
    args = parser.parse_args()
    started = perf_counter()
    (img_features, txt_features, img_codes, txt_codes,
     train_items, held_items, prerequisite) = load_or_fit()
    train, held, sha = mine_and_verify(img_codes, txt_codes, train_items, held_items)
    pairs = choose_swap_pairs(0.5 * (img_codes + txt_codes), held_items,
                              held, EpisodeMiningConfig(seed=SEED))
    if len(pairs) != 61 or len({b for _, b in pairs}) != 30:
        raise AssertionError("Task 7 swap-pair reproduction differs")
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results.update({"episode_sha256": sha, "sha_matches_task8": sha == EXPECTED_SHA,
                    "prerequisite": prerequisite, "swap_pairs": len(pairs),
                    "train_episodes": len(train), "held_episodes": len(held)})
    if args.section in ("base", "all") or "q1_q2" not in results:
        model = load_or_train_head(img_codes, txt_codes, train)
        results["q1_q2"], _ = analyze_base(img_features, txt_features, img_codes,
                                             txt_codes, train_items, train, held, pairs, model)
        RESULTS_PATH.write_text(json.dumps(results))
        print("Q1/Q2 finished", flush=True)
    if args.section in ("q3a", "all"):
        from q3_cross_seed import analyze_cross_seed
        results["q3a"] = analyze_cross_seed(
            img_features, txt_features, img_codes, txt_codes,
            train_items, held_items, train, held, pairs, BASE_GRID,
        )
        RESULTS_PATH.write_text(json.dumps(results))
        print("Q3a finished", flush=True)
    if args.section in ("q3b", "all"):
        from q3_emotion import analyze_emotion
        results["q3b"] = analyze_emotion(img_features, txt_features, img_codes, txt_codes,
                                           train_items, held_items)
        RESULTS_PATH.write_text(json.dumps(results))
        print("Q3b finished", flush=True)
    if args.section in ("q4", "all"):
        from q4_head import analyze_heads
        model = load_or_train_head(img_codes, txt_codes, train)
        beta = results["q1_q2"]["selected_beta"]["naive"]
        results["q4"] = analyze_heads(img_features, txt_features, img_codes, txt_codes,
                                       train, held, pairs, beta, model)
        RESULTS_PATH.write_text(json.dumps(results))
        print("Q4 finished", flush=True)
    results["last_run_seconds"] = perf_counter() - started
    RESULTS_PATH.write_text(json.dumps(results))
    print("RESULT_JSON=" + json.dumps({"sections": list(results),
                                        "last_run_seconds": results["last_run_seconds"],
                                        "episode_sha256": sha}), flush=True)


if __name__ == "__main__":
    main()
