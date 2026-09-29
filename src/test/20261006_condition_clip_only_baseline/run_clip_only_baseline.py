"""Evaluate the exact zero-conditioning row on Task 7's held-out episodes.

Run from the repository root with the CoSiR environment. Data loading, the
item-disjoint split, train-only factor fit, episode checks, swap-pair selection,
and score_pool are imported unchanged from Task 7. Only Task 7's inline
split-local-to-global episode remap is repeated here; no condition head is fit.
"""

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261005_condition_ranking_evaluation"))

from run_ranking_eval import (  # noqa: E402
    BETAS, EVAL_EPISODES, EXPECTED_SAMPLES, SEED, TRAIN_EPISODES,
    EpisodeMiningConfig, choose_swap_pairs, episode_arrays, load_real_features,
    mine_episodes, prepare_item_disjoint_codes, score_pool, split_items,
    validate_roles,
)


def zero_weights(episodes: list, num_factors: int) -> np.ndarray:
    """Give every condition the literal zero vector expected by score_pool."""
    if num_factors != 32:
        raise ValueError("Expected 32 trained factors")
    return np.zeros((len(episodes), num_factors), dtype=np.float32)


def evaluate_clip_only_ranking(img_features: np.ndarray, txt_features: np.ndarray,
                               img_codes: np.ndarray, txt_codes: np.ndarray,
                               episodes: list) -> dict:
    """Use Task 7's score_pool and strict-greater rank convention."""
    _, _, candidates, _ = episode_arrays(episodes)
    anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
    weights = zero_weights(episodes, img_codes.shape[1])
    out = {}
    for beta in BETAS:
        directions = {}
        for direction, af, cf, ac, cc in (
            ("i2t", img_features, txt_features, img_codes, txt_codes),
            ("t2i", txt_features, img_features, txt_codes, img_codes),
        ):
            scores, cosine, factor = score_pool(
                af[anchors], cf[candidates], ac[anchors], cc[candidates], weights, beta
            )
            if not np.all(factor == 0):
                raise AssertionError("CLIP-only factor contribution is nonzero")
            all_tied = bool(np.all(scores == scores[:, :1]))
            ranks = 1 + (scores[:, 1:] > scores[:, :1]).sum(axis=1)
            # Task 7's strict-greater rule assigns rank 1 to every all-zero
            # row. At beta=0 this is an optimistic tie artifact, not Recall.
            directions[direction] = {
                "ranks": ranks.tolist() if not all_tied else None,
                "recall1": float(np.mean(ranks == 1)) if not all_tied else None,
                "recall3": float(np.mean(ranks <= 3)) if not all_tied else None,
                "all_tied": all_tied,
                "mean_cosine": float(cosine.mean()),
                "mean_abs_factor": float(np.abs(factor).mean()),
            }
        out[str(beta)] = directions
    for direction in ("i2t", "t2i"):
        if out["0.03"][direction]["ranks"] != out["0.3"][direction]["ranks"]:
            raise AssertionError("Positive beta scaling changed CLIP-only ranks")
    return out


def evaluate_clip_only_swaps(img_features: np.ndarray, txt_features: np.ndarray,
                             img_codes: np.ndarray, txt_codes: np.ndarray,
                             episodes: list, pairs: list[tuple[int, int]]) -> dict:
    """Exercise Task 7's pair pool and strict reversal criterion with zeros."""
    weights = zero_weights(episodes, img_codes.shape[1])
    out = {}
    for beta in BETAS:
        directions = {}
        for direction, af, cf, ac, cc in (
            ("i2t", img_features, txt_features, img_codes, txt_codes),
            ("t2i", txt_features, img_features, txt_codes, img_codes),
        ):
            reversals = 0
            rank_changes = 0
            for a, b in pairs:
                ep_a, ep_b = episodes[a], episodes[b]
                pool = list(dict.fromkeys([
                    ep_a.positive_idx, ep_b.positive_idx,
                    *ep_a.hard_negative_idxs, *ep_a.condition_distractor_idxs,
                    *ep_a.anchor_distractor_idxs, *ep_b.hard_negative_idxs,
                    *ep_b.condition_distractor_idxs, *ep_b.anchor_distractor_idxs,
                ]))
                anchor_feat = np.broadcast_to(af[ep_a.anchor_idx], (2, af.shape[1]))
                pool_feat = np.broadcast_to(cf[pool], (2, len(pool), cf.shape[1]))
                anchor_code = np.broadcast_to(ac[ep_a.anchor_idx], (2, ac.shape[1]))
                pool_code = np.broadcast_to(cc[pool], (2, len(pool), cc.shape[1]))
                scores, _, factor = score_pool(anchor_feat, pool_feat, anchor_code,
                                               pool_code, weights[[a, b]], beta)
                if not np.all(factor == 0):
                    raise AssertionError("CLIP-only swap factor contribution is nonzero")
                rank_changes += int(not np.array_equal(np.argsort(-scores[0]),
                                                       np.argsort(-scores[1])))
                margin_a = scores[0, 0] - scores[0, 1]
                margin_b = scores[1, 1] - scores[1, 0]
                reversals += int(margin_a > 1e-8 and margin_b > 1e-8)
            directions[direction] = {
                "appropriate_reversals": reversals,
                "reversal_rate": reversals / len(pairs),
                "any_rank_change": rank_changes / len(pairs),
            }
        out[str(beta)] = directions
    return out


def main() -> None:
    started = perf_counter()
    img_features, txt_features = load_real_features()
    train_items, held_items = split_items(EXPECTED_SAMPLES)
    img_codes, txt_codes, prerequisite = prepare_item_disjoint_codes(
        img_features, txt_features, train_items
    )
    config = EpisodeMiningConfig(seed=SEED)
    # This is Task 7's inline mining/remap block, copied without changing its
    # call order or split-local-to-global index mapping.
    with threadpool_limits(limits=1, user_api="blas"):
        train_episodes = mine_episodes(img_codes[train_items], txt_codes[train_items],
                                       config, TRAIN_EPISODES)
        eval_episodes = mine_episodes(img_codes[held_items], txt_codes[held_items],
                                      config, EVAL_EPISODES)
    for episodes, items in ((train_episodes, train_items), (eval_episodes, held_items)):
        for ep in episodes:
            ep.anchor_idx = int(items[ep.anchor_idx])
            ep.support_idxs = items[ep.support_idxs].astype(int).tolist()
            ep.contrast_idxs = items[ep.contrast_idxs].astype(int).tolist()
            ep.positive_idx = int(items[ep.positive_idx])
            ep.hard_negative_idxs = items[ep.hard_negative_idxs].astype(int).tolist()
            ep.condition_distractor_idxs = items[ep.condition_distractor_idxs].astype(int).tolist()
            ep.anchor_distractor_idxs = items[ep.anchor_distractor_idxs].astype(int).tolist()
        validate_roles(episodes, items)
        episode_arrays(episodes)

    pair_codes = 0.5 * (img_codes + txt_codes)
    supports, contrasts, _, targets = episode_arrays(eval_episodes)
    gap = pair_codes[supports].mean(axis=1) - pair_codes[contrasts].mean(axis=1)
    target_rank = 1 + (gap > gap[np.arange(len(gap)), targets, None]).sum(axis=1)
    rank_bins = [int(np.count_nonzero(target_rank == k)) for k in (1, 2, 3)] + [
        int(np.count_nonzero((target_rank >= 4) & (target_rank <= 10))),
        int(np.count_nonzero(target_rank >= 11)),
    ]
    pairs = choose_swap_pairs(pair_codes, held_items, eval_episodes, config)
    checks = {
        "train_items": len(train_items), "held_items": len(held_items),
        "train_episodes": len(train_episodes), "eval_episodes": len(eval_episodes),
        "edges": prerequisite["edges"], "communities": prerequisite["communities"],
        "target_gap_rank_bins": rank_bins, "swap_pairs": len(pairs),
        "distinct_swap_b": len({b for _, b in pairs}),
    }
    expected = {
        "train_items": 246978, "held_items": 61745,
        "train_episodes": 4096, "eval_episodes": 1024,
        "edges": 2463679, "communities": 21,
        "target_gap_rank_bins": [245, 122, 89, 414, 154],
        "swap_pairs": 61, "distinct_swap_b": 30,
    }
    if checks != expected:
        raise ValueError(f"Task 7 reproduction checks differ: observed={checks}, expected={expected}")

    ranking = evaluate_clip_only_ranking(img_features, txt_features, img_codes,
                                         txt_codes, eval_episodes)
    swaps = evaluate_clip_only_swaps(img_features, txt_features, img_codes,
                                     txt_codes, eval_episodes, pairs)
    episode_records = [vars(ep) for ep in eval_episodes]
    fingerprint = hashlib.sha256(json.dumps(episode_records, sort_keys=True).encode()).hexdigest()
    result = {
        "seed": SEED, "betas": BETAS, "checks": checks,
        "episode_sha256": fingerprint,
        "episode_spot_check": [
            {"episode": i, "anchor": eval_episodes[i].anchor_idx,
             "positive": eval_episodes[i].positive_idx,
             "targeted_factor": eval_episodes[i].targeted_factor}
            for i in (0, 1, 2)
        ],
        "ranking": ranking, "swaps": swaps,
        "seconds": perf_counter() - started,
    }
    for beta, directions in swaps.items():
        for direction, values in directions.items():
            if values["appropriate_reversals"]:
                print(f"WARNING: nonzero CLIP-only swap reversal at beta={beta} {direction}: "
                      f"{values['appropriate_reversals']}", flush=True)
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
