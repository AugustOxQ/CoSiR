"""Mine factor-conditioned support, contrast, and candidate-pool episodes."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeMiningConfig:
    """Mining sizes and percentile cutoffs.

    ``high_activation_percentile`` must exceed ``low_activation_percentile``.
    Defaults leave room for an anchor, support examples, and a held-out positive
    in a reasonably sized factor pool. Tied values at both cutoffs are assigned
    to the low pool only, keeping the pools disjoint.
    """

    num_support: int = 4
    num_contrast: int = 4
    num_hard_negatives: int = 4
    num_condition_distractors: int = 4
    num_anchor_distractors: int = 4
    high_activation_percentile: float = 90.0
    low_activation_percentile: float = 50.0
    min_pool_size: int = 50
    seed: int = 42


@dataclass
class Episode:
    """Indices into the same ``(N, L)`` image and text code arrays.

    ``targeted_factor`` is ground truth for validation only. Later training
    must infer the condition from support and contrast codes without using it.
    """

    anchor_idx: int
    targeted_factor: int
    support_idxs: list[int]
    contrast_idxs: list[int]
    positive_idx: int
    hard_negative_idxs: list[int]
    condition_distractor_idxs: list[int]
    anchor_distractor_idxs: list[int]


def _available(pool: np.ndarray, used: set[int]) -> np.ndarray:
    """Return pool members not assigned to an earlier role in this episode."""
    if not used:
        return pool
    return pool[~np.isin(pool, list(used))]


def _cosine_without_factor(
    pair_codes: np.ndarray, candidates: np.ndarray, reference: int, factor: int
) -> np.ndarray:
    """Cosine similarity after zeroing the targeted factor on both sides."""
    other_codes = pair_codes[candidates].copy()
    other_codes[:, factor] = 0
    reference_code = pair_codes[reference].copy()
    reference_code[factor] = 0
    reference_norm = np.linalg.norm(reference_code)
    candidate_norms = np.linalg.norm(other_codes, axis=1)
    denominator = candidate_norms * reference_norm
    return np.divide(
        other_codes @ reference_code,
        denominator,
        out=np.zeros(len(candidates), dtype=np.float64),
        where=denominator > 0,
    )


def _ranked_candidates(
    pair_codes: np.ndarray,
    pool: np.ndarray,
    used: set[int],
    reference: int,
    factor: int,
    count: int,
    *,
    highest: bool,
) -> list[int]:
    candidates = _available(pool, used)
    if count <= 0 or len(candidates) == 0:
        return []
    similarities = _cosine_without_factor(pair_codes, candidates, reference, factor)
    # Index order breaks ties deterministically, including zero-norm code ties.
    order = np.lexsort((candidates, -similarities if highest else similarities))
    return candidates[order[:count]].astype(int).tolist()


def mine_episodes(
    img_codes: np.ndarray,
    txt_codes: np.ndarray,
    config: EpisodeMiningConfig,
    num_episodes: int,
) -> list[Episode]:
    """Build episodes from paired factor activations, with unique indices.

    The pair code is the image/text mean, computed once. For each episode,
    sample a factor whose disjoint high and low percentile pools meet the
    minimum size. Draw an anchor and support from high, contrast from low,
    then a distinct held-out positive from high. Support is shortened if
    needed to reserve that positive. Hard negatives are low on the condition
    but similar to the *positive* on every other factor. Condition-only
    distractors are high on the condition like the anchor but otherwise
    dissimilar. Anchor-only distractors resemble the *anchor* on everything
    else but are low on the condition. Each later role excludes every earlier
    role; exhausted pools yield shorter lists, never padding or reuse.
    """
    img_codes = np.asarray(img_codes)
    txt_codes = np.asarray(txt_codes)
    if (
        img_codes.ndim != 2
        or img_codes.shape != txt_codes.shape
        or img_codes.shape[1] == 0
        or not np.issubdtype(img_codes.dtype, np.number)
        or not np.issubdtype(txt_codes.dtype, np.number)
        or not np.isfinite(img_codes).all()
        or not np.isfinite(txt_codes).all()
    ):
        raise ValueError("img_codes and txt_codes must be finite, matching (N, L) arrays")
    if not (0 <= config.low_activation_percentile < config.high_activation_percentile <= 100):
        raise ValueError("high_activation_percentile must exceed low_activation_percentile")
    if config.min_pool_size < 1 or num_episodes < 0:
        raise ValueError("min_pool_size must be positive and num_episodes nonnegative")
    if any(
        count < 0
        for count in (
            config.num_support,
            config.num_contrast,
            config.num_hard_negatives,
            config.num_condition_distractors,
            config.num_anchor_distractors,
        )
    ):
        raise ValueError("episode role counts must be nonnegative")
    if num_episodes == 0:
        return []

    pair_codes = 0.5 * (img_codes + txt_codes)
    num_factors = pair_codes.shape[1]
    high_cutoffs = np.percentile(pair_codes, config.high_activation_percentile, axis=0)
    low_cutoffs = np.percentile(pair_codes, config.low_activation_percentile, axis=0)
    high_pools = []
    low_pools = []
    valid_factors = set()
    for factor in range(num_factors):
        column = pair_codes[:, factor]
        # A tied percentile boundary could otherwise put the same item in
        # both pools (including a nearly dead factor with mostly zero codes).
        high_pool = np.flatnonzero((column >= high_cutoffs[factor]) & (column > low_cutoffs[factor]))
        low_pool = np.flatnonzero(column <= low_cutoffs[factor])
        high_pools.append(high_pool)
        low_pools.append(low_pool)
        if len(high_pool) >= max(config.min_pool_size, 2) and len(low_pool) >= config.min_pool_size:
            valid_factors.add(factor)
    if not valid_factors:
        raise ValueError("no valid factor has sufficiently large disjoint high/low pools")

    rng = np.random.default_rng(config.seed)
    episodes = []
    for _ in range(num_episodes):
        # Rejection sampling is uniform over valid factors. The bound avoids
        # an infinite loop when most factors have near-empty activation pools.
        for _attempt in range(32 * num_factors):
            factor = int(rng.integers(num_factors))
            if factor in valid_factors:
                break
        else:
            raise ValueError("no valid factor found within bounded sampling attempts")

        high_pool = high_pools[factor]
        low_pool = low_pools[factor]
        anchor_idx = int(rng.choice(high_pool))
        used = {anchor_idx}

        support_available = _available(high_pool, used)
        support_count = min(config.num_support, len(support_available) - 1)
        support_idxs = rng.choice(support_available, size=support_count, replace=False).astype(int).tolist()
        used.update(support_idxs)

        contrast_available = _available(low_pool, used)
        contrast_count = min(config.num_contrast, len(contrast_available))
        contrast_idxs = rng.choice(contrast_available, size=contrast_count, replace=False).astype(int).tolist()
        used.update(contrast_idxs)

        positive_idx = int(rng.choice(_available(high_pool, used)))
        used.add(positive_idx)

        hard_negative_idxs = _ranked_candidates(
            pair_codes, low_pool, used, positive_idx, factor,
            config.num_hard_negatives, highest=True,
        )
        used.update(hard_negative_idxs)

        condition_distractor_idxs = _ranked_candidates(
            pair_codes, high_pool, used, anchor_idx, factor,
            config.num_condition_distractors, highest=False,
        )
        used.update(condition_distractor_idxs)

        anchor_distractor_idxs = _ranked_candidates(
            pair_codes, low_pool, used, anchor_idx, factor,
            config.num_anchor_distractors, highest=True,
        )
        used.update(anchor_distractor_idxs)

        episodes.append(Episode(
            anchor_idx=anchor_idx,
            targeted_factor=factor,
            support_idxs=support_idxs,
            contrast_idxs=contrast_idxs,
            positive_idx=positive_idx,
            hard_negative_idxs=hard_negative_idxs,
            condition_distractor_idxs=condition_distractor_idxs,
            anchor_distractor_idxs=anchor_distractor_idxs,
        ))
    return episodes
