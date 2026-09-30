"""Group-aware multi-positive condition episodes and swap episodes for stage (d) (spec §3)."""

from dataclasses import dataclass

import numpy as np

from src.data.sampling import draw_distinct


@dataclass(frozen=True)
class ConditionEpisodes:
    anchor: np.ndarray
    supports: np.ndarray
    contrasts: np.ndarray
    candidates: np.ndarray        # positives, then hard negatives, then random negatives
    positive_mask: np.ndarray


@dataclass(frozen=True)
class SwapEpisodes:
    anchor: np.ndarray
    supports_a: np.ndarray
    contrasts_a: np.ndarray
    supports_b: np.ndarray
    contrasts_b: np.ndarray
    candidates: np.ndarray        # A-only, then B-only, then hard and random negatives outside both
    positive_mask_a: np.ndarray
    positive_mask_b: np.ndarray


def pair_feature_units(img_features, txt_features) -> np.ndarray:
    pair = 0.5 * (np.asarray(img_features, dtype=np.float32) + np.asarray(txt_features, dtype=np.float32))
    return pair / np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1e-12)


def _hard_negatives(rng, anchor, pool, units, keys, used, count, hard_pool):
    """The `count` rows of `pool` most CLIP-similar to the anchor (among a random `hard_pool` sample)."""
    if count == 0:
        return []
    sample = pool if len(pool) <= hard_pool else rng.choice(pool, hard_pool, replace=False)
    sample = sample[~np.isin(keys[sample], list(used))]
    order = np.argsort(-(units[sample] @ units[anchor]), kind="stable")
    picked = []
    for idx in order:
        row = int(sample[idx])
        if keys[row] in used:
            continue
        used.add(keys[row])
        picked.append(row)
        if len(picked) == count:
            return picked
    raise ValueError("Not enough distinct outside paintings for hard negatives")


def mine_condition_episodes(source, units, keys, n_episodes, rng, episodes_per_condition=4, num_support=4,
                            num_contrast=4, num_positive=4, num_hard=6, num_random=6, hard_pool=2048,
                            max_failures=1000) -> ConditionEpisodes:
    if num_hard > 0 and units is None:
        raise ValueError("hard negatives need CLIP units (pair_feature_units); pass num_hard=0 to mine without them")
    fields = {k: [] for k in ("anchor", "supports", "contrasts", "candidates")}
    failures = 0
    while len(fields["anchor"]) < n_episodes:
        condition = source.sample_condition(rng)
        for _ in range(episodes_per_condition):
            if len(fields["anchor"]) == n_episodes:
                break
            used: set = set()
            try:
                anchor = draw_distinct(rng, condition.inside, keys, used, 1)[0]
                supports = draw_distinct(rng, condition.inside, keys, used, num_support)
                positives = draw_distinct(rng, condition.inside, keys, used, num_positive)
                contrasts = draw_distinct(rng, condition.outside, keys, used, num_contrast)
                hard = _hard_negatives(rng, anchor, condition.outside, units, keys, used, num_hard, hard_pool)
                random_neg = draw_distinct(rng, condition.outside, keys, used, num_random)
            except ValueError:
                failures += 1
                if failures > max_failures:
                    raise RuntimeError("Too many conditions could not fill an episode")
                break                                   # resample a condition
            fields["anchor"].append(anchor)
            fields["supports"].append(supports)
            fields["contrasts"].append(contrasts)
            fields["candidates"].append(positives + hard + random_neg)
    mask = np.zeros((n_episodes, num_positive + num_hard + num_random), dtype=bool)
    mask[:, :num_positive] = True
    return ConditionEpisodes(*(np.asarray(fields[k], dtype=np.int64) for k in fields), positive_mask=mask)


def mine_swap_episodes(source, units, keys, n_episodes, rng, episodes_per_pair=4, num_support=4,
                       num_contrast=4, num_each=3, num_hard=5, num_random=5, hard_pool=2048,
                       max_failures=1000) -> SwapEpisodes:
    if not source.swap_capable:
        raise ValueError(f"{source.name} cannot form swap pairs")
    names = ("anchor", "supports_a", "contrasts_a", "supports_b", "contrasts_b", "candidates")
    fields = {k: [] for k in names}
    failures = 0
    while len(fields["anchor"]) < n_episodes:
        a, b = source.sample_swap(rng)
        both = np.intersect1d(a.inside, b.inside, assume_unique=True)
        a_only = np.setdiff1d(a.inside, b.inside, assume_unique=True)
        b_only = np.setdiff1d(b.inside, a.inside, assume_unique=True)
        neither = np.intersect1d(a.outside, b.outside, assume_unique=True)
        for _ in range(episodes_per_pair):
            if len(fields["anchor"]) == n_episodes:
                break
            used: set = set()
            try:
                anchor = draw_distinct(rng, both, keys, used, 1)[0]
                row = [anchor,
                       draw_distinct(rng, a_only, keys, used, num_support),
                       draw_distinct(rng, a.outside, keys, used, num_contrast),
                       draw_distinct(rng, b_only, keys, used, num_support),
                       draw_distinct(rng, b.outside, keys, used, num_contrast)]
                cands = (draw_distinct(rng, a_only, keys, used, num_each)
                         + draw_distinct(rng, b_only, keys, used, num_each)
                         + _hard_negatives(rng, anchor, neither, units, keys, used, num_hard, hard_pool)
                         + draw_distinct(rng, neither, keys, used, num_random))
            except ValueError:
                failures += 1
                if failures > max_failures:
                    raise RuntimeError("Too many swap pairs could not fill an episode")
                break
            for key, value in zip(names, row + [cands]):
                fields[key].append(value)
    width = 2 * num_each + num_hard + num_random
    mask_a = np.zeros((n_episodes, width), dtype=bool)
    mask_b = np.zeros((n_episodes, width), dtype=bool)
    mask_a[:, :num_each] = True
    mask_b[:, num_each:2 * num_each] = True
    return SwapEpisodes(*(np.asarray(fields[k], dtype=np.int64) for k in names),
                        positive_mask_a=mask_a, positive_mask_b=mask_b)
