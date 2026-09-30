import numpy as np
import pytest

from src.train.condition_episodes import (
    ConditionEpisodes, SwapEpisodes, mine_condition_episodes, mine_swap_episodes, pair_feature_units,
)
from src.train.condition_sources import ClipClusterSource, CommunitySource, FactorComboSource


def _world(n=4000, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(n, dim)).astype(np.float32)
    txt = (img + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)
    codes = np.maximum(0.0, rng.normal(size=(n, 8)) - 0.2).astype(np.float32)
    keys = np.arange(n) // 2                                        # two annotation rows per painting
    return img, txt, codes, keys


def _all_rows(ep, i, fields):
    out = []
    for f in fields:
        v = getattr(ep, f)[i]
        out.extend(np.atleast_1d(v).tolist())
    return out


def test_condition_episodes_roles_and_distinct_paintings():
    img, txt, codes, keys = _world()
    rows = np.arange(0, 4000)
    src = FactorComboSource(codes, rows, min_group_rows=100)
    units = pair_feature_units(img, txt)
    ep = mine_condition_episodes(src, units, keys, 40, np.random.default_rng(1))
    assert isinstance(ep, ConditionEpisodes) and ep.candidates.shape == (40, 16)
    assert ep.positive_mask[:, :4].all() and not ep.positive_mask[:, 4:].any()
    for i in range(40):
        members = _all_rows(ep, i, ("anchor", "supports", "contrasts", "candidates"))
        assert len({keys[r] for r in members}) == len(members)      # no painting twice


def test_positives_inside_negatives_outside_the_same_condition():
    img, txt, codes, keys = _world()
    labels = np.repeat(np.arange(8), 500)
    rows = np.arange(4000)
    src = CommunitySource(labels, rows, min_group_rows=100)
    ep = mine_condition_episodes(src, pair_feature_units(img, txt), keys, 30, np.random.default_rng(2))
    for i in range(30):
        group = labels[ep.anchor[i]]
        assert (labels[ep.supports[i]] == group).all()
        assert (labels[ep.candidates[i, :4]] == group).all()
        assert (labels[ep.contrasts[i]] != group).all()
        assert (labels[ep.candidates[i, 4:]] != group).all()


def test_hard_negatives_are_the_nearest_outside_items_when_the_pool_is_exhaustive():
    img, txt, codes, _ = _world(n=1200)
    keys = np.arange(1200)                                          # one row per painting: exact top-6
    labels = np.repeat(np.arange(4), 300)
    rows = np.arange(1200)
    units = pair_feature_units(img, txt)
    src = CommunitySource(labels, rows, min_group_rows=50)
    ep = mine_condition_episodes(src, units, keys, 10, np.random.default_rng(3), hard_pool=10_000)
    for i in range(10):
        anchor, hard = ep.anchor[i], ep.candidates[i, 4:10]
        used = {keys[r] for r in _all_rows(ep, i, ("anchor", "supports", "contrasts")) + ep.candidates[i, :4].tolist()}
        outside = rows[(labels != labels[anchor]) & ~np.isin(keys, list(used))]
        sims = units[outside] @ units[anchor]
        best = np.sort(sims)[::-1][:6]
        assert np.allclose(np.sort(units[hard] @ units[anchor])[::-1], best, atol=1e-6)


def test_miner_only_uses_fit_rows():
    img, txt, codes, keys = _world()
    rows = np.arange(0, 4000, 2)
    src = FactorComboSource(codes, rows, min_group_rows=50)
    ep = mine_condition_episodes(src, pair_feature_units(img, txt), keys, 20, np.random.default_rng(4))
    used = np.concatenate([ep.anchor, ep.supports.ravel(), ep.contrasts.ravel(), ep.candidates.ravel()])
    assert set(used.tolist()) <= set(rows.tolist())


def test_swap_episodes_have_disjoint_roles_and_anchor_in_both():
    img, txt, codes, keys = _world()
    rows = np.arange(4000)
    src = FactorComboSource(codes, rows, min_group_rows=100)
    rng = np.random.default_rng(5)
    ep = mine_swap_episodes(src, pair_feature_units(img, txt), keys, 20, rng)
    assert isinstance(ep, SwapEpisodes) and ep.candidates.shape == (20, 16)
    assert ep.positive_mask_a[:, :3].all() and not ep.positive_mask_a[:, 3:].any()
    assert ep.positive_mask_b[:, 3:6].all() and not ep.positive_mask_b[:, :3].any() and not ep.positive_mask_b[:, 6:].any()
    for i in range(20):
        members = _all_rows(ep, i, ("anchor", "supports_a", "contrasts_a", "supports_b", "contrasts_b", "candidates"))
        assert len({keys[r] for r in members}) == len(members)


def test_swap_mining_refuses_a_non_swap_source():
    img, txt, codes, keys = _world()
    src = CommunitySource(np.repeat(np.arange(8), 500), np.arange(4000), min_group_rows=100)
    with pytest.raises(ValueError):
        mine_swap_episodes(src, pair_feature_units(img, txt), keys, 4, np.random.default_rng(6))


def test_mining_is_deterministic_for_a_seed():
    img, txt, codes, keys = _world()
    src = FactorComboSource(codes, np.arange(4000), min_group_rows=100)
    units = pair_feature_units(img, txt)
    a = mine_condition_episodes(src, units, keys, 12, np.random.default_rng(7))
    b = mine_condition_episodes(src, units, keys, 12, np.random.default_rng(7))
    assert np.array_equal(a.candidates, b.candidates) and np.array_equal(a.supports, b.supports)


def test_hard_negatives_sampled_branch_takes_the_top_k_of_the_random_sample():
    """With len(pool) > hard_pool, hard negatives are the nearest among a random sample, not the global nearest."""
    from src.train.condition_episodes import _hard_negatives
    img, txt, _, keys = _world()                                    # two rows per painting
    units = pair_feature_units(img, txt)
    labels = np.repeat(np.arange(4), 1000)
    anchor, hard_pool, count = 0, 256, 6
    pool = np.flatnonzero(labels != labels[anchor])                 # the outside rows: 3,000 > hard_pool
    used = {keys[anchor]}
    picked = _hard_negatives(np.random.default_rng(11), anchor, pool, units, keys, used, count, hard_pool)
    sample = np.random.default_rng(11).choice(pool, hard_pool, replace=False)
    sample = sample[keys[sample] != keys[anchor]]
    expected, seen = [], {keys[anchor]}
    for idx in np.argsort(-(units[sample] @ units[anchor]), kind="stable"):
        if keys[sample[idx]] not in seen:
            seen.add(keys[sample[idx]])
            expected.append(int(sample[idx]))
        if len(expected) == count:
            break
    assert picked == expected
    assert set(picked) <= set(pool.tolist())                        # outside the anchor's group
    assert len({keys[r] for r in picked}) == count and keys[anchor] not in {keys[r] for r in picked}
    assert used == {keys[anchor], *(keys[r] for r in picked)}
    global_top = pool[np.argsort(-(units[pool] @ units[anchor]), kind="stable")[:count]]
    assert set(picked) != set(global_top.tolist())                  # the sample, not all outside items
