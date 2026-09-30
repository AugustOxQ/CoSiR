import numpy as np
import pytest

from src.data.sampling import draw_distinct
from src.data.splits import grouped_subsplit
from src.train.condition_sources import ClipClusterSource, CommunitySource, Condition, FactorComboSource


def _blobs(n_per=400, dim=16, seed=0):
    """4 well-separated blobs in image space and 4 differently arranged blobs in caption space."""
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:4] * 10
    img_lab = np.repeat(np.arange(4), n_per)
    txt_lab = (img_lab + np.tile(np.arange(4), n_per)) % 4          # caption blob differs from image blob
    img = centers[img_lab] + rng.normal(size=(4 * n_per, dim))
    txt = centers[txt_lab][:, ::-1] + rng.normal(size=(4 * n_per, dim))
    return img.astype(np.float32), txt.astype(np.float32), img_lab, txt_lab


def test_grouped_subsplit_keeps_groups_whole_and_hits_fraction():
    groups = np.repeat(np.arange(3000), 3)
    rows = np.arange(0, 9000, 2)                                    # a subset, like split.train
    first, second = grouped_subsplit(groups, rows, 0.15, seed=42)
    assert not set(groups[first]) & set(groups[second])
    assert set(first) | set(second) == set(rows) and not set(first) & set(second)
    assert abs(len(second) / len(rows) - 0.15) < 0.01
    again = grouped_subsplit(groups, rows, 0.15, seed=42)
    assert np.array_equal(again[1], second)
    with pytest.raises(ValueError):
        grouped_subsplit(groups, rows, 1.0)


def test_draw_distinct_never_repeats_a_key():
    rng = np.random.default_rng(0)
    keys = np.repeat(np.arange(50), 4)
    used: set = set()
    picked = draw_distinct(rng, np.arange(200), keys, used, 30)
    assert len({keys[r] for r in picked}) == 30 and used == {keys[r] for r in picked}


def test_factor_combo_conditions_are_disjoint_nonzero_and_inside_fit_rows():
    rng = np.random.default_rng(1)
    codes = np.maximum(0.0, rng.normal(size=(5000, 8)) - 0.3).astype(np.float32)
    codes[:, 7] = 0.0                                               # a dead factor: its top decile is all zero
    rows = np.arange(0, 5000, 2)
    src = FactorComboSource(codes, rows, min_group_rows=50)
    for _ in range(30):
        c = src.sample_condition(rng)
        assert isinstance(c, Condition) and c.source == "factor_combo"
        assert set(c.inside) <= set(rows) and set(c.outside) <= set(rows)
        assert not set(c.inside) & set(c.outside)
        factors, weights = np.array(c.key[0]), np.array(c.key[1])
        score_in = codes[c.inside][:, factors] @ weights
        score_out = codes[c.outside][:, factors] @ weights
        assert (score_in > 0).all() and score_in.min() >= score_out.max()
    a, b = src.sample_swap(rng)
    assert not set(a.key[0]) & set(b.key[0])
    assert len(np.intersect1d(a.inside, b.inside)) >= 20


def test_factor_combo_raises_when_no_valid_condition_exists():
    codes = np.zeros((1000, 4), dtype=np.float32)
    with pytest.raises(RuntimeError):
        FactorComboSource(codes, np.arange(1000), min_group_rows=10, max_tries=5).sample_condition(
            np.random.default_rng(0))


def test_clip_clusters_recover_blobs_and_swap_uses_both_views():
    img, txt, img_lab, _ = _blobs()
    rows = np.arange(len(img))
    src = ClipClusterSource(img, txt, rows, n_clusters=4, seed=42, min_group_rows=50)
    rng = np.random.default_rng(2)
    c = src.sample_condition(rng)
    assert c.key[0] in ("image", "caption") and set(c.inside) <= set(rows)
    if c.key[0] == "image":
        assert len(np.unique(img_lab[c.inside])) == 1               # a recovered blob
    a, b = src.sample_swap(rng)
    assert a.key[0] == "image" and b.key[0] == "caption"
    assert len(np.intersect1d(a.inside, b.inside)) >= 20
    assert not set(a.inside) & set(a.outside)


def test_sources_never_return_rows_outside_the_fit_rows():
    img, txt, _, _ = _blobs()
    rows = np.arange(0, len(img), 3)
    src = ClipClusterSource(img, txt, rows, n_clusters=4, seed=42, min_group_rows=20)
    rng = np.random.default_rng(3)
    for _ in range(20):
        c = src.sample_condition(rng)
        assert set(c.inside) <= set(rows) and set(c.outside) <= set(rows)


def test_community_source_skips_small_groups_and_cannot_swap():
    labels = np.full(3000, -1)
    rows = np.arange(0, 3000, 2)
    labels[rows] = np.repeat(np.arange(5), len(rows) // 5 + 1)[: len(rows)]
    labels[rows[:10]] = 99                                          # a tiny community
    src = CommunitySource(labels, rows, min_group_rows=50)
    rng = np.random.default_rng(4)
    keys = {src.sample_condition(rng).key for _ in range(50)}
    assert ("community", 99) not in keys
    assert src.swap_capable is False
    with pytest.raises(NotImplementedError):
        src.sample_swap(rng)
