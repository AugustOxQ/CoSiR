import numpy as np
import pytest

from src.eval.aspect_episodes import (
    AspectEpisodes, PaintingValueIndex, build_aspect_episodes, concat_episodes, episodes_sha256,
    validate_aspect_episodes,
)


def _world(n_paintings=3000, seed=0, genre_missing=0.2):
    """Three aspects, 6 values each; two rows per painting; the first aspect varies per row like emotion."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    style = np.repeat(rng.integers(0, 6, n_paintings), 2)
    genre = np.repeat(rng.integers(0, 6, n_paintings), 2)
    genre[np.repeat(rng.random(n_paintings) < genre_missing, 2)] = -1
    emotion = rng.integers(0, 6, 2 * n_paintings)
    return {"emotion": emotion, "style": style, "genre": genre}, groups


def test_roles_follow_the_rules_and_validate():
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 200, seed=1, third="genre", index=index)
    assert isinstance(ep, AspectEpisodes) and ep.candidates.shape == (200, 13) and ep.pairs_a_img.shape == (200, 4)
    validate_aspect_episodes(ep, labels, groups, index, third="genre")
    a, b = labels["emotion"][ep.anchor], labels["style"][ep.anchor]
    assert (labels["emotion"][ep.candidates[:, 0]] == a).all() and (labels["style"][ep.candidates[:, 1]] == b).all()
    assert (labels["emotion"][ep.pairs_a_img] == labels["emotion"][ep.pairs_a_txt]).all()
    assert (labels["style"][ep.pairs_a_img] != labels["style"][ep.pairs_a_txt]).all()
    assert (labels["emotion"][ep.pairs_a_img] != a[:, None]).all()


def test_condition_swaps_roles():
    labels, groups = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "emotion", "style", 20, seed=2)
    si, st, ci, ct, col = ep.condition("a")
    assert col == 0 and (si == ep.pairs_a_img).all() and (ci == ep.pairs_b_img).all()
    si, st, ci, ct, col = ep.condition("b")
    assert col == 1 and (si == ep.pairs_b_img).all() and (ci == ep.pairs_a_img).all()
    with pytest.raises(ValueError):
        ep.condition("c")


def test_seed_determinism_and_hash():
    labels, groups = _world()
    rows = np.arange(len(groups))
    e1 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=3)
    e2 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=3)
    e3 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=4)
    assert episodes_sha256(e1) == episodes_sha256(e2) != episodes_sha256(e3)


def test_rows_scope_respected():
    labels, groups = _world()
    rows = np.arange(0, len(groups), 1)[: len(groups) // 2]                   # first half of the paintings only
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 50, seed=5)
    assert np.isin(ep.rows(), rows).all()


def test_unlabelled_rows_never_used():                                   # Review Focus 1
    labels, groups = _world(genre_missing=0.5)
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "genre", "style", 100, seed=6)
    assert (labels["genre"][ep.rows()] >= 0).all()


def test_exhausted_pool_raises():                                         # Review Focus 5
    labels, groups = _world(n_paintings=40)
    with pytest.raises(RuntimeError, match="emotion.*style"):
        build_aspect_episodes(labels, groups, np.arange(len(groups)), "emotion", "style", 50, seed=7,
                              min_paintings=1)


def test_concat_keeps_order():
    labels, groups = _world()
    rows = np.arange(len(groups))
    e1 = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=8)
    e2 = build_aspect_episodes(labels, groups, rows, "style", "genre", 10, seed=9)
    both = concat_episodes([e1, e2])
    assert both.anchor.tolist() == e1.anchor.tolist() + e2.anchor.tolist() and both.aspect_a == "mixed"
