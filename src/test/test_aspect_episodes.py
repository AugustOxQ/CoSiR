import numpy as np
import pytest
from dataclasses import replace

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


def test_third_aspect_labelling_enforced():
    """Third-aspect labels must be >= 0 on all rows when third is given."""
    labels, groups = _world(genre_missing=0.5)
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 100, seed=10, third="genre", index=index)
    # All rows must have genre labelled
    assert (labels["genre"][ep.rows()] >= 0).all()
    # Validate should pass with third="genre"
    validate_aspect_episodes(ep, labels, groups, index, third="genre")


def test_validator_rejects_unlabelled_third_aspect():
    """Validator must reject a candidate with unlabelled third aspect."""
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=11, third="genre", index=index)

    # Find a row with unlabelled genre
    unlabelled_row = np.where(labels["genre"] == -1)[0][0]

    # Corrupt: replace a candidate in the first episode with an unlabelled row
    corrupted_candidates = ep.candidates.copy()
    corrupted_candidates[0, 2] = unlabelled_row  # replace a negative with unlabelled row

    corrupted_ep = replace(ep, candidates=corrupted_candidates)

    with pytest.raises(AssertionError, match="unlabelled third aspect"):
        validate_aspect_episodes(corrupted_ep, labels, groups, index, third="genre")


def test_validator_rejects_pair_with_anchor_aspect_a_value():
    """Validator must reject a pair row that carries the anchor's aspect-A value."""
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=12)

    # Find a row with the anchor's emotion value but different style
    anchor_emotion = labels["emotion"][ep.anchor[0]]
    candidate_row = np.where((labels["emotion"] == anchor_emotion) &
                              (labels["style"] != labels["style"][ep.anchor[0]]))[0][0]

    # Corrupt: replace a pairs_a_img row with one that has the anchor's emotion value
    corrupted_pairs_a_img = ep.pairs_a_img.copy()
    corrupted_pairs_a_img[0, 0] = candidate_row

    corrupted_ep = replace(ep, pairs_a_img=corrupted_pairs_a_img)

    with pytest.raises(AssertionError):
        validate_aspect_episodes(corrupted_ep, labels, groups, index)


def test_validator_rejects_painting_duplication():
    """Validator must reject when a painting appears in two different roles."""
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=13)

    # Find another row from the same painting as the anchor
    anchor_painting = groups[ep.anchor[0]]
    duplicate_row = np.where((groups == anchor_painting) & (np.arange(len(groups)) != ep.anchor[0]))[0][0]

    # Corrupt: replace a candidate with a row from the anchor's painting
    corrupted_candidates = ep.candidates.copy()
    corrupted_candidates[0, 2] = duplicate_row

    corrupted_ep = replace(ep, candidates=corrupted_candidates)

    with pytest.raises(AssertionError, match="a painting appears twice"):
        validate_aspect_episodes(corrupted_ep, labels, groups, index)


def test_validator_rejects_pair_with_matching_differ_aspect():
    """Validator must reject a pair where both rows agree on the aspect they should differ on."""
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=14)

    # Find two rows that have the same emotion and style
    same_emotion_style = np.where((labels["emotion"] == labels["emotion"][ep.pairs_a_img[0, 0]]) &
                                    (labels["style"] == labels["style"][ep.pairs_a_img[0, 0]]))[0]
    if len(same_emotion_style) >= 2:
        bad_row = same_emotion_style[1]

        # Corrupt: replace pairs_a_txt with a row that has the same style as pairs_a_img
        corrupted_pairs_a_txt = ep.pairs_a_txt.copy()
        corrupted_pairs_a_txt[0, 0] = bad_row

        corrupted_ep = replace(ep, pairs_a_txt=corrupted_pairs_a_txt)

        with pytest.raises(AssertionError):
            validate_aspect_episodes(corrupted_ep, labels, groups, index)
