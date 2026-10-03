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


def test_validator_rejects_pair_values_anchor_emotion():
    """Validator must reject a pair where the shared aspect equals the anchor's value (pair values rule)."""
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=12)

    anchor_emotion = labels["emotion"][ep.anchor[0]]

    # Find two rows with the anchor's emotion, different styles, from different paintings not in episode 0
    used_paintings = {groups[ep.anchor[0]], groups[ep.candidates[0, 0]], groups[ep.candidates[0, 1]]}
    candidates_paintings = set(groups[ep.candidates[0]])
    used_paintings.update(candidates_paintings)
    # Also mark paintings used in pairs as used
    for arr in [ep.pairs_a_img[0], ep.pairs_a_txt[0], ep.pairs_b_img[0], ep.pairs_b_txt[0]]:
        used_paintings.update(groups[arr])

    rows_with_anchor_emotion = np.where(labels["emotion"] == anchor_emotion)[0]
    candidate_rows = []
    for row in rows_with_anchor_emotion:
        if groups[row] not in used_paintings:
            candidate_rows.append(row)
            if len(candidate_rows) == 2 and labels["style"][candidate_rows[0]] != labels["style"][candidate_rows[1]]:
                break

    assert len(candidate_rows) >= 2, "Could not find suitable rows for test"
    img_row, txt_row = candidate_rows[0], candidate_rows[1]
    assert labels["style"][img_row] != labels["style"][txt_row], "Rows must differ in style"

    # Corrupt: replace both pairs_a_img[0,0] and pairs_a_txt[0,0] with rows sharing anchor's emotion
    corrupted_pairs_a_img = ep.pairs_a_img.copy()
    corrupted_pairs_a_txt = ep.pairs_a_txt.copy()
    corrupted_pairs_a_img[0, 0] = img_row
    corrupted_pairs_a_txt[0, 0] = txt_row

    corrupted_ep = replace(ep, pairs_a_img=corrupted_pairs_a_img, pairs_a_txt=corrupted_pairs_a_txt)

    with pytest.raises(AssertionError, match="pair values"):
        validate_aspect_episodes(corrupted_ep, labels, groups, index)


def test_validator_rejects_example_shows_anchor_value():
    """Validator must reject when an example painting contains the anchor's aspect value (an example shows the anchor's value rule)."""
    labels, groups = _world()
    rows = np.arange(len(groups))
    index = PaintingValueIndex(labels, groups)
    ep = build_aspect_episodes(labels, groups, rows, "emotion", "style", 10, seed=15)

    anchor_emotion = labels["emotion"][ep.anchor[0]]
    
    # Get the emotions used in the original pairs_a (excluding the pair we'll replace)
    original_pair_emotions = np.unique(labels["emotion"][ep.pairs_a_img[0, 1:]])  # emotions of pairs 1-3
    assert len(original_pair_emotions) == 3, "Need 3 distinct emotions in other pairs"
    
    # Find an emotion not used in those 3 pairs and not the anchor emotion
    all_emotions = np.unique(labels["emotion"][labels["emotion"] >= 0])
    unused_emotion = None
    for emo in all_emotions:
        if emo != anchor_emotion and emo not in original_pair_emotions:
            unused_emotion = emo
            break
    assert unused_emotion is not None, "Need an unused emotion for pair values rule"
    
    # Find two rows with unused_emotion from a painting with anchor emotion, different styles
    used_in_ep0 = set(ep.rows())
    used_paintings = {groups[r] for r in used_in_ep0}
    paintings_with_anchor = set(np.unique(groups[labels["emotion"] == anchor_emotion]))
    
    rows_with_unused = np.where(labels["emotion"] == unused_emotion)[0]
    rows_with_unused = rows_with_unused[~np.isin(groups[rows_with_unused], list(used_paintings))]
    
    img_row = None
    txt_row = None
    for row in rows_with_unused:
        if groups[row] in paintings_with_anchor:
            img_row = row
            # Find txt_row with different style, different painting
            txt_cand = rows_with_unused[(groups[rows_with_unused] != groups[row]) &
                                        (labels["style"][rows_with_unused] != labels["style"][row])]
            if len(txt_cand) > 0:
                txt_row = txt_cand[0]
                break
    
    assert img_row is not None and txt_row is not None, "Could not find suitable rows"
    
    # Corrupt: replace pair 0 in pairs_a_img and pairs_a_txt
    corrupted_pairs_a_img = ep.pairs_a_img.copy()
    corrupted_pairs_a_txt = ep.pairs_a_txt.copy()
    corrupted_pairs_a_img[0, 0] = img_row
    corrupted_pairs_a_txt[0, 0] = txt_row
    
    corrupted_ep = replace(ep, pairs_a_img=corrupted_pairs_a_img, pairs_a_txt=corrupted_pairs_a_txt)
    
    with pytest.raises(AssertionError, match="an example shows the anchor's value"):
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
    assert len(same_emotion_style) >= 2, "Precondition: need rows with same emotion and style"
    bad_row = same_emotion_style[1]

    # Corrupt: replace pairs_a_txt with a row that has the same style as pairs_a_img
    corrupted_pairs_a_txt = ep.pairs_a_txt.copy()
    corrupted_pairs_a_txt[0, 0] = bad_row

    corrupted_ep = replace(ep, pairs_a_txt=corrupted_pairs_a_txt)

    with pytest.raises(AssertionError, match="episode 0: pair"):
        validate_aspect_episodes(corrupted_ep, labels, groups, index)
