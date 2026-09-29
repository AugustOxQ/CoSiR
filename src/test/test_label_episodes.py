import numpy as np
import pytest
import torch

from src.eval.label_episodes import (
    build_label_episodes, condition_lift, label_episode_recall, tie_aware_rank,
)


def _fixture(seed=0):
    """3 labels with unequal sizes; 3 annotation rows per painting; label is per painting."""
    rng = np.random.default_rng(seed)
    painting_labels = np.repeat(np.array(["a", "b", "c"]), [400, 120, 60])
    paintings = np.repeat(np.arange(len(painting_labels)), 3).astype(str)
    labels = np.repeat(painting_labels, 3)
    return labels, paintings, rng


def test_episodes_respect_labels_and_never_repeat_a_painting():
    labels, paintings, _ = _fixture()
    rows = np.arange(len(labels))
    ep = build_label_episodes(labels, paintings, rows, 300, seed=42)
    for i in range(300):
        same = [ep.anchor[i], ep.positive[i], *ep.supports[i]]
        other = [*ep.contrasts[i], *ep.distractors[i]]
        assert all(labels[r] == ep.labels[i] for r in same)
        assert all(labels[r] != ep.labels[i] for r in other)
        members = same + other
        assert len({paintings[r] for r in members}) == len(members)


def test_target_labels_are_balanced_and_rows_are_respected():
    labels, paintings, _ = _fixture()
    ep = build_label_episodes(labels, paintings, np.arange(len(labels)), 900, seed=42)
    shares = np.array([(ep.labels == x).mean() for x in ("a", "b", "c")])
    assert np.allclose(shares, 1 / 3, atol=0.06)                        # balanced despite 400/120/60
    subset = np.arange(0, len(labels), 2)                                # every label still present
    ep_subset = build_label_episodes(labels, paintings, subset, 50, seed=42)
    used = np.concatenate([ep_subset.anchor, ep_subset.positive, ep_subset.supports.ravel(),
                           ep_subset.contrasts.ravel(), ep_subset.distractors.ravel()])
    assert set(used.tolist()) <= set(subset.tolist())


def test_small_and_excluded_labels_are_never_targets_and_two_are_required():
    labels, paintings, _ = _fixture()
    rows = np.arange(len(labels))
    ep = build_label_episodes(labels, paintings, rows, 200, seed=42, min_paintings_per_label=100)
    assert set(ep.labels.tolist()) == {"a", "b"}                       # "c" has 60 paintings
    ep2 = build_label_episodes(labels, paintings, rows, 200, seed=42, exclude_target_labels=("a",))
    assert "a" not in set(ep2.labels.tolist())
    assert any(labels[r] == "a" for r in ep2.distractors.ravel())       # still usable as a negative
    with pytest.raises(ValueError):
        build_label_episodes(labels, paintings, rows, 10, seed=42, min_paintings_per_label=100,
                             exclude_target_labels=("b",))


def test_builder_is_deterministic():
    labels, paintings, _ = _fixture()
    rows = np.arange(len(labels))
    a = build_label_episodes(labels, paintings, rows, 40, seed=42)
    b = build_label_episodes(labels, paintings, rows, 40, seed=42)
    assert np.array_equal(a.distractors, b.distractors) and np.array_equal(a.labels, b.labels)


def test_tie_aware_rank_counts_ties_against_the_positive():
    assert tie_aware_rank(torch.zeros(2, 13)).tolist() == [7.0, 7.0]
    best = torch.tensor([[5.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
    assert tie_aware_rank(best).tolist() == [1.0, 2.5]


def _label_onehot(labels):
    return (labels[:, None] == np.array(["a", "b", "c"])[None, :]).astype(np.float32)


def test_recall_uses_clip_when_weights_are_zero_and_ties_when_everything_is_zero():
    labels, paintings, rng = _fixture()
    n = len(labels)
    # CLIP features encode the label; the positive is the only same-label candidate.
    img = np.concatenate([3 * _label_onehot(labels), 0.1 * rng.normal(size=(n, 13))], axis=1).astype(np.float32)
    txt = np.concatenate([3 * _label_onehot(labels), 0.1 * rng.normal(size=(n, 13))], axis=1).astype(np.float32)
    codes = np.zeros((n, 4), dtype=np.float32)
    ep = build_label_episodes(labels, paintings, np.arange(n), 100, seed=42)
    zeros = torch.zeros(100, 4)
    clip = label_episode_recall(img, txt, codes, codes, ep, zeros, beta=1.0)
    assert clip["i2t"]["recall1"] >= 0.99 and clip["t2i"]["recall1"] >= 0.99
    tied = label_episode_recall(img, txt, codes, codes, ep, zeros, beta=0.0)
    assert tied["i2t"]["recall1"] == 0.0 and tied["i2t"]["tied_episodes"] == 100


def test_condition_lift_is_positive_when_a_factor_encodes_the_label_and_near_zero_for_noise():
    labels, paintings, rng = _fixture()
    n = len(labels)
    img = rng.normal(size=(n, 16)).astype(np.float32)
    txt = rng.normal(size=(n, 16)).astype(np.float32)
    # 3 label factors + 20 irrelevant factors: uniform weighting is swamped by the irrelevant
    # ones (R@1 roughly 0.1-0.2); naive weighting concentrates on the label factor (R@1 near 1).
    informative = np.concatenate([_label_onehot(labels), rng.random((n, 20)).astype(np.float32)], axis=1)
    noise = rng.random((n, 8)).astype(np.float32)
    ep = build_label_episodes(labels, paintings, np.arange(n), 400, seed=42)
    assert condition_lift(img, txt, informative, informative, ep)["lift_mean"] > 0.3
    assert abs(condition_lift(img, txt, noise, noise, ep)["lift_mean"]) < 0.1


# ---- Fix round 1: clean negatives for per-annotation labels, exclusion validation, non-finite ranks ----

def _per_row_fixture(seed=3, n_paintings=300, rows_per_painting=3):
    """Per-ANNOTATION labels (like emotion): every row's label is drawn independently, so a painting
    usually carries several labels and an other-label row often shows a painting that also has the target."""
    rng = np.random.default_rng(seed)
    paintings = np.repeat(np.arange(n_paintings), rows_per_painting).astype(str)
    labels = rng.choice(np.array(["a", "b", "c"]), size=len(paintings))
    return labels, paintings


def _negatives_with_target_painting(ep, labels, paintings):
    """Count contrast/distractor rows whose painting has ANY row (full label array) labeled as the episode target."""
    bad = 0
    for i in range(len(ep.labels)):
        has_target = set(paintings[labels == ep.labels[i]])
        bad += sum(paintings[r] in has_target for r in [*ep.contrasts[i], *ep.distractors[i]])
    return bad


def test_clean_negatives_flag_removes_target_paintings_from_contrasts_and_distractors():
    labels, paintings = _per_row_fixture()
    n = len(labels)
    for rows in (np.arange(n), np.arange(0, n, 2)):      # the subset proves "target paintings" use the FULL label array
        clean = build_label_episodes(labels, paintings, rows, 200, seed=42,
                                     exclude_target_paintings_from_negatives=True)
        assert _negatives_with_target_painting(clean, labels, paintings) == 0
        assert all(labels[r] != clean.labels[i] for i in range(200)
                   for r in [*clean.contrasts[i], *clean.distractors[i]])
        assert set(np.concatenate([clean.anchor, clean.positive, clean.supports.ravel(),
                                   clean.contrasts.ravel(), clean.distractors.ravel()]).tolist()) <= set(rows.tolist())
        default = build_label_episodes(labels, paintings, rows, 200, seed=42)
        assert _negatives_with_target_painting(default, labels, paintings) > 0    # the flag is what removes them


def test_clean_negatives_flag_default_off_is_identical_to_omitting_it():
    labels, paintings = _per_row_fixture()
    rows = np.arange(len(labels))
    omitted = build_label_episodes(labels, paintings, rows, 60, seed=42)
    explicit = build_label_episodes(labels, paintings, rows, 60, seed=42,
                                    exclude_target_paintings_from_negatives=False)
    for name in ("anchor", "positive", "supports", "contrasts", "distractors", "labels"):
        assert np.array_equal(getattr(omitted, name), getattr(explicit, name))


def test_clean_negatives_raises_when_no_clean_pool_can_fill_an_episode():
    # Every painting carries every label, so no painting is free of the target label.
    paintings = np.repeat(np.arange(100), 3).astype(str)
    labels = np.tile(np.array(["a", "b", "c"]), 100)
    with pytest.raises(ValueError):
        build_label_episodes(labels, paintings, np.arange(len(labels)), 5, seed=42,
                             exclude_target_paintings_from_negatives=True)
    build_label_episodes(labels, paintings, np.arange(len(labels)), 5, seed=42)   # default still builds


def test_exclude_target_labels_must_be_present_in_the_rows():
    labels, paintings, _ = _fixture()
    rows = np.arange(len(labels))
    with pytest.raises(ValueError):
        build_label_episodes(labels, paintings, rows, 10, seed=42, exclude_target_labels=("Something else",))
    with pytest.raises(ValueError):                                     # present in `labels`, absent from `rows`
        build_label_episodes(labels, paintings, np.flatnonzero(labels != "c"), 10, seed=42,
                             exclude_target_labels=("c",))
    ep = build_label_episodes(labels, paintings, rows, 10, seed=42, exclude_target_labels=("c",))
    assert "c" not in set(ep.labels.tolist())


def test_tie_aware_rank_gives_non_finite_rows_the_worst_rank():
    scores = torch.tensor([[float("nan"), 1.0, 2.0], [5.0, 1.0, 2.0], [1.0, float("inf"), 0.0]])
    assert tie_aware_rank(scores).tolist() == [3.0, 1.0, 3.0]
    assert tie_aware_rank(torch.tensor([[1.0, float("-inf"), 0.0]])).tolist() == [3.0]   # any non-finite entry
