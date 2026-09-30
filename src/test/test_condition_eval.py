from types import SimpleNamespace

import numpy as np
import torch

from src.eval.condition_eval import (
    build_human_swap_episodes, ceiling_ranks, condition_use_gain, human_swap_success, label_ranks,
    paired_bootstrap, swap_success_difference, wrong_condition,
)
from src.eval.label_episodes import LabelEpisodes, build_label_episodes, label_episodes_sha256
from src.model.condition_interface import ConditionalScorer, ResidualConditionInterface


def _episodes(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return LabelEpisodes(anchor=np.arange(n), positive=np.arange(n) + 100,
                         supports=rng.integers(0, 1000, (n, 4)), contrasts=rng.integers(0, 1000, (n, 4)),
                         distractors=rng.integers(0, 1000, (n, 12)), labels=np.array(["a"] * n))


def test_wrong_condition_is_a_derangement_and_keeps_everything_else():
    ep = _episodes()
    wrong = wrong_condition(ep, seed=42)
    assert np.array_equal(wrong.anchor, ep.anchor) and np.array_equal(wrong.distractors, ep.distractors)
    same = [(wrong.supports[i] == ep.supports[i]).all() for i in range(50)]
    assert not any(same)
    assert np.array_equal(wrong_condition(ep, seed=42).supports, wrong.supports)


def test_paired_bootstrap_and_condition_use_gain_arithmetic():
    out = paired_bootstrap(np.full(40, 0.25))
    assert out["point"] == 0.25 and out["ci95"] == [0.25, 0.25]
    ranks = {"i2t": np.array([1, 1, 2, 1.0]), "t2i": np.array([1, 2, 2, 2.0])}
    wrong = {"i2t": np.array([2, 2, 2, 2.0]), "t2i": np.array([2, 2, 2, 2.0])}
    naive = {"i2t": np.array([1, 2, 2, 2.0]), "t2i": np.array([2, 2, 2, 2.0])}
    gain = condition_use_gain(ranks, wrong, naive, wrong, n_boot=200)
    assert gain["i2t"]["point"] == 0.5 and gain["t2i"]["point"] == 0.25 and gain["mean"]["point"] == 0.375


def test_label_episode_hash_is_stable_and_sensitive():
    ep = _episodes()
    h = label_episodes_sha256(ep)
    assert h == label_episodes_sha256(_episodes()) and len(h) == 64
    changed = LabelEpisodes(**{**ep.__dict__, "positive": ep.positive + 1})
    assert label_episodes_sha256(changed) != h


def _art_world(seed=0):
    """120 paintings x 3 annotations; style per painting; emotion per annotation (mixed within paintings)."""
    rng = np.random.default_rng(seed)
    n_paint = 120
    styles_p = np.array(["s0", "s1", "s2", "s3"])[np.arange(n_paint) % 4]
    groups = np.repeat(np.arange(n_paint), 3)
    styles = styles_p[groups]
    emotions = np.array(["awe", "fear", "sadness", "something else"])[rng.integers(0, 4, len(groups))]
    return SimpleNamespace(emotions=emotions, art_styles=styles), groups


def test_human_swap_episodes_are_one_aspect_clean():
    data, groups = _art_world()
    rows = np.arange(len(groups))
    ep = build_human_swap_episodes(data, groups, rows, 60, seed=42, min_paintings=5)
    emo_paintings = {e: set(groups[data.emotions == e]) for e in np.unique(data.emotions)}
    for i in range(60):
        a = ep.anchor[i]
        e, s = data.emotions[a], data.art_styles[a]
        assert e != "something else" and ep.emotions[i] == e and ep.styles[i] == s
        p_emo, p_style, negs = ep.candidates[i, 0], ep.candidates[i, 1], ep.candidates[i, 2:]
        assert data.emotions[p_emo] == e and data.art_styles[p_emo] != s
        assert data.art_styles[p_style] == s and groups[p_style] not in emo_paintings[e]
        assert all(data.art_styles[r] != s and groups[r] not in emo_paintings[e] for r in negs)
        assert all(data.emotions[r] == e and data.art_styles[r] != s for r in ep.supports_emo[i])
        assert all(data.art_styles[r] == s and groups[r] not in emo_paintings[e] for r in ep.supports_style[i])
        assert all(groups[r] not in emo_paintings[e] for r in ep.contrasts_emo[i])
        assert all(data.art_styles[r] != s for r in ep.contrasts_style[i])
        members = [a, *ep.supports_emo[i], *ep.contrasts_emo[i], *ep.supports_style[i], *ep.contrasts_style[i],
                   *ep.candidates[i]]
        assert len({groups[r] for r in members}) == len(members)


def test_human_swap_success_counts_ties_as_failure_and_detects_a_correct_flip():
    data, groups = _art_world(1)
    rows = np.arange(len(groups))
    ep = build_human_swap_episodes(data, groups, rows, 20, seed=0, min_paintings=5)
    n = len(groups)
    feats = np.random.default_rng(2).normal(size=(n, 6)).astype(np.float32)
    zero = np.zeros((n, 8), dtype=np.float32)
    naive = ConditionalScorer(ResidualConditionInterface(torch.ones(8)), beta_init=1e-6)
    zero_feats = np.zeros_like(feats)                              # all scores exactly 0 -> every pair tied
    tied = human_swap_success(naive, zero_feats, zero_feats, zero, zero, ep)
    assert not tied["i2t"].any() and not tied["t2i"].any()
    codes = np.zeros((n, 8), dtype=np.float32)                      # factor 0-3: emotion, 4-7: style
    for j, e in enumerate(["awe", "fear", "sadness", "something else"]):
        codes[data.emotions == e, j] = 1.0
    for j, s in enumerate(["s0", "s1", "s2", "s3"]):
        codes[data.art_styles == s, 4 + j] = 1.0
    good = human_swap_success(naive, feats, feats, codes, codes, ep)
    assert good["i2t"].mean() > 0.9 and good["t2i"].mean() > 0.9
    diff = swap_success_difference(good, tied, n_boot=200)
    assert diff["pooled"]["point"] > 0.9


def test_ceiling_reaches_perfect_recall_when_one_factor_separates_the_positive():
    n = 400
    rng = np.random.default_rng(3)
    codes = rng.random((n, 8)).astype(np.float32) * 0.1
    ep = _episodes(n=30, seed=4)
    ep = LabelEpisodes(**{**ep.__dict__, "positive": np.arange(30) + 200, "distractors": rng.integers(250, 400, (30, 12))})
    codes[ep.anchor, 0] = 1.0
    codes[ep.positive, 0] = 1.0
    feats = rng.normal(size=(n, 6)).astype(np.float32)
    ranks = ceiling_ranks(feats, feats, codes, codes, ep, beta=0.0)
    assert (ranks["i2t"] == 1).all() and (ranks["t2i"] == 1).all()


def test_label_ranks_step_zero_scorer_matches_naive_recall():
    from src.eval.label_episodes import label_episode_recall, label_episode_weights
    rng = np.random.default_rng(5)
    n = 500
    labels = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    paintings = np.arange(n)
    ep = build_label_episodes(labels, paintings, np.arange(n), 40, seed=1, min_paintings_per_label=5)
    codes = rng.random((n, 8)).astype(np.float32)
    feats = rng.normal(size=(n, 6)).astype(np.float32)
    scorer = ConditionalScorer(ResidualConditionInterface(torch.ones(8)), beta_init=0.3)
    ours = label_ranks(scorer, feats, feats, codes, codes, ep)
    ref = label_episode_recall(feats, feats, codes, codes, ep, label_episode_weights(codes, codes, ep), 0.3)
    assert np.allclose(ours["i2t"], ref["i2t"]["ranks"]) and np.allclose(ours["t2i"], ref["t2i"]["ranks"])
