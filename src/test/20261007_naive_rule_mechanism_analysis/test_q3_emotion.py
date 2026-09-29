"""Small synthetic checks for human-label episode roles and Q3b output."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location("q3_emotion", Path(__file__).with_name("q3_emotion.py"))
q3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = q3
SPEC.loader.exec_module(q3)


def test_emotion_episode_shapes_and_painting_disjointness(monkeypatch):
    rng = np.random.default_rng(123)
    # Two annotations of each painting; two emotions; each split has 25
    # distinct paintings per emotion, enough for every full 22-role episode.
    paintings = np.repeat(np.arange(100), 2)
    emotions = np.repeat(np.where(np.arange(100) % 2 == 0, "awe", "sadness"), 2)
    img = rng.normal(size=(200, 8)).astype(np.float32)
    txt = rng.normal(size=(200, 8)).astype(np.float32)
    codes_i = rng.random(size=(200, 4)).astype(np.float32)
    codes_t = rng.random(size=(200, 4)).astype(np.float32)
    train_items, held_items = np.arange(100), np.arange(100, 200)
    episodes = q3.mine_emotion_episodes(img, txt, held_items, emotions, paintings, 8)
    for ep in episodes:
        for direction in q3.DIRECTIONS:
            roles = [ep.anchor_idx, *ep.support_idxs, *ep.contrast_idxs,
                     ep.positive_idx, *ep.random_idxs, *ep.nearest_idxs[direction]]
            assert len(roles) == 22
            assert len(np.unique(paintings[roles])) == 22
            assert all(emotions[i] == ep.emotion for i in
                       [ep.anchor_idx, *ep.support_idxs, ep.positive_idx])
            assert all(emotions[i] != ep.emotion for i in
                       [*ep.contrast_idxs, *ep.random_idxs, *ep.nearest_idxs[direction]])
    monkeypatch.setattr(q3, "_load_labels", lambda n: (emotions, paintings))
    monkeypatch.setattr(q3, "TRAIN_EPISODES", 8)
    monkeypatch.setattr(q3, "HELD_EPISODES", 8)
    monkeypatch.setattr(q3, "BOOTSTRAPS", 20)
    result = q3.analyze_emotion(img, txt, codes_i, codes_t, train_items, held_items)
    assert result["metadata"]["held_episodes"] == 8
    assert sum(result["metadata"]["held_emotion_counts"].values()) == 8
    assert set(result["held_recall"]) == set(q3.VARIANTS)
    json.dumps(result, allow_nan=False)


def test_beta_zero_midrank_for_all_tied_pool():
    cosine = np.ones((2, 13), dtype=np.float32)
    factor = np.zeros_like(cosine)
    r1, r3 = q3._recall(cosine, factor, 0.0)
    assert r1.tolist() == [0, 0]
    assert r3.tolist() == [0, 0]


def test_clip_only_selects_positive_beta_when_train_recall_ties_at_zero():
    cosine = np.ones((2, 13), dtype=np.float32)
    cosine[:, 0] = 0
    factor = np.zeros_like(cosine)
    train = {direction: {variant: (cosine, factor) for variant in q3.VARIANTS}
             for direction in q3.DIRECTIONS}
    selected, _ = q3._select_betas(train)
    assert selected["clip_only"] == 0.001
