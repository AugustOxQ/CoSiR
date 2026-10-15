"""Focused checks for the fixed-episode cross-seed analyzer."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.sparse import csr_matrix


PATH = Path(__file__).with_name("q3_cross_seed.py")
SPEC = importlib.util.spec_from_file_location("q3_cross_seed", PATH)
q3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q3)


def _episode(start: int) -> SimpleNamespace:
    ids = list(range(start, start + 22))
    return SimpleNamespace(
        anchor_idx=ids[0], targeted_factor=0,
        support_idxs=ids[1:5], contrast_idxs=ids[5:9], positive_idx=ids[9],
        hard_negative_idxs=ids[10:14], condition_distractor_idxs=ids[14:18],
        anchor_distractor_idxs=ids[18:22],
    )


def test_half_rank_ties_and_alignment():
    scores = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [2, 1, 0, 0]])
    summary = q3._rank_summary(scores)
    assert summary == {"r1": 1 / 3, "r3": 1.0, "tie_episodes": 2,
                       "all_tied_episodes": 1}
    a = np.random.default_rng(3).normal(size=(10, 2)).astype(np.float32)
    b = a[:, ::-1]
    aligned = q3._alignment(a, a, b, b, np.arange(10))
    assert np.allclose(aligned["per_seed42_factor_max_abs_pearson"], 1)
    assert aligned["matched_seed43_factor"] == [1, 0]


def test_analyzer_uses_seed43_and_fixed_episodes(monkeypatch):
    rng = np.random.default_rng(5)
    features = rng.normal(size=(44, 2)).astype(np.float32)
    codes42 = features.copy()
    episodes = [_episode(0)]
    held_episodes = [_episode(22)]
    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def encode_image(self, x):
            return x

        def encode_text(self, x):
            return x

    def fake_train(img, txt, graph, config):
        seen["config"] = config
        seen["train_rows"] = len(img)
        return FakeModel(), img.copy(), txt.copy()

    monkeypatch.setattr(q3, "build_content_graph", lambda i, t, c: csr_matrix(
        ([1, 1], ([0, 1], [1, 0])), shape=(len(i), len(i))
    ))
    monkeypatch.setattr(q3, "train_factors", fake_train)
    result = q3.analyze_cross_seed(
        features, features, codes42, codes42,
        np.arange(22), np.arange(22, 44), episodes, held_episodes,
        [], (0.0, 0.1),
    )
    assert seen["config"].seed == 43
    assert seen["config"].epochs == 2000
    assert seen["config"].lambda_usage_balance == 0.1
    assert seen["train_rows"] == 22
    assert result["train_episodes"] == result["held_episodes"] == 1
    assert result["held_beta_zero_ranking"]["clip_only"]["i2t"]["all_tied_episodes"] == 1
    assert result["selected_swap"]["naive"]["i2t"]["reversal_rate"] is None
