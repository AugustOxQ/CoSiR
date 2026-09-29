"""Focused checks for Task 9 head evaluation helpers."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from src.model.condition import ConditionEncoder


_SPEC = importlib.util.spec_from_file_location("q4_head", Path(__file__).with_name("q4_head.py"))
q4_head = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(q4_head)


def test_l1_weights_preserve_zero_rows():
    weights = np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(q4_head.l1_weights(weights), [[0.4, 0.6], [0.0, 0.0]])


def test_positive_ranks_use_task7_strict_greater_rule():
    scores = np.array([[1.0, 1.0, 2.0], [3.0, 2.0, 1.0]])
    assert q4_head.positive_ranks(scores).tolist() == [2, 1]


def test_regression_r2_centers_each_factor_separately():
    target = np.array([[0.0, 10.0], [1.0, 11.0]])
    prediction = np.array([[0.0, 10.0], [0.0, 10.0]])
    assert q4_head.regression_r2(prediction, target) == -1.0


def test_analyze_heads_smoke_with_short_training(monkeypatch):
    monkeypatch.setattr(q4_head, "EPOCHS", 2)
    monkeypatch.setattr(q4_head, "CHECKPOINTS", frozenset((0, 2)))
    rng = np.random.default_rng(42)
    img_features = rng.normal(size=(40, 6)).astype(np.float32)
    txt_features = rng.normal(size=(40, 6)).astype(np.float32)
    img_codes = rng.random(size=(40, 3)).astype(np.float32)
    txt_codes = rng.random(size=(40, 3)).astype(np.float32)

    def episodes(offset):
        out = []
        for i in range(9):
            ids = (np.arange(22) + i + offset) % 40
            out.append(SimpleNamespace(
                anchor_idx=int(ids[0]), support_idxs=ids[1:5].tolist(),
                contrast_idxs=ids[5:9].tolist(), positive_idx=int(ids[9]),
                hard_negative_idxs=ids[10:14].tolist(),
                condition_distractor_idxs=ids[14:18].tolist(),
                anchor_distractor_idxs=ids[18:22].tolist(),
                targeted_factor=i % 3,
            ))
        return out

    result = q4_head.analyze_heads(img_features, txt_features, img_codes, txt_codes,
                                   episodes(0), episodes(10), [(0, 1)], 0.3,
                                   ConditionEncoder())
    assert [point["epoch"] for point in result["q4b"]["checkpoints"]] == [0, 2]
    assert result["q4c"]["ranking"]["i2t"]["count1"] <= 9
    json.dumps(result, allow_nan=False)
