import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import text_anchor as ta


def test_direction_from_embeddings_is_unit_norm():
    emb_a = np.array([[1.0, 0.0], [1.0, 0.0]])
    emb_b = np.array([[0.0, 1.0], [0.0, 1.0]])
    d = ta.direction_from_embeddings(emb_a, emb_b)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert np.allclose(d, [1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_direction_from_embeddings_rejects_degenerate():
    emb_a = np.array([[1.0, 2.0]])
    emb_b = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        ta.direction_from_embeddings(emb_a, emb_b)


def test_build_control_directions_uses_three_paraphrases(monkeypatch):
    calls = []

    def fake_build_direction(model, tokenizer, prompts_a, prompts_b, device):
        calls.append((prompts_a, prompts_b, device))
        return np.array([1.0, 0.0])

    monkeypatch.setattr(ta, "build_direction", fake_build_direction)
    ta.build_control_directions(None, None, n=1, seed=42)

    prompts_a, prompts_b, device = calls[0]
    word_a = prompts_a[0].removeprefix("a photo of a ")
    word_b = prompts_b[0].removeprefix("a photo of a ")
    assert prompts_a == [template.format(w=word_a) for template in ta.CONTROL_TEMPLATES]
    assert prompts_b == [template.format(w=word_b) for template in ta.CONTROL_TEMPLATES]
    assert device == "cpu"


@pytest.mark.slow
def test_build_direction_live_clip_smoke():
    model, tokenizer = ta.load_clip_text_tower(device="cpu")
    d = ta.build_direction(model, tokenizer, ["a happy photo"], ["a sad photo"], device="cpu")
    assert d.shape == (512,)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-4)


@pytest.mark.slow
def test_build_control_directions_live_clip_smoke():
    model, tokenizer = ta.load_clip_text_tower(device="cpu")
    dirs = ta.build_control_directions(model, tokenizer, n=3, seed=42, device="cpu")
    assert len(dirs) == 3
    assert all(d.shape == (512,) for d in dirs)
    # deterministic given the fixed seed
    dirs2 = ta.build_control_directions(model, tokenizer, n=3, seed=42, device="cpu")
    assert all(np.allclose(a, b) for a, b in zip(dirs, dirs2))
