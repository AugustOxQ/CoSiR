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
