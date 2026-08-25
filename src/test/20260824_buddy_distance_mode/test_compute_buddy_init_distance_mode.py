"""
Test: compute_buddy_init's distance_mode parameter. "blend" (the default) must
reproduce the EXACT original fixed-alpha behavior for full backward compatibility;
"typed" must route through Task 1's mix_distances_typed and produce a different,
hand-verifiable result on a graph engineered to have real modality disagreement.

Run:
    python src/test/20260824_buddy_distance_mode/test_compute_buddy_init_distance_mode.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.conditional_buddy.compute_buddies import compute_buddy_init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()


def _two_cluster_features(n_per=60, dim=32, seed=1):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(0, 1, dim)
    c1 = rng.normal(5, 1, dim)
    labels = np.array([0] * n_per + [1] * n_per)
    centers = np.stack([c0, c1])
    img = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    txt = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    return img.astype(np.float32), txt.astype(np.float32)


def test_blend_default_matches_no_arg_call():
    """distance_mode='blend' (explicit) must be numerically identical to calling
    compute_buddy_init with no distance_mode argument at all -- the core backward-
    compatibility property this whole plan depends on."""
    img, txt = _two_cluster_features(seed=2)
    emb_no_arg = compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF, seed=42)
    emb_explicit_blend = compute_buddy_init(
        img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF, seed=42, distance_mode="blend"
    )
    np.testing.assert_allclose(emb_no_arg, emb_explicit_blend, atol=1e-6)
    print("PASS test_blend_default_matches_no_arg_call")


def test_invalid_distance_mode_raises():
    img, txt = _two_cluster_features(seed=3)
    try:
        compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF,
                            distance_mode="bogus")
        raise AssertionError("expected ValueError for invalid distance_mode")
    except ValueError as e:
        assert "distance_mode" in str(e), str(e)
        print("PASS test_invalid_distance_mode_raises")


def test_typed_mode_changes_output_on_engineered_disagreement():
    """On a graph engineered to have real image/text disagreement (scramble some text
    rows relative to their images), 'typed' must produce a DIFFERENT embedding than
    'blend' -- if identical, the new code path was not actually exercised."""
    img, txt = _two_cluster_features(seed=4)
    rng = np.random.default_rng(5)
    scramble = rng.permutation(len(txt))[:20]
    txt = txt.copy()
    txt[scramble] = txt[scramble][::-1]

    emb_blend = compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF,
                                    seed=42, distance_mode="blend")
    emb_typed = compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF,
                                    seed=42, distance_mode="typed")
    assert not np.allclose(emb_blend, emb_typed), (
        "typed mode produced identical output to blend -- new code path not exercised"
    )
    print("PASS test_typed_mode_changes_output_on_engineered_disagreement")


if __name__ == "__main__":
    test_blend_default_matches_no_arg_call()
    test_invalid_distance_mode_raises()
    test_typed_mode_changes_output_on_engineered_disagreement()
    print("ALL TESTS PASSED")
