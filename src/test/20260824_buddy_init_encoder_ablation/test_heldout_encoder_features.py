"""
Test: load_encoder_pair_features loads clip_img x clip_txt identically to RedCaps'
own load_data(), and raises when sample-id order doesn't match feature_manager.

Requires the RedCaps-150k FeatureManager on disk (STORAGE in redcaps_buddy.py) and the
held-out feature cache from src/test/20260708_heldout_grid/extract_heldout.py (at least
one non-CLIP model, e.g. dinov2, already extracted — the survival study already did this).

Run:
    python src/test/20260824_buddy_init_encoder_ablation/test_heldout_encoder_features.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

import numpy as np

from src.conditional_buddy.heldout_encoder_features import load_encoder_pair_features
from src.utils import FeatureManager
import redcaps_buddy as rb


def test_clip_pair_matches_load_data():
    fm = FeatureManager(rb.STORAGE)
    img, txt, sample_ids = load_encoder_pair_features("redcaps", "clip_img", "clip_txt", fm)
    data = rb.load_data()
    assert sample_ids == data.sample_ids
    np.testing.assert_allclose(img, data.img, atol=1e-5)
    np.testing.assert_allclose(txt, data.txt, atol=1e-5)
    print(f"PASS test_clip_pair_matches_load_data ({len(sample_ids)} rows)")


def test_nonclip_pair_shape_and_alignment():
    fm = FeatureManager(rb.STORAGE)
    img, txt, sample_ids = load_encoder_pair_features("redcaps", "dinov2", "clip_txt", fm)
    data = rb.load_data()
    assert sample_ids == data.sample_ids
    assert img.shape[0] == len(sample_ids)
    np.testing.assert_allclose(txt, data.txt, atol=1e-5)  # clip_txt side unchanged
    print(f"PASS test_nonclip_pair_shape_and_alignment (img dim={img.shape[1]})")


def test_mismatched_sample_ids_raises():
    class _FakeFM:
        def get_all_sample_ids(self):
            return [0, 1, 2]  # deliberately wrong length/order
    try:
        load_encoder_pair_features("redcaps", "clip_img", "clip_txt", _FakeFM())
        raise AssertionError("expected AssertionError on sample-id mismatch")
    except AssertionError as e:
        assert "sample" in str(e).lower()
        print("PASS test_mismatched_sample_ids_raises")


if __name__ == "__main__":
    test_clip_pair_matches_load_data()
    test_nonclip_pair_shape_and_alignment()
    test_mismatched_sample_ids_raises()
    print("ALL TESTS PASSED")
