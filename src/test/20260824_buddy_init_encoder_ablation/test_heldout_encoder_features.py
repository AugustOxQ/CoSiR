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
    """The guard under test lives in load_encoder_pair_features's `assert data.sample_ids ==
    fm_ids` (heldout_encoder_features.py). NOTE: the AssertionError raised by that guard must
    be distinguished from "no exception was raised at all" -- an earlier version of this test
    wrapped the call in try/except AssertionError with a raise AssertionError(...) as its own
    "did not raise" signal INSIDE the try block, which its own except then silently swallowed
    (the message happened to contain "sample" too), making the test pass even with the real
    guard deleted. This version uses an explicit `raised` flag set only inside the except
    block, so a silently-not-raising code path is distinguishable from the real guard firing.
    """
    class _FakeFM:
        def get_all_sample_ids(self):
            return [0, 1, 2]  # deliberately wrong length/order
    raised = False
    try:
        load_encoder_pair_features("redcaps", "clip_img", "clip_txt", _FakeFM())
    except AssertionError as e:
        raised = True
        assert "sample" in str(e).lower()
    assert raised, "expected AssertionError on sample-id mismatch, none was raised"
    print("PASS test_mismatched_sample_ids_raises")


def test_stale_heldout_cache_row_count_raises():
    """Important Finding #2 (final review): the sample-id assert above only compares two reads
    of the SAME CLIP-backed store (dataset.load_data() vs. feature_manager) -- it never touches
    the actual held-out .npy cache file being loaded for a non-CLIP encoder. A stale cache with
    the same N but a different row order (e.g. regenerated from a re-shuffled annotation file)
    would previously pass silently. This test truncates a real held-out cache file by one row
    and confirms the new row-count assert in heldout_encoder_features.py's _load fires -- using
    the same real, distinguishable-failure pattern as test_mismatched_sample_ids_raises (an
    explicit `raised` flag), not a vacuous try/except.
    """
    import shutil
    import tempfile

    from extract_heldout import cache_path  # src/test/20260708_heldout_grid

    fm = FeatureManager(rb.STORAGE)
    real_path = cache_path("redcaps", "dinov2", 0)
    if not os.path.exists(real_path):
        print("SKIP test_stale_heldout_cache_row_count_raises (no dinov2 held-out cache on disk)")
        return

    tmp_dir = tempfile.mkdtemp(prefix="test_stale_heldout_cache_")
    try:
        # Point cache_path's underlying lookup at a truncated copy by monkeypatching the
        # module-level function heldout_encoder_features imports at call time.
        import extract_heldout

        real_arr = np.load(real_path)
        truncated_path = os.path.join(tmp_dir, "truncated.npy")
        np.save(truncated_path, real_arr[:-1])  # one row short of the real cache

        orig_cache_path = extract_heldout.cache_path

        def _fake_cache_path(dataset, model, smoke):
            if model == "dinov2":
                return truncated_path
            return orig_cache_path(dataset, model, smoke)

        extract_heldout.cache_path = _fake_cache_path
        try:
            raised = False
            try:
                load_encoder_pair_features("redcaps", "dinov2", "clip_txt", fm)
            except AssertionError as e:
                raised = True
                assert "rows" in str(e).lower() or "row" in str(e).lower()
            assert raised, "expected AssertionError on truncated held-out cache, none was raised"
            print("PASS test_stale_heldout_cache_row_count_raises")
        finally:
            extract_heldout.cache_path = orig_cache_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_clip_pair_matches_load_data()
    test_nonclip_pair_shape_and_alignment()
    test_mismatched_sample_ids_raises()
    test_stale_heldout_cache_row_count_raises()
    print("ALL TESTS PASSED")
