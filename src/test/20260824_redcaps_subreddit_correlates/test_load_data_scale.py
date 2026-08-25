"""
Test: redcaps_buddy.load_data() accepts optional storage_dir/annotation_path overrides
so the subreddit-correlates analysis can run at 300k/500k/1M/full RedCaps scales, while
staying fully backward-compatible for every existing caller that relies on the 150k default
(cross_vlm_buddy.py, phase2_vlm.py, run_phase1.py, run_structure.py, analyze_subreddit_correlates.py).

Run:
    python src/test/20260824_redcaps_subreddit_correlates/test_load_data_scale.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

import numpy as np

import redcaps_buddy as rb

# NOTE: this is the uniform-random, all-350-subreddit "diverse" sample built by
# build_subsample.py — NOT Experiment 1's same-named training feature store
# (/data/SSD2/pre_extract/redcaps_300k/features), which is a raw prefix of a
# subreddit-grouped file and spans only 15 of 350 subreddits. See
# docs/reports/2026-08-24_redcaps_subreddit_signal_correlates.md's "Multi-scale
# extension" section for why the two are not interchangeable.
SCALE_300K = dict(
    storage_dir="/data/SSD2/pre_extract/redcaps_300k_diverse/features",
    annotation_path="/data/PDD/redcaps/redcaps_plus/redcaps_300k_diverse.json",
)


def test_default_unchanged():
    """load_data() with no args must be identical to calling it with the module's own
    STORAGE/ANNOT constants explicitly -- backward compatibility for every existing caller."""
    data_default = rb.load_data()
    data_explicit = rb.load_data(storage_dir=rb.STORAGE, annotation_path=rb.ANNOT)
    assert data_default.sample_ids == data_explicit.sample_ids
    assert data_default.n == data_explicit.n == 150000
    np.testing.assert_array_equal(data_default.img, data_explicit.img)
    np.testing.assert_array_equal(data_default.sub_id, data_explicit.sub_id)
    print(f"PASS test_default_unchanged ({data_default.n} rows)")


def test_300k_scale_loads_correctly():
    """load_data() at the 300k scale returns 300k rows, with a real (non-degenerate)
    subreddit label distribution -- confirms the override path actually re-points both
    the FeatureManager AND the metadata JSON, not just one of the two."""
    data = rb.load_data(**SCALE_300K)
    assert data.n == 300000, f"expected 300000 rows, got {data.n}"
    assert len(data.sub_names) > 50, (
        f"expected a real multi-subreddit distribution, got {len(data.sub_names)} names "
        "-- possible sign the annotation_path override didn't take effect"
    )
    # sample-ID consistency: data.sample_ids must match the 300k FeatureManager's own order
    from src.utils import FeatureManager
    fm = FeatureManager(SCALE_300K["storage_dir"])
    assert data.sample_ids == list(fm.get_all_sample_ids())
    print(f"PASS test_300k_scale_loads_correctly ({data.n} rows, {len(data.sub_names)} subreddits)")


if __name__ == "__main__":
    test_default_unchanged()
    test_300k_scale_loads_correctly()
    print("ALL TESTS PASSED")
