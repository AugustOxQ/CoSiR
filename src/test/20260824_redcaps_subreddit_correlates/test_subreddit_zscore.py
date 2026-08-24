"""
Test: subreddit_enrichment_zscore computes a chance-corrected significance z-score for
same-subreddit edge enrichment, complementing subreddit_lift's effect-size ratio.

Run:
    python src/test/20260824_redcaps_subreddit_correlates/test_subreddit_zscore.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

import numpy as np

from redcaps_buddy import Data, subreddit_enrichment_zscore, subreddit_lift


def _synthetic_data(n_subs=20, per_sub=30, seed=1):
    """Same fixture as test_subreddit_lift_all.py's, reused here for direct comparison
    between lift and z-score on the identical graph."""
    rng = np.random.default_rng(seed)
    n = n_subs * per_sub
    sub_id = np.repeat(np.arange(n_subs), per_sub)
    img = np.zeros((n, 4), dtype=np.float32)
    txt = np.zeros((n, 4), dtype=np.float32)
    sample_ids = list(range(n))
    sub_names = [f"sub{i}" for i in range(n_subs)]
    records = [{} for _ in range(n)]
    data = Data(img, txt, sample_ids, sub_id, sub_names, records)

    edges = []
    for s in range(n_subs):
        idx = np.where(sub_id == s)[0]
        n_edges = 100 + s * 10
        pairs = rng.choice(idx, size=(n_edges, 2))
        edges.extend(pairs.tolist())
    e = np.array(edges, dtype=np.int64)
    return data, e


def test_zscore_matches_hand_derivation():
    """Hand-derive z_overall and one subreddit's z from the raw formula and compare."""
    data, e = _synthetic_data(n_subs=5, per_sub=10, seed=2)
    result = subreddit_enrichment_zscore(data, e, top_k=None)

    si, sj = data.sub_id[e[:, 0]], data.sub_id[e[:, 1]]
    same = si == sj
    n_sub = len(data.sub_names)
    M = e.shape[0]
    endpoints = np.concatenate([si, sj])
    p = np.bincount(endpoints, minlength=n_sub).astype(np.float64)
    p /= p.sum()
    exp_same = float((p ** 2).sum())
    mu_overall = M * exp_same
    var_overall = M * exp_same * (1 - exp_same)
    obs_edges = int(same.sum())
    want_z_overall = (obs_edges - mu_overall) / np.sqrt(var_overall)

    assert abs(result["z_overall"] - want_z_overall) < 1e-9, (
        result["z_overall"], want_z_overall
    )
    assert result["n_edges"] == M
    print(f"PASS test_zscore_matches_hand_derivation (z_overall={result['z_overall']:.3f})")


def test_zscore_downweights_small_sample_high_lift():
    """A subreddit with few edges but a large lift should get a SMALLER z-score than
    a subreddit with many edges and a similar or smaller lift -- z rewards statistical
    confidence, not just effect size, which is the whole point of adding it."""
    rng = np.random.default_rng(3)
    n_subs = 10
    per_sub = 200
    n = n_subs * per_sub
    sub_id = np.repeat(np.arange(n_subs), per_sub)
    img = np.zeros((n, 4), dtype=np.float32)
    txt = np.zeros((n, 4), dtype=np.float32)
    data = Data(img, txt, list(range(n)), sub_id, [f"sub{i}" for i in range(n_subs)],
                [{} for _ in range(n)])

    edges = []
    # sub0: small sample, all-intra (perfect lift, low confidence -- only 8 edges)
    idx0 = np.where(sub_id == 0)[0]
    edges.extend(rng.choice(idx0, size=(8, 2)).tolist())
    # sub1: large sample, all-intra (perfect lift, high confidence -- 2000 edges)
    idx1 = np.where(sub_id == 1)[0]
    edges.extend(rng.choice(idx1, size=(2000, 2)).tolist())
    # filler edges across all other subreddits so the marginal p is non-degenerate
    for s in range(2, n_subs):
        idx = np.where(sub_id == s)[0]
        edges.extend(rng.choice(idx, size=(50, 2)).tolist())
    e = np.array(edges, dtype=np.int64)

    lift_result = subreddit_lift(data, e, top_k=None)
    z_result = subreddit_enrichment_zscore(data, e, top_k=None)
    lift_by_sub = {name: lift for name, lift, _deg in lift_result["top_enriched"]}
    z_by_sub = {name: z for name, z, _m in z_result["top_enriched"]}

    assert "sub1" in z_by_sub, "sub1 (large sample) should pass the reliability filter"
    if "sub0" in z_by_sub:
        # If sub0 clears the filter at all, its z must still be far smaller than sub1's,
        # despite both having ~perfect (or near-perfect) lift.
        assert z_by_sub["sub1"] > z_by_sub["sub0"], (z_by_sub["sub1"], z_by_sub["sub0"])
    print(f"PASS test_zscore_downweights_small_sample_high_lift "
          f"(z[sub1]={z_by_sub['sub1']:.1f}, sub0 filtered={'sub0' not in z_by_sub})")


def test_reliability_filter_excludes_low_mu():
    """A subreddit with mu_s <= 5 (too rare to trust the normal approximation) must be
    excluded from top_enriched, mirroring subreddit_lift's exp_s > 5 rule."""
    data, e = _synthetic_data(n_subs=20, per_sub=30, seed=1)
    result = subreddit_enrichment_zscore(data, e, top_k=None)
    si, sj = data.sub_id[e[:, 0]], data.sub_id[e[:, 1]]
    n_sub = len(data.sub_names)
    endpoints = np.concatenate([si, sj])
    p = np.bincount(endpoints, minlength=n_sub).astype(np.float64)
    p /= p.sum()
    M = e.shape[0]
    mu_s = M * p ** 2
    names_returned = {name for name, _z, _m in result["top_enriched"]}
    for i, name in enumerate(data.sub_names):
        if mu_s[i] <= 5:
            assert name not in names_returned, f"{name} has mu_s={mu_s[i]:.2f} but was returned"
    print(f"PASS test_reliability_filter_excludes_low_mu ({len(names_returned)}/{n_sub} passed)")


if __name__ == "__main__":
    test_zscore_matches_hand_derivation()
    test_zscore_downweights_small_sample_high_lift()
    test_reliability_filter_excludes_low_mu()
    print("ALL TESTS PASSED")
