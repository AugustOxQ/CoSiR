"""
Test: subreddit_lift(..., top_k=None) returns every subreddit passing the exp_s > 5
reliability filter, not just the top 15 — and top_k=15 (the existing default) is unchanged.

Run:
    python src/test/20260824_redcaps_subreddit_correlates/test_subreddit_lift_all.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

import numpy as np

from redcaps_buddy import Data, subreddit_lift


def _synthetic_data(n_subs=20, per_sub=30, seed=1):
    """n_subs subreddits x per_sub samples each; every subreddit gets some same-sub edges
    so all pass the exp_s > 5 filter."""
    rng = np.random.default_rng(seed)
    n = n_subs * per_sub
    sub_id = np.repeat(np.arange(n_subs), per_sub)
    img = np.zeros((n, 4), dtype=np.float32)  # unused by subreddit_lift
    txt = np.zeros((n, 4), dtype=np.float32)
    sample_ids = list(range(n))
    sub_names = [f"sub{i}" for i in range(n_subs)]
    records = [{} for _ in range(n)]
    data = Data(img, txt, sample_ids, sub_id, sub_names, records)

    # Build enough same-subreddit edges per subreddit to clear exp_s > 5, with varying
    # density per subreddit so lift genuinely differs across subreddits.
    edges = []
    for s in range(n_subs):
        idx = np.where(sub_id == s)[0]
        n_edges = 100 + s * 10  # increasing density -> DECREASING lift by construction:
        # every edge here is same-subreddit, so obs_s ~ deg_s ~ 2*n_edges and
        # exp_s = deg_s * p_s = deg_s^2 / total_endpoints, giving lift_s = obs_s/exp_s
        # ~ total_endpoints / deg_s — inversely proportional to this subreddit's own
        # edge density, not increasing with it.
        pairs = rng.choice(idx, size=(n_edges, 2))
        edges.extend(pairs.tolist())
    e = np.array(edges, dtype=np.int64)
    return data, e


def test_top_k_none_returns_all_qualifying():
    data, e = _synthetic_data(n_subs=20)
    result_all = subreddit_lift(data, e, top_k=None)
    result_15 = subreddit_lift(data, e, top_k=15)
    assert len(result_15["top_enriched"]) == 15
    assert len(result_all["top_enriched"]) == 20, (
        f"expected exactly 20 qualifying subreddits (the synthetic fixture guarantees "
        f"all 20 clear exp_s > 5), got {len(result_all['top_enriched'])}"
    )
    # The top-15 (by lift, descending) from the top_k=None result must exactly match the
    # top_k=15 result — same ranking, just not truncated.
    names_all_top15 = [name for name, _, _ in result_all["top_enriched"][:15]]
    names_15 = [name for name, _, _ in result_15["top_enriched"]]
    assert names_all_top15 == names_15, (names_all_top15, names_15)
    print(f"PASS test_top_k_none_returns_all_qualifying "
          f"({len(result_all['top_enriched'])} qualifying subreddits)")


def test_default_top_k_unchanged():
    data, e = _synthetic_data(n_subs=20)
    result = subreddit_lift(data, e)  # no top_k passed -> must still default to 15
    assert len(result["top_enriched"]) == 15
    print("PASS test_default_top_k_unchanged")


if __name__ == "__main__":
    test_top_k_none_returns_all_qualifying()
    test_default_top_k_unchanged()
    print("ALL TESTS PASSED")
