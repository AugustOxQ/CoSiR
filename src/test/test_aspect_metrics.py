import numpy as np
import pytest

from src.eval.aspect_metrics import cluster_bootstrap, compare, first_place, per_anchor, summarize


def _scores(a_i2t, b_i2t, a_t2i=None, b_t2i=None):
    a_t2i = a_i2t if a_t2i is None else a_t2i
    b_t2i = b_i2t if b_t2i is None else b_t2i
    return {"a": {"i2t": np.asarray(a_i2t, float), "t2i": np.asarray(a_t2i, float)},
            "b": {"i2t": np.asarray(b_i2t, float), "t2i": np.asarray(b_t2i, float)}}


def test_condition_blind_scorer_has_zero_gain():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(500, 13))
    m = per_anchor(_scores(s, s))                    # identical under both conditions
    assert np.allclose(m["gain"], 0.0) and m["swap"].sum() == 0


def test_perfect_conditional_scorer():
    s_a = np.zeros((10, 13)); s_a[:, 0] = 1.0
    s_b = np.zeros((10, 13)); s_b[:, 1] = 1.0
    m = per_anchor(_scores(s_a, s_b))
    assert np.allclose(m["r1"], 1) and np.allclose(m["gain"], 1) and np.allclose(m["strict"], 1)


def test_aspect_finder_ignoring_condition_gets_half_r1_zero_gain():
    s = np.zeros((10, 13)); s[:, 0] = 1.0; s[5:, 0] = 0.0; s[5:, 1] = 1.0      # always ranks p_a or p_b first
    m = per_anchor(_scores(s, s))
    assert np.isclose(m["r1"].mean(), 0.5) and np.allclose(m["gain"], 0.0)


def test_ties_are_misses():                                               # Review Focus 2
    s = np.zeros((4, 13))
    assert first_place(s, 0).sum() == 0


def test_nonfinite_scores_are_misses():                                   # Review Focus 4
    s = np.zeros((3, 13)); s[:, 0] = 1.0; s[1, 5] = np.nan
    assert first_place(s, 0).tolist() == [1.0, 0.0, 1.0]


def test_cluster_bootstrap_widens_with_clustering():
    rng = np.random.default_rng(1)
    cluster_effect = rng.normal(size=100)
    clusters = np.repeat(np.arange(100), 20)
    values = cluster_effect[clusters] + 0.1 * rng.normal(size=2000)
    clustered = cluster_bootstrap(values, clusters)
    naive = cluster_bootstrap(values, np.arange(2000))
    assert clustered["n_clusters"] == 100
    assert (clustered["ci95"][1] - clustered["ci95"][0]) > 2 * (naive["ci95"][1] - naive["ci95"][0])
    assert np.isclose(clustered["point"], values.mean())


def test_cluster_bootstrap_needs_two_clusters():
    with pytest.raises(ValueError):
        cluster_bootstrap(np.ones(5), np.zeros(5))


def test_summarize_and_compare_in_points():
    s_a = np.zeros((10, 13)); s_a[:, 0] = 1.0
    s_b = np.zeros((10, 13)); s_b[:, 1] = 1.0
    good = per_anchor(_scores(s_a, s_b))
    blind = per_anchor(_scores(s_a, s_a))
    clusters = np.arange(10)
    assert summarize(good, clusters)["r1"]["point"] == pytest.approx(100.0)
    assert compare(good, blind, clusters, "gain")["point"] == pytest.approx(100.0)


def test_first_place_infinite_values_are_misses():                         # finite guard coverage
    """Verify the finite mask catches +inf and -inf, not just NaN."""
    s = np.zeros((4, 13))
    s[0, 0] = 1.0; s[0, 1] = 0.0                                            # all finite: hit
    s[1, 0] = np.inf; s[1, 1] = 0.0                                         # +inf target: miss (non-finite)
    s[2, 0] = 1.0; s[2, 5] = -np.inf                                        # -inf in other: miss (non-finite)
    s[3, 0] = 1.0; s[3, 5] = np.inf                                         # +inf in other: miss (non-finite)
    assert first_place(s, 0).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_swap_positive_case_and_non_finite_coverage():                     # swap coverage
    """Test positive swap case and non-finite handling in swap."""
    # Perfect scorer case: should have swap=1 for matching pairs
    s_a = np.zeros((2, 13)); s_a[:, 0] = 1.0; s_a[:, 1] = 0.5
    s_b = np.zeros((2, 13)); s_b[:, 0] = 0.5; s_b[:, 1] = 1.0
    m = per_anchor(_scores(s_a, s_b))
    assert np.all(m["swap"] == 1.0)                                         # both rows swap

    # Tie case: p_a == p_b in both conditions, expect swap=0
    s_a = np.zeros((2, 13)); s_a[:, 0] = 1.0; s_a[:, 1] = 1.0
    s_b = np.zeros((2, 13)); s_b[:, 0] = 1.0; s_b[:, 1] = 1.0
    m = per_anchor(_scores(s_a, s_b))
    assert np.all(m["swap"] == 0.0)                                         # ties are misses

    # Non-finite in non-target columns: should be miss (Review Focus 4)
    s_a = np.zeros((2, 13)); s_a[:, 0] = 1.0; s_a[:, 1] = 0.5; s_a[0, 5] = np.nan
    s_b = np.zeros((2, 13)); s_b[:, 0] = 0.5; s_b[:, 1] = 1.0; s_b[0, 6] = np.inf
    m = per_anchor(_scores(s_a, s_b))
    assert m["swap"].tolist() == [0.0, 1.0]                                 # only row 1 is finite


def test_cluster_bootstrap_ratio_of_sums_with_unequal_sizes():              # bootstrap definition
    """Verify ratio-of-sums (actual) vs mean-of-means (wrong) with correlated cluster sizes."""
    # Small clusters (size 1) valued 1, large clusters (size 100) valued 0
    # Mean = (50*1 + 50*0) / (50*1 + 50*100) ≈ 0.0099
    # Mean-of-means = (1 + 0) / 2 = 0.5 (wrong)
    clusters = np.concatenate([np.repeat(np.arange(50), 1), np.repeat(np.arange(50, 100), 100)])
    values = np.concatenate([np.ones(50), np.zeros(5000)])
    result = cluster_bootstrap(values, clusters, n_boot=1000, seed=42)
    assert np.isclose(result["point"], values.mean(), rtol=1e-6)
    assert result["ci95"][0] < values.mean() < result["ci95"][1]
    assert not (result["ci95"][0] <= 0.5 <= result["ci95"][1])              # excludes mean-of-means
