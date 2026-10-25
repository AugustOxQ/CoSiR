"""Part 1 partitions, scores and draws (ticket 03, rule §3.1): the statistic's seams, hand-checked values."""
import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

import rb_paths  # noqa: F401  (sys.path)
import rb_ami


def blobs(n=600, width=16, seed=1):
    """Unit-norm rows around 20 well-separated centres, float16 like the saved exports."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((20, width))
    x = centres[rng.integers(0, 20, n)] + 0.3 * rng.standard_normal((n, width))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float16)


def test_partitions_are_the_rules_three_kmeans_runs_on_the_float32_cast():
    x = blobs()
    parts = rb_ami.partitions(x)
    assert parts.shape == (3, 600) and parts.dtype == np.int64
    x32 = x.astype(np.float32)
    for state in (0, 1, 2):
        expect = MiniBatchKMeans(n_clusters=20, random_state=state, n_init=3, batch_size=4096).fit_predict(x32)
        np.testing.assert_array_equal(parts[state], expect)


def test_partitions_do_not_renormalize_and_reject_rows_that_are_not_unit_norm():
    x = blobs().astype(np.float32)
    ok = x * np.float32(1.0005)                      # |norm - 1| ~ 5e-4 < 1e-3: accepted, left as it is
    expect = MiniBatchKMeans(n_clusters=20, random_state=0, n_init=3, batch_size=4096).fit_predict(ok)
    np.testing.assert_array_equal(rb_ami.partitions(ok)[0], expect)
    with pytest.raises(AssertionError):
        rb_ami.partitions(x * np.float32(1.002))     # 2e-3 off
    bad = x.copy()
    bad[3, 0] = np.nan
    with pytest.raises(AssertionError):
        rb_ami.partitions(bad)
    with pytest.raises(AssertionError):
        rb_ami.partitions(x.astype(np.float64))      # float16 or float32 only


def test_score_is_the_mean_of_the_three_partitions_amis_on_the_indexed_rows():
    labels = np.array([0, 0, 1, 1, 2, 2, -1, -1])
    parts = np.array([[5, 5, 6, 6, 7, 7, 0, 1],           # the labels renamed: AMI 1
                      [1, 1, 0, 0, 2, 2, 1, 0],           # a permutation of them: AMI 1
                      [3, 3, 3, 3, 3, 3, 3, 3]])          # one cluster: AMI 0
    assert rb_ami.score(labels, parts, np.arange(6)) == pytest.approx(2 / 3)
    # with repeats: rows 0, 0, 2, 2, 4 only change the counts, not the identity of the partitions
    assert rb_ami.score(labels, parts, np.array([0, 0, 2, 2, 4, 4])) == pytest.approx(2 / 3)
    parts[1] = [0, 1, 0, 1, 0, 1, 0, 1]                   # unrelated to the labels on rows 0 to 5
    expect = (1 + adjusted_mutual_info_score(labels[:6], parts[1, :6]) + 0) / 3
    assert rb_ami.score(labels, parts, np.arange(6)) == pytest.approx(expect, abs=1e-12)


def test_score_rejects_masked_rows_and_misshaped_inputs():
    labels = np.array([0, 1, -1, 0])
    parts = np.zeros((3, 4), dtype=np.int64)
    with pytest.raises(AssertionError):
        rb_ami.score(labels, parts, np.array([0, 2]))     # row 2 has no label
    with pytest.raises(AssertionError):
        rb_ami.score(labels, parts[:2], np.array([0, 1]))
    with pytest.raises(AssertionError):
        rb_ami.score(labels, np.zeros((3, 5), dtype=np.int64), np.array([0, 1]))
    with pytest.raises(AssertionError):
        rb_ami.score(labels, parts, np.array([0.0, 1.0]))


def test_draws_are_the_rules_formula_one_default_rng_0_per_call():
    # rule §3.1: rng = default_rng(0); b = 1..2000: I_b = L_a[rng.integers(0, len(L_a), size=len(L_a))]
    n = 37
    rng = np.random.default_rng(0)
    expect = np.stack([rng.integers(0, n, size=n) for _ in range(25)])
    got = rb_ami.draws(n, 25)
    assert got.shape == (25, n)
    np.testing.assert_array_equal(got, expect)
    np.testing.assert_array_equal(rb_ami.draws(n, 25), got)       # a fresh generator each call
    assert rb_ami.N_DRAWS == 2000
    np.testing.assert_array_equal(rb_ami.draws(n)[:25], expect)
