import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cross_vlm_buddy as cvb


def test_jaccard_identical():
    a = np.array([1, 2, 5], dtype=np.int64)
    jac, ov, inter = cvb.jaccard(a, a)
    assert jac == 1.0 and ov == 1.0 and inter == 3


def test_jaccard_disjoint():
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([3, 4], dtype=np.int64)
    jac, ov, inter = cvb.jaccard(a, b)
    assert jac == 0.0 and inter == 0


def test_perm_null_identical_has_high_lift():
    N = 50
    # edges among distinct node pairs, encoded i*N+j (i<j)
    a = np.sort(np.array([0 * N + 1, 2 * N + 3, 4 * N + 5], dtype=np.int64))
    res = cvb.perm_null_jaccard(a, a, N, n_perm=100, seed=0)
    assert res["observed"] == 1.0
    assert res["lift"] > 5.0            # identical sets crush the permuted null
    assert res["percentile"] == 1.0


def test_perm_null_random_lift_near_one():
    N = 200
    rng = np.random.default_rng(3)

    def rand_keys(m):
        i = rng.integers(0, N, m); j = rng.integers(0, N, m)
        ok = i != j
        lo = np.minimum(i[ok], j[ok]); hi = np.maximum(i[ok], j[ok])
        return np.unique(lo.astype(np.int64) * N + hi.astype(np.int64))

    a, b = rand_keys(300), rand_keys(300)
    res = cvb.perm_null_jaccard(a, b, N, n_perm=100, seed=1)
    assert 0.3 < res["lift"] < 3.0      # independent graphs: no real agreement


def test_agreement_matrix_shape_and_diag():
    N = 20
    cells = {"a": np.array([1, 2], dtype=np.int64),
             "b": np.array([1, 3], dtype=np.int64),
             "c": np.array([7, 8], dtype=np.int64)}
    out = cvb.agreement_matrix(cells, N, n_perm=20, seed=0)
    assert out["jaccard"].shape == (3, 3)
    assert np.allclose(np.diag(out["jaccard"]), 1.0)
    assert out["jaccard"][0, 1] > 0.0 and out["jaccard"][0, 2] == 0.0
    assert np.isfinite(out["median_offdiag_jaccard"])
