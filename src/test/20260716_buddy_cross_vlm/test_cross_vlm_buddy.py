import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
import cross_vlm_buddy as cvb


def test_adj_to_keys_encodes_upper_triangle_and_skips_stored_zeros():
    N = 5
    dense = np.zeros((N, N), dtype=np.float64)
    dense[0, 2] = dense[2, 0] = 1.0
    dense[1, 3] = dense[3, 1] = 1.0
    A = csr_matrix(dense)
    keys = cvb.adj_to_keys(A)
    assert keys.tolist() == [0 * N + 2, 1 * N + 3] == [2, 8]

    # now inject an explicit stored zero into the sparsity pattern (upper
    # triangle) and confirm it is NOT reported as an edge.
    rows = np.array([0, 2, 1, 3, 0, 4], dtype=np.int32)
    cols = np.array([2, 0, 3, 1, 4, 0], dtype=np.int32)
    data = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    A_zero = csr_matrix((data, (rows, cols)), shape=(N, N))
    # (0, 4) is a structurally-stored zero: it must not appear in the keys.
    assert A_zero[0, 4] == 0.0
    assert A_zero.nnz > A.nnz  # confirm the zero entry is actually stored
    keys_zero = cvb.adj_to_keys(A_zero)
    assert keys_zero.tolist() == [2, 8]
    assert (0 * N + 4) not in keys_zero.tolist()


def test_perm_null_edges_stay_canonical():
    N = 30
    b = np.sort(np.array(
        [0 * N + 1, 2 * N + 3, 4 * N + 5, 6 * N + 7, 8 * N + 9, 10 * N + 11],
        dtype=np.int64,
    ))
    res = cvb.perm_null_jaccard(b, b, N, n_perm=50, seed=7)
    # b against itself: perfect observed agreement.
    assert res["observed"] == 1.0
    # If the min/max re-canonicalization + dedup step inside perm_null_jaccard
    # were broken (e.g. keys not re-sorted into i<j form, or duplicate keys
    # left in after relabeling), the permuted-vs-original null jaccard would
    # not behave like a random relabeling: it would either spuriously match
    # more often (broken canonicalization keeping some raw alignment) or
    # produce degenerate near-zero unions. A correctly canonicalized,
    # deduplicated null must land strictly below the observed value and
    # above zero (six edges over 30 nodes is not so sparse that random
    # relabelings never collide).
    assert 0.0 < res["null_mean"] < res["observed"]


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
