"""rt_vh's AP and nDCG against scikit-learn on tie-free inputs, and the tie rule by hand (ties keep gallery row order;
scikit-learn groups ties). Nothing else: the module is itself the check."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, ndcg_score

import rt_vh


def _unit(rng, n, d):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_ap_matches_sklearn_tie_free():
    rng = np.random.default_rng(1)
    G, lab = _unit(rng, 300, 16), rng.integers(0, 5, size=300)
    ap = rt_vh.ap_core(G, lab, np.arange(300))[0]
    for i in range(300):
        keep = np.arange(300) != i
        want = average_precision_score(lab[keep] == lab[i], (G @ G[i])[keep])
        assert ap[i] == pytest.approx(want, abs=1e-12)


def test_ndcg_matches_sklearn_tie_free():
    rng = np.random.default_rng(2)
    G = _unit(rng, 200, 16)
    P = rng.dirichlet(np.ones(9), size=200)
    nd = rt_vh.ndcg_core(G, P, np.arange(200))
    for i in range(200):
        keep = np.arange(200) != i
        gain = np.minimum(P[i], P[keep]).sum(axis=1)
        want = ndcg_score([gain], [(G @ G[i])[keep]])
        assert nd[i] == pytest.approx(want, abs=1e-12)


def test_ties_keep_gallery_row_order():
    # query row 0 = (1,0); rows 1 and 2 are identical (0,1), both at cosine 0; row 3 = (-1,0).
    G = np.array([[1.0, 0], [0, 1], [0, 1], [-1, 0]])
    lab = np.array([0, 0, 1, 0])                                  # row 1 relevant, row 2 not: ties keep row order
    ap = rt_vh.ap_core(G, lab, np.array([0]))[0][0]
    assert ap == pytest.approx((1 / 1 + 2 / 3) / 2)               # ranking 1, 2, 3 -> relevant at ranks 1 and 3
    # scikit-learn groups the two tied rows, giving another number
    sk = average_precision_score([1, 0, 1], [0.0, 0.0, -1.0])
    assert sk != pytest.approx(ap)
    # the same tie with the labels of rows 1 and 2 swapped: relevant at ranks 2 and 3
    lab2 = np.array([0, 1, 0, 0])
    assert rt_vh.ap_core(G, lab2, np.array([0]))[0][0] == pytest.approx((1 / 2 + 2 / 3) / 2)
