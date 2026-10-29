"""rt_vh's AP and nDCG against scikit-learn on tie-free inputs, and the tie rule by hand (ties keep gallery row order;
scikit-learn groups ties), and the round log's Numeric convention. The module is itself the independent check."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, ndcg_score

import rt_vh


def _unit(rng, n, d):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_none_readout_normalizes_in_float64_before_single_cast():
    x = np.random.default_rng(26).normal(size=(20, 17)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    want = x.astype(np.float64)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    np.testing.assert_array_equal(rt_vh.apply_w('none', 0, None, x), want.astype(np.float32))


def test_crl_loader_casts_only_after_float64_cosines_and_normalization():
    rng = np.random.default_rng(27)
    image = _unit(rng, 20, 768).astype(np.float32)
    text = _unit(rng, 50, 768).astype(np.float32)
    data = rt_vh.Data.__new__(rt_vh.Data)
    data.names = {'val': list(map(str, range(20)))}
    data._cache = {('clip', 'val', None, None): image}
    data.crl_text = lambda aspect: text
    want = image.astype(np.float64) @ text.astype(np.float64).T
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    np.testing.assert_array_equal(data.emb('crl', 'val', 'emotion'), want.astype(np.float32))


@pytest.mark.parametrize('k', [0, 3])
def test_stage5_none_readout_uses_the_same_normalization_convention(k, monkeypatch):
    from types import SimpleNamespace
    rng = np.random.default_rng(28)
    x = rng.normal(size=(30, 17)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    lab = np.arange(30) % 3
    data = SimpleNamespace(labels={'held': {'emotion': lab}}, emb=lambda *args: x)
    run = rt_vh.Run(data, SimpleNamespace(workers=1))
    want = x.astype(np.float64)
    want /= np.linalg.norm(want, axis=1, keepdims=True)

    def check_rows(X32, labels, workers, ks):
        assert X32.dtype == np.float32 and ks == (k,)
        np.testing.assert_array_equal(X32, want.astype(np.float32))
        return {k: np.zeros(len(labels))}

    monkeypatch.setattr(rt_vh, 'ap_all', check_rows)
    run.perq('held', 'clip', None, 'emotion', ro=('none', 0, k))


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
