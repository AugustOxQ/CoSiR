import numpy as np
import pytest

import r20_ret as ret


def fixture_rows():
    return (np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
            np.array([0, 0, 1, 1], dtype=np.int64),
            np.array(['painting-a', 'painting-b', 'painting-c', 'painting-d']))


def test_perfect_ranking():
    x, y, names = fixture_rows()
    result = ret.score_rows(x, y, names, workers=1)
    np.testing.assert_array_equal(result['ap'], np.ones(4))


def test_bootstrap_point_is_difference_of_means():
    rng = np.random.default_rng(12)
    ids = np.array(['a', 'b', 'c', 'd'])
    a = dict(ap=rng.random((5, 4)), query_ids=ids, dropped=0)
    b = dict(ap=rng.random((5, 4)), query_ids=ids.copy(), dropped=0)
    p = ret.paired(a, b, draws=40)
    assert p['d'] == a['ap'].mean() - b['ap'].mean()
    assert set(p) == {'d', 'lo', 'hi'}


def test_query_join_uses_names_and_preserves_labelled_order():
    x, y, names = fixture_rows()
    x = np.vstack([x, np.array([[1, 1]], dtype=np.float32)])
    y = np.append(y, -1)
    names = np.append(names, 'unlabelled')
    left = ret.score_rows(x, y, names, workers=1)
    right = ret.score_rows(x.copy(), y.copy(), names.copy(), workers=1)
    ret.same_queries(left, right)
    np.testing.assert_array_equal(left['query_ids'], names[:4])
    with pytest.raises(AssertionError, match='order differs'):
        ret.same_queries(left, dict(right, query_ids=right['query_ids'][::-1]))


def test_two_seed_smoke_bootstrap():
    ids = np.array(['a', 'b', 'c'])
    b = dict(ap=np.full((2, 3), 0.3), query_ids=ids, dropped=0)
    a = dict(b, ap=b['ap'] + 0.1)
    p = ret.paired(a, b, draws=20, smoke=True)
    np.testing.assert_allclose([p['d'], p['lo'], p['hi']], 0.1)
