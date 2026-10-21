"""Unit tests for ab_graphs: synthetic small inputs, CPU, fast."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ab_graphs
from ab_graphs import (degree_preserving_swaps, dview_pairs, pair_degrees,
                       reciprocal_topk)


def brute_reciprocal(score, k):
    n = len(score)
    nn = []
    for a in range(n):
        order = [b for b in np.argsort(-score[a], kind='stable') if b != a]
        nn.append(set(order[:k]))
    return {(a, b) for a in range(n) for b in nn[a] if a < b and a in nn[b]}


def random_graph(n, m, rng):
    edges = set()
    while len(edges) < m:
        a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
        edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64)


def as_set(pairs):
    return set(map(tuple, pairs.tolist()))


def test_reciprocal_topk_matches_brute_force():
    rng = np.random.default_rng(1)
    score = rng.normal(size=(70, 70))
    score = (score + score.T) / 2  # symmetric matrix, as a cosine matrix is
    for k in (3, 5, 20):
        got = reciprocal_topk(score, k, block=16)
        assert as_set(got) == brute_reciprocal(score, k)
        assert got.dtype == np.int64 and got.shape[1] == 2


def test_reciprocal_topk_asymmetric_scores_and_no_self():
    rng = np.random.default_rng(2)
    score = rng.normal(size=(50, 50))
    got = reciprocal_topk(score, 4, block=7)
    assert as_set(got) == brute_reciprocal(score, 4)
    assert (got[:, 0] < got[:, 1]).all()


def test_dview_pairs_is_reciprocal_topk_of_standardized_mean_cosine():
    rng = np.random.default_rng(3)
    n = 80
    d = rng.normal(size=(n, 16)).astype(np.float32)
    i = rng.normal(size=(n, 16)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    i /= np.linalg.norm(i, axis=1, keepdims=True)
    got, sds = dview_pairs(d, i, k=5, block=32)
    draws = ab_graphs.ags_graphs.draw_pairs(n)
    sd = lambda x: float(np.std(np.sum(x[draws[0]].astype(np.float64) * x[draws[1]], axis=1)))
    content = ((d @ d.T) / sd(d) + (i @ i.T) / sd(i)) / 2
    assert as_set(got) == brute_reciprocal(content, 5)


@pytest.mark.parametrize('n,m', [(60, 150), (200, 700)])
def test_swaps_keep_degrees_and_stay_simple(n, m):
    rng = np.random.default_rng(4)
    pairs = random_graph(n, m, rng)
    out, info = degree_preserving_swaps(pairs, n, np.random.default_rng([0, 1, 0]), 10)
    assert out.dtype == np.int64 and out.shape == pairs.shape
    assert np.array_equal(pair_degrees(out, n), pair_degrees(pairs, n))
    assert (out[:, 0] < out[:, 1]).all()  # no self pairs, canonical order
    codes = out[:, 0] * n + out[:, 1]
    assert (np.diff(codes) > 0).all()  # sorted, no duplicates
    assert info['accepted'] >= 10 * m  # the swap target is met
    assert len(as_set(out) & as_set(pairs)) < 0.5 * m  # mixed well


def test_swaps_deterministic_per_seed_and_differ_between_seeds():
    n = 120
    pairs = random_graph(n, 300, np.random.default_rng(5))
    run = lambda seq: degree_preserving_swaps(pairs, n, np.random.default_rng(seq), 10)[0]
    a, a2, b, c = run([0, 1, 0]), run([0, 1, 0]), run([1, 1, 0]), run([0, 2, 0])
    assert np.array_equal(a, a2)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)
    assert not np.array_equal(a, pairs)


def test_swaps_on_graph_that_cannot_change():
    # A triangle's only simple graph with these degrees is itself: no swap is possible.
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    out, info = degree_preserving_swaps(pairs, 3, np.random.default_rng(0), 10)
    assert np.array_equal(out, pairs) and info['accepted'] == 0
