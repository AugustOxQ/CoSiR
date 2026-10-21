"""Known-answer tests for ab_metrics (ticket 03). Small synthetic arrays; runs in seconds."""
import os
import sys

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_metrics as m  # noqa: E402


def rng(seed=0):
    return np.random.default_rng(seed)


def rot(d, seed=1):
    q, _ = np.linalg.qr(rng(seed).standard_normal((d, d)))
    return q


# ---- linear CKA
def test_cka_self_is_one():
    x = rng().standard_normal((200, 10))
    assert m.linear_cka(x, x) == pytest.approx(1.0, abs=1e-10)


def test_cka_rotation_and_scale_invariant():
    x = rng().standard_normal((200, 10))
    y = 3.7 * x @ rot(10)
    assert m.linear_cka(x, y) == pytest.approx(1.0, abs=1e-10)


def test_cka_different_widths_and_float32():
    x = rng().standard_normal((200, 10)).astype(np.float32)
    y = np.concatenate([x, x], axis=1)[:, :14]
    v = m.linear_cka(x, y)
    assert 0.0 < v <= 1.0 + 1e-9


def test_cka_independent_near_zero():
    r = rng(2)
    v = m.linear_cka(r.standard_normal((2000, 5)), r.standard_normal((2000, 5)))
    assert v < 0.02


def test_cka_matches_sample_gram_form():
    # the n x n Gram definition on centred features equals the d x d feature-Gram form
    r = rng(3)
    x = r.standard_normal((50, 6))
    y = r.standard_normal((50, 4)) + x[:, :4]
    xc, yc = x - x.mean(0), y - y.mean(0)
    kx, ky = xc @ xc.T, yc @ yc.T
    ref = (kx * ky).sum() / np.sqrt((kx * kx).sum() * (ky * ky).sum())
    assert m.linear_cka(x, y) == pytest.approx(ref, rel=1e-10)


# ---- kNN overlap
def test_knn_self_is_one():
    x = rng().standard_normal((120, 8))
    assert m.knn_overlap(x, x, k=5) == pytest.approx(1.0)


def test_knn_independent_near_chance():
    r = rng(4)
    n, k = 600, 20
    v = m.knn_overlap(r.standard_normal((n, 8)), r.standard_normal((n, 8)), k=k)
    assert v == pytest.approx(m.knn_chance(n, k), abs=0.01)


def test_knn_blocking_does_not_change_result():
    r = rng(5)
    x = r.standard_normal((130, 6))
    y = x + 0.5 * r.standard_normal((130, 6))
    a = m.knn_overlap(x, y, k=7, block=1000)
    b = m.knn_overlap(x, y, k=7, block=17)
    assert a == pytest.approx(b)
    assert 0.0 < a < 1.0


def test_knn_known_answer():
    # two tight groups of 3 points, k=2: neighbours of a point are its two group mates
    x = np.array([[1, 0], [1, .01], [1, -.01], [0, 1], [.01, 1], [-.01, 1]], float)
    assert m.knn_overlap(x, x @ rot(2), k=2) == pytest.approx(1.0)


def test_knn_exact_zero_overlap():
    # x: points 0..3 on a line of angles so that nearest are consecutive; y: reversed neighbourhood structure
    ang = np.array([0.0, 0.1, 1.5, 1.6])
    x = np.stack([np.cos(ang), np.sin(ang)], 1)
    ang_y = np.array([0.0, 1.5, 0.1, 1.6])  # swap rows 1 and 2
    y = np.stack([np.cos(ang_y), np.sin(ang_y)], 1)
    # k=1: x nn: 0->1, 1->0, 2->3, 3->2.  y nn: 0->2, 1->3, 2->0, 3->1.  overlap 0
    assert m.knn_overlap(x, y, k=1) == pytest.approx(0.0)


def test_knn_chance():
    assert m.knn_chance(5000, 20) == pytest.approx(20 / 4999)


# ---- size entropy
def test_size_entropy():
    assert m.size_entropy(np.repeat(np.arange(4), 25), 4) == pytest.approx(1.0)
    assert m.size_entropy(np.zeros(100, int), 4) == pytest.approx(0.0)
    assert m.size_entropy(np.array([0] * 50 + [1] * 50), 4) == pytest.approx(0.5)  # log2 / log4
    assert 0.0 < m.size_entropy(np.array([0] * 90 + [1] * 10), 2) < 1.0


# ---- effective rank
def test_effective_rank_isotropic_rank_r():
    q, _ = np.linalg.qr(rng(6).standard_normal((300, 5)) - 0.0)
    q, _ = np.linalg.qr(q - q.mean(0))          # orthonormal, centred columns: equal singular values
    x = 10.0 * q @ np.eye(5, 20)
    assert m.effective_rank(x) == pytest.approx(5.0, abs=1e-6)


def test_effective_rank_rank_one_and_noise():
    r = rng(8)
    assert m.effective_rank(r.standard_normal((500, 10))) > 9.0
    v = r.standard_normal((100, 1)) @ r.standard_normal((1, 12))
    assert m.effective_rank(v) == pytest.approx(1.0, abs=1e-6)


# ---- principal angles
def test_principal_angles_same_subspace():
    r = rng(9)
    x = r.standard_normal((300, 4)) @ r.standard_normal((4, 12))
    cos, msq = m.principal_angles(x, 2.0 * x, 3)
    assert cos.shape == (3,) and np.allclose(cos, 1.0)
    assert msq == pytest.approx(1.0)


def test_principal_angles_orthogonal_subspaces():
    r = rng(10)
    a = np.zeros((200, 8))
    a[:, :3] = r.standard_normal((200, 3)) * [5, 4, 3]
    b = np.zeros((200, 8))
    b[:, 3:6] = r.standard_normal((200, 3)) * [5, 4, 3]
    cos, msq = m.principal_angles(a, b, 3)
    assert np.allclose(cos, 0.0, atol=1e-8)
    assert msq == pytest.approx(0.0, abs=1e-10)


def test_subspace_chance():
    assert m.subspace_chance(5, 100) == pytest.approx(0.05)


# ---- probe
def _blobs(n_per=40, classes=4, d=6, sep=20.0, seed=11):
    r = rng(seed)
    centres = r.standard_normal((classes, d)) * sep
    y = np.repeat(np.arange(classes), n_per)
    return centres[y] + r.standard_normal((len(y), d)), y


def test_probe_separable_scores_one():
    x, y = _blobs()
    out = m.probe(x, y)
    assert out['balanced_accuracy'] == pytest.approx(1.0)
    assert np.allclose(out['per_class_recall'], 1.0)
    assert out['chance'] == pytest.approx(0.25)
    assert out['majority_rate'] == pytest.approx(0.25)
    assert out['n'] == len(y) and out['n_classes'] == 4
    assert isinstance(out['convergence_warnings'], int)


def test_probe_shuffled_labels_near_chance():
    x, y = _blobs(n_per=60, sep=5.0)
    y = rng(12).permutation(y)
    out = m.probe(x, y)
    assert abs(out['balanced_accuracy'] - 0.25) < 0.1


def test_probe_drops_negative_labels_and_reports_majority():
    x, y = _blobs(n_per=50, classes=3)
    y = y.copy()
    y[:30] = -1                       # drops 30 of the 50 class-0 rows
    out = m.probe(x, y)
    assert out['n'] == len(y) - 30
    assert out['n_classes'] == 3
    assert out['majority_rate'] == pytest.approx(50 / 120)
    assert out['chance'] == pytest.approx(1 / 3)
    assert out['classes'].tolist() == [0, 1, 2]


def test_probe_counts_and_reemits_convergence_warnings():
    x, y = _blobs(n_per=60, sep=1.0, d=30)
    with pytest.warns(ConvergenceWarning):
        out = m.probe(x, y, max_iter=1)
    assert out['convergence_warnings'] >= 1
    assert m.probe(*_blobs())['convergence_warnings'] == 0


def test_probe_deterministic():
    x, y = _blobs(sep=3.0)
    a, b = m.probe(x, y, seed=3), m.probe(x, y, seed=3)
    assert a['balanced_accuracy'] == b['balanced_accuracy']
