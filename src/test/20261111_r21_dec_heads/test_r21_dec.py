"""Unit tests for r21_dec, against hand-computed values and numpy references."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r21_dec  # noqa: E402
import pc_sept  # noqa: E402  (on the path through r21_dec)

# Two unit points on the centres themselves: d2 = [[0, 2], [2, 0]].
Z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
C = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
Q_SOFT = np.array([[3 / 4, 1 / 4], [1 / 4, 3 / 4]])
# Used directly as q in the target / KL tests (no head output involved).
Q_HAND = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
P_HAND = np.array([[0.8, 0.2], [0.2, 0.8]])


def np_kl(p, q):
    """KL(p || q) summed over columns, averaged over rows."""
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


def test_soft_assign_hand_values_and_rows_sum_to_one():
    q = r21_dec.soft_assign(Z, C)
    np.testing.assert_allclose(q.numpy(), Q_SOFT, atol=1e-6)
    np.testing.assert_allclose(q.sum(1).numpy(), 1.0, atol=1e-6)


def test_soft_assign_uses_the_squared_distance():
    # Opposite unit points: d2 = 4 (d = 2). Raw q = [1, 1/5] -> [5/6, 1/6];
    # an unsquared distance would give [3/4, 1/4].
    z = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    q = r21_dec.soft_assign(z, z.clone())
    np.testing.assert_allclose(q[0].numpy(), [5 / 6, 1 / 6], atol=1e-6)


def test_soft_assign_chunked_without_grad_is_bit_identical():
    g = torch.Generator().manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(10, 5, generator=g), dim=1)
    c = torch.randn(4, 5, generator=g)
    ref = pc_sept.soft_assignments(z, c)
    with torch.no_grad():
        got = r21_dec.soft_assign(z, c, chunk=3)  # 3 + 3 + 3 + 1 rows
    assert torch.equal(got, ref)
    # with grad enabled the unchunked path runs and still agrees
    assert torch.equal(r21_dec.soft_assign(z, c, chunk=3), ref)


def test_soft_assign_checks_k():
    with pytest.raises(AssertionError):
        r21_dec.soft_assign(Z, C, k=3)
    with pytest.raises(AssertionError):
        r21_dec.soft_assign(Z, C[:1])  # k = 1


def test_target_sharpens_and_matches_hand_values():
    q = torch.from_numpy(Q_HAND).float()
    p = r21_dec.target(q)
    np.testing.assert_allclose(p.numpy(), P_HAND, atol=1e-6)
    assert (p.max(1).values > q.max(1).values).all()
    assert not p.requires_grad


def test_target_uses_column_frequency_of_the_whole_pass():
    # Column 0 is used by three rows, column 1 by one: the frequency divides q^2.
    q = torch.tensor([[.6, .4], [.6, .4], [.6, .4], [.4, .6]])
    w = q ** 2 / q.sum(0, keepdim=True)
    ref = (w / w.sum(1, keepdim=True)).numpy()
    np.testing.assert_allclose(r21_dec.target(q).numpy(), ref, atol=1e-6)


def test_dec_kl_equals_numpy_reference_and_hand_value():
    q = torch.from_numpy(Q_HAND).float()
    p = torch.from_numpy(P_HAND).float()
    got = r21_dec.dec_kl(p, q).item()
    assert abs(got - np_kl(P_HAND, Q_HAND)) < 1e-6
    assert abs(got - 0.04369213) < 1e-6  # .8 ln 1.2 + .2 ln .6


def test_kl_terms_assert_targets_carry_no_gradient():
    q = torch.from_numpy(Q_HAND).float()
    p = torch.from_numpy(P_HAND).float().requires_grad_(True)
    with pytest.raises(AssertionError):
        r21_dec.dec_kl(p, q)
    with pytest.raises(AssertionError):
        r21_dec.pull_kl(p, q)


def test_pull_kl_hand_value_and_zero_on_itself():
    q = torch.from_numpy(Q_HAND).float()
    r = torch.full((2, 2), 0.5)
    assert abs(r21_dec.pull_kl(r, q).item() - 0.5 * np.log(1.125)) < 1e-6
    assert abs(r21_dec.pull_kl(q, q).item()) < 1e-7


def test_balance_zero_for_uniform_mean_and_hand_value_otherwise():
    q_even = torch.from_numpy(Q_HAND).float()  # mean q = [.5, .5]
    assert abs(r21_dec.balance_kl(q_even).item()) < 1e-7
    q_skew = torch.tensor([[.9, .1], [.9, .1]])
    assert abs(r21_dec.balance_kl(q_skew).item() - np.log(5 / 3)) < 1e-6


def test_balance_clamps_an_empty_column():
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert np.isfinite(r21_dec.balance_kl(q).item())


def _inputs():
    q = torch.tensor([[.9, .1], [.7, .3], [.5, .5]])
    p = r21_dec.target(q)
    r = torch.tensor([[.6, .4], [.6, .4], [.2, .8]])
    return q, p, r


def test_dec_block_plain_has_only_the_dec_term():
    q, p, r = _inputs()
    s = dict(form='plain', lam=2.0, warmup=1, stop='S1', balance=False)
    total, parts = r21_dec.dec_block(q, p, None, s)
    ref = 2.0 * np_kl(p.numpy(), q.numpy())
    assert abs(total.item() - ref) < 1e-6
    assert set(parts) == {'kl_p'}
    assert all(isinstance(v, float) for v in parts.values())
    assert abs(parts['kl_p'] - np_kl(p.numpy(), q.numpy())) < 1e-6


def test_dec_block_pull_and_balance_sum_with_the_balance_ratio():
    q, p, r = _inputs()
    s = dict(form='start', lam=3.0, warmup=1, stop='S2', balance=True)
    total, parts = r21_dec.dec_block(q, p, r, s)
    qm = q.numpy().mean(0)
    bal = float(np.sum(0.5 * np.log(0.5 / qm)))
    ref = 3.0 * (np_kl(p.numpy(), q.numpy()) + np_kl(r.numpy(), q.numpy()) + 1000 * bal)
    assert abs(total.item() - ref) < 1e-4 * max(1.0, abs(ref))
    assert set(parts) == {'kl_p', 'pull', 'balance'}
    assert abs(parts['balance'] - bal) < 1e-6  # unweighted parts
    assert abs(parts['kl_p'] - np_kl(p.numpy(), q.numpy())) < 1e-6
    assert abs(parts['pull'] - np_kl(r.numpy(), q.numpy())) < 1e-6


def test_dec_block_lambda_zero_is_exactly_zero_with_pull_and_balance_on():
    q, p, r = _inputs()
    q = q.clone().requires_grad_(True)
    s = dict(form='teacher', lam=0.0, warmup=1, stop='S1', balance=True)
    total, _ = r21_dec.dec_block(q, p, r, s)
    assert total.item() == 0.0


def test_dec_block_gradient_reaches_q_and_matches_finite_difference():
    q, p, r = _inputs()
    s = dict(form='plain', lam=1.0, warmup=1, stop='S1', balance=False)
    q = q.clone().requires_grad_(True)
    total, _ = r21_dec.dec_block(q, p, None, s)
    total.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    # p is fixed, so d/dq of mean_rows sum p log(p/q) = -p / (q n)
    np.testing.assert_allclose(q.grad.numpy(), -(p / q.detach() / 3).numpy(), atol=1e-6)


def test_dec_block_asserts_pull_matches_form():
    q, p, r = _inputs()
    plain = dict(form='plain', lam=1.0, warmup=1, stop='S1', balance=False)
    with pytest.raises(AssertionError):
        r21_dec.dec_block(q, p, r, plain)
    start = dict(plain, form='start')
    with pytest.raises(AssertionError):
        r21_dec.dec_block(q, p, None, start)
    off = dict(form='off', lam=None, warmup=None, stop=None, balance=None)
    with pytest.raises(AssertionError):
        r21_dec.dec_block(q, p, None, off)


def test_shape_and_finiteness_asserts():
    with pytest.raises(AssertionError):
        r21_dec.soft_assign(torch.zeros(3, 2), torch.zeros(2, 3))
    with pytest.raises(AssertionError):
        r21_dec.soft_assign(torch.tensor([[float('nan'), 0.0]]), C)
    with pytest.raises(AssertionError):
        r21_dec.soft_assign(torch.tensor([[2.0, 0.0]]), C)  # not unit length
    with pytest.raises(AssertionError):
        r21_dec.target(torch.ones(2, 2))  # rows do not sum to 1
    with pytest.raises(AssertionError):
        r21_dec.dec_kl(torch.full((3, 2), .5), torch.full((2, 2), .5))


def _blobs(seed=0, n=40, d=4):
    rng = np.random.RandomState(seed)
    a = rng.normal(0, 0.05, (n, d)) + np.eye(d)[0]
    b = rng.normal(0, 0.05, (n, d)) + np.eye(d)[1]
    c = rng.normal(0, 0.05, (n, d)) + np.eye(d)[2]
    x = np.concatenate([a, b, c]).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_kmeans_centres_deterministic_and_shaped():
    z = _blobs()
    c1 = r21_dec.kmeans_centres(torch.from_numpy(z), 3, seed=4)
    c2 = r21_dec.kmeans_centres(torch.from_numpy(z), 3, seed=4)
    assert c1.shape == (3, 4) and c1.dtype == torch.float32
    assert torch.equal(c1, c2)
    # the three blobs are found: each centre sits near one basis vector
    assert sorted(c1.argmax(1).tolist()) == [0, 1, 2]


def test_kmeans_centres_uses_the_seed():
    rng = np.random.RandomState(0)
    z = rng.normal(size=(300, 6)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)  # no cluster structure: seeds differ
    c0 = r21_dec.kmeans_centres(torch.from_numpy(z), 8, seed=0)
    c1 = r21_dec.kmeans_centres(torch.from_numpy(z), 8, seed=1)
    assert not torch.equal(c0, c1)
    assert torch.equal(c0, r21_dec.kmeans_centres(torch.from_numpy(z), 8, seed=0))


def test_teacher_inputs_must_be_unit_and_k_at_least_two():
    x = _blobs(seed=1)
    with pytest.raises(AssertionError):
        r21_dec.teacher_clusters(x * 2, 3)
    with pytest.raises(AssertionError):
        r21_dec.teacher_clusters(x, 1)
    c, _ = r21_dec.teacher_clusters(x, 3)
    with pytest.raises(AssertionError):
        r21_dec.teacher_soft(x * 2, c)


def test_teacher_clusters_deterministic_and_recover_blobs():
    x = _blobs(seed=1)
    c1, l1 = r21_dec.teacher_clusters(x, 3)
    c2, l2 = r21_dec.teacher_clusters(x, 3)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(l1, l2)
    assert c1.shape == (3, 4) and l1.shape == (120,)
    for i in range(3):  # each blob of 40 rows is one cluster
        assert len(set(l1[40 * i:40 * (i + 1)])) == 1
    assert len(set(l1)) == 3


def test_teacher_soft_rows_sum_to_one_and_peak_at_own_centre():
    x = _blobs(seed=1)
    c, l = r21_dec.teacher_clusters(x, 3)
    t = r21_dec.teacher_soft(x, c)
    assert t.shape == (120, 3)
    np.testing.assert_allclose(t.sum(1).numpy(), 1.0, atol=1e-5)
    np.testing.assert_array_equal(t.argmax(1).numpy(), l)


def test_cluster_means_hand_value_and_empty_cluster_asserts():
    z = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 4.0]])
    m = r21_dec.cluster_means(z, np.array([0, 0, 1]), 2)
    np.testing.assert_allclose(m.numpy(), [[1.0, 0.0], [10.0, 4.0]])
    assert m.dtype == torch.float32
    with pytest.raises(AssertionError):
        r21_dec.cluster_means(z, np.array([0, 0, 0]), 2)


def test_changed_fraction_and_cluster_shares():
    prev = np.array([0, 0, 1, 1, 2])
    cur = np.array([0, 1, 1, 2, 2])
    assert r21_dec.changed_fraction(prev, cur) == pytest.approx(0.4)
    assert r21_dec.changed_fraction(prev, prev) == 0.0
    hi, lo = r21_dec.cluster_shares(np.array([0, 0, 0, 1, 2]), 4)
    assert (hi, lo) == (pytest.approx(0.6), 0.0)
    with pytest.raises(AssertionError):
        r21_dec.changed_fraction(prev, cur[:3])
    with pytest.raises(AssertionError):
        r21_dec.cluster_shares(np.array([0, 4]), 4)


def test_changed_fraction_accepts_torch_argmax():
    a = torch.tensor([0, 1, 1, 0])
    b = torch.tensor([0, 1, 0, 1])
    assert r21_dec.changed_fraction(a, b) == pytest.approx(0.5)


def test_cluster_means_and_helpers_accept_tensor_labels():
    z = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 4.0]])
    m = r21_dec.cluster_means(z, torch.tensor([0, 0, 1]), 2)
    np.testing.assert_allclose(m.numpy(), [[1.0, 0.0], [10.0, 4.0]])
    assert r21_dec.cluster_shares(torch.tensor([0, 0, 1]), 2) == (pytest.approx(2 / 3), pytest.approx(1 / 3))


def test_kmeans_and_teacher_soft_return_cpu_tensors():
    z = torch.from_numpy(_blobs())
    assert r21_dec.kmeans_centres(z, 3, seed=0).device.type == 'cpu'
    x = _blobs(seed=1)
    c, _ = r21_dec.teacher_clusters(x, 3)
    t = r21_dec.teacher_soft(x, c)
    assert t.device.type == 'cpu' and not t.requires_grad


@pytest.mark.parametrize('dev', [
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')),
])
def test_every_helper_runs_on_the_device(dev):
    z = torch.from_numpy(_blobs()).to(dev)
    c = r21_dec.kmeans_centres(z, 3, seed=0)  # CPU result, as documented
    assert c.device.type == 'cpu'
    c = c.to(dev).requires_grad_(True)  # learnable centres
    with torch.no_grad():
        q_full = r21_dec.soft_assign(z, c, k=3, chunk=50)
        p_full = r21_dec.target(q_full)
    assert q_full.device.type == dev and p_full.device.type == dev
    q = r21_dec.soft_assign(z[:30], c)
    p = p_full[:30] / p_full[:30].sum(1, keepdim=True)
    r = q.detach().clone()
    s = dict(form='start', lam=1.0, warmup=1, stop='S1', balance=True)
    total, parts = r21_dec.dec_block(q, p, r, s)
    total.backward()
    assert total.device.type == dev and np.isfinite(parts['kl_p'])
    assert c.grad is not None and torch.isfinite(c.grad).all()
    am = q_full.argmax(1)
    assert r21_dec.changed_fraction(am, am) == 0.0
    hi, lo = r21_dec.cluster_shares(am, 3)
    assert 0 < lo <= hi <= 1
    lab = torch.arange(120, device=dev) // 40
    assert r21_dec.cluster_means(z, lab, 3).device.type == dev
    assert r21_dec.cluster_means(z, lab.cpu(), 3).device.type == dev
