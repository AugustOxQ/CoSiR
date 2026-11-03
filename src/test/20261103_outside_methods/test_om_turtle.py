"""Tests for om_turtle (r18 ticket 02): our TURTLE loop against the official run's saved reference (gate (ii), CPU).

Verdict code: each test names the break that makes it fail.
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m pytest -q test_om_turtle.py
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om_paths as P  # noqa: E402
import om_turtle as T  # noqa: E402


@pytest.fixture(scope='module')
def ref():
    inputs = np.load(P.parity_file('inputs'))
    turtle = np.load(P.parity_file('turtle'))
    return inputs['qwen'], turtle


@pytest.fixture(scope='module')
def cpu_fit(ref):
    z, _ = ref
    c = P.PARITY_TURTLE
    return T.fit(z, c['K'], c['seed'], 'cpu', outer_steps=c['outer_steps'], threads=c['threads'])


def test_cpu_gate_weights_bias_partition(ref, cpu_fit):
    # Fails if the loss, the init draw order, the cold start or the weight norm differs from the run.
    _, t = ref
    assert cpu_fit['weight'].shape == t['weight'].shape and cpu_fit['weight'].dtype == np.float32
    assert cpu_fit['bias'].shape == t['bias'].shape and cpu_fit['bias'].dtype == np.float32
    assert np.abs(cpu_fit['weight'] - t['weight']).max() < P.TOL_TURTLE
    assert np.abs(cpu_fit['bias'] - t['bias']).max() < P.TOL_TURTLE
    assert np.array_equal(cpu_fit['labels'], t['partition'])
    assert cpu_fit['labels'].dtype == np.int32


def test_fit_moves_the_weights(ref, cpu_fit):
    # A fit that never updates the task encoder would keep its init: one step and 100 steps must differ.
    z, _ = ref
    first = T.fit(z, 9, 0, 'cpu', outer_steps=1, threads=8)
    assert np.abs(cpu_fit['weight'] - first['weight']).max() > 1e-4


def test_result_fields(ref, cpu_fit):
    z, _ = ref
    r = cpu_fit
    assert set(r) >= {'labels', 'weight', 'bias', 'loss', 'entropy', 'n_clusters', 'seconds'}
    assert r['labels'].shape == (len(z),)
    assert r['n_clusters'] == len(np.unique(r['labels']))
    assert 0 < r['entropy'] <= np.log(9) + 1e-6          # entropy of a mean prediction over 9 clusters
    assert np.isfinite(r['loss']) and r['seconds'] > 0


def test_labels_are_argmax_of_effective_weights(ref, cpu_fit):
    # Fails if the returned weight is the raw direction (before the norm) or the labels come from another head.
    z, _ = ref
    logits = z @ cpu_fit['weight'].T + cpu_fit['bias']
    assert np.array_equal(logits.argmax(1), cpu_fit['labels'])


def test_separated_blobs_recovered():
    # Behavioural: three far-apart blobs come out as three clusters with the right members, K 3.
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.RandomState(0)
    centres = np.eye(3, 16, dtype=np.float32) * 5
    truth = np.repeat(np.arange(3), 100)
    z = (centres[truth] + 0.1 * rng.randn(300, 16)).astype(np.float32)
    r = T.fit(z, 3, 0, 'cpu', outer_steps=300, threads=2)
    assert r['n_clusters'] == 3
    assert adjusted_rand_score(truth, r['labels']) > 0.95


def test_batch_indices_equal_numpy_sequence():
    # Fails if the batch is drawn with replacement, with another size, or from a stream not seeded like the run.
    for n in P.BATCH_DRAW_CHECK['n']:
        np.random.seed(0)
        want = [np.random.choice(n, size=min(10000, n), replace=False) for _ in range(3)]
        got = T.batch_indices(n, 0, 3)
        assert len(got) == 3
        for g, w in zip(got, want):
            assert np.array_equal(g, w)
    two = T.batch_indices(500, 0, 2)
    assert not np.array_equal(two[0], two[1])                              # consecutive draws differ
    assert len(T.batch_indices(12281, 0, 1)[0]) == 10000                   # batch capped at 10,000
    assert len(T.batch_indices(16384, 0, 1)[0]) == 10000
    assert len(T.batch_indices(2000, 0, 1)[0]) == 2000                     # and at n


def test_fit_does_not_touch_global_numpy_stream():
    np.random.seed(123)
    a = np.random.rand()
    np.random.seed(123)
    z = np.random.RandomState(1).randn(50, 8).astype(np.float32)
    T.fit(z, 2, 0, 'cpu', outer_steps=2, threads=2)
    assert np.random.rand() == a


def test_cold_start_and_init_draw_order(monkeypatch):
    # The torch stream: task encoder init, one inner init before the loop, then one fresh inner init each outer step.
    # Fails if the inner model is kept between steps (warm start), or if the draws come in another order.
    d, K, steps, seed = 12, 4, 3, 5
    z = np.random.RandomState(0).randn(40, d).astype(np.float32)
    seen = []
    orig = T._init_inner

    def spy(*a, **k):
        out = orig(*a, **k)
        seen.append([t.detach().clone() for t in out])
        return out

    monkeypatch.setattr(T, '_init_inner', spy)
    T.fit(z, K, seed, 'cpu', outer_steps=steps, threads=1)
    assert len(seen) == steps + 1
    torch.manual_seed(seed)
    torch.nn.Linear(d, K)                                   # the task encoder's init
    for got in seen:
        lin = torch.nn.Linear(d, K)
        assert torch.equal(got[0], lin.weight) and torch.equal(got[1], lin.bias)
    assert not torch.equal(seen[1][0], seen[2][0])           # different draws


def test_no_label_argument():
    params = inspect.signature(T.fit).parameters
    assert list(params) == ['z', 'K', 'seed', 'device', 'outer_steps', 'threads']
    assert not any('label' in p or p == 'y' for p in params)


def test_rejects_bad_input():
    z = np.zeros((10, 4), np.float32)
    with pytest.raises(AssertionError):
        T.fit(z.astype(np.float64), 3, 0, 'cpu', outer_steps=1)
    with pytest.raises(AssertionError):
        T.fit(z, 1, 0, 'cpu', outer_steps=1)
    bad = z.copy()
    bad[0, 0] = np.nan
    with pytest.raises(AssertionError):
        T.fit(bad, 3, 0, 'cpu', outer_steps=1)


def test_defaults_are_the_official_ones():
    # Fails if a default drifts from om_paths.TURTLE (gamma 10, 10 inner steps, both rates 1e-3, batch 10,000).
    assert T.fit.__defaults__[0] == P.TURTLE['outer_steps'] == 6000
    assert (T.GAMMA, T.INNER_STEPS, T.OUTER_LR, T.INNER_LR, T.BATCH) == (
        P.TURTLE['gamma'], P.TURTLE['inner_steps'], P.TURTLE['outer_lr'], P.TURTLE['inner_lr'], P.TURTLE['batch_size'])
    assert P.TURTLE['warm_start'] is False


def test_gate_ii_cpu_part_writes_json(tmp_path, monkeypatch):
    # The CLI's CPU part: the gate file carries both differences and the pass flag; no CUDA part, so pass stays false.
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(T, 'OUT', tmp_path)
    res = T.gate_ii('cpu')
    assert res['cpu']['pass'] is True and res['cpu']['partition_identical'] is True
    assert res['cpu']['batch_indices_identical'] is True
    assert res['cpu']['max_abs_weight'] < P.TOL_TURTLE
    import json
    on_disk = json.loads((tmp_path / 'stage0' / 'gate_ii.json').read_text())
    assert on_disk['cpu']['pass'] is True and 'cuda' not in on_disk and on_disk['pass'] is False


def _oracle(z, K, seed, steps):
    """The official order restated plainly: global numpy stream, torch's own weight norm, a fresh inner model per step."""
    n, d = z.shape
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = torch.nn.utils.weight_norm(torch.nn.Linear(d, K))
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    new_inner = lambda: (lambda m: (m, torch.optim.Adam(m.parameters(), lr=1e-3)))(torch.nn.Linear(d, K))
    new_inner()
    for _ in range(steps):
        zb = torch.from_numpy(z[np.random.choice(n, size=min(10000, n), replace=False)])
        p = torch.softmax(enc(zb), 1)
        m, o = new_inner()
        for _ in range(10):
            o.zero_grad()
            torch.nn.functional.cross_entropy(m(zb), p.detach()).backward()
            o.step()
        opt.zero_grad()
        err = torch.nn.functional.cross_entropy(m(zb).detach(), p)
        (err - 10.0 * torch.special.entr(p.mean(0)).sum()).backward()
        opt.step()
    torch.nn.utils.remove_weight_norm(enc)
    return enc.weight.detach().numpy(), enc.bias.detach().numpy()


@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('seed', [0, 3])
def test_fit_equals_independent_oracle_on_subset_batches(seed):
    # n 12,281 > 10,000, so each batch is a real subset. Fails if fit's batch stream is seeded another way, the batch
    # is not the capped draw, one batch is reused, or fit ignores the drawn indices (all four sabotages differ > 1e-4).
    rng = np.random.RandomState(7)
    z = (rng.randn(5, 32)[rng.randint(5, size=12281)] * 3 + rng.randn(12281, 32)).astype(np.float32)
    w, b = _oracle(z, 5, seed, 6)
    f = T.fit(z, 5, seed, 'cpu', outer_steps=6, threads=8)
    assert np.abs(f['weight'] - w).max() < 1e-6 and np.abs(f['bias'] - b).max() < 1e-6


def test_fit_uses_the_batch_draws_generator(monkeypatch):
    # Gate (ii) checks batch_indices; fit must take its batches from the same generator.
    calls = []
    orig = T._batch_draws
    monkeypatch.setattr(T, '_batch_draws', lambda n, seed: calls.append((n, seed)) or orig(n, seed))
    T.fit(np.random.RandomState(1).randn(50, 8).astype(np.float32), 2, 4, 'cpu', outer_steps=1, threads=1)
    assert calls == [(50, 4)]


def test_gate_pass_needs_both_parts_with_the_current_code_hash(tmp_path, monkeypatch, cpu_fit):
    # Fails if a part from older code (different hash) or a missing hash can leave the top-level pass true.
    import json
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(T, 'OUT', tmp_path)
    monkeypatch.setattr(T, 'fit', lambda *a, **k: cpu_fit)                 # the same fit, run once
    path = tmp_path / 'stage0' / 'gate_ii.json'
    path.parent.mkdir(parents=True)
    cuda = {'pass': True, 'code_sha256': 'old'}
    path.write_text(json.dumps({'cuda': cuda}))
    res = T.gate_ii('cpu')
    assert res['cpu']['code_sha256'] == T._code_sha256() and res['pass'] is False     # stale cuda part
    cuda['code_sha256'] = T._code_sha256()
    path.write_text(json.dumps({'cuda': cuda}))
    assert T.gate_ii('cpu')['pass'] is True
    path.write_text(json.dumps({'cuda': {'pass': True}}))
    assert T.gate_ii('cpu')['pass'] is False                                           # part without a hash
