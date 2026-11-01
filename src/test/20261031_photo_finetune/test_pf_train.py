"""Ticket 05: the emotion-only trainer on tiny synthetic data (CPU), and gate (vi) (slow, GPU, controller)."""
import json

import numpy as np
import pytest
import torch

import pf_paths
import pf_train
import rs_vh

N_FIT, N_VAL = 300, 80
CPU = torch.device('cpu')


def _unit(rng, n):
    x = rng.standard_normal((n, 2048)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _part(rng, n, tag):
    return dict(keys=[f'{tag}/img{i:04d}.jpg' for i in range(n)], I=_unit(rng, n), D=_unit(rng, n),
                E=_unit(rng, n), T=_unit(rng, n))


def _pairs(rng, n, m):
    a = rng.integers(0, n, 4 * m)
    b = rng.integers(0, n, 4 * m)
    keep = a != b
    p = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1)[keep], axis=0)
    return p[:m].astype(np.int64)


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    fit, val = _part(rng, N_FIT, 'fit'), _part(rng, N_VAL, 'val')
    graph = dict(fit_pairs=_pairs(rng, N_FIT, 700), dev_pairs=_pairs(rng, N_VAL, 150), manifest_sha256='ab' * 32)
    return fit, val, graph


def _settings(**kw):
    return dict(pf_paths.GRID[1], **kw)


def _run(name, settings, data, tmp_path, seed=100, **kw):
    fit, val, graph = data
    return pf_train.train_run(name, settings, 0, seed, fit, val, graph, 'cpu', outputs=tmp_path, **kw)


def test_random_start_two_epochs(tmp_path, data):
    fit = data[0]
    r = _run('sweep/row8', _settings(start='random'), data, tmp_path, max_epochs=2)
    assert r['status'] == 'ok' and r['failure_kind'] is None and r['error'] is None
    assert json.loads((tmp_path / 'runs/sweep/row8/run.json').read_text()) == r
    for key in ('settings', 'cstar_seed', 'cstar_weights_sha256', 'train_seed', 'weights_sha256', 'best_epoch',
                'best_dev_loss', 'history', 'graph_manifest_sha256', 'fit_keys_sha256', 'val_keys_sha256',
                'device', 'seconds', 'status', 'failure_kind', 'error'):
        assert key in r
    assert r['cstar_weights_sha256'] is None and r['graph_manifest_sha256'] == 'ab' * 32
    assert r['train_seed'] == 100 and r['best_epoch'] in (1, 2)
    assert [h['epoch'] for h in r['history']] == [1, 2]
    assert all(set(h) >= {'train_loss', 'dev_loss', 'seconds'} for h in r['history'])
    assert r['best_dev_loss'] == min(h['dev_loss'] for h in r['history'])
    assert r['fit_keys_sha256'] == rs_vh.names_sha(fit['keys'])
    ckpt = tmp_path / 'runs/sweep/row8/emotion.pt'
    assert rs_vh.file_sha(ckpt) == r['weights_sha256']
    state = torch.load(ckpt, weights_only=True)
    assert tuple(state) == pf_train.CKPT_KEYS
    assert state['img.trunk.0.weight'].shape == (1024, 2048) and state['cap.output.weight'].shape == (2048, 1024)
    assert all(v.dtype == torch.float32 for v in state.values())


def test_same_seed_same_weights_and_refuses_overwrite(tmp_path, data):
    s = _settings(start='random')
    a, b = _run('a', s, data, tmp_path, 5, max_epochs=1), _run('b', s, data, tmp_path, 5, max_epochs=1)
    c = _run('c', s, data, tmp_path, 6, max_epochs=1)
    assert a['weights_sha256'] == b['weights_sha256'] != c['weights_sha256']
    with pytest.raises(AssertionError):
        _run('a', s, data, tmp_path, 5, max_epochs=1)


def _fake_cstar(root, seed=3):
    root.mkdir(parents=True)
    _, view = pf_train.build_emotion(CPU, seed=seed)
    path = root / 'emotion__s0.pt'
    torch.save({k: v.detach().clone() for k, v in view.state_dict().items()}, path)
    path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256=rs_vh.file_sha(path))))
    return path


def test_cstar_start_hash_checked(tmp_path, data):
    path = _fake_cstar(tmp_path / 'cstar')
    r = _run('x', _settings(), data, tmp_path, max_epochs=1, cstar_root=tmp_path / 'cstar')
    assert r['status'] == 'ok' and r['cstar_weights_sha256'] == rs_vh.file_sha(path)
    # a start point that does not match its sidecar is a crash record, no checkpoint
    path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256='0' * 64)))
    r = _run('y', _settings(), data, tmp_path, max_epochs=1, cstar_root=tmp_path / 'cstar')
    assert r['status'] == 'failed' and r['failure_kind'] == 'crash' and 'sidecar' in r['error']
    assert not (tmp_path / 'runs/y/emotion.pt').exists()


def test_cstar_start_is_used(tmp_path, data):
    """Same training seed, c* start against random start: different end weights."""
    _fake_cstar(tmp_path / 'cstar')
    a = _run('a', _settings(), data, tmp_path, max_epochs=1, cstar_root=tmp_path / 'cstar')
    b = _run('b', _settings(start='random'), data, tmp_path, max_epochs=1)
    assert a['weights_sha256'] != b['weights_sha256']


def test_nan_gives_failed_record(tmp_path, data):
    data[0]['I'][3] = np.nan
    r = _run('n', _settings(start='random'), data, tmp_path, max_epochs=2)
    assert r['status'] == 'failed' and r['failure_kind'] == 'nonfinite_loss' and 'train' in r['error']
    assert r['weights_sha256'] is None and r['best_epoch'] is None
    assert not (tmp_path / 'runs/n/emotion.pt').exists()
    assert json.loads((tmp_path / 'runs/n/run.json').read_text())['status'] == 'failed'


def test_nan_in_validation_gives_failed_record(tmp_path, data):
    data[1]['D'][:] = np.nan
    r = _run('v', _settings(start='random'), data, tmp_path, max_epochs=2)
    assert r['status'] == 'failed' and r['failure_kind'] == 'nonfinite_loss' and 'validation' in r['error']


def test_bad_pairs_is_a_crash_record(tmp_path, data):
    data[2]['fit_pairs'] = data[2]['fit_pairs'] + N_FIT                 # rows out of range
    r = _run('c', _settings(start='random'), data, tmp_path, max_epochs=2)
    assert r['status'] == 'failed' and r['failure_kind'] == 'crash' and r['error']
    assert not (tmp_path / 'runs/c/emotion.pt').exists()


def test_dev_loss_batching(data):
    """dev_loss is the pair-weighted mean over fixed-order batches: one batch of all pairs equals batches of 64
    up to float error (the in-batch negatives differ, so only the one-batch case is exact against itself)."""
    _, val, graph = data
    row, switches = pf_train.loss_recipe(0.25)
    _, view = pf_train.build_emotion(CPU, seed=1)
    t = pf_train._tensors(val, CPU)
    p = graph['dev_pairs']
    a = pf_train.dev_loss(view, t, p, row, switches, batch_size=len(p))
    assert np.isfinite(a) and a == pf_train.dev_loss(view, t, p, row, switches, batch_size=len(p))
    # the loss is the model's: with a mean over 2 equal halves the weighted mean is the mean of the halves
    h = len(p) // 2
    left = pf_train.dev_loss(view, t, p[:h], row, switches, batch_size=h)
    right = pf_train.dev_loss(view, t, p[h:2 * h], row, switches, batch_size=h)
    both = pf_train.dev_loss(view, t, p[:2 * h], row, switches, batch_size=h)
    assert abs(both - (left + right) / 2) < 1e-6


@pytest.mark.slow
@pytest.mark.real_data
def test_gate_vi_loss_replay(tmp_path):
    """Controller, GPU: c*'s seed-0 emotion checkpoint on its full-scale ArtELingo validation inputs gives
    BEST_DEV_LOSS_S0 within 1e-4. Writes OUTPUTS/stage0a/loss.json (here: under tmp_path)."""
    result = pf_train.gate_vi(outputs=tmp_path, device='cuda')
    assert result['ok'], result
    assert abs(result['value'] - pf_paths.BEST_DEV_LOSS_S0) <= pf_paths.TOL['vi']
    assert (tmp_path / 'stage0a/loss.json').is_file()
