"""Tests for r21_train: the DEC-off parity with the frozen loop, and the DEC hooks per head (CPU, tiny inputs)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r21_dec  # noqa: E402
import r21_train  # noqa: E402

ct2_d2, ms_train = r21_train.ct2_d2, r21_train.ms_train
ASPECTS = list(ms_train.ASPECTS)
GRAPH_ROWS = dict(emotion='Fres1', style='Fres05', genre='R0o')
BATCH = 64  # the tests shrink c*'s batch size (512) so that CPU steps are fast
N_FIT, N_DEV, N_ALL = 300, 100, 400
PAIRS = dict(emotion=100, style=160, genre=230)  # 2, 3, 4 steps per epoch at batch 64
K = 3


def _pairs(rng, n_nodes, count):
    ab = np.sort(rng.integers(0, n_nodes, size=(4 * count, 2)), axis=1)
    ab = np.unique(ab[ab[:, 0] < ab[:, 1]], axis=0)
    ab = ab[rng.permutation(len(ab))[:count]]
    assert len(ab) == count
    return ab.astype(np.int64), np.ones(count, dtype=np.float32)


@pytest.fixture(autouse=True)
def small_batch(monkeypatch):
    monkeypatch.setitem(ms_train.CONFIG, 'batch_size', BATCH)


@pytest.fixture
def world():
    """Three heads on 2048-dim inputs with three blobs; fit nodes 0..299, dev nodes 300..399."""
    rng = np.random.default_rng(5)
    row = copy.deepcopy(ms_train.ROWS['T1'])
    inputs, graphs = {}, {}
    for aspect in ASPECTS:
        blob = rng.normal(size=(K, 2048)).astype(np.float32)
        img = blob[rng.integers(0, K, N_ALL)] + 0.5 * rng.normal(size=(N_ALL, 2048)).astype(np.float32)
        cap = img + 0.5 * rng.normal(size=(N_ALL, 2048)).astype(np.float32)
        inputs[aspect] = dict(img=img, cap=cap)
        fit, dev = np.arange(N_FIT), np.arange(N_FIT, N_ALL)
        graphs[aspect] = dict(fit_node_ids=fit, dev_node_ids=dev, rows={GRAPH_ROWS[aspect]: dict(
            fit=_pairs(rng, N_FIT, PAIRS[aspect]), dev=_pairs(rng, N_DEV, 60))})
    row['graphs'] = dict(GRAPH_ROWS)
    return dict(row=row, inputs=inputs, graphs=graphs)


class Saved:
    """Stand-in for ct2_d2.save_output: keeps what the real one reads from the state, with its checks."""
    def __init__(self):
        self.out = {}

    def __call__(self, s, view, inputs, state, path, config, start):
        assert 1 <= state['best_epoch'] <= len(state['history']) <= config['max_epochs']
        assert np.isfinite(state['best'])
        self.out[config['aspect']] = dict(
            history=copy.deepcopy(state['history']), best_epoch=state['best_epoch'], best=state['best'],
            best_state=copy.deepcopy(state['best_state']), final_state=copy.deepcopy(view.state_dict()),
            state={k: copy.deepcopy(v) for k, v in state.items() if k in ('dec_start_epoch', 'stop_reason')})
        return dict(history=state['history'], best_epoch=state['best_epoch'])


def _make_config(s, row, name, aspect, aspects, manifests, seed, device, smoke, max_epochs):
    return dict(aspect=aspect, graph_row=row['graphs'][aspect], seed=seed, device=str(device), max_epochs=max_epochs)


def go(runner, world, tmp_path, monkeypatch, seed=0, max_epochs=3, **kwargs):
    saved = Saved()
    monkeypatch.setattr(ct2_d2, 'make_config', _make_config)
    monkeypatch.setattr(ct2_d2, 'save_output', saved)
    runner(None, world['inputs'], world['graphs'], None, world['row'], 'T', ASPECTS, seed,
           torch.device('cpu'), tmp_path, max_epochs=max_epochs, **kwargs)
    return saved.out


def _loss_history(entries):
    return [{k: v for k, v in e.items() if k in ('epoch', 'step', 'train_loss', 'dev_loss')} for e in entries]


def test_dec_off_equals_frozen_loop_bit_for_bit(world, tmp_path, monkeypatch):
    frozen = go(ct2_d2.run, world, tmp_path, monkeypatch)
    for dec in (None, dict(form='off', lam=None, warmup=None, stop=None, balance=None)):
        new = go(r21_train.run, world, tmp_path, monkeypatch, dec=dec)
        for aspect in ASPECTS:
            assert _loss_history(new[aspect]['history']) == _loss_history(frozen[aspect]['history'])
            assert all(set(e) == set(f) for e, f in zip(new[aspect]['history'], frozen[aspect]['history']))
            assert new[aspect]['best_epoch'] == frozen[aspect]['best_epoch']
            assert new[aspect]['best'] == frozen[aspect]['best']
            for key, value in frozen[aspect]['best_state'].items():
                assert torch.equal(new[aspect]['best_state'][key], value), key


# --- HeadDec on a fixed set of rows ------------------------------------------------------------------------------

def _blobs(n_per=30, d=16, k=K, seed=1):
    rng = np.random.default_rng(seed)
    centres = np.eye(d, dtype=np.float32)[:k]
    z = np.concatenate([centres[i] + 0.05 * rng.normal(size=(n_per, d)) for i in range(k)]).astype(np.float32)
    z = torch.from_numpy(z)
    return torch.nn.functional.normalize(z, dim=1)


def _settings(form='plain', lam=1.0, warmup=1, stop='S1', balance=False):
    return dict(form=form, lam=lam, warmup=warmup, stop=stop, balance=balance)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_lambda_zero_adds_exactly_zero_to_the_loss():
    z = _blobs()
    hd = r21_train.HeadDec(_settings(lam=0.0, balance=True), K, 0, torch.device('cpu'))
    hd.start(z)
    base = torch.tensor(1.2345678, requires_grad=True)
    ids = torch.arange(0, 90, 3)
    term = hd.block(Identity(), z, ids)
    assert term.item() == 0.0
    assert torch.equal(base + term, base)


def test_centres_receive_gradients_and_start_from_kmeans():
    z = _blobs()
    hd = r21_train.HeadDec(_settings(), K, 0, torch.device('cpu'))
    hd.start(z)
    assert isinstance(hd.centres, torch.nn.Parameter) and hd.centres.shape == (K, 16)
    # k-means finds the three blobs: each centre is near one coordinate axis.
    assert sorted(hd.centres.detach().argmax(1).tolist()) == [0, 1, 2]
    x = z.clone().requires_grad_(True)
    hd.block(Identity(), x, torch.arange(0, 90, 2)).backward()
    assert hd.centres.grad is not None and hd.centres.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_p_and_pull_targets_are_detached_and_indexed_by_fit_position():
    z = _blobs()
    hd = r21_train.HeadDec(_settings('start'), K, 0, torch.device('cpu'))
    hd.start(z)
    assert hd.p.shape == (90, K) and not hd.p.requires_grad and not hd.r.requires_grad
    # q0 is the start q, fixed: row i is the soft assignment of fit painting i.
    with torch.no_grad():
        expected = r21_dec.soft_assign(z, hd.centres.detach())
    assert torch.allclose(hd.r, expected)
    assert torch.allclose(hd.p.sum(0).sum(), torch.tensor(90.0), atol=1e-3)


def test_changed_fraction_first_epoch_vs_start_then_vs_previous():
    z = _blobs()
    hd = r21_train.HeadDec(_settings(), K, 0, torch.device('cpu'))
    hd.start(z)

    def epoch(swap=None):
        hd.block(Identity(), z, torch.arange(0, 90, 2))  # one batch, so the epoch has batch statistics
        if swap is not None:
            hd.centres.data = hd.centres.data[swap]
        return hd.end_epoch(z)

    first = epoch()
    assert first['changed'] == 0.0 and first['away'] == 0.0
    # Relabel the clusters (0->1->2->0): every painting changes cluster against the previous pass and the start.
    second = epoch(swap=[1, 2, 0])
    assert second['changed'] == 1.0 and second['away'] == 1.0
    # Unchanged since the previous epoch, still away from the start.
    third = epoch()
    assert third['changed'] == 0.0 and third['away'] == 1.0
    assert second['share_max'] == pytest.approx(1 / 3) and second['share_min'] == pytest.approx(1 / 3)


def test_teacher_form_starts_at_cluster_means_with_fixed_t():
    z = _blobs()
    feats = torch.nn.functional.normalize(z[:, :4] + 0.0, dim=1).numpy()  # the "teacher" sees a coarser view
    tcentres = np.eye(4, dtype=np.float32)[:K]
    hd = r21_train.HeadDec(_settings('teacher'), K, 0, torch.device('cpu'),
                           teacher=dict(feats=torch.from_numpy(feats), centres=torch.from_numpy(tcentres)))
    hd.start(z)
    t = r21_dec.teacher_soft(feats, tcentres)
    labels = t.argmax(1).numpy()
    for j in range(K):
        np.testing.assert_allclose(hd.centres.detach()[j].numpy(), z[labels == j].mean(0).numpy(), atol=1e-6)
    assert torch.allclose(hd.r, t)
    hd.block(Identity(), z, torch.arange(90))
    stats = hd.end_epoch(z)
    assert stats['pull'] is not None and stats['balance'] is None


# --- run with DEC on ---------------------------------------------------------------------------------------------

def _dec_data(world, form='plain', k=K):
    data = {}
    for aspect in ASPECTS:
        data[aspect] = dict(k=k)
        if form == 'teacher':
            rng = np.random.default_rng(3)
            centres = rng.normal(size=(k, 2048)).astype(np.float32)
            centres /= np.linalg.norm(centres, axis=1, keepdims=True)
            feats = centres[rng.integers(0, k, N_FIT)] + 0.1 * rng.normal(size=(N_FIT, 2048)).astype(np.float32)
            feats /= np.linalg.norm(feats, axis=1, keepdims=True)
            data[aspect].update(feats=feats.astype(np.float32), centres=centres)
    return data


def test_warmup_without_dec_then_dec_starts_per_head_at_its_own_epoch_w(world, tmp_path, monkeypatch):
    monkeypatch.setattr(r21_train, 'S1_CHANGED', -1.0)  # never stop on the change; the cap of epochs ends the run
    off = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=4)
    on = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=4, dec=_settings(warmup=2),
            dec_data=_dec_data(world))
    # Emotion has the fewest steps per epoch: its warm-up (epochs 1, 2) ends before any head starts DEC, so it
    # is the frozen loop's, bit for bit.
    assert _loss_history(on['emotion']['history'][:2]) == _loss_history(off['emotion']['history'][:2])
    steps_per_epoch = {a: -(-PAIRS[a] // BATCH) for a in ASPECTS}
    assert len(set(steps_per_epoch.values())) == 3
    for aspect in ASPECTS:
        history = on[aspect]['history']
        assert [e['dec']['phase'] for e in history] == ['warmup', 'warmup', 'dec', 'dec']
        assert history[1]['dec']['start_epoch'] == 3
        assert on[aspect]['state']['dec_start_epoch'] == 3
        # Every batch of a DEC epoch carries the DEC block: its own clock, its own step count.
        assert [e['dec']['batches'] for e in history[2:]] == [steps_per_epoch[aspect]] * 2
    # Heads reach epoch 2 at different steps.
    assert len({on[a]['history'][1]['step'] for a in ASPECTS}) == 3


def test_s1_stops_on_small_change_and_keeps_the_last_state(world, tmp_path, monkeypatch):
    monkeypatch.setattr(r21_train, 'S1_CHANGED', 1.1)  # always below: stop after the first DEC epoch
    on = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=6, dec=_settings(warmup=1),
            dec_data=_dec_data(world))
    for aspect in ASPECTS:
        o = on[aspect]
        assert len(o['history']) == 2 and o['best_epoch'] == 2
        assert o['best'] == o['history'][-1]['dev_loss']
        assert o['history'][-1]['dec']['stop_reason'] == 'changed' and o['state']['stop_reason'] == 'changed'
    # Heads stop on their own clocks; the last stopped head's last state is the final model state.
    last = max(ASPECTS, key=lambda a: on[a]['history'][-1]['step'])
    for key, value in on[last]['final_state'].items():
        assert torch.equal(on[last]['best_state'][key], value), key


def test_s1_stops_at_the_dec_epoch_cap(world, tmp_path, monkeypatch):
    monkeypatch.setattr(r21_train, 'S1_CHANGED', -1.0)
    monkeypatch.setattr(r21_train, 'S1_MAX_DEC_EPOCHS', 2)
    on = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=8, dec=_settings(warmup=1),
            dec_data=_dec_data(world))
    for aspect in ASPECTS:
        assert len(on[aspect]['history']) == 3 and on[aspect]['best_epoch'] == 3
        assert on[aspect]['state']['stop_reason'] == 'dec_epochs'


def _script_dev(monkeypatch, dev_losses):
    """Patch ms_train.head_loss: train calls as before, dev calls return scripted values and record the state."""
    real = ms_train.head_loss
    seen = {a: [] for a in ASPECTS}

    def head_loss(models, *args, aspect=None, include_mask_penalty=True, **kwargs):
        if include_mask_penalty:
            return real(models, *args, aspect=aspect, **kwargs)
        seen[aspect].append(copy.deepcopy(models.state_dict()))
        return torch.tensor(dev_losses[aspect][len(seen[aspect]) - 1])

    monkeypatch.setattr(ms_train, 'head_loss', head_loss)
    return seen


def test_s2_patience_counts_from_dec_start_and_best_state_is_among_dec_epochs(world, tmp_path, monkeypatch):
    monkeypatch.setitem(ms_train.CONFIG, 'patience', 2)
    # Epoch 1 (warm-up) has the lowest loss of all; DEC epochs: 5, 4 (best), 4.5, 4.75 -> stale 2 at epoch 5.
    script = [1.0, 5.0, 4.0, 4.5, 4.75, 0.125, 0.125]
    seen = _script_dev(monkeypatch, {a: script for a in ASPECTS})
    on = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=9, dec=_settings(warmup=1, stop='S2'),
            dec_data=_dec_data(world))
    for aspect in ASPECTS:
        o = on[aspect]
        assert len(o['history']) == 5 and o['best_epoch'] == 3 and o['best'] == 4.0
        assert [e['dev_loss'] for e in o['history']] == script[:5]  # the dev loss carries no DEC term
        assert o['state']['stop_reason'] == 'patience' and o['history'][-1]['dec']['stop_reason'] == 'patience'
        for key, value in seen[aspect][2].items():  # the state at the end of epoch 3
            assert torch.equal(o['best_state'][key], value), key


def test_s2_best_state_is_not_taken_in_warmup_when_dec_never_improves(world, tmp_path, monkeypatch):
    monkeypatch.setitem(ms_train.CONFIG, 'patience', 3)
    script = [1.0, 6.0, 6.0, 6.0, 6.0]  # DEC epoch 2 is the first of DEC: best_epoch 2 although epoch 1 was lower
    seen = _script_dev(monkeypatch, {a: script for a in ASPECTS})
    on = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=9, dec=_settings(warmup=1, stop='S2'),
            dec_data=_dec_data(world))
    for aspect in ASPECTS:
        assert on[aspect]['best_epoch'] == 2 and len(on[aspect]['history']) == 5
        for key, value in seen[aspect][1].items():
            assert torch.equal(on[aspect]['best_state'][key], value), key


def test_cap_holds_for_s2(world, tmp_path, monkeypatch):
    on = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=3, dec=_settings(warmup=1, stop='S2'),
            dec_data=_dec_data(world))
    for aspect in ASPECTS:
        assert len(on[aspect]['history']) == 3 and on[aspect]['state']['stop_reason'] == 'max_epochs'


@pytest.mark.parametrize('form', ['plain', 'start', 'teacher'])
def test_every_form_runs_and_records_its_dec_stats(world, tmp_path, monkeypatch, form):
    monkeypatch.setattr(r21_train, 'S1_CHANGED', -1.0)
    on = go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=3, dec=_settings(form, balance=True),
            dec_data=_dec_data(world, form))
    for aspect in ASPECTS:
        entry = on[aspect]['history'][-1]['dec']
        assert entry['phase'] == 'dec' and np.isfinite(entry['kl']) and np.isfinite(entry['balance'])
        assert (entry['pull'] is None) == (form == 'plain')
        assert 0.0 <= entry['changed'] <= 1.0 and 0.0 <= entry['away'] <= 1.0
        assert 0.0 <= entry['share_min'] <= entry['share_max'] <= 1.0
        assert on[aspect]['history'][1]['dec']['share_max'] > 0  # the shares at DEC start, in epoch W's entry
        json.dumps(on[aspect]['history'])  # the sidecar writes the history as JSON


def test_centres_are_their_own_adamw_group_and_get_updated(world, tmp_path, monkeypatch):
    made = []

    class Spy(torch.optim.AdamW):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    monkeypatch.setattr(torch.optim, 'AdamW', Spy)
    monkeypatch.setattr(r21_train, 'S1_CHANGED', -1.0)
    go(r21_train.run, world, tmp_path, monkeypatch, max_epochs=3, dec=_settings(), dec_data=_dec_data(world))
    opt = made[0]
    centre_groups = opt.param_groups[1:]
    assert len(opt.param_groups) == 1 + len(ASPECTS)
    for group in centre_groups:
        assert group['lr'] == ms_train.CONFIG['lr'] and group['weight_decay'] == 0.0
        (p,) = group['params']
        assert p.shape == (K, 2048) and len(opt.state[p]) > 0  # AdamW stepped it: it got gradients
    assert opt.param_groups[0]['weight_decay'] == ms_train.CONFIG['weight_decay']


def test_dec_settings_are_asserted(world, tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        go(r21_train.run, world, tmp_path, monkeypatch, dec=_settings(warmup=3), dec_data=_dec_data(world))  # W = cap
    with pytest.raises(AssertionError):
        go(r21_train.run, world, tmp_path, monkeypatch, dec=_settings(), dec_data=None)
