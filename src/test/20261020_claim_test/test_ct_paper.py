"""Paper recipe, epoch-level DEC and dual-cut artifact contract."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_percept as ct
from test_ct_percept import sample


def paper():
    return importlib.import_module('ct_paper_sept')


def test_paper_pretraining_uses_mirrored_third_layer_and_noisy_latent_clean_target(monkeypatch):
    p = paper()
    encoder, decoder = p.build_autoencoder(3)
    assert [(m.in_features, m.out_features) for m in encoder if isinstance(m, nn.Linear)] == [
        (3, 500), (500, 500), (500, 500), (500, 2000), (2000, 128)]
    assert [(m.in_features, m.out_features) for m in decoder if isinstance(m, nn.Linear)] == [
        (128, 2000), (2000, 500), (500, 500), (500, 500), (500, 3)]
    encoder, decoder = nn.Linear(2, 2), nn.Linear(2, 2)
    with torch.no_grad():
        encoder.weight.zero_()
        encoder.bias.zero_()
        decoder.weight.copy_(torch.eye(2))
        decoder.bias.zero_()
    monkeypatch.setattr(torch, 'randn_like', lambda z: torch.ones_like(z))
    losses = p.pretrain_autoencoder(encoder, decoder, torch.zeros(512, 2), 'cpu', epochs=1)
    assert losses == pytest.approx([.01])
    assert encoder.bias.detach().tolist() == pytest.approx([-.001, -.001])


@pytest.mark.parametrize('epochs', [100, 200])
def test_cosine_uses_each_phase_maximum_and_reaches_paper_floor(epochs):
    optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=.001)
    scheduler = paper().cosine_scheduler(optimizer, epochs)
    assert optimizer.param_groups[0]['lr'] == .001
    for epoch in range(epochs):
        optimizer.step()
        scheduler.step()
        if epoch + 1 == epochs // 2:
            assert optimizer.param_groups[0]['lr'] == pytest.approx(.000505)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.00001)


class EpochAssignments(nn.Module):
    """A fixed test encoder with one hard topic changing after the first epoch."""
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1))
        self.evaluations = 0

    def forward(self, x):
        z = x + self.offset
        if not self.training:
            if self.evaluations:
                z = z.clone()
                z[0] = 10
            self.evaluations += 1
        return z


@pytest.mark.parametrize(('rows', 'epochs'), [(1001, 1), (1000, 2), (999, 2)])
def test_dec_stops_strictly_below_one_tenth_percent(rows, epochs):
    encoder = EpochAssignments()
    decoder = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        decoder.weight.fill_(1)
    centers = nn.Parameter(torch.tensor([[0.], [10.]]))
    history, reason, stop_epoch = paper().train_dec_until_stable(
        encoder, decoder, centers, torch.zeros(rows, 1), 'cpu', max_epochs=4)
    assert stop_epoch == epochs and reason == 'stability criterion'
    assert history[0]['fraction_changed'] == 1 / rows
    assert len(history) == epochs


def test_dec_refreshes_detached_global_target_once_per_epoch_and_shuffles_full_fit(monkeypatch):
    p = paper()
    torch.manual_seed(3)
    x = torch.linspace(-1, 1, 1025)[:, None]
    encoder, decoder = nn.Linear(1, 1), nn.Linear(1, 1)
    centers = nn.Parameter(torch.tensor([[-1.], [2.]]))
    refreshes, minibatches = [], []
    target = ct.pc_sept.target_distribution

    def observe(q):
        assert not q.requires_grad and q.shape == (1025, 2)
        with torch.no_grad():
            expected = ct.pc_sept.soft_assignments(encoder(x), centers)
        torch.testing.assert_close(q, expected)
        result = target(q)
        assert not result.requires_grad
        refreshes.append(result.clone())
        return result

    def batch(module, args):
        if torch.is_grad_enabled():
            minibatches.append(args[0].clone())

    monkeypatch.setattr(ct.pc_sept, 'target_distribution', observe)
    handle = encoder.register_forward_pre_hook(batch)
    history, reason, stop_epoch = p.train_dec_until_stable(
        encoder, decoder, centers, x, 'cpu', max_epochs=3, stability_threshold=0)
    handle.remove()
    assert stop_epoch == 3 and reason == 'epoch ceiling' and len(refreshes) == 3
    assert not torch.equal(refreshes[0], refreshes[1])
    assert len(minibatches) == 9
    for epoch in range(3):
        batches = minibatches[3 * epoch:3 * epoch + 3]
        assert list(map(len, batches)) == [512, 512, 1]
        ordered = torch.cat(batches).ravel()
        torch.testing.assert_close(ordered.sort().values, x.ravel())
        assert not torch.equal(ordered, x.ravel())
    for row in history:
        assert row['total'] == pytest.approx(row['kl'] + row['reconstruction'], abs=1e-7)
        assert 'balance' not in row


@pytest.mark.parametrize('version', ['clip_roberta', 'qwen_qwen'])
def test_paper_stage1_saves_one_run_with_both_cuts_and_rcol_inputs(sample, version):
    path = ct.run_stage1(sample, 'paper', version, 0, smoke=True, device='cpu')
    meta = ct.verify_output(path)
    assert meta['K'] == 40 and meta['K_by_cut']['k40'] == 40
    config = meta['config']
    assert config['encoder'] == [config['input_dim'], 500, 500, 500, 2000, 128]
    assert config['decoder'] == config['encoder'][::-1]
    assert config['pretrain_batch_size'] == config['joint_batch_size'] == 512
    assert config['pretrain_lr'] == config['joint_lr'] == .001
    assert config['latent_noise_std'] == .1 and config['balance_weight'] == 0
    assert config['scheduler'] == 'cosine per epoch' and config['eta_min'] == 1e-5
    assert config['initial_centers'] == 100
    state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
    encoder, _ = paper().build_autoencoder(config['input_dim'])
    encoder.load_state_dict(state['encoder'])
    with np.load(path) as z:
        assert z['fit_latent'].shape == (2000, 128) and z['fit_latent'].dtype == np.float32
        assert z['centers'].shape == (100, 128) and z['centers'].dtype == np.float32
        assert z['stop_epoch'].item() == meta['epochs']['joint'] == meta['stop_epoch']
        with torch.no_grad():
            q = ct.pc_sept.soft_assignments(torch.from_numpy(z['fit_latent']), state['centers']).numpy()
        np.testing.assert_array_equal(z['fit_counts'], np.bincount(q.argmax(1), minlength=100))
        import ct_rcol
        for cut, mask in [('strict', ct_rcol.strict_mask(z['center_norms'])),
                          ('k40', ct_rcol.k40_mask(z['fit_counts']))]:
            np.testing.assert_array_equal(z[f'kept_{cut}'], mask)
            np.testing.assert_array_equal(z[f'fit_topics_{cut}'], q[:, mask].argmax(1))
            np.testing.assert_array_equal(z[f'fit_topics_{cut}'], z[f'topic_{cut}'][:2000])
            assert z[f'topic_{cut}'].shape == sample['pid'].shape
            assert z[f'q_paint_eval_{cut}'].shape == (5000, mask.sum())
            np.testing.assert_allclose(z[f'q_paint_eval_{cut}'].sum(1), 1, atol=1e-6)
            topic, k = ct.load_stage1(sample, path, version, 0, cut=cut)
            assert k == mask.sum() == meta['K_by_cut'][cut]
            np.testing.assert_array_equal(topic, z[f'topic_{cut}'])


def test_near_tied_fit_q_gives_counts_and_topics_that_lose_no_pairs(sample, monkeypatch):
    """Counts and topics must come from one Q: after training, Q is uniform to 1e-8 and
    differs on every call, so any second pass over the fit pairs breaks the R-COL check."""
    import ct_paper_sept
    original_q = ct.pc_sept.soft_assignments
    original_train = ct_paper_sept.train_dec_until_stable
    state = dict(noisy=False, calls=0)

    def noisy_q(z, centers):
        if not state['noisy']:
            return original_q(z, centers)
        state['calls'] += 1
        generator = torch.Generator().manual_seed(state['calls'])
        noise = torch.randn(len(z), len(centers), generator=generator) * 1e-8
        q = (1 / len(centers) + noise).to(z.device)
        return (q / q.sum(dim=1, keepdim=True)).float()

    def train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        state['noisy'] = True
        return result

    monkeypatch.setattr(ct.pc_sept, 'soft_assignments', noisy_q)
    monkeypatch.setattr(ct_paper_sept, 'train_dec_until_stable', train)
    path = ct.run_stage1(sample, 'paper', 'clip_roberta', 0, smoke=True, device='cpu')
    ct.verify_output(path)
    with np.load(path) as z:
        counts = z['fit_counts']
        assert counts.sum() == 2000
        for cut in ('strict', 'k40'):
            kept, topics = z[f'kept_{cut}'], z[f'fit_topics_{cut}']
            local = np.bincount(topics, minlength=int(kept.sum()))
            assert int(np.maximum(counts[kept] - local, 0).sum()) == 0, cut
            if kept.all():
                assert np.array_equal(local, counts)
    assert state['calls'] > 0


def test_paper_stage2_width_must_equal_its_stage1_kept_count(sample):
    import shutil
    stage1 = ct.run_stage1(sample, 'paper', 'clip_roberta', 0, smoke=True, device='cpu')
    previous = ct.verify_output(stage1)
    good = ct.run_stage2(sample, 'paper', 'k40', 'P1', 0, smoke=True, device='cpu')
    assert ct.verify_output(good)['K'] == previous['K_by_cut']['k40'] == 40
    assert previous['K_by_cut']['strict'] != 40
    # A k40-width mapper filed under the strict cut: payload hashes are all intact.
    bad = ct.stage2_path('paper', 'strict', 'P1', 0)
    bad.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(good.with_suffix('.npz'), bad)
    shutil.copy(good.with_suffix('.pt'), bad.with_suffix('.pt'))
    meta = json.loads(good.with_suffix('.json').read_text())
    meta['config']['cut'] = 'strict'
    bad.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match='Stage 1 kept count'):
        ct.verify_output(bad)
    expected = dict(settings='paper', cut='strict', arm='P1', seed=0, smoke=True)
    assert not ct.complete_output(bad, expected, 'synthetic-sample', previous)
    with pytest.raises(AssertionError, match='Stage 1 kept count'):
        ct.run_held('paper', 'strict', 'P1', 0, smoke=True)


def test_paper_mapper_uses_64_row_batches_and_weights_tail_loss_by_rows(monkeypatch):
    p = paper()
    config = dict(seed=7, device='cpu', input_dim=2, mapper='attention', epochs=1, arm='P1')
    torch.manual_seed(7)
    initial = ct.build_mapper(config, 2)
    bias = initial.classifier.bias.detach()
    # Worked two-batch example: 64 positive rows, then one negative row.
    first = torch.nn.functional.softplus(-bias).mean().item()
    tail = torch.nn.functional.softplus(bias + .001).mean().item()
    expected = (64 * first + tail) / 65
    monkeypatch.setattr(torch, 'randperm', lambda n, **kwargs: torch.arange(n, **kwargs))
    inputs = {part: torch.zeros(n, 50, 2) for part, n in [('fit', 65), ('dev', 2), ('eval', 3)]}
    targets = dict(fit=np.r_[np.ones((64, 2)), np.zeros((1, 2))].astype(np.float32))
    eval_prob, dev_prob, history, seconds, state = p.train_mapper(inputs, targets, config, 2)
    assert history == pytest.approx([expected], abs=1e-7)
    assert abs(history[0] - (first + tail) / 2) > .01
    assert len(seconds) == 1 and seconds[0] >= 0
    initial.load_state_dict(state)
    with torch.no_grad():
        np.testing.assert_array_equal(eval_prob, initial(inputs['eval']).sigmoid().numpy())
    assert dev_prob.shape == (2, 2)


@pytest.mark.parametrize('arm', ['P1', 'P4'])
def test_each_paper_cut_trains_its_mapper_and_replays_probabilities(sample, arm):
    stage1 = ct.run_stage1(sample, 'paper', ct.ARMS[arm][0], 0, smoke=True, device='cpu')
    before = ct.file_sha256(stage1)
    widths = []
    for cut in ('strict', 'k40'):
        path = ct.run_stage2(sample, 'paper', cut, arm, 0, smoke=True, device='cpu')
        meta = ct.verify_output(path)
        assert meta['config']['batch_size'] == 64
        assert meta['config']['lr'] == .001 and meta['config']['eta_min'] == 1e-5
        assert meta['config']['scheduler'] == 'cosine per epoch'
        assert len(meta['loss_history']) == 3
        assert meta['stage1_sha256'] == before
        widths.append(meta['K'])
        names = sample['all_painting_ids'][sample['subset_pid']].tolist()
        prob = ct.infer_stage2('paper', cut, arm, 0, names)
        with np.load(path) as z:
            np.testing.assert_array_equal(prob, z['eval_prob'])
        assert prob.shape == (5000, meta['K']) and prob.dtype == np.float32
        # The same image-only seam used by held inference preserves caller order.
        reordered = ct.infer_stage2('paper', cut, arm, 0, names[::-1])
        np.testing.assert_allclose(reordered, prob[::-1], atol=1e-7)
    assert widths[0] != widths[1] == 40
    assert ct.file_sha256(stage1) == before


def test_core_calls_original_trainers_without_importing_paper(sample, monkeypatch):
    calls = []
    for module, name in [(ct.pc_sept, 'pretrain_autoencoder'),
                         (ct.pc_sept, 'train_dec_until_stable'), (ct.pc_stage2, 'train_mapper')]:
        original = getattr(module, name)

        def observe(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, observe)
    monkeypatch.setitem(sys.modules, 'ct_paper_sept', None)
    ct.run_stage1(sample, 'core', 'clip_roberta', 0, smoke=True, device='cpu')
    ct.run_stage2(sample, 'core', 'k40', 'P2', 0, smoke=True, device='cpu')
    assert calls == ['pretrain_autoencoder', 'train_dec_until_stable', 'train_mapper']


def test_paper_cli_resumes_cut_mapper_and_held_uses_checkpoint_device(sample, monkeypatch):
    monkeypatch.setattr(ct, 'load_frozen', lambda: sample)
    stage1_args = ['--stage', '1', '--settings', 'paper', '--version', 'clip_roberta',
                   '--seeds', '0', '--smoke', '--device', 'cpu']
    [stage1] = ct.main(stage1_args)
    stamp = stage1.stat().st_mtime_ns
    assert ct.main(stage1_args) == [stage1] and stage1.stat().st_mtime_ns == stamp
    args = ['--settings', 'paper', '--cut', 'strict', '--arm', 'P1', '--seeds', '0', '--smoke']
    [mapper] = ct.main(['--stage', '2', '--device', 'cpu'] + args)
    stamp = mapper.stat().st_mtime_ns
    assert ct.main(['--stage', '2', '--device', 'cpu'] + args) == [mapper]
    assert mapper.stat().st_mtime_ns == stamp
    names = sorted(f'held-{i}' for i in range(12281))
    sha = ct.ct_held.names_sha256(names)
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: (names, sha))
    # Full held-shaped output, without reading a held feature store.
    patches = np.zeros((12281, 50, 512), dtype=np.float32)
    monkeypatch.setattr(ct.ct_held, 'clip_patches', lambda names: patches)
    [held] = ct.main(['--stage', 'held'] + args)
    info = json.loads(held.with_suffix('.json').read_text())
    source = ct.verify_output(mapper)
    assert info['inference_device'] == 'cpu' and info['weights_sha256'] == source['weights_sha256']
    assert held.parent.name == 'paper__strict'
    with np.load(held) as z:
        assert z['prob'].shape == (12281, source['K']) and z['prob'].dtype == np.float32
    assert ct.main(['--stage', 'held'] + args) == [held]


def write_rcol_fixture(seed, version, kept_count=12, *, smoke=False):
    """A collapsed, but trainable, run whose assignments are exact and tied."""
    config = dict(settings='paper', cut='k40', version=version, seed=seed, input_dim=2,
                  fit_pairs=67327, captions=67327, eval_paintings=5000, max_joint_steps=200,
                  device='cpu' if smoke else 'cuda', smoke=smoke)
    encoder, _ = paper().build_autoencoder(2)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
    centers = np.zeros((100, 128), dtype=np.float32)
    centers[:kept_count, 0] = 1
    centers[kept_count:, 0] = 100
    counts = np.r_[67327, np.zeros(99, dtype=np.int64)]
    kept_k40 = ct.pc_fixes.top_used(counts, 40)
    kept_strict = np.arange(100) < kept_count
    topics = np.zeros(67327, dtype=np.int16)
    q40 = np.tile(np.r_[np.full(kept_count, .5), np.full(40-kept_count, 1/10001)], (5000, 1))
    q40 = (q40 / q40.sum(axis=1, keepdims=True)).astype(np.float32)
    strict_q = np.full((5000, kept_count), 1/kept_count, dtype=np.float32)
    arrays = dict(centers=centers, center_norms=np.linalg.norm(centers, axis=1),
                  fit_counts=counts, fit_latent=np.zeros((67327, 128), dtype=np.float32),
                  kept=kept_k40, kept_k40=kept_k40, kept_strict=kept_strict,
                  topic=topics, topic_k40=topics, topic_strict=topics,
                  fit_topics_k40=topics, fit_topics_strict=topics,
                  q_paint_eval=q40, q_paint_eval_k40=q40, q_paint_eval_strict=strict_q,
                  stop_epoch=np.asarray([1], dtype=np.int64))
    meta = dict(config=config, K=40, K_by_cut=dict(k40=40, strict=kept_count),
                epochs=dict(joint=1), stop_epoch=1, sample_sha256='synthetic-sample',
                K_gamma={key: kept_count for key in ('1.5', '2', '3', '5')})
    return ct.save_output(ct.stage1_path('paper', version, seed), arrays,
                          dict(encoder=ct.cpu_state(encoder), centers=torch.from_numpy(centers)), meta)


def test_rcol_is_written_only_after_both_versions_have_all_five_stage1_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, 'OUTPUTS', tmp_path)
    output = tmp_path / 'percept/rcol_stage1.json'
    for version in ct.VERSIONS:
        for seed in ct.SEEDS:
            assert ct.finalize_stage1() is None
            assert not output.exists()
            write_rcol_fixture(seed, version)
    monkeypatch.setattr(ct, 'load_frozen', lambda: {})
    monkeypatch.setattr(ct.pc_data, 'fingerprint', lambda s: 'synthetic-sample')
    ct.main(['--stage', '1', '--settings', 'paper', '--version', 'qwen_qwen', '--seeds', '4'])
    assert output.exists()
    record = ct.finalize_stage1()
    assert record == json.loads(output.read_text())
    assert len(record['runs']) == 20
    assert all(r['collapsed'] and r['share'] == 1 for r in record['runs'])
    assert all(entry == dict(eligible=False, seeds=[]) for entry in record['entries'].values())
    stamp = output.stat().st_mtime_ns
    assert ct.finalize_stage1() == record and output.stat().st_mtime_ns == stamp


def test_one_topic_cut_skips_stage2_and_held_without_creating_outputs(sample, monkeypatch):
    write_rcol_fixture(0, 'clip_roberta', kept_count=1, smoke=True)
    s = dict(sample, pid=np.resize(sample['pid'], 67327))
    monkeypatch.setattr(ct, 'load_frozen', lambda: s)
    assert ct.run_stage2(s, 'paper', 'strict', 'P1', 0, smoke=True, device='cpu') is None
    args = ['--settings', 'paper', '--cut', 'strict', '--arm', 'P1', '--seeds', '0', '--smoke']
    assert ct.main(['--stage', '2', '--device', 'cpu'] + args) == []
    assert ct.run_held('paper', 'strict', 'P1', 0, smoke=True) is None
    assert ct.main(['--stage', 'held'] + args) == []
    assert not ct.stage2_path('paper', 'strict', 'P1', 0).exists()
    assert not (ct.OUTPUTS / 'held').exists()


def test_full_recipe_limits_and_training_cuda_default(sample, monkeypatch):
    cfg = ct.pc_stage1.make_config(sample, 'clip_roberta', 0, False, np.arange(2000), 'cuda')
    cfg = paper().stage1_config(cfg)
    assert cfg['pretrain_epochs'] == 100 and cfg['max_joint_steps'] == 200
    assert cfg['stability_threshold'] == .001
    parts = np.asarray(['fit'] * 13400 + ['dev'] * 1489 + ['eval'] * 5000)
    monkeypatch.setattr(ct.pc_data, 'partitions', lambda s: parts)
    cfg2 = paper().stage2_config(ct.pc_stage2.make_config(sample, 'P4', 0, False, 'cuda'))
    assert cfg2['epochs'] == 100 and cfg2['batch_size'] == 64 and cfg2['lr'] == .001
    monkeypatch.setattr(ct, 'load_frozen', lambda: sample)
    devices = []

    def train(s, settings, version, seed, *, smoke, device):
        devices.append(device)

    monkeypatch.setattr(ct, 'run_stage1', train)
    ct.main(['--stage', '1', '--version', 'clip_roberta', '--seeds', '0', '--smoke'])
    assert devices == ['cuda']


def test_dec_optimizes_global_sharpened_target_and_no_balance_term():
    import math

    torch.manual_seed(1)
    encoder, decoder = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        encoder.weight.fill_(1)
        decoder.weight.fill_(1)
    centers = nn.Parameter(torch.tensor([[0.], [1.]]))
    # Q=(2/3,1/3) for each of three zeros, (1/3,2/3) for one one.
    # Global frequencies give P=(20/27,7/27) and (5/33,28/33).
    zero_kl = (20/27) * math.log((20/27)/(2/3)) + (7/27) * math.log((7/27)/(1/3))
    one_kl = (5/33) * math.log((5/33)/(1/3)) + (28/33) * math.log((28/33)/(2/3))
    expected = (3 * zero_kl + one_kl) / 4
    history, _, _ = paper().train_dec_until_stable(
        encoder, decoder, centers, torch.tensor([[0.], [0.], [0.], [1.]]), 'cpu', max_epochs=1)
    assert history[0]['kl'] == pytest.approx(expected, abs=1e-7)
    assert history[0]['reconstruction'] == 0
    assert history[0]['total'] == pytest.approx(expected, abs=1e-7)


def test_pretraining_shuffles_512_row_batches_and_includes_tail():
    torch.manual_seed(5)
    encoder, decoder = nn.Linear(1, 1), nn.Linear(1, 1)
    x = torch.arange(513, dtype=torch.float32)[:, None] / 513
    batches = []
    handle = encoder.register_forward_pre_hook(lambda module, args: batches.append(args[0].clone()))
    paper().pretrain_autoencoder(encoder, decoder, x, 'cpu', epochs=2)
    handle.remove()
    assert list(map(len, batches)) == [512, 1, 512, 1]
    for epoch in range(2):
        ordered = torch.cat(batches[2 * epoch:2 * epoch + 2]).ravel()
        torch.testing.assert_close(ordered.sort().values, x.ravel())
        assert not torch.equal(ordered, x.ravel())


@pytest.mark.parametrize(('phase', 'epochs'), [('pretrain', 100), ('joint', 200), ('mapper', 100)])
def test_each_training_phase_advances_cosine_once_per_epoch(monkeypatch, phase, epochs):
    optimizers, used_rates = [], []
    adam = torch.optim.Adam

    def observe(*args, **kwargs):
        optimizer = adam(*args, **kwargs)
        optimizer.register_step_pre_hook(lambda opt, args, kwargs: used_rates.append(opt.param_groups[0]['lr']))
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, 'Adam', observe)
    encoder, decoder = nn.Linear(2, 2), nn.Linear(2, 2)
    if phase == 'pretrain':
        paper().pretrain_autoencoder(encoder, decoder, torch.zeros(2, 2), 'cpu', epochs=epochs)
    elif phase == 'joint':
        paper().train_dec_until_stable(encoder, decoder, nn.Parameter(torch.zeros(2, 2)),
                                       torch.zeros(2, 2), 'cpu', max_epochs=epochs, stability_threshold=0)
    else:
        inputs = {part: torch.zeros(2, 50, 2) for part in ('fit', 'dev', 'eval')}
        config = dict(seed=2, device='cpu', input_dim=2, epochs=epochs, arm='P1')
        paper().train_mapper(inputs, dict(fit=np.zeros((2, 2), dtype=np.float32)), config, 2)
    assert len(used_rates) == epochs and len(optimizers) == 1
    assert used_rates[0] == .001
    assert used_rates[epochs // 2] == pytest.approx(.000505)
    assert optimizers[0].param_groups[0]['lr'] == pytest.approx(.00001)


# ---- V4: a paper seed with non-finite outputs is saved and counts as collapsed ----

def nan_bce(monkeypatch):
    """Make every paper mapper loss NaN so its weights end non-finite."""
    p = paper()
    real = p.F.binary_cross_entropy_with_logits
    monkeypatch.setattr(p.F, 'binary_cross_entropy_with_logits',
                        lambda *args, **kwargs: real(*args, **kwargs) * float('nan'))


def held_fixture(monkeypatch):
    names = sorted(f'held-{i}' for i in range(12281))
    sha = ct.ct_held.names_sha256(names)
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: (names, sha))
    monkeypatch.setattr(ct.ct_held, 'clip_patches',
                        lambda names: np.ones((len(names), 50, 512), dtype=np.float32))
    return names, sha


def test_paper_stage2_with_nan_loss_saves_its_artifact_marked_nonfinite(sample, monkeypatch):
    ct.run_stage1(sample, 'paper', 'clip_roberta', 0, smoke=True, device='cpu')
    finite = ct.run_stage2(sample, 'paper', 'strict', 'P1', 0, smoke=True, device='cpu')
    assert ct.verify_output(finite)['nonfinite'] is False
    nan_bce(monkeypatch)
    path = ct.run_stage2(sample, 'paper', 'k40', 'P1', 0, smoke=True, device='cpu')
    meta = ct.verify_output(path)
    assert meta['nonfinite'] is True
    state = torch.load(path.with_suffix('.pt'), weights_only=True)['mapper']
    assert not all(torch.isfinite(v).all() for v in state.values())
    with np.load(path) as z:
        assert not np.isfinite(z['eval_prob']).all()
    assert meta['loss_history'] == [None] * 3


def test_nonfinite_weights_are_accepted_only_for_a_paper_sidecar_that_says_so(sample, monkeypatch):
    ct.run_stage1(sample, 'core', 'clip_roberta', 0, smoke=True, device='cpu')
    core = ct.run_stage2(sample, 'core', 'k40', 'P2', 0, smoke=True, device='cpu')
    ct.run_stage1(sample, 'paper', 'clip_roberta', 0, smoke=True, device='cpu')
    nan_bce(monkeypatch)
    path = ct.run_stage2(sample, 'paper', 'k40', 'P1', 0, smoke=True, device='cpu')
    sidecar = path.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    meta['nonfinite'] = False
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        ct.verify_output(path)
    assert 'nonfinite' not in ct.verify_output(core)
    state = torch.load(core.with_suffix('.pt'), weights_only=True)
    first = next(iter(state['mapper']))
    state['mapper'][first] = torch.full_like(state['mapper'][first], float('nan'))
    torch.save(state, core.with_suffix('.pt'))
    meta = json.loads(core.with_suffix('.json').read_text())
    meta['weights_sha256'] = ct.file_sha256(core.with_suffix('.pt'))
    meta['nonfinite'] = True
    core.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        ct.verify_output(core)


def test_nonfinite_paper_held_output_is_saved_and_scores_as_collapsed(sample, monkeypatch):
    import ct_score
    ct.run_stage1(sample, 'paper', 'clip_roberta', 0, smoke=True, device='cpu')
    nan_bce(monkeypatch)
    ct.run_stage2(sample, 'paper', 'strict', 'P1', 0, smoke=True, device='cpu')
    names, sha = held_fixture(monkeypatch)
    held = ct.run_held('paper', 'strict', 'P1', 0, smoke=True)
    info = json.loads(held.with_suffix('.json').read_text())
    assert info['nonfinite'] is True
    with np.load(held) as z:
        assert z['prob'].dtype == np.float32 and z['prob'].shape[0] == 12281
        assert not np.isfinite(z['prob']).all()
    assert ct.run_held('paper', 'strict', 'P1', 0, smoke=True) == held
    assert ct_score.load_held_output(held, 'B2', names, sha) is None


def test_finite_paper_held_output_records_nonfinite_false_and_scores(sample, monkeypatch):
    import ct_score
    ct.run_stage1(sample, 'paper', 'clip_roberta', 0, smoke=True, device='cpu')
    ct.run_stage2(sample, 'paper', 'strict', 'P1', 0, smoke=True, device='cpu')
    names, sha = held_fixture(monkeypatch)
    held = ct.run_held('paper', 'strict', 'P1', 0, smoke=True)
    assert json.loads(held.with_suffix('.json').read_text())['nonfinite'] is False
    assert ct_score.load_held_output(held, 'B2', names, sha).shape[0] == 12281


def test_core_held_inference_still_refuses_nonfinite_probabilities(sample, monkeypatch):
    ct.run_stage1(sample, 'core', 'clip_roberta', 0, smoke=True, device='cpu')
    ct.run_stage2(sample, 'core', 'k40', 'P2', 0, smoke=True, device='cpu')
    names = sample['all_painting_ids'][sample['subset_pid']].tolist()
    real = ct.build_mapper

    def nan_mapper(config, k):
        module = real(config, k)
        module.register_forward_hook(lambda m, a, out: out * float('nan'))
        return module

    monkeypatch.setattr(ct, 'build_mapper', nan_mapper)
    with pytest.raises(AssertionError):
        ct.infer_stage2('core', 'k40', 'P2', 0, names)


def test_paper_held_check_needs_a_truthful_flag_and_bounds_finite_entries(sample, monkeypatch):
    ct.run_stage1(sample, 'paper', 'clip_roberta', 0, smoke=True, device='cpu')
    ct.run_stage2(sample, 'paper', 'strict', 'P1', 0, smoke=True, device='cpu')
    names, sha = held_fixture(monkeypatch)
    held = ct.run_held('paper', 'strict', 'P1', 0, smoke=True)
    source = ct.verify_output(ct.stage2_path('paper', 'strict', 'P1', 0))
    with np.load(held) as z:
        prob = z['prob'].copy()

    def rewrite(values, flag):
        np.savez_compressed(held, prob=values)
        meta = json.loads(held.with_suffix('.json').read_text())
        meta.update(nonfinite=flag, array_sha256s=dict(prob=ct.pc_stage1.array_sha256(values)))
        held.with_suffix('.json').write_text(json.dumps(meta))

    nan_prob = prob.copy()
    nan_prob[0, 0] = np.nan
    rewrite(nan_prob, True)
    assert ct.verify_held_output(held, source, names, sha, 'cpu')['nonfinite'] is True
    rewrite(nan_prob, False)
    with pytest.raises(AssertionError):
        ct.verify_held_output(held, source, names, sha, 'cpu')
    big = nan_prob.copy()
    big[1, 0] = 1.5
    rewrite(big, True)
    with pytest.raises(AssertionError):
        ct.verify_held_output(held, source, names, sha, 'cpu')
