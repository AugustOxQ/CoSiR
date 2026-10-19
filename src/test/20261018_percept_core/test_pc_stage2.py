"""September mapper parity and Stage 2 caption-to-prediction contracts."""
import hashlib
import json
import os
import subprocess
import textwrap

import numpy as np
import pytest
import torch


SEPTEMBER = 'src/test/20260922_percept_topic_pipeline/'


def test_mapper_loss_and_auc_match_fresh_september_process(tmp_path):
    import pc_sept

    for name in ('run_percept_stage1_pilot.py',
                 'run_percept_stage1_cluster_count_sweep_pilot.py',
                 'run_percept_stage2_sweep_pilot.py'):
        source = subprocess.check_output(
            ['git', '-C', '/project/CoSiR', 'show', '1b70d7f:' + SEPTEMBER + name])
        (tmp_path / name).write_bytes(source)
    script = textwrap.dedent('''
        import importlib.util
        import sys
        import numpy as np
        import torch
        sys.path.insert(0, '/project/CoSiR-buddy_prototype_conditioning')
        spec = importlib.util.spec_from_file_location('original', sys.argv[1])
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        torch.manual_seed(0)
        mapper = original.AttentionPoolingMapper(n_topics=5)
        rng = np.random.default_rng(17)
        x = torch.from_numpy(rng.normal(size=(9, 50, 512)).astype(np.float32))
        targets = rng.integers(0, 2, size=(9, 5)).astype(np.float32)
        targets[:, 3] = 0
        targets[:, 4] = 1
        logits = mapper(x)
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.from_numpy(targets))
        loss.backward()
        aucs, skipped = original.evaluate_auc(
            logits.detach().sigmoid().numpy(), targets, lambda _: None, 'dev')
        np.savez(sys.argv[2], logits=logits.detach().numpy(), loss=loss.item(),
                 query_grad=mapper.query.grad.numpy(),
                 classifier_grad=mapper.classifier.weight.grad.numpy(),
                 topics=list(aucs), aucs=list(aucs.values()), skipped=skipped,
                 summary=list(original.auc_summary(aucs).values()))
    ''')
    subprocess.run([os.sys.executable, '-c', script,
                    str(tmp_path / 'run_percept_stage2_sweep_pilot.py'),
                    str(tmp_path / 'reference.npz')], check=True, timeout=90,
                   cwd='/project/CoSiR-buddy_prototype_conditioning')
    torch.manual_seed(0)
    mapper = pc_sept.AttentionPoolingMapper(n_topics=5)
    rng = np.random.default_rng(17)
    x = torch.from_numpy(rng.normal(size=(9, 50, 512)).astype(np.float32))
    targets = rng.integers(0, 2, size=(9, 5)).astype(np.float32)
    targets[:, 3] = 0
    targets[:, 4] = 1
    logits = mapper(x)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.from_numpy(targets))
    loss.backward()
    aucs, skipped = pc_sept.evaluate_auc(
        logits.detach().sigmoid().numpy(), targets, lambda _: None, 'dev')
    assert skipped == [3, 4] and len(aucs) == 3
    # Unequal AUCs over three topics: the macro mean must differ from median.
    summary = pc_sept.auc_summary(aucs)
    assert abs(summary['macro'] - summary['median']) > 1e-3
    with np.load(tmp_path / 'reference.npz') as expected:
        for name, actual in dict(logits=logits.detach().numpy(), loss=loss.item(),
                                 query_grad=mapper.query.grad.numpy(),
                                 classifier_grad=mapper.classifier.weight.grad.numpy(),
                                 topics=list(aucs), aucs=list(aucs.values()), skipped=skipped,
                                 summary=list(pc_sept.auc_summary(aucs).values())).items():
            np.testing.assert_allclose(actual, expected[name], rtol=0, atol=1e-6)


def fake_stage1(root, s, topic, version='clip_roberta', seed=0, k=40):
    import pc_data

    folder = root / 'stage1'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{version}__s{seed}.npz'
    kept = np.arange(100) < k
    q = np.full((len(s['subset_pid']), k), 1 / k if k else 0, dtype=np.float32)
    np.savez(path, center_norms=np.ones(100, dtype=np.float32), kept=kept,
             fit_counts=np.bincount(topic, minlength=100).astype(np.int64),
             topic=topic, q_paint_eval=q)
    path.with_suffix('.json').write_text(json.dumps(dict(
        config=dict(version=version, seed=seed), sample_sha256=pc_data.fingerprint(s),
        K=k, K_gamma={gamma: 100 for gamma in ('1.5', '2', '3', '5')},
        kept_topics=dict(fit_pairs=len(topic), with_fit_pair=3, with_1pct=3, share_before_cut=1.))))
    return path


def test_fit_and_dev_targets_are_their_own_caption_topic_unions(tmp_path, monkeypatch):
    import pc_data
    import pc_fixes
    import pc_stage2

    s = dict(pid=np.array([4, 0, 3, 1, 4, 2, 0, 5, 1, 3, 2]),
             first=np.array([1, 3, 5, 2, 0, 7]),
             sample_ids=np.array([99, 17, 31, 5, 12, 64, 73, 8, 90, 14, 18]),
             subset_pid=np.array([5, 3]), subset_image_indices=np.array([40, 20]),
             all_painting_ids=np.array([f'painting_{p}' for p in range(6)]))
    partition = np.array(['fit', 'dev', 'dev', 'eval', 'fit', 'eval'])
    monkeypatch.setattr(pc_data, 'partitions', lambda sample: partition.copy())
    topic = np.array([1, 0, 2, 1, 2, 0, 1, 2, 1, 0, 2], dtype=np.int16)
    path = fake_stage1(tmp_path, s, topic)
    loaded, k = pc_stage2.load_stage1(s, path, 'clip_roberta', 0)
    np.testing.assert_array_equal(loaded, topic)
    assert k == 40
    ids, targets = pc_stage2.caption_targets(s, loaded, k)
    np.testing.assert_array_equal(ids['fit'], [0, 4])
    np.testing.assert_array_equal(ids['dev'], [1, 2])
    np.testing.assert_array_equal(ids['eval'], [5, 3])
    np.testing.assert_array_equal(targets['fit'], np.pad([[1, 1, 0], [0, 1, 1]], ((0, 0), (0, 37))))
    np.testing.assert_array_equal(targets['dev'], np.pad([[0, 1, 0], [1, 0, 1]], ((0, 0), (0, 37))))
    for part in ('fit', 'dev'):
        rows = pc_fixes.pair_rows(s['pid'], partition, part)
        np.testing.assert_array_equal(targets[part], pc_fixes.multi_hot(
            s['pid'][rows], topic[rows], ids[part], k))
    # Alter only the evaluation captions: neither training nor dev targets change.
    topic[partition[s['pid']] == 'eval'] = 1
    _, changed = pc_stage2.caption_targets(s, topic, k)
    for part in ('fit', 'dev'):
        np.testing.assert_array_equal(changed[part], targets[part])


def smoke_data(monkeypatch, k=3, constant_dev=False):
    import pc_data

    # 201 fit paintings detect a smoke limit of 200; shuffled evaluation IDs
    # detect accidental sorting, and repeated captions exercise topic unions.
    n = 208
    pid = np.tile(np.arange(n, dtype=np.int64), 2)
    s = dict(pid=pid, first=np.arange(n, dtype=np.int64),
             sample_ids=np.arange(len(pid), dtype=np.int64) * 7 + 19,
             subset_pid=np.array([207, 205, 206]), subset_image_indices=np.array([9, 3, 6]),
             all_painting_ids=np.array([f'painting_{p}' for p in range(n)]))
    partition = np.array(['fit'] * 201 + ['dev'] * 4 + ['eval'] * 3)
    topic = (np.arange(len(pid)) % k).astype(np.int16)
    if constant_dev:
        topic[partition[pid] == 'dev'] = 0
    monkeypatch.setattr(pc_data, 'partitions', lambda sample: partition.copy())
    monkeypatch.setattr(pc_data, 'load_frozen', lambda: s)
    return s, topic


@pytest.mark.parametrize('arm,version,kind,width', [
    ('P1', 'clip_roberta', 'patch', 512),
    ('P2', 'clip_roberta', 'clip_image', 512),
    ('P3', 'qwen_qwen', 'qwen_default', 2048),
    ('P4', 'qwen_qwen', 'patch', 512),
    ('P5', 'qwen_roberta', 'qwen_default', 2048),
])
def test_smoke_cli_trains_and_writes_eval_in_subset_order(
        arm, version, kind, width, tmp_path, monkeypatch, capsys):
    import pc_data
    import pc_stage2

    s, topic = smoke_data(monkeypatch)
    source = fake_stage1(tmp_path, s, topic, version=version, seed=1)
    rng = np.random.default_rng(29)
    shape = (len(s['first']), 50, width) if kind == 'patch' else (len(s['first']), width)
    x = rng.normal(size=shape).astype(np.float32) * 7
    def images(sample, requested):
        assert kind != 'patch' and requested == kind
        return x.copy()
    def patches(sample, pids):
        assert kind == 'patch'
        return x[pids].copy()
    monkeypatch.setattr(pc_data, 'image_features', images)
    monkeypatch.setattr(pc_data, 'patch_tokens', patches)
    monkeypatch.setattr(os.sys, 'argv', ['pc_stage2.py', '--arm', arm, '--seed', '1',
                                        '--smoke', str(tmp_path)])
    pc_stage2.main()
    path = tmp_path / 'stage2' / f'{arm}__s1.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['stage1_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert meta['K'] == 40 and meta['dev_valid_topics'] == 3
    assert 0 <= meta['dev_auc'] <= 1
    assert len(meta['loss_history']) == 3 and np.isfinite(meta['loss_history']).all()
    assert meta['config']['fit_paintings'] == 200
    assert meta['config']['dev_paintings'] == 4 and meta['config']['eval_paintings'] == 3
    assert meta['config']['epochs'] == 3 and meta['config']['lr'] == .003
    assert meta['config']['batch'] == 'full fit paintings'
    assert meta['config']['input_dim'] == width
    assert meta['config']['version'] == version and meta['config']['seed'] == 1
    assert meta['config']['mapper'] == ('attention' if kind == 'patch' else 'mlp')
    assert meta['seconds'] >= 0
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'eval_prob'}
        prob = z['eval_prob']
        assert prob.shape == (3, 40) and prob.dtype == np.float32
        assert np.isfinite(prob).all() and ((prob > 0) & (prob < 1)).all()
    output = capsys.readouterr().out
    for epoch in range(1, 4):
        assert f'epoch={epoch}' in output
    assert 'seconds_per_epoch=' in output
    # Reseed the mapper per run, even when the interpreter RNG has moved.
    # Globally rescaling vectors leaves normalized MLP inputs unchanged.
    torch.manual_seed(876)
    if kind != 'patch':
        x *= 2
    pc_stage2.main()
    with np.load(path) as z:
        np.testing.assert_array_equal(z['eval_prob'], prob)
    # Reordering only evaluation paintings must reorder the emitted rows.
    s['subset_pid'] = s['subset_pid'][[1, 2, 0]]
    fake_stage1(tmp_path, s, topic, version=version, seed=1)
    pc_stage2.main()
    with np.load(path) as z:
        np.testing.assert_array_equal(z['eval_prob'], prob[[1, 2, 0]])


@pytest.mark.parametrize('k', [1, 39, 41])
def test_stage1_requires_fixed_k40(k, tmp_path, monkeypatch):
    import pc_stage2

    s, topic = smoke_data(monkeypatch)
    path = fake_stage1(tmp_path, s, topic, k=k)
    with pytest.raises(AssertionError):
        pc_stage2.load_stage1(s, path, 'clip_roberta', 0)


def test_stage1_rejects_a_wrong_kept_mask_with_40_centers(tmp_path, monkeypatch):
    import pc_stage2

    s, topic = smoke_data(monkeypatch)
    path = fake_stage1(tmp_path, s, topic)
    with np.load(path) as z:
        arrays = dict(z)
    arrays['kept'][0], arrays['kept'][99] = False, True
    np.savez(path, **arrays)
    with pytest.raises(AssertionError, match='kept mask'):
        pc_stage2.load_stage1(s, path, 'clip_roberta', 0)


def test_no_valid_dev_topics_still_writes_predictions_and_null_auc(tmp_path, monkeypatch):
    import pc_data
    import pc_stage2

    s, topic = smoke_data(monkeypatch, constant_dev=True)
    fake_stage1(tmp_path, s, topic)
    monkeypatch.setattr(pc_data, 'image_features', lambda sample, kind:
                        np.ones((len(s['first']), 512), dtype=np.float32))
    path = pc_stage2.run(s, 'P2', 0, smoke=tmp_path, device='cpu')
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['dev_auc'] is None and meta['dev_valid_topics'] == 0
    assert meta['dev_skipped_topics'] == list(range(40)) and meta['dev_auc_note']
    assert 'reason' not in meta
    assert len(meta['loss_history']) == 3
    with np.load(path) as z:
        assert z['eval_prob'].shape == (3, 40)


def test_constant_dev_columns_are_excluded_from_valid_topic_count(tmp_path, monkeypatch):
    import pc_data
    import pc_stage2

    s, topic = smoke_data(monkeypatch, k=2)
    # Dev captions use only topics 0 and 1 of the 40 kept centers.
    fake_stage1(tmp_path, s, topic)
    monkeypatch.setattr(pc_data, 'image_features', lambda sample, kind:
                        np.ones((len(s['first']), 512), dtype=np.float32))
    path = pc_stage2.run(s, 'P2', 0, smoke=tmp_path, device='cpu')
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['dev_valid_topics'] == 2 and meta['dev_skipped_topics'] == list(range(2, 40))
    assert meta['dev_auc'] == .5


@pytest.mark.parametrize('arm,kind,width', [
    ('P1', 'patch', 512), ('P2', 'clip_image', 512),
    ('P3', 'qwen_default', 2048), ('P4', 'patch', 512), ('P5', 'qwen_default', 2048),
])
def test_image_inputs_preserve_patches_and_normalize_global_vectors(arm, kind, width, monkeypatch):
    import pc_data
    import pc_stage2

    s = dict(first=np.arange(3))
    ids = dict(fit=np.array([2, 0]), dev=np.array([1]), eval=np.array([0, 2]))
    global_x = np.zeros((3, width), dtype=np.float32)
    global_x[:, :2] = [[3, 4], [0, 5], [12, 5]]
    patch_x = np.repeat(global_x[:, None, :], 50, axis=1)
    def images(sample, requested):
        assert kind == requested
        return global_x.copy()
    def patches(sample, pids):
        assert kind == 'patch'
        return patch_x[pids].copy()
    monkeypatch.setattr(pc_data, 'image_features', images)
    monkeypatch.setattr(pc_data, 'patch_tokens', patches)
    inputs = pc_stage2.image_inputs(s, arm, ids, 'cpu')
    expected = np.zeros_like(global_x)
    expected[:, :2] = [[.6, .8], [0, 1], [12 / 13, 5 / 13]]
    for part, pids in ids.items():
        want = patch_x[pids] if kind == 'patch' else expected[pids]
        np.testing.assert_allclose(inputs[part].numpy(), want, rtol=0, atol=1e-7)


@pytest.mark.parametrize('field,value,message', [
    ('sample_sha256', 'wrong', 'sample mismatch'),
    ('version', 'qwen_qwen', 'version mismatch'),
    ('seed', 2, 'seed mismatch'),
])
def test_stage1_identity_must_match_before_training(field, value, message, tmp_path, monkeypatch):
    import pc_stage2

    s, topic = smoke_data(monkeypatch)
    path = fake_stage1(tmp_path, s, topic)
    sidecar = path.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    if field == 'sample_sha256':
        meta[field] = value
    else:
        meta['config'][field] = value
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match=message):
        pc_stage2.load_stage1(s, path, 'clip_roberta', 0)


def test_predictions_are_sealed_before_the_sidecar(tmp_path, monkeypatch):
    import pc_data
    import pc_stage2

    s, topic = smoke_data(monkeypatch)
    fake_stage1(tmp_path, s, topic)
    monkeypatch.setattr(pc_data, 'image_features', lambda sample, kind:
                        np.ones((len(s['first']), 512), dtype=np.float32))
    seen = []
    real = pc_stage2.write_sidecar
    def spy(path, meta):
        folder = path.parent
        seen.append((path.with_suffix('.npz').exists(),
                     (folder / 'P2__s0.pending.npz').exists()))
        return real(path, meta)
    monkeypatch.setattr(pc_stage2, 'write_sidecar', spy)
    pc_stage2.run(s, 'P2', 0, smoke=tmp_path, device='cpu')
    # A crash during the sidecar write never leaves a new sidecar beside old predictions.
    assert seen == [(True, False)]


class RecordedAdam(torch.optim.Adam):
    made = []

    def __init__(self, params, **kwargs):
        params = list(params)
        self.start = [p.detach().clone() for p in params]
        super().__init__(params, **kwargs)
        RecordedAdam.made.append(self)


@pytest.mark.parametrize('mapper,x_shape', [('mlp', (6, 8)), ('attention', (6, 50, 512))])
def test_train_mapper_uses_configured_lr_and_architecture(mapper, x_shape, monkeypatch):
    import pc_stage2

    RecordedAdam.made = []
    monkeypatch.setattr(torch.optim, 'Adam', RecordedAdam)
    k = 3
    rng = np.random.default_rng(5)
    inputs = dict(fit=torch.from_numpy(rng.normal(size=x_shape).astype(np.float32)),
                  dev=torch.from_numpy(rng.normal(size=(2,) + x_shape[1:]).astype(np.float32)),
                  eval=torch.from_numpy(rng.normal(size=(2,) + x_shape[1:]).astype(np.float32)))
    targets = dict(fit=rng.integers(0, 2, size=(6, k)).astype(np.float32))
    config = dict(seed=4, device='cpu', mapper=mapper, input_dim=x_shape[-1],
                  lr=.003, epochs=1, arm='T')
    pc_stage2.train_mapper(inputs, targets, config, k)
    optimizer, = RecordedAdam.made
    assert optimizer.param_groups[0]['lr'] == .003
    shapes = [tuple(p.shape) for p in optimizer.start]
    if mapper == 'mlp':
        assert shapes == [(1024, 8), (1024,), (k, 1024), (k,)]
    else:
        assert (k, 512) in shapes


def test_train_mapper_epoch_loss_is_the_mean_bce(monkeypatch):
    import pc_stage2

    RecordedAdam.made = []
    monkeypatch.setattr(torch.optim, 'Adam', RecordedAdam)
    k, d = 3, 5
    rng = np.random.default_rng(8)
    x = rng.normal(size=(7, d)).astype(np.float32)
    y = rng.integers(0, 2, size=(7, k)).astype(np.float32)
    inputs = dict(fit=torch.from_numpy(x), dev=torch.from_numpy(x[:2]),
                  eval=torch.from_numpy(x[:2]))
    config = dict(seed=2, device='cpu', mapper='mlp', input_dim=d, lr=.003,
                  epochs=1, arm='T')
    _, _, history, _ = pc_stage2.train_mapper(inputs, dict(fit=y), config, k)
    w1, b1, w2, b2 = [p.numpy().astype(np.float64) for p in RecordedAdam.made[0].start]
    z = np.maximum(x @ w1.T + b1, 0) @ w2.T + b2
    expected = np.mean(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - y * z)
    assert history[0] == pytest.approx(expected, rel=1e-5)
