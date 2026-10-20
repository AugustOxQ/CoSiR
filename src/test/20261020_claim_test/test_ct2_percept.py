"""Stage 2 PercepT recipes, joined inputs and bounded DEC memory."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ct2_extract
import ct2_percept as ct
from test_ct2_extract import features, setup  # noqa: F401  (tiny old stores, fake encoders)


@pytest.mark.parametrize('rows', [13, 1031])
def test_chunked_assignments_match_original_values_and_both_gradients(rows):
    torch.manual_seed(19)
    latent = torch.randn(rows, 128, requires_grad=True)
    centers = torch.randn(100, 128, requires_grad=True)
    weights = torch.randn(rows, 100)
    expected = ct.ORIGINAL_SOFT_ASSIGNMENTS(latent, centers)
    expected_grads = torch.autograd.grad((expected * weights).sum(), (latent, centers))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(x.shape), x)[1], lambda x: x):
        actual = ct.chunked_soft_assignments(latent, centers)
        actual_grads = torch.autograd.grad((actual * weights).sum(), (latent, centers))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=1e-5)
    assert all(len(shape) < 3 for shape in saved), 'checkpoint must not retain distance broadcasts'
    with torch.no_grad():
        torch.testing.assert_close(ct.chunked_soft_assignments(latent, centers), expected, rtol=0, atol=1e-6)


def chunk_inputs(rows):
    torch.manual_seed(3)
    return torch.randn(rows, 16, requires_grad=True), torch.randn(100, 16, requires_grad=True)


def record_original(monkeypatch):
    calls = []
    original = ct.ORIGINAL_SOFT_ASSIGNMENTS

    def recorder(latent, centers):
        calls.append(len(latent))
        return original(latent, centers)

    monkeypatch.setattr(ct, 'ORIGINAL_SOFT_ASSIGNMENTS', recorder)
    return calls


def test_chunking_happens_with_default_chunk_rows(monkeypatch):
    """3 full chunks and a tail of 5: a no-op chunker (one big call) must fail."""
    calls = record_original(monkeypatch)
    rows = 3 * ct.DEC_CHUNK_ROWS + 5
    latent, centers = chunk_inputs(rows)
    q = ct.chunked_soft_assignments(latent, centers)
    assert calls == [ct.DEC_CHUNK_ROWS] * 3 + [5]
    q.sum().backward()  # checkpoint recomputes per chunk, never more than a chunk
    assert len(calls) == 8 and max(calls) <= ct.DEC_CHUNK_ROWS


def test_runtime_binds_dec_chunk_rows_to_the_assignment_function(monkeypatch, tmp_path):
    calls = record_original(monkeypatch)
    latent, centers = chunk_inputs(3 * 64 + 5)
    with ct.runtime(tmp_path, dec_chunk_rows=64):
        ct.pc_sept.base.soft_assignments(latent, centers)
    assert calls == [64, 64, 64, 5]
    calls.clear()
    latent, centers = chunk_inputs(3 * ct.DEC_CHUNK_ROWS + 5)
    with ct.runtime(tmp_path, ct.data.split2('full')):
        ct.pc_sept.base.soft_assignments(latent, centers)
    assert len(calls) == 4 and max(calls) <= 1024


def test_sample_mode_uses_the_original_dec_function_and_full_mode_the_chunked_one(tmp_path):
    sample, full = ct.data.split2('sample'), ct.data.split2('full')
    assert ct.default_chunk_rows('sample') is None and ct.default_chunk_rows('full') == 1024
    original = ct.pc_sept.base.soft_assignments
    with ct.runtime(tmp_path, sample):
        assert ct.pc_sept.base.soft_assignments is original is ct.ORIGINAL_SOFT_ASSIGNMENTS
        config = ct.pc_stage1.make_config(sample, 'qwen_qwen', 0, False, sample['fit_pairs'], 'cuda')
        assert config['dec_chunk_rows'] is None
    with ct.runtime(tmp_path, sample, dec_chunk_rows=256):
        assert ct.pc_sept.base.soft_assignments is not original
        assert ct.pc_stage1.make_config(sample, 'qwen_qwen', 0, False, sample['fit_pairs'], 'cuda')[
            'dec_chunk_rows'] == 256
    with ct.runtime(tmp_path, full):
        assert ct.pc_sept.base.soft_assignments is not original
        assert ct.pc_stage1.make_config(full, 'qwen_qwen', 0, False, full['fit_pairs'], 'cuda')[
            'dec_chunk_rows'] == 1024
    assert ct.pc_sept.base.soft_assignments is original


REVISION = 'd75048347613a25d77de8cf6412eaae9fa7b26be'
MODEL = 'SamLowe/roberta-base-go_emotions'


def onehot(ids):
    return features(ids, 768, np.float32)


def reseal(path, ids, names=None, values=None):
    """Rewrite an s2 store the way ticket 02 writes it: ID hash, count and shape follow."""
    ids = np.asarray(ids, dtype=np.int64)
    names = np.asarray([f'p{i:03}' for i in ids] if names is None else names)
    np.savez(path / 'ids.npz', sample_ids=ids, painting_ids=names)
    np.save(path / 'txt.npy', onehot(ids) if values is None else np.asarray(values, dtype=np.float32))
    done = json.loads((path / 'done.json').read_text())
    done.update(count=len(ids), shape=[len(ids), 768],
                ids_sha256=ct.data.ids_sha256(dict(sample_ids=ids, painting_ids=names)))
    (path / 'done.json').write_text(json.dumps(done))
    return done


@pytest.fixture
def stores(tmp_path, monkeypatch, setup):
    """Old store (ids 0..31) plus an s2 store written by ct2_extract.extract (ids 35, 40, 47)."""
    old = ct.data.ROBERTA
    (old / 'done.json').write_text(json.dumps(dict(model_id=MODEL, snapshot_revision=REVISION, count=32)))
    original = ct2_extract.load_encoder
    monkeypatch.setattr(ct2_extract, 'load_encoder', lambda store, device: (
        original(store, device)[0], dict(model_id=MODEL, model_revision=REVISION)))
    extra = ct2_extract.extract('roberta', 'cpu')
    assert extra == ct2_extract.store_path('roberta') == old.with_name(old.name + '_s2')
    monkeypatch.setattr(ct, 'ROBERTA', old)
    monkeypatch.setattr(ct, 'ROBERTA_S2', extra)
    return old, extra


def request(ids, mode='full'):
    return dict(mode=mode, sample_ids=np.asarray(ids, dtype=np.int64),
                painting_ids=np.asarray([f'p{i:03}' for i in ids]))


def test_roberta_union_reads_a_ticket_02_s2_store_and_prefers_old_rows(stores):
    old, extra = stores
    done = json.loads((extra / 'done.json').read_text())
    assert 'snapshot_revision' not in done and done['model_revision'] == REVISION  # ticket 02's format
    np.testing.assert_array_equal(ct.caption_features(request([35, 4, 47, 31]), 'roberta'),
                                  onehot([35, 4, 47, 31]))
    reseal(extra, [4, 40], values=np.stack([onehot([4])[0] * 2, onehot([40])[0]]))  # cosine 1; keep old bytes
    np.testing.assert_array_equal(ct.caption_features(request([40, 4]), 'roberta'), onehot([40, 4]))


@pytest.mark.parametrize('damage', ['overlap', 'painting', 'missing', 'duplicate', 'revision', 'parity', 'ids'])
def test_roberta_union_refuses_damaged_s2_stores(stores, damage):
    old, extra = stores
    ids = [35, 4, 47]
    if damage == 'overlap':
        reseal(extra, [4, 40], values=np.stack([-onehot([4])[0], onehot([40])[0]]))
    elif damage == 'painting':
        reseal(extra, [35, 40, 47], names=['wrong', 'p040', 'p047'])
    elif damage == 'missing':
        ids = [35, 4, 999]
    elif damage == 'duplicate':
        reseal(extra, [35, 35, 47])
    elif damage == 'revision':
        done = json.loads((extra / 'done.json').read_text())
        (extra / 'done.json').write_text(json.dumps(dict(done, model_revision='wrong')))
    elif damage == 'parity':
        done = json.loads((extra / 'done.json').read_text())
        done['parity_gate']['passed'] = False
        (extra / 'done.json').write_text(json.dumps(done))
    else:  # ids.npz no longer matches the sealed hash
        np.savez(extra / 'ids.npz', sample_ids=np.array([35, 40, 48]), painting_ids=np.array(['p035', 'p040', 'p048']))
    with pytest.raises(AssertionError):
        ct.caption_features(request(ids), 'roberta')


def test_roberta_old_store_keeps_its_snapshot_revision_check(stores):
    old, extra = stores
    (old / 'done.json').write_text(json.dumps(dict(model_id=MODEL, snapshot_revision='wrong', count=32)))
    with pytest.raises(AssertionError):
        ct.caption_features(request([4, 31], 'sample'), 'roberta')


def test_sample_mode_reads_only_the_old_store(stores):
    """The frozen sample is fully covered by the old store; a broken s2 store must not matter."""
    old, extra = stores
    (extra / 'done.json').write_text('not json')
    np.testing.assert_array_equal(ct.caption_features(request([4, 31], 'sample'), 'roberta'), onehot([4, 31]))
    with pytest.raises(Exception):
        ct.caption_features(request([4, 31], 'full'), 'roberta')
    with pytest.raises(AssertionError, match='missing'):
        ct.caption_features(request([4, 35], 'sample'), 'roberta')  # no fallback to the s2 store


@pytest.mark.parametrize(('provider', 'kind'), [
    ('caption', 'clip_text'), ('caption', 'roberta'), ('caption', 'qwen_default'),
    ('caption', 'qwen_emotion_focus'), ('image', 'clip_image'), ('image', 'qwen_default'),
    ('patch', None)])
def test_sample_providers_preserve_stage1_input_bytes(provider, kind):
    s = ct.data.split2('sample')
    if provider == 'patch':
        ids = s['subset_pid'][[0, 7, -1]]
        expected = ct.pc_stage1.array_sha256(ct.pc_data.patch_tokens(s, ids))
        actual = ct.patch_tokens(s, ids)
    else:
        original = ct.pc_data.caption_features if provider == 'caption' else ct.pc_data.image_features
        expected = ct.pc_stage1.array_sha256(original(s, kind))
        actual = ct.caption_features(s, kind) if provider == 'caption' else ct.image_features(s, kind)
    assert actual.dtype == np.float32
    assert ct.pc_stage1.array_sha256(actual) == expected


def tiny_roles(mode):
    # Multiple captions per fit/dev painting; no full-mode evaluation subset.
    pid = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    names = np.array(['a', 'b', 'c', 'd'])
    s = dict(mode=mode, pid=pid, first=np.array([0, 2, 4, 6]),
             sample_ids=np.arange(8, dtype=np.int64), painting_ids=names[pid],
             all_painting_ids=names, fit=np.array([1, 0]), dev=np.array([2]),
             subset_pid=np.array([3]) if mode == 'sample' else np.array([], dtype=np.int64))
    if mode == 'full':
        s['dev'] = np.array([2, 3])
    return s


@pytest.mark.parametrize('mode', ['sample', 'full'])
def test_targets_use_all_fit_captions_and_mode_dev_paintings(mode):
    s = tiny_roles(mode)
    topic = np.array([0, 1, 1, 2, 0, 2, 1, 2], dtype=np.int16)
    ids, targets = ct.caption_targets(s, topic, 3)
    np.testing.assert_array_equal(ids['fit'], [0, 1])
    np.testing.assert_array_equal(targets['fit'], [[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(ids['dev'], s['dev'])
    np.testing.assert_array_equal(targets['dev'], [[1, 0, 1]] if mode == 'sample' else [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(ids['eval'], s['subset_pid'] if mode == 'sample' else s['dev'])


def test_sample_mapper_config_keeps_every_recipe_setting():
    s = ct.data.split2('sample')
    expected = ct.pc_stage2.make_config(s, 'P2', 0, False, 'cuda')
    actual = ct.mapper_config(s, 'P2', 0, False, 'cuda')
    assert all(actual[key] == value for key, value in expected.items())
    full = tiny_roles('full')
    config = ct.mapper_config(full, 'P3', 1, False, 'cuda')
    assert (config['fit_paintings'], config['dev_paintings'], config['eval_paintings']) == (2, 2, 2)
    assert (config['epochs'], config['lr'], config['batch'], config['scheduler']) == (100, .003, 'full fit paintings', None)


@pytest.mark.parametrize(('mode', 'settings', 'arm'), [
    ('full', 'core', 'P1'), ('full', 'core', 'P4'), ('full', 'paper', 'P2'), ('sample', 'paper', 'P3')])
def test_unsupported_arms_fail_before_loading_any_data(tmp_path, mode, settings, arm):
    with pytest.raises(AssertionError, match='arm'):
        ct.run_stage2(mode, settings, 'k40', arm, 0, root=tmp_path, device='cpu')
    assert not list(tmp_path.iterdir())


@pytest.fixture
def synthetic(monkeypatch):
    """Feature-source stand-ins; real fusion, PCA, cuts, training and artifacts."""
    n = 209
    names = np.array([f'painting-{i:03}' for i in range(n)])
    pid = np.repeat(np.arange(n, dtype=np.int64), 10)
    rng = np.random.default_rng(82)
    clip = rng.normal(size=(n, 512)).astype(np.float32)
    qwen = rng.normal(size=(n, 2048)).astype(np.float32)
    qwen /= np.linalg.norm(qwen, axis=1, keepdims=True)
    text = rng.normal(size=(len(pid), 512)).astype(np.float32)
    emotion = rng.normal(size=(len(pid), 768)).astype(np.float32)
    qtext = rng.normal(size=(len(pid), 2048)).astype(np.float32)
    qemotion = rng.normal(size=(len(pid), 2048)).astype(np.float32)
    splits = {}
    for mode in ('sample', 'full'):
        evaluation = np.array([208, 204, 207, 205, 206]) if mode == 'sample' else np.array([], dtype=np.int64)
        splits[mode] = dict(mode=mode, sample_ids=np.arange(len(pid), dtype=np.int64),
            pid=pid, first=np.arange(n, dtype=np.int64) * 10, all_painting_ids=names,
            painting_ids=names[pid], subset_pid=evaluation, subset_image_indices=evaluation.copy(),
            fit=np.arange(200, dtype=np.int64)[::-1],
            dev=np.arange(200, 204 if mode == 'sample' else n, dtype=np.int64),
            fit_pairs=np.arange(2000, dtype=np.int64))
    monkeypatch.setattr(ct.data, 'split2', lambda mode: splits[mode])
    monkeypatch.setattr(ct, 'image_features', lambda s, kind: (qwen if kind == 'qwen_default' else clip).copy())
    monkeypatch.setattr(ct, 'caption_features', lambda s, kind: dict(
        clip_text=text, roberta=emotion, qwen_default=qtext, qwen_emotion_focus=qemotion)[kind].copy())
    patches = lambda s, ids: np.broadcast_to(clip[ids, None, :], (len(ids), 50, 512)).copy()
    monkeypatch.setattr(ct, 'patch_tokens', patches)
    lookup = {name: row for row, name in enumerate(names)}
    rows = lambda names: np.asarray([lookup[name] for name in names])
    monkeypatch.setattr(ct.base.ct_held, 'clip_global_images',
                        lambda names: ct.data.l2(clip[rows(names)]))
    monkeypatch.setattr(ct.base.ct_held, 'qwen_default_images', lambda names: qwen[rows(names)].copy())
    monkeypatch.setattr(ct.base.ct_held, 'clip_patches', lambda names: patches(None, rows(names)))
    return splits


@pytest.mark.parametrize('mode', ['sample', 'full'])
@pytest.mark.parametrize('settings', ['core', 'paper'])
@pytest.mark.parametrize('version', ct.VERSIONS)
def test_runners_seal_all_fit_pairs_dev_auc_and_reloadable_mappers(synthetic, tmp_path, mode, settings, version):
    s = synthetic[mode]
    root = tmp_path / mode
    original_root = ct.base.OUTPUTS
    original_q = ct.pc_sept.base.soft_assignments
    original_eval = s['subset_pid'].copy()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(x.shape), x)[1], lambda x: x):
        stage1 = ct.run_stage1(mode, settings, version, 0, root=root, device='cpu', smoke=True)
    if mode == 'full':
        assert all(len(shape) < 3 for shape in saved), 'joint loop must use checkpointed assignments'
    else:
        assert any(len(shape) == 3 for shape in saved), 'sample mode keeps stage 1 unchunked function'
    assert ct.base.OUTPUTS == original_root and ct.pc_sept.base.soft_assignments is original_q
    np.testing.assert_array_equal(s['subset_pid'], original_eval)
    meta = ct.verify_output(stage1, root=root)
    assert meta['config']['fit_pairs'] == 2000
    assert meta['config']['mode'] == mode
    assert meta['sample_sha256'] == ct.pc_data.fingerprint(s)
    with np.load(stage1) as arrays:
        assert arrays['fit_counts'].sum() == 2000
        assert arrays['fit_latent'].shape == (2000, 128)
        if settings == 'paper':
            assert arrays['fit_topics_strict'].shape == arrays['fit_topics_k40'].shape == (2000,)
    arms = ['P1', 'P2'] if version == 'clip_roberta' else ['P3', 'P4']
    arms = [arm for arm in arms if (arm in ('P1', 'P4') if settings == 'paper' else mode == 'sample' or arm in ('P2', 'P3'))]
    for arm in arms:
        for cut in (('strict', 'k40') if settings == 'paper' else ('k40',)):
            path = ct.run_stage2(mode, settings, cut, arm, 0, root=root, device='cpu', smoke=True)
            if settings == 'paper' and meta['K_by_cut'][cut] < 2:
                assert path is None
                continue
            mapper_meta = ct.verify_output(path, root=root)
            assert mapper_meta['config']['dev_paintings'] == len(s['dev'])
            with np.load(stage1) as z:
                topic = z[f'topic_{cut}'] if settings == 'paper' else z['topic']
            _, targets = ct.caption_targets(s, topic, mapper_meta['K'], smoke=True)
            positives = targets['dev'].sum(axis=0)
            valid = (positives > 0) & (positives < len(s['dev']))
            assert mapper_meta['dev_valid_topics'] == int(valid.sum())
            if not valid.any():
                assert mapper_meta['dev_auc'] is None
                assert 'No dev topic' in mapper_meta['dev_auc_note']
            else:
                dev_names = s['all_painting_ids'][s['dev']].tolist()
                dev_prob = ct.infer_stage2(mode, settings, cut, arm, 0, dev_names, root=root, device='cpu')
                from sklearn.metrics import roc_auc_score
                expected_auc = np.mean([roc_auc_score(targets['dev'][:, i], dev_prob[:, i])
                                        for i in np.flatnonzero(valid)])
                assert mapper_meta['dev_auc'] == expected_auc
            ids = s['subset_pid'] if mode == 'sample' else s['dev']
            names = s['all_painting_ids'][ids].tolist()
            replay = ct.infer_stage2(mode, settings, cut, arm, 0, names, root=root, device='cpu')
            with np.load(path) as arrays:
                np.testing.assert_array_equal(replay, arrays['eval_prob'])
            assert mapper_meta['eval_names_sha256'] == ct.base.ct_held.names_sha256(names)
            assert path == root / 'percept' / settings / 'stage2' / cut / f'{arm}__s0.npz'
    assert ct.base.OUTPUTS == original_root


def test_held_runner_uses_root_frozen_list_and_training_checkpoint(synthetic, tmp_path, monkeypatch):
    mode, root = 'full', tmp_path / 'full'
    ct.run_stage1(mode, 'core', 'clip_roberta', 2, root=root, device='cpu', smoke=True)
    source = ct.run_stage2(mode, 'core', 'k40', 'P2', 2, root=root, device='cpu', smoke=True)
    names = [f'held-{i:05}' for i in range(12281)]
    digest = ct.base.ct_held.names_sha256(names)
    def held_list(path):
        if path == ct.FROZEN_HELD_LIST:
            return names, digest
        assert path == root / 'held/held_paintings.json'
        return names, digest
    monkeypatch.setattr(ct.base.ct_held, 'read_held_list', held_list)
    features = ct.data.l2(np.random.default_rng(2).normal(size=(12281, 512)).astype(np.float32))
    monkeypatch.setattr(ct.base.ct_held, 'clip_global_images', lambda names: features)
    path = ct.run_held(mode, 'core', 'k40', 'P2', 2, root=root, device='cpu', smoke=True)
    expected = ct.infer_stage2(mode, 'core', 'k40', 'P2', 2, names, root=root, device='cpu')
    with np.load(path) as z:
        np.testing.assert_array_equal(z['prob'], expected)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['held_sha256'] == digest
    assert meta['weights_sha256'] == ct.verify_output(source, root=root)['weights_sha256']
    assert path == root / 'held/percept/core__k40/P2__s2.npz'
    modified = path.stat().st_mtime_ns
    assert ct.run_held(mode, 'core', 'k40', 'P2', 2, root=root, smoke=True) == path
    assert path.stat().st_mtime_ns == modified
    with pytest.raises(AssertionError, match='mode'):
        ct.run_held('sample', 'core', 'k40', 'P2', 2, root=root, device='cpu', smoke=True)


def test_cli_routes_mode_root_recipe_and_seeds(synthetic, tmp_path):
    args = ['--mode', 'full', '--settings', 'core', '--stage', '1', '--version', 'clip_roberta',
            '--seeds', '0', '--device', 'cpu', '--root', str(tmp_path), '--smoke']
    [path] = ct.main(args)
    assert path == tmp_path / 'percept/core/stage1/clip_roberta__s0.npz'
    [stage2] = ct.main(['--mode', 'full', '--settings', 'core', '--stage', '2', '--cut', 'k40',
                      '--arm', 'P2', '--seeds', '0', '--device', 'cpu', '--root', str(tmp_path), '--smoke'])
    assert ct.verify_output(stage2, root=tmp_path)['config']['mode'] == 'full'
    with pytest.raises(SystemExit):
        ct.main(['--mode', 'full', '--stage', '2', '--arm', 'P1', '--root', str(tmp_path)])


@pytest.mark.parametrize('mode', ['sample', 'full'])
def test_runtime_selects_every_real_fit_pair_and_preserves_frozen_recipe(mode, tmp_path):
    s = ct.data.split2(mode)
    identity = ct.pc_data.fingerprint(s)
    original_eval = s['subset_pid'].copy()
    original = (ct.base.OUTPUTS, ct.pc_data.partitions, ct.pc_data.caption_features,
                ct.pc_stage2.make_config, ct.pc_sept.base.soft_assignments)
    with pytest.raises(RuntimeError, match='restore'):
        with ct.runtime(tmp_path, s) as view:
            pairs = ct.pc_fixes.pair_rows(view['pid'], ct.pc_data.partitions(view))
            np.testing.assert_array_equal(pairs, s['fit_pairs'])
            assert len(pairs) == (67327 if mode == 'sample' else 216107)
            config = ct.pc_stage1.make_config(view, 'qwen_qwen', 0, False, pairs, 'cuda')
            assert (config['pretrain_epochs'], config['pretrain_batch_size'], config['max_joint_steps']) == (100, 1024, 500)
            assert config['fit_pairs'] == len(pairs)
            assert config['dec_chunk_rows'] == (None if mode == 'sample' else 1024)
            mapper = ct.pc_stage2.make_config(view, 'P3', 0, False, 'cuda')
            assert mapper['dev_paintings'] == (1489 if mode == 'sample' else 6152)
            assert ct.pc_data.fingerprint(view) == identity
            raise RuntimeError('restore')
    assert (ct.base.OUTPUTS, ct.pc_data.partitions, ct.pc_data.caption_features,
            ct.pc_stage2.make_config, ct.pc_sept.base.soft_assignments) == original
    np.testing.assert_array_equal(s['subset_pid'], original_eval)


@pytest.mark.parametrize('arm', ['P1', 'P4'])
def test_paper_strict_mapper_trains_at_selected_non40_width(synthetic, tmp_path, arm):
    """A schema-valid noncollapsed Stage 1 source, independent of DEC convergence."""
    import ct_paper_sept as paper
    s, mode, k = synthetic['full'], 'full', 30
    version = ct.ARMS[arm][0]
    config = ct.stage1_config(s, version, 0, True, s['fit_pairs'], 'cpu')
    paper.stage1_config(config)
    config.update(settings='paper', cut='k40', eval_paintings=len(s['dev']))
    encoder, _ = paper.build_autoencoder(config['input_dim'])
    centers = torch.zeros(100, 128)
    centers[:, 0] = torch.cat([torch.arange(30) * .01 + 1, torch.arange(70) * .01 + 10])
    norms = torch.linalg.vector_norm(centers, dim=1).numpy()
    counts = np.full(100, 20, dtype=np.int64)
    kept = ct.pc_fixes.top_used(counts, 40)
    strict = np.arange(100) < 30
    topic = (np.arange(len(s['pid'])) % 40).astype(np.int16)
    topic_strict = (np.arange(len(s['pid'])) % k).astype(np.int16)
    q = np.full((len(s['dev']), 40), 1 / 40, dtype=np.float32)
    qs = np.full((len(s['dev']), k), 1 / k, dtype=np.float32)
    arrays = dict(center_norms=norms, fit_counts=counts, kept=kept, topic=topic,
        q_paint_eval=q, centers=centers.numpy(), fit_latent=np.zeros((2000, 128), dtype=np.float32),
        kept_k40=kept, fit_topics_k40=topic[s['fit_pairs']], kept_strict=strict,
        fit_topics_strict=topic_strict[s['fit_pairs']], topic_strict=topic_strict, topic_k40=topic,
        q_paint_eval_strict=qs, q_paint_eval_k40=q, stop_epoch=np.array([1], dtype=np.int64))
    meta = dict(config=config, K=40, K_by_cut=dict(strict=k, k40=40), stop_epoch=1,
        epochs=dict(joint=1), sample_sha256=ct.pc_data.fingerprint(s),
        K_gamma={gamma: int(ct.pc_fixes.kept_centers(norms, float(gamma)).sum())
                 for gamma in ('1.5', '2', '3', '5')})
    source = ct.stage1_path('paper', version, 0, root=tmp_path)
    ct.base.save_output(source, arrays, dict(encoder=ct.base.cpu_state(encoder), centers=centers), meta)
    ct.verify_output(source, root=tmp_path)
    path = ct.run_stage2(mode, 'paper', 'strict', arm, 0, root=tmp_path, device='cpu', smoke=True)
    meta = ct.verify_output(path, root=tmp_path)
    assert meta['K'] == k
    assert meta['config']['lr'] == .001 and meta['config']['batch_size'] == 64
    assert meta['dev_valid_topics'] == k and meta['dev_auc'] is not None
    names = s['all_painting_ids'][s['dev']].tolist()
    with np.load(path) as z:
        assert z['eval_prob'].shape == (9, k)
        np.testing.assert_array_equal(z['eval_prob'], ct.infer_stage2(
            mode, 'paper', 'strict', arm, 0, names, root=tmp_path, device='cpu'))


def test_caption_targets_keep_stage1_partition_checks(monkeypatch):
    s = tiny_roles('sample')
    topic = np.zeros(8, dtype=np.int16)
    good = ct.partitions(s)
    monkeypatch.setattr(ct, 'partitions', lambda _: good[:-1])
    with pytest.raises(AssertionError):  # partition must have one label per painting
        ct.caption_targets(s, topic, 3)
    wrong = good.copy()
    wrong[s['subset_pid']] = 'dev'
    monkeypatch.setattr(ct, 'partitions', lambda _: wrong)
    with pytest.raises(AssertionError):  # evaluation paintings must hold the eval role
        ct.caption_targets(s, topic, 3)
    full = tiny_roles('full')
    wrong = ct.partitions(full)
    wrong[full['dev'][0]] = 'fit'
    monkeypatch.setattr(ct, 'partitions', lambda _: wrong)
    with pytest.raises(AssertionError):  # full mode evaluates the dev role
        ct.caption_targets(full, topic, 3)


class Gate:
    """Fake ct2_stage2a: records the order of gate calls and base-trainer calls."""

    def __init__(self, monkeypatch, refuse=False):
        self.events = []
        self.refuse = refuse
        monkeypatch.setitem(sys.modules, 'ct2_stage2a', SimpleNamespace(require_pass=self.require_pass))

    def require_pass(self, **kwargs):
        self.events.append(('gate', kwargs))
        if self.refuse:
            raise RuntimeError('stage 2a has not passed')

    def runner(self, name):
        def call(*args, **kwargs):
            self.events.append((name, kwargs.get('device')))
            return Path(name)
        return call


@pytest.fixture
def trained(synthetic, tmp_path, monkeypatch):
    """Smoke artifacts for seed 0 (smoke is ungated) and a frozen-list stand-in."""
    root = tmp_path / 'full'
    ct.run_stage1('full', 'core', 'clip_roberta', 0, root=root, device='cpu', smoke=True)
    ct.run_stage2('full', 'core', 'k40', 'P2', 0, root=root, device='cpu', smoke=True)
    digest = ct.base.ct_held.names_sha256(['a'])
    monkeypatch.setattr(ct.base.ct_held, 'read_held_list', lambda path: (['a'], digest))
    return root


@pytest.mark.parametrize('refuse', [True, False])
@pytest.mark.parametrize('stage', ['1', '2', 'held'])
def test_stage2a_gate_precedes_full_mode_training_and_held_writes(trained, monkeypatch, stage, refuse):
    gate = Gate(monkeypatch, refuse)
    for name in ('run_stage1', 'run_stage2', 'run_held'):
        monkeypatch.setattr(ct.base, name, gate.runner(name))
    calls = dict(
        **{'1': lambda **kw: ct.run_stage1('full', 'core', 'clip_roberta', 0, root=trained, **kw),
           '2': lambda **kw: ct.run_stage2('full', 'core', 'k40', 'P2', 0, root=trained, **kw),
           'held': lambda **kw: ct.run_held('full', 'core', 'k40', 'P2', 0, root=trained, **kw)})
    device = 'cpu' if stage == 'held' else None  # held replays the training device (cpu here)
    if refuse:
        with pytest.raises(RuntimeError, match='stage 2a'):
            calls[stage](device=device)
        assert gate.events == [('gate', dict(device='cpu' if stage == 'held' else 'cuda'))]
    else:
        calls[stage](device=device)
        name = dict([('1', 'run_stage1'), ('2', 'run_stage2'), ('held', 'run_held')])[stage]
        assert [e[0] for e in gate.events] == ['gate', name]
        assert gate.events[0][1] == dict(device='cpu' if stage == 'held' else 'cuda')


def test_stage2a_gate_uses_the_requested_training_device_and_skips_sample_and_smoke(trained, monkeypatch):
    gate = Gate(monkeypatch)
    for name in ('run_stage1', 'run_stage2', 'run_held'):
        monkeypatch.setattr(ct.base, name, gate.runner(name))
    ct.run_stage1('full', 'core', 'clip_roberta', 1, root=trained, device='cuda:1')
    assert gate.events[0] == ('gate', dict(device='cuda:1'))
    gate.events.clear()
    ct.run_stage1('sample', 'core', 'clip_roberta', 1, root=trained)  # sample mode is stage 2a itself
    ct.run_stage1('full', 'core', 'clip_roberta', 1, root=trained, device='cpu', smoke=True)
    ct.run_held('full', 'core', 'k40', 'P2', 0, root=trained, device='cpu', smoke=True)
    assert [e[0] for e in gate.events] == ['run_stage1', 'run_stage1', 'run_held']


def test_held_list_must_match_the_frozen_stage1_list(trained, monkeypatch):
    gate = Gate(monkeypatch)
    monkeypatch.setattr(ct.base, 'run_held', gate.runner('run_held'))
    digest = ct.base.ct_held.names_sha256(['a'])

    def read(path):
        return (['a'], digest if path == ct.FROZEN_HELD_LIST else 'other')
    monkeypatch.setattr(ct.base.ct_held, 'read_held_list', read)
    with pytest.raises(AssertionError, match='frozen'):
        ct.run_held('full', 'core', 'k40', 'P2', 0, root=trained, device='cpu', smoke=True)
    assert gate.events == []
    monkeypatch.setattr(ct.base.ct_held, 'read_held_list', lambda path: (['a'], digest))
    ct.run_held('full', 'core', 'k40', 'P2', 0, root=trained, device='cpu', smoke=True)
    assert gate.events == [('run_held', 'cpu')]
    assert ct.FROZEN_HELD_LIST == ct.ct_score.OUTPUTS / 'held/held_paintings.json'


def test_cli_dec_chunk_rows_default_override_and_separate_root(synthetic, tmp_path):
    def stage1(root, *extra):
        [path] = ct.main(['--mode', 'sample', '--stage', '1', '--version', 'clip_roberta', '--seeds', '0',
                          '--device', 'cpu', '--root', str(root), '--smoke', *extra])
        return path, ct.verify_output(path, root=root)['config']

    gating, config = stage1(tmp_path / 'gating')
    assert config['dec_chunk_rows'] is None and config['mode'] == 'sample'
    before = {p: p.read_bytes() for p in (tmp_path / 'gating').rglob('*') if p.is_file()}
    extra, config = stage1(tmp_path / 'chunked', '--dec-chunk-rows', '64')
    assert config['dec_chunk_rows'] == 64 and extra.is_relative_to(tmp_path / 'chunked')
    assert before == {p: p.read_bytes() for p in (tmp_path / 'gating').rglob('*') if p.is_file()}
    [path] = ct.main(['--mode', 'full', '--stage', '1', '--version', 'clip_roberta', '--seeds', '0',
                      '--device', 'cpu', '--root', str(tmp_path / 'full'), '--smoke', '--dec-chunk-rows', '32'])
    assert ct.verify_output(path, root=tmp_path / 'full')['config']['dec_chunk_rows'] == 32
    with pytest.raises(SystemExit):  # the flag only concerns Stage 1
        ct.main(['--mode', 'full', '--stage', '2', '--arm', 'P2', '--root', str(tmp_path), '--dec-chunk-rows', '32'])


def test_sample_stage1_never_calls_the_chunked_function_unless_asked(synthetic, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('chunked assignments used')
    monkeypatch.setattr(ct, 'chunked_soft_assignments', forbidden)
    ct.run_stage1('sample', 'core', 'clip_roberta', 0, root=tmp_path, device='cpu', smoke=True)
    with pytest.raises(AssertionError, match='chunked'):
        ct.run_stage1('sample', 'core', 'clip_roberta', 0, root=tmp_path / 'b', device='cpu', smoke=True,
                      dec_chunk_rows=64)


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('held_name', ['d2/emotion__s0.npz', 'percept/paper__strict/P4__s4.npz'])
def test_v4_refuses_full_training_after_any_held_output(tmp_path, monkeypatch, stage, held_name):
    held = tmp_path / 'held' / held_name
    held.parent.mkdir(parents=True)
    held.write_bytes(b'held output exists')
    monkeypatch.setattr(ct, 'require_stage2a', lambda *a: None)
    monkeypatch.setattr(ct.data, 'split2', lambda *a: pytest.fail('training inputs read after held output'))
    with pytest.raises(AssertionError, match='^V4'):
        if stage == 1:
            ct.run_stage1('full', 'paper', 'qwen_qwen', 0, root=tmp_path)
        else:
            ct.run_stage2('full', 'paper', 'k40', 'P4', 0, root=tmp_path)
