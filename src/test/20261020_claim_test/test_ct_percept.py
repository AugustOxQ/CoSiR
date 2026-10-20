"""Core recipe parity and retained PercepT inference artifacts."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_percept as ct


@pytest.fixture
def sample(monkeypatch, tmp_path):
    """Synthetic feature stores; real core fusion, topics and training."""
    rng = np.random.default_rng(11)
    n = 7004
    names = np.asarray([f'painting-{i}' for i in range(n)])
    s = dict(pid=np.arange(n, dtype=np.int64), first=np.arange(n, dtype=np.int64),
             subset_pid=np.arange(2004, n, dtype=np.int64), all_painting_ids=names,
             painting_ids=names, sample_ids=np.arange(n, dtype=np.int64))
    parts = np.asarray(['fit'] * 2000 + ['dev'] * 4 + ['eval'] * 5000)
    image = rng.normal(size=(n, 512)).astype(np.float32)
    qwen = rng.normal(size=(n, 2048)).astype(np.float32)
    qwen /= np.linalg.norm(qwen, axis=1, keepdims=True)
    text = rng.normal(size=(n, 512)).astype(np.float32)
    emotion = rng.normal(size=(n, 768)).astype(np.float32)
    monkeypatch.setattr(ct.pc_data, 'partitions', lambda s: parts)
    monkeypatch.setattr(ct.pc_data, 'fingerprint', lambda s: 'synthetic-sample')
    monkeypatch.setattr(ct.pc_data, 'image_features',
                        lambda s, kind: (qwen if kind == 'qwen_default' else image).copy())
    monkeypatch.setattr(ct.pc_data, 'caption_features',
                        lambda s, kind: (emotion if kind == 'roberta' else
                                         qwen if kind.startswith('qwen_') else text).copy())

    def patches(s, ids):
        return np.broadcast_to(image[ids, None, :], (len(ids), 50, 512)).copy()

    lookup = {name: i for i, name in enumerate(names)}

    def rows(names):
        return np.asarray([lookup[name] for name in names])

    monkeypatch.setattr(ct.pc_data, 'patch_tokens', patches)
    monkeypatch.setattr(ct.ct_held, 'clip_patches', lambda names: patches(s, rows(names)))
    def clip_global(names):
        x = image[rows(names)]
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    monkeypatch.setattr(ct.ct_held, 'clip_global_images', clip_global)
    monkeypatch.setattr(ct.ct_held, 'qwen_default_images', lambda names: qwen[rows(names)].copy())
    monkeypatch.setattr(ct, 'OUTPUTS', tmp_path / 'outputs')
    return s


@pytest.mark.parametrize(('version', 'width'), [('clip_roberta', 512), ('qwen_qwen', 2048)])
def test_stage1_matches_core_arrays_and_retains_encoder(sample, tmp_path, version, width):
    reference = ct.pc_stage1.run(sample, version, 0, smoke=tmp_path / 'core', device='cpu')
    reference_rng = torch.get_rng_state().clone()
    path = ct.run_stage1(sample, 'core', version, 0, smoke=True, device='cpu')
    assert torch.equal(torch.get_rng_state(), reference_rng)
    with np.load(reference) as expected, np.load(path) as actual:
        assert set(actual.files) == set(expected.files) | {'centers', 'fit_latent', 'kept_k40', 'fit_topics_k40'}
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key])
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['array_sha256s'] == {
            key: hashlib.sha256(actual[key].tobytes()).hexdigest() for key in actual.files}
        state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        encoder, _ = ct.pc_sept.build_autoencoder(width)
        encoder.load_state_dict(state['encoder'])
        centers = state['centers']
        assert centers.shape == (100, 128) and centers.dtype == torch.float32
        np.testing.assert_array_equal(torch.linalg.vector_norm(centers, dim=1).numpy(),
                                      actual['center_norms'])
        kept = actual['kept']
        assert kept.sum() == 40
        fit_rows = np.arange(2000, dtype=np.int64)
        h, _ = ct.pc_stage1.prepare_inputs(sample, version, fit_rows)
        encoder.eval()
        with torch.no_grad():
            fit_latent = np.concatenate([
                encoder(torch.from_numpy(h[fit_rows[start:start + ct.pc_stage1.BLOCK_SIZE]])).numpy()
                for start in range(0, len(fit_rows), ct.pc_stage1.BLOCK_SIZE)])
        assert actual['fit_latent'].dtype == np.float32
        np.testing.assert_array_equal(actual['fit_latent'], fit_latent)
        np.testing.assert_array_equal(actual['centers'], centers.numpy())
        np.testing.assert_array_equal(actual['kept_k40'], kept)
        np.testing.assert_array_equal(actual['fit_topics_k40'], actual['topic'][fit_rows])
        topics, q = ct.pc_stage1.assign_captions(encoder, centers, h, sample['pid'],
                                                sample['subset_pid'], kept, 'cpu')
        np.testing.assert_array_equal(topics, actual['topic'])
        np.testing.assert_array_equal(q, actual['q_paint_eval'])
    assert meta['weights_sha256'] == hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()
    assert ct.verify_output(path)['config']['seed'] == 0
    topics, k = ct.load_stage1(sample, path, version, 0)
    expected_topics, expected_k = ct.pc_stage2.load_stage1(sample, reference, version, 0)
    assert k == expected_k
    np.testing.assert_array_equal(topics, expected_topics)


@pytest.mark.parametrize(('arm', 'seed'), [('P1', 0), ('P2', 1), ('P3', 3), ('P4', 4)])
def test_reloaded_mapper_equals_all_5000_evaluation_probabilities(sample, arm, seed):
    version = ct.ARMS[arm][0]
    stage1 = ct.run_stage1(sample, 'core', version, seed, smoke=True, device='cpu')
    path = ct.run_stage2(sample, 'core', 'k40', arm, seed, smoke=True, device='cpu')
    names = sample['all_painting_ids'][sample['subset_pid']].tolist()
    prob = ct.infer_stage2('core', 'k40', arm, seed, names)
    with np.load(path) as saved:
        np.testing.assert_array_equal(prob, saved['eval_prob'])
    assert prob.shape == (5000, 40) and prob.dtype == np.float32
    meta = ct.verify_output(path)
    assert meta['stage1_array_sha256s'] == ct.verify_output(stage1)['array_sha256s']
    state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
    assert state['mapper'] and meta['config']['seed'] == seed


@pytest.mark.parametrize('damage', ['arrays', 'weights', 'sidecar'])
def test_cli_skips_only_verified_complete_runs(sample, monkeypatch, damage):
    monkeypatch.setattr(ct, 'load_frozen', lambda: sample)
    args = ['--stage', '2', '--settings', 'core', '--cut', 'k40', '--arm', 'P2',
            '--seeds', '2', '--smoke', '--device', 'cpu']
    ct.run_stage1(sample, 'core', 'clip_roberta', 2, smoke=True, device='cpu')
    [path] = ct.main(args)
    original = np.load(path)['eval_prob']
    modified = path.stat().st_mtime_ns
    [same] = ct.main(args)
    assert same == path and path.stat().st_mtime_ns == modified
    if damage == 'arrays':
        np.savez_compressed(path, eval_prob=original + np.float32(.01))
    elif damage == 'weights':
        path.with_suffix('.pt').write_bytes(b'incomplete weights')
    else:
        path.with_suffix('.json').write_text('{}')
    ct.main(args)
    np.testing.assert_array_equal(np.load(path)['eval_prob'], original)
    assert ct.verify_output(path)['config']['seed'] == 2


@pytest.mark.parametrize('damage', ['arrays', 'held_sha256', 'weights_sha256'])
def test_held_cli_saves_frozen_order_and_binds_checkpoint(sample, monkeypatch, damage):
    ct.run_stage1(sample, 'core', 'clip_roberta', 2, smoke=True, device='cpu')
    source = ct.run_stage2(sample, 'core', 'k40', 'P2', 2, smoke=True, device='cpu')
    names = sorted(f'held-{i}' for i in range(12281))
    sha = ct.ct_held.names_sha256(names)
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: (names, sha))
    features = np.random.default_rng(87).normal(size=(12281, 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    monkeypatch.setattr(ct.ct_held, 'clip_global_images', lambda names: features.copy())
    args = ['--stage', 'held', '--arm', 'P2', '--seeds', '2', '--smoke']
    [path] = ct.main(args)
    assert path == ct.OUTPUTS / 'held/percept/core__k40/P2__s2.npz'
    prob = np.load(path)['prob']
    np.testing.assert_array_equal(prob, ct.infer_stage2('core', 'k40', 'P2', 2, names))
    meta = json.loads(path.with_suffix('.json').read_text())
    assert prob.shape == (12281, 40) and prob.dtype == np.float32
    assert meta['held_sha256'] == sha and meta['rows'] == 12281
    assert meta['weights_sha256'] == hashlib.sha256(source.with_suffix('.pt').read_bytes()).hexdigest()
    assert meta['array_sha256s']['prob'] == hashlib.sha256(prob.tobytes()).hexdigest()
    stamp = path.stat().st_mtime_ns

    def unread(names):
        pytest.fail('verified held output must be skipped without another feature read')
    monkeypatch.setattr(ct.ct_held, 'clip_global_images', unread)
    ct.main(args)
    assert path.stat().st_mtime_ns == stamp
    if damage == 'arrays':
        np.savez_compressed(path, prob=prob + np.float32(.01))
    else:
        meta[damage] = 'wrong hash'
        path.with_suffix('.json').write_text(json.dumps(meta))
    monkeypatch.setattr(ct.ct_held, 'clip_global_images', lambda names: features.copy())
    ct.main(args)
    np.testing.assert_array_equal(np.load(path)['prob'], prob)
    fixed = json.loads(path.with_suffix('.json').read_text())
    assert fixed['held_sha256'] == sha
    assert fixed['weights_sha256'] == hashlib.sha256(source.with_suffix('.pt').read_bytes()).hexdigest()


def test_stage1_cli_defaults_to_all_five_seeds_and_skips_without_features(sample, monkeypatch):
    monkeypatch.setattr(ct, 'load_frozen', lambda: sample)
    args = ['--stage', '1', '--version', 'clip_roberta', '--smoke', '--device', 'cpu']
    paths = ct.main(args)
    assert [path.stem for path in paths] == [f'clip_roberta__s{seed}' for seed in range(5)]
    stamps = [path.stat().st_mtime_ns for path in paths]

    def unread(s, kind):
        pytest.fail('verified Stage 1 output must not reload features')
    monkeypatch.setattr(ct.pc_data, 'image_features', unread)
    assert ct.main(args) == paths
    assert [path.stat().st_mtime_ns for path in paths] == stamps


@pytest.mark.skipif(os.environ.get('CT_PERCEPT_FROZEN_SMOKE') != '1',
                    reason='opt-in smoke run using main-checkout feature stores')
@pytest.mark.parametrize(('version', 'arms'), [('clip_roberta', ('P1', 'P2')),
                                              ('qwen_qwen', ('P3', 'P4'))])
def test_frozen_evaluation_replay(tmp_path, monkeypatch, version, arms):
    """Exercise real name joins on the frozen 5,000; read no held features."""
    monkeypatch.setattr(ct, 'OUTPUTS', tmp_path / 'outputs')
    s = ct.load_frozen()
    names = s['subset_painting_ids'].tolist()
    assert len(names) == 5000
    ct.run_stage1(s, 'core', version, 0, smoke=True, device='cpu')
    for arm in arms:
        path = ct.run_stage2(s, 'core', 'k40', arm, 0, smoke=True, device='cpu')
        prob = ct.infer_stage2('core', 'k40', arm, 0, names)
        with np.load(path) as saved:
            np.testing.assert_array_equal(prob, saved['eval_prob'])


@pytest.mark.parametrize('stage', [1, 2])
def test_full_training_refuses_cpu_before_loading_data(stage):
    with pytest.raises(AssertionError, match=f'full Stage {stage} requires GPU'):
        if stage == 1:
            ct.run_stage1(None, 'core', 'clip_roberta', 0, device='cpu')
        else:
            ct.run_stage2(None, 'core', 'k40', 'P2', 0, device='cpu')


def test_held_cli_leaves_default_device_to_inference(sample, monkeypatch):
    ct.run_stage1(sample, 'core', 'clip_roberta', 0, smoke=True, device='cpu')
    ct.run_stage2(sample, 'core', 'k40', 'P2', 0, smoke=True, device='cpu')
    names = sorted(f'held-{i}' for i in range(12281))
    sha = ct.ct_held.names_sha256(names)
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: (names, sha))
    devices = []

    def infer(settings, cut, arm, seed, names, *, device):
        devices.append(device)
        return np.zeros((len(names), 40), dtype=np.float32)

    monkeypatch.setattr(ct, 'infer_stage2', infer)
    [path] = ct.main(['--stage', 'held', '--arm', 'P2', '--seeds', '0', '--smoke'])
    assert devices == [None]
    assert json.loads(path.with_suffix('.json').read_text())['inference_device'] == 'cpu'


@pytest.mark.parametrize('fault', [
    None, 'sample', 'version', 'seed', 'gammas', 'k_type', 'k_value',
    'center_norms_shape', 'center_norms_dtype', 'center_norms_nan',
    'fit_counts_shape', 'fit_counts_dtype', 'fit_counts_negative',
    'kept_shape', 'kept_dtype', 'kept_count', 'kept_mask',
    'topic_shape', 'topic_dtype', 'topic_range',
    'q_paint_eval_shape', 'q_paint_eval_dtype', 'q_paint_eval_nan', 'q_paint_eval_range', 'q_paint_eval_sum'])
def test_stage1_loader_preserves_core_file_checks(tmp_path, fault):
    ids = np.arange(6, dtype=np.int64)
    s = dict(sample_ids=ids, pid=ids, first=ids, all_painting_ids=np.asarray(list('abcdef')),
             subset_pid=ids[:3], subset_image_indices=ids[:3])
    meta = dict(sample_sha256=ct.pc_data.fingerprint(s),
                config=dict(settings='core', version='clip_roberta', seed=0),
                K_gamma={key: 40 for key in ('1.5', '2', '3', '5')}, K=40)
    counts = np.zeros(100, dtype=np.int64)
    counts[:6] = 1
    kept = ct.pc_fixes.top_used(counts, 40)
    arrays = dict(center_norms=np.zeros(100, dtype=np.float32), fit_counts=counts,
                  kept=kept, topic=np.arange(6, dtype=np.int16),
                  q_paint_eval=np.full((3, 40), 1 / 40, dtype=np.float32))
    if fault == 'sample':
        meta['sample_sha256'] = 'wrong sample'
    elif fault in ('version', 'seed'):
        meta['config'][fault] = 'qwen_qwen' if fault == 'version' else 1
    elif fault == 'gammas':
        meta['K_gamma'] = {}
    elif fault in ('k_type', 'k_value'):
        meta['K'] = 40.0 if fault == 'k_type' else 39
    elif fault is not None:
        key, defect = fault.rsplit('_', 1)
        if defect == 'shape':
            arrays[key] = arrays[key][:-1]
        elif defect == 'dtype':
            dtype = {'center_norms': np.float64, 'fit_counts': np.int32,
                     'kept': np.uint8, 'topic': np.int64, 'q_paint_eval': np.float64}[key]
            arrays[key] = arrays[key].astype(dtype)
        elif defect in ('nan', 'negative', 'range'):
            arrays[key].flat[0] = {'nan': np.nan, 'negative': -1,
                                   'range': 40 if key == 'topic' else 2}[defect]
        elif defect == 'count':
            arrays[key][0] = False
        elif defect == 'mask':
            arrays[key] = arrays[key][::-1].copy()
        elif defect == 'sum':
            arrays[key][0] = 0
    reference = tmp_path / 'reference.npz'
    path = tmp_path / 'clip_roberta__s0.npz'
    for target in (reference, path):
        target.with_suffix('.json').write_text(json.dumps(meta))
    np.savez_compressed(reference, **arrays)
    np.savez_compressed(path, **arrays, centers=np.zeros((100, 128), dtype=np.float32),
                        fit_latent=np.zeros((6, 128), dtype=np.float32),
                        kept_k40=kept, fit_topics_k40=np.arange(6, dtype=np.int16))
    if fault is None:
        expected, expected_k = ct.pc_stage2.load_stage1(s, reference, 'clip_roberta', 0)
        actual, k = ct.load_stage1(s, path, 'clip_roberta', 0)
        assert k == expected_k
        np.testing.assert_array_equal(actual, expected)
    else:
        with pytest.raises(AssertionError):
            ct.pc_stage2.load_stage1(s, reference, 'clip_roberta', 0)
        with pytest.raises(AssertionError):
            ct.load_stage1(s, path, 'clip_roberta', 0)


@pytest.mark.parametrize('entrypoint', ['infer', 'held'])
def test_inference_refuses_a_device_different_from_the_checkpoint(sample, monkeypatch, entrypoint):
    ct.run_stage1(sample, 'core', 'clip_roberta', 0, smoke=True, device='cpu')
    source = ct.run_stage2(sample, 'core', 'k40', 'P2', 0, smoke=True, device='cpu')
    sidecar = source.with_suffix('.json')
    meta = json.loads(sidecar.read_text())
    meta['config']['device'] = 'cuda:1'
    sidecar.write_text(json.dumps(meta))
    names = sorted(f'held-{i}' for i in range(12281))
    sha = ct.ct_held.names_sha256(names)
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: (names, sha))
    with pytest.raises(AssertionError, match='inference device differs from training device'):
        if entrypoint == 'infer':
            ct.infer_stage2('core', 'k40', 'P2', 0, names, device='cpu')
        else:
            ct.run_held('core', 'k40', 'P2', 0, smoke=True, device='cpu')
