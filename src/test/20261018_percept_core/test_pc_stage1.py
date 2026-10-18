"""CPU September parity and Stage 1 pair-to-artifact contracts."""
import hashlib
import json
import os
import subprocess
import textwrap

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA


SEPTEMBER = 'src/test/20260922_percept_topic_pipeline/'


def test_september_port_matches_fresh_original_process(tmp_path, monkeypatch):
    import pc_sept

    for name in ('run_percept_stage1_pilot.py',
                 'run_percept_stage1_cluster_count_sweep_pilot.py'):
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
        sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweep)
        base = sweep.base
        np.random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        base.PRETRAIN_EPOCHS = 2
        base.PRETRAIN_BATCH_SIZE = 64
        sweep.MAX_DEC_EPOCHS = 5
        sweep.STABILITY_THRESHOLD = -1
        x = torch.from_numpy(np.random.default_rng(0).normal(size=(133, 7)).astype(np.float32))
        encoder, decoder = base.build_autoencoder(7)
        losses = base.pretrain_autoencoder(encoder, decoder, x, 'cpu', lambda _: None)
        with torch.no_grad():
            pre = encoder(x).numpy().copy()
        centers = sweep.initialize_cluster_centers(encoder, x, 'cpu', 100, 0)
        initial = centers.detach().numpy().copy()
        history, reason, epoch = sweep.train_dec_until_stable(
            encoder, decoder, centers, x, 'cpu', lambda _: None, 100)
        with torch.no_grad():
            post = encoder(x).numpy()
            q = base.soft_assignments(encoder(x), centers).numpy()
            heldout = torch.tensor([[2., -1., 0., 4., 3., -2., 1.]])
            heldout_latent = encoder(heldout).numpy()
            heldout_q = base.soft_assignments(encoder(heldout), centers).numpy()
        np.savez(sys.argv[2], pre=pre, post=post, initial=initial,
                 centers=centers.detach().numpy(), q=q, labels=q.argmax(1),
                 heldout_latent=heldout_latent, heldout_q=heldout_q,
                 losses=losses, history=np.array([list(row.values()) for row in history]))
    ''')
    subprocess.run([os.sys.executable, '-c', script,
                    str(tmp_path / 'run_percept_stage1_cluster_count_sweep_pilot.py'),
                    str(tmp_path / 'reference.npz')], check=True, timeout=180,
                   cwd='/project/CoSiR-buddy_prototype_conditioning')
    assert pc_sept.PRETRAIN_EPOCHS == 100 and pc_sept.MAX_DEC_EPOCHS == 500
    assert pc_sept.STABILITY_THRESHOLD == .001
    assert pc_sept.LAMBDA_BALANCE == 1000 and pc_sept.LAMBDA_RECONSTRUCTION == 1
    monkeypatch.setattr(pc_sept, 'PRETRAIN_EPOCHS', 2)
    monkeypatch.setattr(pc_sept, 'PRETRAIN_BATCH_SIZE', 64)
    monkeypatch.setattr(pc_sept, 'MAX_DEC_EPOCHS', 5)
    monkeypatch.setattr(pc_sept, 'STABILITY_THRESHOLD', -1)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    x = torch.from_numpy(np.random.default_rng(0).normal(size=(133, 7)).astype(np.float32))
    encoder, decoder = pc_sept.build_autoencoder(7)
    losses = pc_sept.pretrain_autoencoder(encoder, decoder, x, 'cpu', lambda _: None)
    with torch.no_grad():
        pre = encoder(x).numpy().copy()
    centers = pc_sept.initialize_cluster_centers(encoder, x, 'cpu', 100, 0)
    initial = centers.detach().numpy().copy()
    history, reason, epoch = pc_sept.train_dec_until_stable(
        encoder, decoder, centers, x, 'cpu', lambda _: None, 100)
    assert reason == 'epoch ceiling' and epoch == 5
    with torch.no_grad():
        post = encoder(x).numpy()
        q = pc_sept.soft_assignments(encoder(x), centers).numpy()
        heldout = torch.tensor([[2., -1., 0., 4., 3., -2., 1.]])
        heldout_latent = encoder(heldout).numpy()
        heldout_q = pc_sept.soft_assignments(encoder(heldout), centers).numpy()
    with np.load(tmp_path / 'reference.npz') as expected:
        for name, actual in dict(pre=pre, post=post, initial=initial,
                                 centers=centers.detach().numpy(), q=q,
                                 heldout_latent=heldout_latent, heldout_q=heldout_q,
                                 labels=q.argmax(1), losses=losses,
                                 history=[list(row.values()) for row in history]).items():
            np.testing.assert_allclose(actual, expected[name], rtol=0, atol=1e-6)


def test_clip_inputs_use_fit_pca_mean_for_every_caption(monkeypatch):
    import pc_data
    import pc_stage1

    rng = np.random.default_rng(4)
    n = 527
    s = dict(pid=np.arange(n) % 9)
    fit = np.arange(520, dtype=np.int64)
    image = rng.normal(size=(9, 512)).astype(np.float32)
    text = rng.normal(size=(n, 512)).astype(np.float32)
    emotion = rng.normal(size=(n, 768)).astype(np.float32) + 10
    emotion[520:] -= 15
    monkeypatch.setattr(pc_data, 'image_features', lambda sample, kind: image.copy())
    monkeypatch.setattr(pc_data, 'caption_features',
                        lambda sample, kind: (text if kind == 'clip_text' else emotion).copy())
    h, provenance = pc_stage1.prepare_inputs(s, 'clip_roberta', fit, block_size=37)
    # Independent sklearn transform and Eq. 2, with the map's specified fixed signs.
    pca = PCA(n_components=512, svd_solver='full').fit(emotion[fit])
    signs = np.sign(pca.components_[np.arange(512), np.abs(pca.components_).argmax(1)])
    e = pca.transform(emotion) * signs
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    i, t = image[s['pid']], text.copy()
    i /= np.linalg.norm(i, axis=1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    c = (i + t) / 2
    expected = (2 * c + e * np.linalg.norm(c, axis=1, keepdims=True)) / 3
    np.testing.assert_allclose(h, expected, atol=3e-5, rtol=3e-4)
    assert h.shape == (n, 512) and h.dtype == np.float32
    assert set(provenance['input_sha256s']) == {'image', 'text', 'emotion'}
    maps = provenance['maps']
    assert len(maps['pca_mean_sha256']) == len(maps['pca_projection_sha256']) == 64
    assert maps['orthogonal_map_sha256'] is None
    mean = emotion[fit].mean(0, dtype=np.float64).astype(np.float32)
    assert maps['pca_mean_sha256'] == pc_stage1.array_sha256(mean)
    with pytest.raises(TypeError):
        pc_stage1.transform_emotion(emotion, projection=pca.components_.T.copy())


def test_assignment_uses_frozen_kept_centers_and_eval_painting_order():
    import pc_stage1

    encoder = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(2))
    centers = torch.nn.Parameter(torch.tensor([[0., 0.], [99., 99.], [2., 0.]]))
    h = np.array([[0, 0], [2, 0], [1, 0], [3, 0], [1, 0]], dtype=np.float32)
    pid = np.array([0, 2, 1, 0, 2])
    kept = np.array([True, False, True])
    weights_before = encoder.weight.detach().clone()
    centers_before = centers.detach().clone()
    topic, painting_q = pc_stage1.assign_captions(
        encoder, centers, h, pid, np.array([2, 0]), kept, 'cpu', block_size=2)
    np.testing.assert_array_equal(topic, [0, 1, 0, 1, 0])
    # Q at x=0: (5/6,1/6), x=1: (1/2,1/2), x=2 or 3: (1/6,5/6).
    np.testing.assert_allclose(painting_q, [[1/3, 2/3], [.5, .5]], atol=1e-6)
    assert topic.dtype == np.int16 and painting_q.dtype == np.float32
    torch.testing.assert_close(encoder.weight, weights_before, rtol=0, atol=0)
    torch.testing.assert_close(centers, centers_before, rtol=0, atol=0)
    assert encoder.weight.grad is None and centers.grad is None


def test_smoke_trains_2000_fit_pairs_and_writes_stage1_schema(tmp_path, monkeypatch, capsys):
    import pc_data
    import pc_stage1

    rng = np.random.default_rng(5)
    pid = np.concatenate((rng.integers(0, 4, 2017), [4, 4, 4, 5, 6, 5, 6])).astype(np.int64)
    first = np.array([np.flatnonzero(pid == p)[0] for p in range(7)])
    names = np.array([f'painting_{p}' for p in range(7)])
    s = dict(sample_ids=rng.permutation(len(pid)).astype(np.int64) + 123,
             pid=pid, first=first, all_painting_ids=names, painting_ids=names[pid],
             subset_pid=np.array([6, 5]), subset_image_indices=np.array([44, 22]))
    image = rng.normal(size=(7, 512)).astype(np.float32)
    text = rng.normal(size=(len(pid), 512)).astype(np.float32)
    emotion = rng.normal(size=(len(pid), 768)).astype(np.float32) + 4
    monkeypatch.setattr(pc_data, 'partitions',
                        lambda sample: np.array(['fit'] * 4 + ['dev', 'eval', 'eval']))
    monkeypatch.setattr(pc_data, 'image_features', lambda sample, kind: image.copy())
    monkeypatch.setattr(pc_data, 'caption_features',
                        lambda sample, kind: (text if kind == 'clip_text' else emotion).copy())
    path = pc_stage1.run(s, 'clip_roberta', 0, smoke=tmp_path, device='cpu')
    assert path == tmp_path / 'stage1' / 'clip_roberta__s0.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sample_sha256'] == pc_data.fingerprint(s)
    assert set(meta['K']) == {'1.5', '2', '3', '5'}
    assert meta['config']['fit_pairs'] == 2000
    assert meta['config']['fit_rows_sha256'] == hashlib.sha256(
        np.arange(2000, dtype=np.int64).tobytes()).hexdigest()
    assert meta['config']['pretrain_epochs'] == meta['epochs']['pretrain'] == 2
    assert meta['config']['max_joint_steps'] == meta['epochs']['joint'] == 5
    assert meta['config']['stability_threshold'] == 0
    assert meta['config']['full_stability_threshold'] == .001
    assert meta['config']['smoke'] is True
    assert meta['config']['balance_weight'] == 1000
    assert meta['config']['initial_centers'] == 100
    assert meta['config']['fusion'] == 'Eq.2; normalized I,T,E; unnormalized h'
    assert 0 <= meta['final_change_fraction'] <= 1
    assert len(meta['input_sha256s']['emotion']) == 64
    assert len(meta['maps']['pca_mean_sha256']) == 64
    assert len(meta['maps']['pca_projection_sha256']) == 64
    assert meta['maps']['orthogonal_map_sha256'] is None
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'center_norms', 'kept', 'topic', 'q_paint_eval'}
        assert z['center_norms'].shape == z['kept'].shape == (100,)
        assert z['center_norms'].dtype == np.float32 and z['kept'].dtype == np.bool_
        assert z['topic'].shape == (2024,) and z['topic'].dtype == np.int16
        assert z['q_paint_eval'].shape == (2, meta['K']['2'])
        assert z['q_paint_eval'].dtype == np.float32
        assert z['kept'].sum() == meta['K']['2']
        assert 0 <= z['topic'].min() <= z['topic'].max() < meta['K']['2']
        np.testing.assert_allclose(z['q_paint_eval'].sum(1), 1, atol=1e-6)
    output = capsys.readouterr().out
    for phase, seconds in meta['seconds'].items():
        assert np.isfinite(seconds) and seconds >= 0
        assert f'{phase}:' in output


@pytest.mark.parametrize('version', ['qwen_qwen', 'qwen_roberta'])
def test_qwen_feature_selection_and_emotion_map(version, monkeypatch):
    import pc_data
    import pc_stage1

    rng = np.random.default_rng(6)
    s = dict(pid=np.array([2, 0, 2, 1, 0]))
    image = rng.normal(size=(3, 2048)).astype(np.float32)
    text = rng.normal(size=(5, 2048)).astype(np.float32)
    width = 2048 if version == 'qwen_qwen' else 768
    emotion = rng.normal(size=(5, width)).astype(np.float32)
    def read_image(sample, kind):
        assert kind == 'qwen_default'
        return image.copy()
    def read_caption(sample, kind):
        if kind == 'qwen_default':
            return text.copy()
        assert kind == ('qwen_emotion_focus' if version == 'qwen_qwen' else 'roberta')
        return emotion.copy()
    monkeypatch.setattr(pc_data, 'image_features', read_image)
    monkeypatch.setattr(pc_data, 'caption_features', read_caption)
    h, provenance = pc_stage1.prepare_inputs(s, version, np.array([0, 1]), block_size=2)
    e = emotion / np.linalg.norm(emotion, axis=1, keepdims=True)
    maps = provenance['maps']
    if version == 'qwen_roberta':
        matrix = np.random.default_rng(0).normal(size=(2048, 768))
        q, r = np.linalg.qr(matrix, mode='reduced')
        orthogonal = (q * np.sign(np.diag(r))).astype(np.float32)
        assert maps['orthogonal_map_sha256'] == hashlib.sha256(orthogonal.tobytes()).hexdigest()
        e = e @ orthogonal.T
        e /= np.linalg.norm(e, axis=1, keepdims=True)
    else:
        assert maps['orthogonal_map_sha256'] is None
    assert maps['pca_mean_sha256'] is maps['pca_projection_sha256'] is None
    i, t = image[s['pid']], text.copy()
    i /= np.linalg.norm(i, axis=1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    c = (i + t) / 2
    np.testing.assert_allclose(h, (2 * c + e * np.linalg.norm(c, axis=1, keepdims=True)) / 3,
                               atol=1e-7, rtol=1e-5)
    for key, raw in dict(image=image, text=text, emotion=emotion).items():
        assert provenance['input_sha256s'][key] == hashlib.sha256(raw.tobytes()).hexdigest()
