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


@pytest.mark.parametrize('threshold, expected_reason, expected_epoch', [
    (-1, 'epoch ceiling', 5),
    (1.0, 'stability criterion', 1),
], ids=['epoch-ceiling', 'early-stop'])
def test_september_port_matches_fresh_original_process(
        tmp_path, monkeypatch, threshold, expected_reason, expected_epoch):
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
        sweep.STABILITY_THRESHOLD = float(sys.argv[3])
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
                 reason=reason, epoch=epoch,
                 losses=losses, history=np.array([list(row.values()) for row in history]))
    ''')
    subprocess.run([os.sys.executable, '-c', script,
                    str(tmp_path / 'run_percept_stage1_cluster_count_sweep_pilot.py'),
                    str(tmp_path / 'reference.npz'), str(threshold)], check=True, timeout=180,
                   cwd='/project/CoSiR-buddy_prototype_conditioning')
    assert pc_sept.PRETRAIN_EPOCHS == 100 and pc_sept.MAX_DEC_EPOCHS == 500
    assert pc_sept.PRETRAIN_BATCH_SIZE == 1024
    assert pc_sept.STABILITY_THRESHOLD == .001
    assert pc_sept.LAMBDA_BALANCE == 1000 and pc_sept.LAMBDA_RECONSTRUCTION == 1
    monkeypatch.setattr(pc_sept, 'PRETRAIN_EPOCHS', 2)
    monkeypatch.setattr(pc_sept, 'PRETRAIN_BATCH_SIZE', 64)
    monkeypatch.setattr(pc_sept, 'MAX_DEC_EPOCHS', 5)
    monkeypatch.setattr(pc_sept, 'STABILITY_THRESHOLD', threshold)
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
    assert reason == expected_reason and epoch == expected_epoch
    if threshold > 0:
        assert history[-1]['fraction_changed'] < threshold
        assert epoch < pc_sept.MAX_DEC_EPOCHS
    with torch.no_grad():
        post = encoder(x).numpy()
        q = pc_sept.soft_assignments(encoder(x), centers).numpy()
        heldout = torch.tensor([[2., -1., 0., 4., 3., -2., 1.]])
        heldout_latent = encoder(heldout).numpy()
        heldout_q = pc_sept.soft_assignments(encoder(heldout), centers).numpy()
    with np.load(tmp_path / 'reference.npz') as expected:
        assert expected['reason'].item() == expected_reason
        assert expected['epoch'].item() == expected_epoch
        for name, actual in dict(pre=pre, post=post, initial=initial,
                                 centers=centers.detach().numpy(), q=q,
                                 heldout_latent=heldout_latent, heldout_q=heldout_q,
                                 labels=q.argmax(1), losses=losses,
                                 history=[list(row.values()) for row in history]).items():
            np.testing.assert_allclose(actual, expected[name], rtol=0, atol=1e-6)


def test_full_run_preserves_september_batch_and_stop_defaults():
    import pc_sept
    import pc_stage1

    s = dict(pid=np.array([0, 1, 0]), first=np.array([0, 1]))
    config = pc_stage1.make_config(s, 'clip_roberta', 0, False, np.arange(3), 'cuda')
    assert pc_sept.PRETRAIN_BATCH_SIZE == config['pretrain_batch_size'] == 1024
    assert pc_sept.STABILITY_THRESHOLD == config['stability_threshold'] == .001
    assert config['pretrain_epochs'] == 100 and config['max_joint_steps'] == 500


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


@pytest.mark.parametrize('controlled_norms', [False, True], ids=['trained-norms', 'gamma-gap'])
def test_smoke_trains_2000_fit_pairs_and_writes_stage1_schema(
        tmp_path, monkeypatch, capsys, controlled_norms):
    import pc_data
    import pc_sept
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
    fit_rows = np.flatnonzero(np.isin(pid, [0, 1, 2, 3]))[:2000]
    prepared, trained = {}, []
    prepare = pc_stage1.prepare_inputs
    pretrain = pc_sept.pretrain_autoencoder
    initialize = pc_sept.initialize_cluster_centers
    joint = pc_sept.train_dec_until_stable

    def capture_inputs(sample, version, rows):
        np.testing.assert_array_equal(rows, fit_rows)
        h, provenance = prepare(sample, version, rows)
        prepared['fit'] = h[fit_rows].copy()
        return h, provenance

    def check_pretrain(encoder, decoder, inputs, *args, **kwargs):
        assert inputs.shape == (2000, 512), 'pretrain must train only smoke fit pairs'
        np.testing.assert_array_equal(inputs.cpu().numpy(), prepared['fit'])
        trained.append('pretrain')
        return pretrain(encoder, decoder, inputs, *args, **kwargs)

    def check_joint(encoder, decoder, centers, inputs, *args, **kwargs):
        assert inputs.shape == (2000, 512), 'joint must train only smoke fit pairs'
        np.testing.assert_array_equal(inputs.cpu().numpy(), prepared['fit'])
        trained.append('joint')
        result = joint(encoder, decoder, centers, inputs, *args, **kwargs)
        if controlled_norms:
            # Largest gap / mean gap = 2.5 / (100.5 / 99), between 2 and 3.
            norms = np.arange(100, dtype=np.float32)
            norms[50:] += 1.5
            norms = norms[np.random.default_rng(9).permutation(100)]
            with torch.no_grad():
                centers.zero_()
                centers[:, 0].copy_(torch.from_numpy(norms).to(centers.device))
        return result

    def check_kmeans(encoder, inputs, device, n_clusters, seed):
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

        assert inputs.shape == (2000, 512) and n_clusters == 100 and seed == 0
        assert _openmp_effective_n_threads() == 1
        trained.append('kmeans')
        return initialize(encoder, inputs, device, n_clusters, seed)

    monkeypatch.setattr(pc_stage1, 'prepare_inputs', capture_inputs)
    monkeypatch.setattr(pc_sept, 'pretrain_autoencoder', check_pretrain)
    monkeypatch.setattr(pc_sept, 'initialize_cluster_centers', check_kmeans)
    monkeypatch.setattr(pc_sept, 'train_dec_until_stable', check_joint)
    path = pc_stage1.run(s, 'clip_roberta', 0, smoke=tmp_path, device='cpu')
    assert trained == ['pretrain', 'kmeans', 'joint']
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
    assert meta['config']['kmeans_threads'] == 1
    assert 'outroot' not in meta['config']
    assert meta['config']['fusion'] == 'Eq.2; normalized I,T,E; unnormalized h'
    assert 0 <= meta['final_change_fraction'] <= 1
    assert len(meta['input_sha256s']['emotion']) == 64
    assert len(meta['maps']['pca_mean_sha256']) == 64
    assert len(meta['maps']['pca_projection_sha256']) == 64
    assert meta['maps']['orthogonal_map_sha256'] is None
    with np.load(path, allow_pickle=False) as z:
        # Independent Algorithm 1 oracle; never call pc_fixes for expectations.
        norms = z['center_norms']
        ordered = np.sort(norms.astype(np.float64))
        gaps = ordered[1:] - ordered[:-1]
        largest = int(np.argmax(gaps))
        for gamma in ('1.5', '2', '3', '5'):
            expected_kept = np.ones(len(norms), dtype=bool)
            if gaps[largest] > float(gamma) * gaps.mean():
                cut = (ordered[largest] + ordered[largest + 1]) / 2
                expected_kept = norms < cut
            assert meta['K'][gamma] == int(expected_kept.sum())
            if gamma == '2':
                np.testing.assert_array_equal(z['kept'], expected_kept)
        if controlled_norms:
            assert meta['K']['2'] == 50 and meta['K']['3'] == 100
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


def test_one_thread_nested_limit_restores_all_pools_even_on_error():
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    from threadpoolctl import threadpool_info, threadpool_limits
    import pc_stage1

    with threadpool_limits(limits=8):
        assert _openmp_effective_n_threads() == 8
        assert all(pool['num_threads'] == 8 for pool in threadpool_info())
        with pytest.raises(RuntimeError, match='leave context'):
            with pc_stage1.one_thread():
                assert _openmp_effective_n_threads() == 1
                assert all(pool['num_threads'] == 1 for pool in threadpool_info())
                raise RuntimeError('leave context')
        assert _openmp_effective_n_threads() == 8
        assert all(pool['num_threads'] == 8 for pool in threadpool_info())


def test_one_thread_rejects_ineffective_limit(monkeypatch):
    from contextlib import nullcontext
    from threadpoolctl import threadpool_limits
    import pc_stage1

    monkeypatch.setattr(pc_stage1, 'threadpool_limits', lambda **kwargs: nullcontext())
    with threadpool_limits(limits=8):
        with pytest.raises(AssertionError):
            with pc_stage1.one_thread():
                pass


def test_a2_three_kmeans_starts_have_one_center_hash():
    from threadpoolctl import threadpool_limits
    import pc_stage0
    import pc_stage1

    with np.load(pc_stage0.SNAPSHOT, allow_pickle=True) as snapshot:
        latents = torch.from_numpy(snapshot['train_latent_pre'])
    hashes = []
    with threadpool_limits(limits=8):
        for _ in range(3):
            centers = pc_stage1.kmeans_start(torch.nn.Identity(), latents, 'cpu', 60, 42)
            assert centers.shape == (60, 128) and centers.dtype == torch.float32
            hashes.append(hashlib.sha256(centers.detach().numpy().tobytes()).hexdigest())
    assert len(set(hashes)) == 1, hashes


def test_outroot_cli_routes_full_run_and_rejects_smoke(tmp_path, monkeypatch):
    import pc_data
    import pc_stage1

    sample = object()
    calls = []
    run = pc_stage1.run
    monkeypatch.setattr(pc_data, 'load_frozen', lambda: sample)
    monkeypatch.setattr(pc_stage1, 'run', lambda *args, **kwargs: calls.append((args, kwargs)))
    args = ['pc_stage1.py', '--version', 'clip_roberta', '--seed', '0']
    monkeypatch.setattr(os.sys, 'argv', args + ['--outroot', str(tmp_path)])
    pc_stage1.main()
    assert calls == [((sample, 'clip_roberta', 0, None), dict(outroot=tmp_path))]
    monkeypatch.setattr(os.sys, 'argv', args)
    pc_stage1.main()
    assert calls[-1] == ((sample, 'clip_roberta', 0, None), dict(outroot=None))
    monkeypatch.setattr(os.sys, 'argv', args + ['--outroot', str(tmp_path), '--smoke', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        pc_stage1.main()
    assert error.value.code == 2 and len(calls) == 2
    with pytest.raises(AssertionError, match='--outroot cannot be used with --smoke'):
        run(sample, 'clip_roberta', 0, smoke=tmp_path, outroot=tmp_path)


def test_full_run_writes_outroot_without_recording_it(tmp_path, monkeypatch):
    import pc_data
    import pc_sept
    import pc_stage1

    sample = dict(pid=np.array([0, 0, 1]), first=np.array([0, 2]), subset_pid=np.array([1]))
    monkeypatch.setattr(pc_data, 'partitions', lambda sample: np.array(['fit', 'fit']))
    monkeypatch.setattr(pc_data, 'fingerprint', lambda sample: 'sample hash')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(pc_stage1, 'prepare_inputs', lambda *args: (
        np.ones((3, 512), dtype=np.float32),
        dict(seconds={}, input_sha256s={}, maps={})))
    monkeypatch.setattr(pc_sept, 'build_autoencoder',
                        lambda width: (torch.nn.Identity(), torch.nn.Identity()))
    monkeypatch.setattr(pc_sept, 'pretrain_autoencoder', lambda *args: [0.])
    monkeypatch.setattr(pc_sept, 'train_dec_until_stable',
                        lambda *args: ([dict(fraction_changed=0.)], 'stability criterion', 1))
    monkeypatch.setattr(pc_stage1, 'assign_captions', lambda *args: (
        np.zeros(3, dtype=np.int16), np.full((1, 100), .01, dtype=np.float32)))
    starts = []

    def initialize(encoder, inputs, device, n_clusters, seed):
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

        assert _openmp_effective_n_threads() == 1
        starts.append((device, n_clusters, seed))
        return torch.nn.Parameter(torch.ones((100, 128)))

    monkeypatch.setattr(pc_sept, 'initialize_cluster_centers', initialize)
    path = pc_stage1.run(sample, 'clip_roberta', 2, device='cuda', outroot=tmp_path)
    assert starts == [('cuda', 100, 2)]
    assert path == tmp_path / 'stage1/clip_roberta__s2.npz' and path.exists()
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['config']['kmeans_threads'] == 1 and meta['config']['smoke'] is False
    assert 'outroot' not in meta['config'] and str(tmp_path) not in json.dumps(meta)
