"""CPU regression checks for D2 at either data scale."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ct_d2
import ct2_data
import ct2_graphs
import ms_train
from ms_rows import ASPECTS, ROWS


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


@pytest.fixture
def tiny_model(monkeypatch):
    # Keep both towers, shared trunks, dropout and all three independent heads.
    class TinyTrunk(torch.nn.Module):
        def __init__(self, aspects):
            super().__init__()
            self.trunk = torch.nn.Sequential(torch.nn.Linear(2048, 8), torch.nn.GELU(),
                                             torch.nn.Dropout(.1))
            self.heads = torch.nn.ModuleDict({a: torch.nn.Linear(8, 2048) for a in aspects})
    monkeypatch.setattr(ms_train, 'TrunkSide', TinyTrunk)
    monkeypatch.setitem(ms_train.CONFIG, 'batch_size', 512)
    torch.set_num_threads(8)


def tiny_data(mode='sample', n=19889):
    rng = np.random.default_rng(31)
    values = rng.standard_normal((12, 2048), dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    x = np.broadcast_to(values[0], (n, 2048)).copy()
    x[:12] = values
    fit = np.array([8, 1, 7, 2, 6, 3], dtype=np.int64)
    dev = np.array([11, 9, 10, 0], dtype=np.int64)
    evaluation = np.full(5000, 4, dtype=np.int64) if mode == 'sample' else np.empty(0, dtype=np.int64)
    s = dict(mode=mode, first=np.arange(n), sample_ids=np.arange(n), pid=np.arange(n),
             all_painting_ids=np.array([f'p{i:05d}' for i in range(n)]), subset_pid=evaluation,
             subset_image_indices=evaluation.copy(), fit=fit, dev=dev, fit_pairs=fit.copy())
    s['names_sha256'] = {role: ct_d2.ct_held.names_sha256(s['all_painting_ids'][ids])
                         for role, ids in [('fit', fit), ('dev', dev), ('eval', evaluation)]}
    inputs = {a: dict(img=x, cap=x) for a in ASPECTS}
    graphs = {a: dict(fit_node_ids=fit[::-1].copy(), dev_node_ids=dev[::-1].copy(),
                     rows={ROWS['D2']['graphs'][a]: {
                         'fit': (np.array([[0, 1], [0, 2], [1, 3], [2, 4], [3, 5]]), np.ones(5, np.float32)),
                         'dev': (np.array([[0, 1], [0, 2], [1, 3]]), np.ones(3, np.float32))}})
              for a in ASPECTS}
    manifest = dict(identity=dict(method='tiny CPU fixture'), archives={a: a for a in ASPECTS},
                    counts={a: {p: {ROWS['D2']['graphs'][a]: len(graphs[a]['rows'][ROWS['D2']['graphs'][a]][p][0])}
                                for p in ('fit', 'dev')} for a in ASPECTS})
    s['data_sha256'] = dict(img=sha(x), cap=sha(x), fit=sha(fit), dev=sha(dev))
    return s, inputs, graphs, dict.fromkeys(ASPECTS, manifest)


def test_generalized_loop_matches_sheet_arrays_losses_and_best_epochs(tmp_path, monkeypatch, tiny_model):
    import ct2_d2
    s, inputs, graphs, manifests = tiny_data()
    # Unequal edge counts exercise distinct RNG streams, partial batches and head clocks.
    for aspect, repeats in zip(ASPECTS, [103, 104, 1]):
        row = ROWS['D2']['graphs'][aspect]
        pairs, weights = graphs[aspect]['rows'][row]['fit']
        graphs[aspect]['rows'][row]['fit'] = np.tile(pairs, (repeats, 1)), np.tile(weights, repeats)
        manifests[aspect]['counts'][aspect]['fit'][row] = len(pairs) * repeats
    monkeypatch.setattr(ms_train, 'disjoint_split', lambda split: (split['fit'], split['dev']))
    teacher = inputs['emotion']['img'][:, ::-1]
    monkeypatch.setattr(ms_train.ags_trainimg, 'aspect_images', lambda split, key: teacher)
    monkeypatch.setattr(ct2_data, 'instructed_images', lambda split, key: teacher)
    original_loss, traces, phase = ms_train.head_loss, [[], []], [0]

    def observe_loss(*args, **kwargs):
        traces[phase[0]].append((kwargs['aspect'], kwargs.get('include_mask_penalty', True),
                                sha(kwargs['pairs']), sha(kwargs['weights']),
                                tuple((key, str(kwargs[key].dtype), sha(kwargs[key].numpy()))
                                      for key in ('img', 'cap', 'teacher'))))
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(ms_train, 'head_loss', observe_loss)
    expected = ms_train.run(s, inputs, graphs, manifests, ROWS['D2'], 'D2', list(ASPECTS),
                            0, torch.device('cpu'), tmp_path / 'reference', max_epochs=3)
    phase[0] = 1
    actual = ct2_d2.run(s, inputs, graphs, manifests, ROWS['D2'], 'D2', list(ASPECTS),
                       0, torch.device('cpu'), tmp_path / 'actual', max_epochs=3)
    assert traces[0] == traces[1]
    for aspect in ASPECTS:
        assert actual[aspect]['best_epoch'] == expected[aspect]['best_epoch']
        assert actual[aspect]['best_dev_loss'] == expected[aspect]['best_dev_loss']
        for got, want in zip(actual[aspect]['history'], expected[aspect]['history']):
            assert {k: v for k, v in got.items() if k != 'seconds'} == {k: v for k, v in want.items() if k != 'seconds'}
        name = f"{aspect}__{ROWS['D2']['graphs'][aspect]}__s0.npz"
        with np.load(tmp_path / 'reference/D2' / name) as ref, np.load(tmp_path / 'actual/D2' / name) as new:
            assert ref.files == new.files == ['img', 'cap']
            for side in ref.files:
                np.testing.assert_array_equal(new[side].view(np.uint16), ref[side].view(np.uint16))


def write_graph_cache(s, graphs, spaces, root):
    folder = root / 'd2/graphs'
    folder.mkdir(parents=True)
    archives = {}
    for aspect, graph in graphs.items():
        arrays = {f'{p}_node_ids': graph[f'{p}_node_ids'] for p in ('fit', 'dev')}
        row = ROWS['D2']['graphs'][aspect]
        for part, (pairs, weights) in graph['rows'][row].items():
            arrays[f'{row}__{part}'], arrays[f'{row}__{part}_w'] = pairs, weights
        path = folder / f'{aspect}.npz'
        np.savez(path, **arrays)
        archives[aspect] = ms_train.head.sha_file(path)
    manifest = dict(schema=1, identity=dict(method=ct2_graphs.METHOD, fit_size=len(s['fit']),
                    dev_size=len(s['dev']), mode=s['mode'], sample_sha256=ms_train.itp_data.fingerprint(s),
                    names_sha256=s['names_sha256']), archives=archives,
                    source_sha256={key: sha(value) for key, value in spaces.items()},
                    counts={a: {p: {r: len(graphs[a]['rows'][r][p][0])} for p in ('fit', 'dev')}
                            for a, r in ct2_graphs.ROWS.items()})
    ms_train.head.atomic_json(folder / 'manifest.json', manifest)


def test_sample_providers_and_recipe_equal_stage1(tmp_path):
    import ct2_d2
    s, inputs, graphs, manifests = ct_d2.load_training_data()
    s2 = ct2_data.split2('sample')
    spaces = dict(D=inputs['emotion']['cap'], I=inputs['emotion']['img'],
                  E=ct2_data.instructed_captions(s2, 'emotion_focus'),
                  **{'style.img': ms_train.ags_trainimg.aspect_images(s, 'style_word'),
                     'genre.img': ms_train.ags_trainimg.aspect_images(s, 'genre_word')})
    write_graph_cache(s2, graphs, spaces, tmp_path)
    actual_s, actual_inputs, actual_graphs, actual_manifests = ct2_d2.load_training_data('sample', tmp_path)
    for key in s:
        np.testing.assert_array_equal(actual_s[key], s[key])
    fit, dev = ms_train.disjoint_split(s)
    np.testing.assert_array_equal(actual_s['fit'], fit)
    np.testing.assert_array_equal(actual_s['dev'], dev)
    for aspect in ASPECTS:
        for side in ('img', 'cap'):
            np.testing.assert_array_equal(actual_inputs[aspect][side], inputs[aspect][side])
        teacher = ms_train.ags_trainimg.aspect_images(s, ms_train.head.ASPECTS[aspect])
        assert actual_s['data_sha256'][f'{aspect}.img'] == sha(teacher)
        for part in ('fit', 'dev'):
            np.testing.assert_array_equal(actual_graphs[aspect][f'{part}_node_ids'], graphs[aspect][f'{part}_node_ids'])
            row = ROWS['D2']['graphs'][aspect]
            for got, want in zip(actual_graphs[aspect]['rows'][row][part], graphs[aspect]['rows'][row][part]):
                np.testing.assert_array_equal(got, want)
        config = ct2_d2.make_config(actual_s, ROWS['D2'], 'D2', aspect, list(ASPECTS),
                                    actual_manifests, 0, torch.device('cuda'), False, 200)
        reference = ms_train.make_config(s, ROWS['D2'], 'D2', aspect, list(ASPECTS),
                                         manifests, 0, torch.device('cuda'), False, 200)
        added = {'mode', 'names_sha256', 'data_sha256', 'output_rows', 'stage2_code_sha256'}
        changed = {'graph_sha256', 'graph_archives', 'graph_identity'}
        assert set(config) - set(reference) == added
        assert {k for k in reference if reference[k] != config[k]} == changed
        assert {k: v for k, v in config.items() if k not in added | changed} == {k: v for k, v in reference.items() if k not in changed}
        path = Path('/project/CoSiR/src/test/20261020_claim_test/outputs/d2/train/D2') / f'{aspect}__{row}__s0.json'
        assert reference == json.loads(path.read_text())['config']
    assert actual_s['data_sha256']['fit'] == sha(fit)
    assert actual_s['data_sha256']['dev'] == sha(dev)
    assert actual_s['data_sha256']['fit_pairs'] == sha(actual_s['fit_pairs'])


def test_full_training_stops_per_head_saves_best_weights_and_validation_rows(tmp_path, monkeypatch, tiny_model):
    import ct2_d2
    s, inputs, graphs, manifests = tiny_data('full', n=12)
    monkeypatch.setattr(ct2_d2, 'load_training_data', lambda mode, root: (s, inputs, graphs, manifests))
    monkeypatch.setattr(ct2_data, 'instructed_images', lambda split, key: inputs['emotion']['img'][::-1].copy())
    monkeypatch.setitem(ms_train.CONFIG, 'max_epochs', 4)
    monkeypatch.setitem(ms_train.CONFIG, 'patience', 1)
    gates = []
    monkeypatch.setitem(sys.modules, 'ct2_stage2a', SimpleNamespace(require_pass=lambda **kw: gates.append(kw)))
    original_loss = ms_train.head_loss
    checkpoints = {a: [] for a in ASPECTS}
    dev_losses = dict(emotion=[.1, .2], style=[.3, .1, .2], genre=[.4, .3, .1, .2])

    def scripted_dev(models, *args, aspect, include_mask_penalty=True, **kwargs):
        if include_mask_penalty:
            return original_loss(models, *args, aspect=aspect, **kwargs)
        checkpoints[aspect].append(copy.deepcopy(models.state_dict()))
        return torch.tensor(dev_losses[aspect][len(checkpoints[aspect]) - 1])

    monkeypatch.setattr(ms_train, 'head_loss', scripted_dev)
    result = ct2_d2.train('full', [4], 'cpu', tmp_path)
    assert gates == [dict(device='cpu')]
    assert set(result) == {4} and set(result[4]) == set(ASPECTS)
    for aspect, best_epoch in zip(ASPECTS, [1, 2, 3]):
        info = result[4][aspect]
        assert info['best_epoch'] == best_epoch and info['epochs_run'] == best_epoch + 1
        assert info['best_dev_loss'] == pytest.approx(.1)
        weight = tmp_path / 'd2/weights' / f'{aspect}__s4.pt'
        saved = torch.load(weight, map_location='cpu', weights_only=True)
        assert set(saved) == set(checkpoints[aspect][0])
        for key in saved:
            assert torch.equal(saved[key], checkpoints[aspect][best_epoch - 1][key])
        assert any(not torch.equal(saved[k], checkpoints[aspect][-1][k]) for k in saved)
        weight_info = json.loads(weight.with_suffix('.json').read_text())
        assert weight_info['weights_sha256'] == ms_train.head.sha_file(weight)
        assert weight_info['output_sha256'] == info['output_sha256']
        assert weight_info['best_epoch'] == best_epoch and weight_info['best_dev_loss'] == info['best_dev_loss']
        assert weight_info['config'] == info['config']
        path = tmp_path / 'd2/train/D2' / f"{aspect}__{ROWS['D2']['graphs'][aspect]}__s4.npz"
        _, views = ms_train.build_model(ROWS['D2'], list(ASPECTS), 'cpu')
        views[aspect].load_state_dict(saved)
        views[aspect].eval()
        with np.load(path) as z, torch.no_grad():
            for side in ('img', 'cap'):
                expected = views[aspect][side](torch.from_numpy(inputs[aspect][side][s['dev']])).numpy().astype(np.float16)
                np.testing.assert_array_equal(z[side], expected)
                assert z[side].shape == (4, 2048) and z[side].dtype == np.float16
    # Complete weights and outputs skip; neither the loss nor weights are regenerated.
    monkeypatch.setattr(ct2_d2, 'run', lambda *args, **kwargs: pytest.fail('complete cohort reran'))
    assert ct2_d2.train('full', [4], 'cpu', tmp_path) == result


@pytest.fixture
def trained(tmp_path, monkeypatch, tiny_model):
    import ct2_d2
    data = tiny_data('full', n=12)
    monkeypatch.setattr(ct2_d2, 'load_training_data', lambda mode, root: data)
    monkeypatch.setattr(ct2_data, 'instructed_images', lambda s, key: data[1]['emotion']['img'])
    monkeypatch.setitem(ms_train.CONFIG, 'max_epochs', 2)
    monkeypatch.setitem(sys.modules, 'ct2_stage2a', SimpleNamespace(require_pass=lambda **kw: None))
    result = ct2_d2.train('full', [0], 'cpu', tmp_path)
    return data, result, tmp_path


def test_infer_reuses_stage1_path_and_replays_validation_from_saved_weights(trained, monkeypatch):
    import ct2_d2
    (s, inputs, _, _), _, root = trained
    names = s['all_painting_ids'][s['dev']].tolist()
    monkeypatch.setattr(ct_d2.ct_held, 'qwen_default_images', lambda requested: inputs['emotion']['img'][s['dev']]
                        if requested == names else pytest.fail('changed name order'))
    original_infer, calls = ct_d2.infer, []

    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return original_infer(*args, **kwargs)

    monkeypatch.setattr(ct_d2, 'infer', observe)
    for aspect in ASPECTS:
        actual = ct2_d2.infer('full', aspect, 0, names, device='cpu', root=root)
        path = root / 'd2/train/D2' / f"{aspect}__{ROWS['D2']['graphs'][aspect]}__s0.npz"
        with np.load(path) as z:
            np.testing.assert_array_equal(actual.view(np.uint16), z['img'].view(np.uint16))
    assert len(calls) == 3


def test_held_writes_12281_unit_float16_rows_and_verified_sidecars(trained, monkeypatch):
    import ct2_d2
    (_, inputs, _, _), _, root = trained
    names = [f'painting-{i:05d}' for i in range(12281)]
    held_hash = ct_d2.ct_held.names_sha256(names)
    ms_train.head.atomic_json(root / 'held/held_paintings.json', dict(names=names, sha256=held_hash))
    reads = []

    def images(requested):
        assert requested == names
        reads.append(len(requested))
        return np.broadcast_to(inputs['emotion']['img'][0], (12281, 2048))

    monkeypatch.setattr(ct_d2.ct_held, 'qwen_default_images', images)
    result = ct2_d2.write_held('full', [0], 'cpu', root)
    assert reads == [12281] * 3
    for aspect in ASPECTS:
        path = root / 'held/d2' / f'{aspect}__s0.npz'
        with np.load(path) as z:
            assert z.files == ['img'] and z['img'].dtype == np.float16
            assert z['img'].shape == (12281, 2048) and np.isfinite(z['img']).all()
            assert np.max(np.abs(np.linalg.norm(z['img'].astype(np.float32), axis=1) - 1)) < 1e-3
        info = json.loads(path.with_suffix('.json').read_text())
        assert info == result[aspect, 0]
        assert info == dict(held_sha256=held_hash, rows=12281,
                            weights_sha256=ms_train.head.sha_file(root / 'd2/weights' / f'{aspect}__s0.pt'),
                            output_sha256=ms_train.head.sha_file(path))
    assert ct2_d2.write_held('full', [0], 'cpu', root) == result
    assert reads == [12281] * 3
    (root / 'held/d2/emotion__s0.npz').write_bytes(b'corrupt')
    with pytest.raises(AssertionError, match='held output checksum mismatch'):
        ct2_d2.write_held('full', [0], 'cpu', root)
    assert reads == [12281] * 3


@pytest.mark.parametrize('mode,dirname', [('sample', 's2a'), ('full', 'full')])
@pytest.mark.parametrize('explicit_root', [False, True])
def test_cli_dispatches_mode_seeds_device_and_root(tmp_path, monkeypatch, mode, dirname, explicit_root):
    import ct2_d2
    calls = []
    monkeypatch.setattr(ct2_d2, 'train', lambda *args: calls.append(('train', args)))
    monkeypatch.setattr(ct2_d2, 'write_held', lambda *args: calls.append(('held', args)))
    argv = ['--mode', mode, '--train', '--held', '--seeds', '0', '4', '--device', 'cpu']
    if explicit_root:
        argv += ['--root', str(tmp_path)]
    ct2_d2.main(argv)
    root = tmp_path if explicit_root else Path(ct2_d2.__file__).parent / 'outputs/stage2' / dirname
    assert calls == [('train', (mode, [0, 4], 'cpu', root)), ('held', (mode, [0, 4], 'cpu', root))]


def test_cli_defaults_to_five_seeds_and_training_cuda(tmp_path, monkeypatch):
    import ct2_d2
    calls = []
    monkeypatch.setattr(ct2_d2, 'train', lambda *args: calls.append(args))
    ct2_d2.main(['--mode', 'sample', '--train', '--root', str(tmp_path)])
    assert calls == [('sample', [0, 1, 2, 3, 4], 'cuda', tmp_path)]


@pytest.mark.parametrize('operation', ['train', 'infer', 'held'])
def test_stage2a_failure_precedes_training_and_held_reads(trained, monkeypatch, operation):
    import ct2_d2
    _, _, root = trained

    def refuse(**kwargs):
        assert kwargs == dict(device='cpu')
        raise AssertionError('stage 2a failed')

    monkeypatch.setitem(sys.modules, 'ct2_stage2a', SimpleNamespace(require_pass=refuse))
    monkeypatch.setattr(ct2_d2, 'load_training_data', lambda *args: pytest.fail('data read before gate'))
    monkeypatch.setattr(ct_d2.ct_held, 'qwen_default_images', lambda *args: pytest.fail('images read before gate'))
    monkeypatch.setattr(ct_d2.ct_held, 'read_held_list', lambda *args: pytest.fail('held list read before gate'))
    monkeypatch.setattr(ct_d2.ct_held, 'write_held_list', lambda *args: pytest.fail('held list built before gate'))
    with pytest.raises(AssertionError, match='stage 2a failed'):
        if operation == 'train':
            ct2_d2.train('full', [0], 'cpu', root)
        elif operation == 'infer':
            ct2_d2.infer('full', 'emotion', 0, ['painting'], root=root)
        else:
            ct2_d2.write_held('full', [0], root=root)


@pytest.mark.parametrize('fault', ['missing_weight', 'corrupt_weight', 'missing_output', 'corrupt_output', 'best_epoch'])
def test_incomplete_cohort_preserves_artifacts_and_retrains(trained, fault):
    import ct2_d2
    _, _, root = trained
    weight = root / 'd2/weights/emotion__s0.pt'
    output = root / 'd2/train/D2/emotion__Gres__s0.npz'
    if fault == 'missing_weight':
        weight.rename(weight.with_suffix('.missing'))
    elif fault == 'corrupt_weight':
        weight.write_bytes(b'corrupt weight')
    elif fault == 'missing_output':
        output.rename(output.with_suffix('.missing'))
    elif fault == 'corrupt_output':
        output.write_bytes(b'corrupt output')
    else:
        path = weight.with_suffix('.json')
        info = json.loads(path.read_text())
        info['best_epoch'] = 99
        path.write_text(json.dumps(info))
    before = {p.relative_to(root / 'd2'): p.read_bytes() for p in (root / 'd2').rglob('*')
              if p.suffix in ('.json', '.npz', '.pt')}
    result = ct2_d2.train('full', [0], 'cpu', root)
    backups = list((root / 'd2/replaced').iterdir())
    assert len(backups) == 1
    assert {p.relative_to(backups[0]): p.read_bytes() for p in backups[0].rglob('*') if p.is_file()} == before
    for aspect in ASPECTS:
        path = root / 'd2/train/D2' / f"{aspect}__{ROWS['D2']['graphs'][aspect]}__s0.npz"
        assert ct2_d2.verify_output(path, result[0][aspect]['config']) == result[0][aspect]
        assert ms_train.head.sha_file(root / 'd2/weights' / f'{aspect}__s0.pt') == json.loads(
            (root / 'd2/weights' / f'{aspect}__s0.json').read_text())['weights_sha256']


def test_resume_rejects_recipe_mismatch_without_changing_artifacts(trained):
    import ct2_d2
    _, _, root = trained
    path = root / 'd2/weights/style__s0.json'
    info = json.loads(path.read_text())
    info['config']['row']['distill_weight'] = 1.
    path.write_text(json.dumps(info))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(AssertionError, match='resume config mismatch'):
        ct2_d2.train('full', [0], 'cpu', root)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_v4_prevents_retraining_once_held_outputs_exist(trained):
    import ct2_d2
    _, _, root = trained
    held = root / 'held/d2/emotion__s0.npz'
    held.parent.mkdir(parents=True)
    held.write_bytes(b'held artifact exists')
    weight = root / 'd2/weights/emotion__s0.pt'
    weight.rename(weight.with_suffix('.missing'))
    with pytest.raises(AssertionError, match='V4'):
        ct2_d2.train('full', [0], 'cpu', root)
    assert not (root / 'd2/replaced').exists()


@pytest.mark.parametrize('fault,message', [('mode', 'weights mode mismatch'),
                                           ('code', 'weights code mismatch'),
                                           ('checksum', 'weights checksum mismatch'),
                                           ('device', 'inference device differs')])
def test_infer_rejects_wrong_provenance_before_feature_reads(trained, monkeypatch, fault, message):
    import ct2_d2
    _, _, root = trained
    path = root / 'd2/weights/emotion__s0.json'
    info = json.loads(path.read_text())
    if fault == 'mode':
        info['config']['mode'] = 'sample'
    elif fault == 'code':
        info['config']['stage2_code_sha256'] = 'old code'
    elif fault == 'checksum':
        info['weights_sha256'] = 'wrong hash'
    else:
        info['config']['device'] = 'cuda'
    path.write_text(json.dumps(info))
    monkeypatch.setattr(ct_d2.ct_held, 'qwen_default_images', lambda *a: pytest.fail('read before validation'))
    with pytest.raises(AssertionError, match=message):
        ct2_d2.infer('full', 'emotion', 0, ['painting'], device='cpu', root=root)


@pytest.mark.parametrize('fault', ['dtype', 'rows', 'norm'])
def test_validation_output_verifier_rejects_invalid_arrays_even_with_valid_checksum(trained, fault):
    import ct2_d2
    _, result, root = trained
    path = root / 'd2/train/D2/emotion__Gres__s0.npz'
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if fault == 'dtype':
        arrays['img'] = arrays['img'].astype(np.float32)
    elif fault == 'rows':
        arrays['img'] = arrays['img'][:-1]
    else:
        arrays['img'][0] = 0
    np.savez(path, **arrays)
    info = json.loads(path.with_suffix('.json').read_text())
    info['output_sha256'] = ms_train.head.sha_file(path)
    path.with_suffix('.json').write_text(json.dumps(info))
    with pytest.raises(AssertionError):
        ct2_d2.verify_output(path, result[0]['emotion']['config'])


def test_loader_builds_selected_graphs_then_checks_cache_sources_and_archives(tmp_path, monkeypatch):
    import ct2_d2
    rng = np.random.default_rng(17)
    spaces = {k: rng.standard_normal((48, 2048), dtype=np.float32) for k in ct2_graphs.KEYS}
    spaces = {k: x / np.linalg.norm(x, axis=1, keepdims=True) for k, x in spaces.items()}
    s = dict(mode='full', first=np.arange(48), sample_ids=np.arange(48), pid=np.arange(48),
             all_painting_ids=np.array([f'p{i:02d}' for i in range(48)]),
             subset_pid=np.empty(0, dtype=np.int64), subset_image_indices=np.empty(0, dtype=np.int64),
             fit=np.arange(24), dev=np.arange(24, 48), fit_pairs=np.arange(24))
    s['names_sha256'] = {p: ct_d2.ct_held.names_sha256(s['all_painting_ids'][rows])
                         for p, rows in [('fit', s['fit']), ('dev', s['dev']), ('eval', s['subset_pid'])]}
    monkeypatch.setattr(ct2_data, 'split2', lambda mode: dict(s))
    monkeypatch.setattr(ct2_data, 'default_images', lambda split: spaces['I'])
    monkeypatch.setattr(ct2_data, 'default_captions', lambda split: spaces['D'])
    monkeypatch.setattr(ct2_data, 'instructed_captions', lambda split, key: spaces['E'])
    monkeypatch.setattr(ct2_data, 'instructed_images', lambda split, key: spaces[
        {'emotion_focus': 'E', 'style_word': 'style.img', 'genre_word': 'genre.img'}[key]])
    loaded, inputs, graphs, manifests = ct2_d2.load_training_data('full', tmp_path)
    assert loaded['mode'] == 'full'
    for aspect, row in ct2_graphs.ROWS.items():
        assert set(graphs[aspect]['rows']) == {row}
        for part in ('fit', 'dev'):
            np.testing.assert_array_equal(graphs[aspect][f'{part}_node_ids'], s[part])
        assert manifests[aspect]['source_sha256'] == {k: sha(x) for k, x in spaces.items()}
        assert inputs[aspect]['img'] is spaces['I']
    monkeypatch.setattr(ct2_graphs, 'build', lambda *args: pytest.fail('rebuilt verified graphs'))
    ct2_d2.load_training_data('full', tmp_path)
    monkeypatch.setattr(ct2_data, 'default_images', lambda split: spaces['I'][::-1].copy())
    with pytest.raises(AssertionError, match='graph source mismatch'):
        ct2_d2.load_training_data('full', tmp_path)
    monkeypatch.setattr(ct2_data, 'default_images', lambda split: spaces['I'])
    (tmp_path / 'd2/graphs/emotion.npz').write_bytes(b'corrupt archive')
    with pytest.raises(AssertionError, match='graph archive checksum mismatch'):
        ct2_d2.load_training_data('full', tmp_path)
