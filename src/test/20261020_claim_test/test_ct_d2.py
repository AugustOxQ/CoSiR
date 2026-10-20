"""CPU checks for D2's saved best views and sheet-compatible outputs."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '20261019_model_sheet'))
import ms_train
from ms_rows import ASPECTS, ROWS

import ct_d2


@pytest.fixture(scope='module')
def sample():
    path = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


@pytest.fixture(scope='module')
def cohort_data(sample):
    # The sheet's serializer still sees its full 5,000-row evaluation boundary.
    rng = np.random.default_rng(31)
    x = rng.standard_normal((5000, 2048), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    inputs = dict(img=x, cap=x[::-1].copy())
    s = dict(sample, subset_pid=np.arange(5000, dtype=np.int64))
    manifests = {a: dict(identity={'method': 'tiny CPU fixture'},
                         archives={a: a + '-graph-hash'},
                         counts={a: {part: {ROWS['D2']['graphs'][a]: 2}
                                     for part in ('fit', 'dev')}}) for a in ASPECTS}
    return s, {a: inputs for a in ASPECTS}, {}, manifests


def test_training_keeps_each_best_view_and_the_original_sheet_bytes(tmp_path, monkeypatch, cohort_data):
    s, inputs, graphs, manifests = cohort_data
    monkeypatch.setattr(ct_d2, 'load_training_data', lambda: cohort_data)
    original_save = ms_train.save_output
    expected, best, calls = {}, {}, []

    def sheet_run(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir,
                  smoke=False, max_epochs=200):
        calls.append((row, name, aspects, seed, device, outdir, smoke, max_epochs))
        torch.manual_seed(seed)
        _, views = ms_train.build_model(row, aspects, device)
        result = {}
        for index, aspect in enumerate(aspects):
            # Distinct trunks per aspect simulate independent best-dev epoch clocks.
            with torch.no_grad():
                views[aspect]['img'].trunk[0].bias.add_(index / 10)
            state = dict(best_state=copy.deepcopy(views[aspect].state_dict()),
                         best_epoch=1, history=[{'epoch': 1}], best=0.25)
            best[aspect] = state['best_state']
            config = ms_train.make_config(s, row, name, aspect, aspects, manifests,
                                          seed, device, smoke, max_epochs)
            path = ms_train.output_path(outdir, name, aspect, config['graph_row'], seed)
            reference = tmp_path / 'reference' / path.name
            original_save(s, views[aspect], inputs[aspect], state, reference, config, time.monotonic())
            expected[aspect] = reference.read_bytes()
            # A live final state is deliberately different from the best state.
            with torch.no_grad():
                views[aspect]['img'].trunk[0].bias.add_(1)
            result[aspect] = ms_train.save_output(s, views[aspect], inputs[aspect], state,
                                                  path, config, time.monotonic())
        return result

    monkeypatch.setattr(ms_train, 'run', sheet_run)
    outdir = tmp_path / 'outputs'
    result = ct_d2.train(seeds=[0], device='cpu', outdir=outdir)
    assert len(calls) == 1
    assert calls[0] == (ROWS['D2'], 'D2', list(ASPECTS), 0, torch.device('cpu'),
                         outdir / 'd2/train', False, 200)
    assert set(result[0]) == set(ASPECTS)
    for aspect in ASPECTS:
        path = ms_train.output_path(outdir / 'd2/train', 'D2', aspect,
                                    ROWS['D2']['graphs'][aspect], 0)
        assert path.read_bytes() == expected[aspect]
        weight = outdir / 'd2/weights' / f'{aspect}__s0.pt'
        saved = torch.load(weight, map_location='cpu', weights_only=True)
        assert set(saved) == set(best[aspect])
        assert any(key.startswith('cap.trunk.') for key in saved)
        assert any(key.startswith('img.output.') for key in saved)
        for key in saved:
            assert torch.equal(saved[key], best[aspect][key])
        meta = json.loads(weight.with_suffix('.json').read_text())
        assert meta['weights_sha256'] == hashlib.sha256(weight.read_bytes()).hexdigest()
        assert meta['output_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert meta['config'] == result[0][aspect]['config']
        assert meta['best_epoch'] == 1
    # The wrapper is scoped to training, including when the sheet raises.
    assert ms_train.save_output is original_save


@pytest.fixture
def completed(tmp_path, cohort_data):
    s, _, _, manifests = cohort_data
    torch.manual_seed(18)
    _, views = ms_train.build_model(ROWS['D2'], list(ASPECTS), 'cpu')
    x = np.zeros((5000, 2048), dtype=np.float16)
    x[:, 0] = 1
    for aspect in ASPECTS:
        config = ms_train.make_config(s, ROWS['D2'], 'D2', aspect, list(ASPECTS), manifests,
                                      0, torch.device('cpu'), False, 200)
        output = ms_train.output_path(tmp_path / 'd2/train', 'D2', aspect,
                                      ROWS['D2']['graphs'][aspect], 0)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, img=x, cap=x)
        info = dict(config=config, best_epoch=1, epochs_run=1, best_dev_loss=0.25,
                    history=[{'epoch': 1}], output_sha256=ms_train.head.sha_file(output))
        ms_train.head.atomic_json(output.with_suffix('.json'), info)
        weight = tmp_path / 'd2/weights' / f'{aspect}__s0.pt'
        weight.parent.mkdir(parents=True, exist_ok=True)
        torch.save(views[aspect].state_dict(), weight)
        ms_train.head.atomic_json(weight.with_suffix('.json'), dict(
            weights_sha256=ms_train.head.sha_file(weight), output_sha256=info['output_sha256'],
            config=config, best_epoch=1, best_dev_loss=0.25))
    return tmp_path


def test_complete_cohort_skips_only_after_verifying_outputs_and_weights(completed, monkeypatch, cohort_data):
    monkeypatch.setattr(ct_d2, 'load_training_data', lambda: cohort_data)

    def unexpected_run(*args, **kwargs):
        pytest.fail('a complete verified cohort must skip training')

    monkeypatch.setattr(ms_train, 'run', unexpected_run)
    result = ct_d2.train([0], outdir=completed)
    assert set(result[0]) == set(ASPECTS)


@pytest.mark.parametrize('fault', ['weight_missing', 'weight_hash', 'weight_meta', 'weight_epoch',
                                   'output_missing', 'output_hash', 'output_meta', 'stale_config'])
def test_incomplete_or_corrupt_cohort_is_preserved_and_retrained(
        completed, monkeypatch, cohort_data, fault):
    weight = completed / 'd2/weights/emotion__s0.pt'
    output = completed / 'd2/train/D2/emotion__Gres__s0.npz'
    if fault == 'weight_missing':
        weight.rename(weight.with_suffix('.missing'))
    elif fault == 'weight_hash':
        weight.write_bytes(b'broken weight file')
    elif fault == 'weight_meta':
        weight.with_suffix('.json').write_text('broken JSON')
    elif fault in ('weight_epoch', 'stale_config'):
        path = weight.with_suffix('.json')
        info = json.loads(path.read_text())
        if fault == 'weight_epoch':
            info['best_epoch'] = 2
        else:
            info['config']['seed'] = 1
        path.write_text(json.dumps(info))
    elif fault == 'output_missing':
        output.rename(output.with_suffix('.missing'))
    elif fault == 'output_hash':
        output.write_bytes(b'broken output file')
    else:
        output.with_suffix('.json').write_text('broken JSON')
    before = {p.relative_to(completed): p.read_bytes()
              for p in completed.rglob('*') if p.suffix in ('.json', '.pt', '.npz')}
    monkeypatch.setattr(ct_d2, 'load_training_data', lambda: cohort_data)
    calls = []

    def replay(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir):
        calls.append(seed)
        # Corrupt artifacts must not reach the sheet's strict resume checks.
        assert not list((outdir / 'D2').glob('*__s0.npz'))
        return {'retrained': seed}

    monkeypatch.setattr(ms_train, 'run', replay)
    assert ct_d2.train([0], outdir=completed)[0] == {'retrained': 0}
    assert calls == [0]
    archived = {p.relative_to(root): p.read_bytes()
                for root in (completed / 'd2/replaced').iterdir()
                for p in root.rglob('*') if p.is_file()}
    assert archived == {p.relative_to('d2'): value for p, value in before.items()}


def test_infer_matches_sheet_outputs_on_all_5000_evaluation_names(completed, sample, cohort_data):
    import ct_held

    names = sample['subset_painting_ids'].tolist()
    x = ct_held.qwen_default_images(names)
    weight = completed / 'd2/weights/emotion__s0.pt'
    _, views = ms_train.build_model(ROWS['D2'], list(ASPECTS), 'cpu')
    view = views['emotion']
    state = torch.load(weight, map_location='cpu', weights_only=True)
    config = json.loads(weight.with_suffix('.json').read_text())['config']
    path = completed / 'reference/emotion.npz'
    ms_train.save_output(cohort_data[0], view, dict(img=x, cap=x),
                          dict(best_state=state, best_epoch=1, history=[{'epoch': 1}], best=0.25),
                          path, config, time.monotonic())
    with np.load(path, allow_pickle=False) as z:
        for side in ('img', 'cap'):
            actual = ct_d2.infer('emotion', 0, names, side=side, outdir=completed, device='cpu')
            assert actual.dtype == np.float16 and actual.shape == (5000, 2048)
            np.testing.assert_array_equal(actual.view(np.uint16), z[side].view(np.uint16))
            assert np.max(np.abs(np.linalg.norm(actual.astype(np.float32), axis=1) - 1)) < 1e-3


def test_infer_verifies_weights_before_loading_image_inputs(completed, monkeypatch):
    import ct_held

    (completed / 'd2/weights/emotion__s0.pt').write_bytes(b'corrupted')
    monkeypatch.setattr(ct_held, 'qwen_default_images', lambda names: pytest.fail('input read too early'))
    with pytest.raises(AssertionError, match='weights checksum mismatch'):
        ct_d2.infer('emotion', 0, ['a-painting'], outdir=completed)


@pytest.mark.parametrize('seed_args,expected', [([], [0, 1, 2, 3, 4]),
                                             (['--seeds', '0', '4'], [0, 4])])
def test_train_cli_dispatches_the_requested_seed_cohorts(tmp_path, monkeypatch, cohort_data,
                                                        seed_args, expected):
    monkeypatch.setattr(ct_d2, 'load_training_data', lambda: cohort_data)
    seen = []

    def sheet_run(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir):
        assert row == ROWS['D2'] and aspects == list(ASPECTS)
        assert device == torch.device('cpu') and outdir == tmp_path / 'd2/train'
        seen.append(seed)
        return {}

    monkeypatch.setattr(ms_train, 'run', sheet_run)
    ct_d2.main(['--train', '--device', 'cpu', '--outdir', str(tmp_path), *seed_args])
    assert seen == expected


def test_held_cli_writes_three_aspects_in_frozen_order_with_sidecars(completed, monkeypatch):
    import ct_held

    names = [f'painting-{i:05d}' for i in range(12281)]
    names_hash = ct_held.names_sha256(names)
    ct_held_list = completed / 'held/held_paintings.json'
    ms_train.head.atomic_json(ct_held_list, dict(names=names, sha256=names_hash))
    reads = []

    def images(requested):
        assert requested == names
        reads.append(len(requested))
        x = np.zeros((len(requested), 2048), dtype=np.float32)
        x[:, 0] = 1
        return x

    monkeypatch.setattr(ct_held, 'qwen_default_images', images)
    argv = ['--held', '--seeds', '0', '--device', 'cpu', '--outdir', str(completed)]
    ct_d2.main(argv)
    assert reads == [12281] * 3
    before = {}
    for aspect in ASPECTS:
        path = completed / 'held/d2' / f'{aspect}__s0.npz'
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['img']
            x = z['img']
            assert x.dtype == np.float16 and x.shape == (12281, 2048)
            assert np.isfinite(x).all()
        info = json.loads(path.with_suffix('.json').read_text())
        weight = completed / 'd2/weights' / f'{aspect}__s0.pt'
        assert info['held_sha256'] == names_hash and info['rows'] == 12281
        assert info['weights_sha256'] == ms_train.head.sha_file(weight)
        assert info['output_sha256'] == ms_train.head.sha_file(path)
        before[path] = (path.stat().st_mtime_ns, path.read_bytes())
    # A repeated invocation verifies the files without rerunning seeds (V4).
    ct_d2.main(argv)
    assert reads == [12281] * 3
    for path, expected in before.items():
        assert (path.stat().st_mtime_ns, path.read_bytes()) == expected
    path = completed / 'held/d2/emotion__s0.npz'
    path.write_bytes(b'corrupted held output')
    with pytest.raises(AssertionError, match='held output checksum mismatch'):
        ct_d2.main(argv)
