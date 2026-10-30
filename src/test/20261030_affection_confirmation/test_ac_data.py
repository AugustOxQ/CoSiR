"""Label verdicts and keyed file boundaries; synthetic files need no extraction or GPU."""
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ac_paths
import ac_data
import rs_vh
import rt_crl
import rt_data


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def annotations(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('dataset', 'image_name', 'split', 'emotion'))
        writer.writerows(rows)


@pytest.fixture
def small_round(tmp_path, monkeypatch):
    path = tmp_path / 'annotations.csv'
    rows = [('z', 'b.jpg', 'val', 'sadness'),
            ('a', 'a.jpg', 'val', 'awe'), ('a', 'a.jpg', 'val', 'amusement'),
            ('a', 'a.jpg', 'ignored', 'sadness'), ('a', 'a.jpg', 'ignored', 'sadness'),
            ('z', 'b.jpg', 'val', 'sadness'), ('z', 'b.jpg', 'val', 'fear'),
            ('a', 'a.jpg', 'test', 'anger'),
            ('ignored', 'only.jpg', 'ignored', 'fear'),
            ('train', 'only.jpg', 'train', 'awe')]
    # Equal image names from different sources remain distinct keys (§3.3).
    rows[7] = ('other', 'a.jpg', 'test', 'anger')
    rows.extend([('z', 'b.jpg', 'val', 'awe'), ('z', 'b.jpg', 'val', 'amusement')])
    annotations(path, rows)
    monkeypatch.setattr(ac_paths, 'ANNOTATIONS', path)
    monkeypatch.setattr(ac_paths, 'N', {'val': 2, 'test': 1})
    for name in ('STORE', 'VIEWS', 'CLIP_VIEW', 'OUTPUTS'):
        monkeypatch.setattr(ac_paths, name, tmp_path / name.lower())
    monkeypatch.setattr(rt_data, 'CSTAR_ROOT', tmp_path / 'r10_cstar')
    monkeypatch.setattr(rt_data.rt_paths, 'PERCEPT_OUT', tmp_path / 'r11_percept')
    return tmp_path


def test_labels_exclude_ignored_and_use_plurality_with_lowest_code_ties(small_round):
    assert ac_data.keys('val') == ['a/a.jpg', 'z/b.jpg']
    assert ac_data.keys('test') == ['other/a.jpg']
    assert ac_data.sources('val') == ['a', 'z']
    lab = ac_data.labels('val')
    assert lab.dtype == np.int64
    np.testing.assert_array_equal(lab, [0, 7])
    tied = ac_data.ties('val')
    assert tied.dtype == np.bool_
    np.testing.assert_array_equal(tied, [True, False])
    dist = ac_data.emotion_dist('val')
    assert dist.dtype == np.float64
    np.testing.assert_allclose(dist, [[.5, 0, .5, 0, 0, 0, 0, 0, 0],
                                      [.2, 0, .2, 0, 0, 0, .2, .4, 0]], rtol=0, atol=1e-15)
    assert ac_data.label_counts('val') == {'classes': [1, 0, 0, 0, 0, 0, 0, 1, 0], 'ties': 1, 'n': 2}
    lab[:] = 8
    dist[:] = 0
    assert ac_data.labels('val').tolist() == [0, 7]
    assert ac_data.emotion_dist('val')[0, 0] == .5


def test_image_cannot_belong_to_two_nonignored_splits(small_round):
    with ac_paths.ANNOTATIONS.open('a') as handle:
        handle.write('a,a.jpg,test,fear\n')
    with pytest.raises(AssertionError, match='split'):
        ac_data.keys('val')


def test_annotation_keys_use_utf8_with_a_non_utf8_locale(small_round, monkeypatch):
    path = ac_paths.ANNOTATIONS
    text = path.read_text(encoding='utf-8').replace('a.jpg', 'été.jpg')
    path.write_text(text, encoding='utf-8')
    original_open = Path.open

    def locale_open(opened, *args, **kwargs):
        if opened == path and 'encoding' not in kwargs:
            kwargs['encoding'] = 'ascii'  # Simulate a host whose default encoding is not UTF-8.
        return original_open(opened, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', locale_open)
    assert ac_data.keys('val') == ['a/été.jpg', 'z/b.jpg']


@pytest.mark.real_data
def test_real_csv_label_gate():
    # Spec §3.4, lines 133–134: independently frozen class/tie counts.
    assert ac_data.label_counts('val') == {
        'classes': [798, 54, 622, 1387, 189, 620, 203, 270, 108], 'ties': 907, 'n': 4251}
    assert ac_data.label_counts('test') == {
        'classes': [1651, 81, 1460, 2626, 351, 1185, 448, 501, 198], 'ties': 1917, 'n': 8501}
    for split in ac_paths.SPLITS:
        assert len(ac_data.keys(split)) == ac_paths.N[split]
        assert ac_data.emotion_dist(split).shape == (ac_paths.N[split], 9)


def basis(rows, width, dtype=np.float16):
    x = np.zeros((rows, width), dtype=dtype)
    x[np.arange(rows), np.arange(rows)] = 1
    return x


def default_store():
    # Deliberately different from annotation and evaluation order; includes train.
    stored = ['z/b.jpg', 'train/only.jpg', 'other/a.jpg', 'a/a.jpg']
    path = ac_paths.STORE
    write_json(path / 'index.json', {
        'dataset': 'affection_local', 'backbone': 'qwen3vl_emb_2b', 'smoke': False,
        'dtype': 'float16', 'dim': 2048, 'n_images': 4,
        'images': [dict(zip(('source', 'image_name'), key.split('/'))) for key in stored]})
    x = basis(4, 2048)
    np.save(path / 'img.npy', x)
    return x


def test_plain_store_gathers_by_source_and_name(small_round):
    x = default_store()
    actual = ac_data.embed('plain', 'val')
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, x[[3, 0]].astype(np.float32))
    np.testing.assert_array_equal(ac_data.embed('plain', 'test'), x[[2]].astype(np.float32))


def test_plain_caches_unchanged_index_keys_and_reloads_features(small_round, monkeypatch):
    x = default_store()
    index = ac_paths.STORE / 'index.json'
    original_read = Path.read_text
    reads = []

    def read_text(path, *args, **kwargs):
        if path == index:
            reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    ac_data.embed('plain', 'val')
    ac_data.embed('plain', 'test')
    changed = x[[1, 0, 3, 2]]
    np.save(ac_paths.STORE / 'img.npy', changed)
    np.testing.assert_array_equal(ac_data.embed('plain', 'val'), changed[[3, 0]].astype(np.float32))
    assert len(reads) == 1, 'unchanged large store index should be parsed once'


@pytest.mark.parametrize('changed', ['mtime', 'size'])
def test_plain_refreshes_cached_keys_when_index_stat_changes(small_round, changed):
    x = default_store()
    index = ac_paths.STORE / 'index.json'
    ac_data.embed('plain', 'val')
    stat = index.stat()
    side = json.loads(index.read_text())
    side['images'][0], side['images'][3] = side['images'][3], side['images'][0]
    if changed == 'size':
        side['note'] = 'size changes while mtime is held constant'
    write_json(index, side)
    os.utime(index, ns=(stat.st_atime_ns, stat.st_mtime_ns + (1_000_000 if changed == 'mtime' else 0)))
    if changed == 'mtime':
        assert index.stat().st_size == stat.st_size
    else:
        assert index.stat().st_mtime_ns == stat.st_mtime_ns
    np.testing.assert_array_equal(ac_data.embed('plain', 'val'), x[[0, 3]].astype(np.float32))


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'dtype', 'shape', 'nonfinite', 'nonunit', 'smoke'])
def test_plain_rejects_invalid_store(small_round, damage):
    x = default_store()
    path = ac_paths.STORE
    side = json.loads((path / 'index.json').read_text())
    if damage == 'missing':
        side['images'][3]['image_name'] = 'missing.jpg'
    elif damage == 'duplicate':
        side['images'][3] = side['images'][0]
    elif damage == 'dtype':
        x = x.astype(np.float32)
    elif damage == 'shape':
        x = x[:, :20]
    elif damage == 'nonfinite':
        x[3, 0] = np.nan
    elif damage == 'nonunit':
        x[3] = 0
    else:
        side['smoke'] = True
    write_json(path / 'index.json', side)
    np.save(path / 'img.npy', x)
    with pytest.raises(AssertionError):
        ac_data.embed('plain', 'val')


@pytest.mark.parametrize('arm,seed,head', [('unknown', None, 'emotion'), ('plain', 0, 'emotion'),
                                          ('cstar', None, 'emotion'), ('percept', 5, 'emotion'),
                                          ('cstar', 0.0, 'emotion'), ('percept', True, 'emotion'),
                                          ('cstar', 0, 'unknown'), ('plain', None, 'style')])
def test_embed_rejects_invalid_request(small_round, arm, seed, head):
    with pytest.raises(AssertionError):
        ac_data.embed(arm, 'val', seed=seed, head=head)


def feature_view(arm):
    is_clip = arm == 'clip'
    path = ac_paths.CLIP_VIEW if is_clip else ac_paths.VIEWS / ac_paths.WORDINGS[arm.removeprefix('teacher_')]
    stored = ['a/a.jpg', 'z/b.jpg', 'other/a.jpg']
    x = basis(3, 768 if is_clip else 2048)
    instruction = None if is_clip else (ac_paths.PHOTO_INSTRUCTION if arm == 'teacher_photo' else
                                        rt_data.ct2_data.itp_data.PROMPTS['emotion_focus'])
    side = {'instruction': instruction,
            'model_id': rt_crl.CLIP_ID if is_clip else 'Qwen/Qwen3-VL-Embedding-2B',
            'model_revision': rt_crl.CLIP_REVISION if is_clip else rt_data.ct2_data.QWEN_REVISION,
            'dtype': 'float16', 'shape': list(x.shape), 'smoke': False,
            'gate': {'ok': True, 'n_nonfinite': 0, 'n_bad_norm': 0, 'max_norm_dev': 0.0},
            'keys_sha256': rs_vh.names_sha(stored), 'img_sha256': rs_vh.array_sha(x)}
    write_json(path / 'keys.json', stored)
    write_json(path / 'meta.json', side)
    np.save(path / 'img.npy', x)
    return path, side, x


@pytest.mark.parametrize('arm', ['teacher_artwork', 'teacher_photo', 'clip'])
def test_views_return_each_split_in_key_order(small_round, arm):
    _, _, x = feature_view(arm)
    np.testing.assert_array_equal(ac_data.embed(arm, 'val'), x[:2].astype(np.float32))
    np.testing.assert_array_equal(ac_data.embed(arm, 'test'), x[2:].astype(np.float32))


@pytest.mark.parametrize('damage', ['keys_sha256', 'img_sha256', 'model_id', 'model_revision',
                                  'instruction', 'dtype', 'shape', 'smoke', 'gate',
                                  'array_dtype', 'array_shape', 'missing_metadata',
                                  'reordered_keys', 'duplicate_keys', 'nonunit', 'nonfinite'])
def test_view_rejects_corrupt_or_wrong_identity_files(small_round, damage):
    path, side, x = feature_view('teacher_photo')
    if damage in ('keys_sha256', 'img_sha256', 'model_id', 'model_revision', 'instruction', 'dtype'):
        side[damage] = 'wrong'
    elif damage == 'shape':
        side['shape'] = [3, 768]
    elif damage == 'smoke':
        side['smoke'] = True
    elif damage == 'gate':
        side['gate'] = {'ok': False}
    elif damage == 'missing_metadata':
        side = {key: value for key, value in side.items() if key != 'keys_sha256'}
    elif damage in ('array_dtype', 'array_shape'):
        x = x.astype(np.float32) if damage == 'array_dtype' else x[:2]
        side['img_sha256'] = rs_vh.array_sha(x)
        np.save(path / 'img.npy', x)
    elif damage in ('reordered_keys', 'duplicate_keys'):
        stored = ['z/b.jpg', 'a/a.jpg', 'other/a.jpg'] if damage == 'reordered_keys' else ['a/a.jpg'] * 3
        write_json(path / 'keys.json', stored)
        side['keys_sha256'] = rs_vh.names_sha(stored)
    else:
        x[0] = 0 if damage == 'nonunit' else np.nan
        side['img_sha256'] = rs_vh.array_sha(x)
        np.save(path / 'img.npy', x)
    write_json(path / 'meta.json', side)
    with pytest.raises(AssertionError):
        ac_data.embed('teacher_photo', 'val')


@pytest.mark.parametrize('field', ['n_nonfinite', 'n_bad_norm', 'max_norm_dev'])
@pytest.mark.parametrize('damage', ['missing', 'nonzero'])
def test_view_requires_complete_clean_gate_even_when_ok_is_true(small_round, field, damage):
    path, side, _ = feature_view('teacher_photo')
    gate = side['gate']
    if damage == 'missing':
        side['gate'] = {key: value for key, value in gate.items() if key != field}
    else:
        gate[field] = .01 if field == 'max_norm_dev' else 1
    write_json(path / 'meta.json', side)
    with pytest.raises(AssertionError):
        ac_data.embed('teacher_photo', 'val')


def frozen_export(arm, split='val', seed=0, head='emotion'):
    percept = arm == 'percept'
    path = ac_paths.OUTPUTS / 'exports' / arm / split / f'{"P3" if percept else head}__s{seed}.npz'
    x = basis(ac_paths.N[split], 40 if percept else 2048, np.float32 if percept else np.float16)
    if percept:
        x[:, 5] = .5  # Stored probabilities are normalized by embed, not assumed unit.
    weights = rs_vh.names_sha([arm, head, str(seed)])
    reference = (rt_data.rt_paths.PERCEPT_OUT / f'held/percept/core__k40/P3__s{seed}.json' if percept else
                 rt_data.CSTAR_ROOT / f'held/k11=40/{head}__s{seed}.json')
    write_json(reference, {'weights_sha256': weights})
    side = {'keys_sha256': rs_vh.names_sha(ac_data.keys(split)), 'array_sha256': rs_vh.array_sha(x),
            'weights_sha256': weights, 'split': split, 'seed': seed, 'smoke': False}
    side.update({'device': 'cuda'} if percept else {'aspect': head})
    write_json(path.with_suffix('.json'), side)
    np.savez(path, **{'prob' if percept else 'img': x})
    return path, side, x


@pytest.mark.parametrize('split', ['val', 'test'])
@pytest.mark.parametrize('arm,seed,head', [('cstar', 0, 'emotion'), ('cstar', 1, 'style'),
                                         ('cstar', 4, 'genre'), ('percept', 0, 'emotion'),
                                         ('percept', 4, 'emotion')])
def test_seeded_exports_follow_keys_hash_and_normalize(small_round, split, arm, seed, head):
    _, _, x = frozen_export(arm, split, seed, head)
    expected = x.astype(np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    actual = ac_data.embed(arm, split, seed, head=head)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('split', ['val', 'test'])
def test_percept_prob_preserves_raw_export_bytes(small_round, split):
    _, _, expected = frozen_export('percept', split, seed=4)
    actual = ac_data.percept_prob(split, 4)
    assert actual.dtype == np.float32 and actual.shape == (ac_paths.N[split], 40)
    assert actual.tobytes() == expected.tobytes()
    assert actual[0, 5] == .5 and actual[0, 0] == 1
    assert not np.array_equal(actual, ac_data.embed('percept', split, 4))


@pytest.mark.parametrize('seed', [None, 5, True, 0.0])
def test_percept_prob_requires_a_valid_integer_seed(small_round, seed):
    with pytest.raises(AssertionError):
        ac_data.percept_prob('val', seed)


def test_percept_own_reading_uses_raw_near_tie_probabilities(small_round):
    path, side, x = frozen_export('percept')
    x[:] = np.float32(.018908909)
    x[:, 0] = np.float32(.95)
    x[:, 1] = np.nextafter(np.float32(.95), np.float32(1))
    side['array_sha256'] = rs_vh.array_sha(x)
    write_json(path.with_suffix('.json'), side)
    np.savez(path, prob=x)
    assert ac_data.percept_prob('val', 0).argmax(axis=1).tolist() == [1, 1]
    assert ac_data.embed('percept', 'val', 0).argmax(axis=1).tolist() == [0, 0]


@pytest.mark.parametrize('arm', ['cstar', 'percept'])
@pytest.mark.parametrize('damage', ['keys_sha256', 'array_sha256', 'weights_sha256', 'split', 'seed',
                                  'other_weights', 'smoke', 'identity', 'dtype', 'shape', 'npz_keys',
                                  'nonfinite', 'bad_values'])
def test_export_rejects_invalid_boundaries(small_round, arm, damage):
    path, side, x = frozen_export(arm)
    key = 'prob' if arm == 'percept' else 'img'
    if damage in ('keys_sha256', 'array_sha256', 'weights_sha256', 'split'):
        side[damage] = 'wrong'
    elif damage == 'other_weights':
        side['weights_sha256'] = rs_vh.names_sha([arm, 'emotion', '1'])  # Valid hash, wrong frozen seed.
    elif damage == 'seed':
        side['seed'] = 1
    elif damage == 'smoke':
        side['smoke'] = True
    elif damage == 'identity':
        side['device' if arm == 'percept' else 'aspect'] = 'wrong'
    else:
        if damage == 'dtype':
            x = x.astype(np.float64)
        elif damage == 'shape':
            x = x[:1]
        elif damage == 'npz_keys':
            key = 'wrong'
        elif damage == 'nonfinite':
            x[0, 0] = np.inf
        elif damage == 'bad_values':
            x[0, 0] = 2
        side['array_sha256'] = rs_vh.array_sha(x)
        np.savez(path, **{key: x})
    write_json(path.with_suffix('.json'), side)
    with pytest.raises(AssertionError):
        ac_data.embed(arm, 'val', seed=0)
    if arm == 'percept':
        with pytest.raises(AssertionError):
            ac_data.percept_prob('val', 0)


def test_crl_fixture_preserves_an_importable_module(small_round, monkeypatch, request):
    module = SimpleNamespace(vectors=lambda img, text: None)
    monkeypatch.setitem(sys.modules, 'ac_crl', module)
    request.getfixturevalue('crl_inputs')
    assert importlib.import_module('ac_crl') is module


@pytest.fixture
def crl_inputs(small_round, monkeypatch):
    """Prefer real ac_crl; provide a loader-only stand-in while ticket 05 is absent.

    The independent reference runs r11's actual raw('crl') and embed('crl')
    with only the frozen image/text input boundaries replaced by small arrays.
    """
    lists_path = small_round / 'lists.json'
    write_json(lists_path, {'aspects': {a: {'values': ['joy', 'fear', 'calm'], 'count': 3}
                                       for a in ac_paths.ASPECTS}})
    monkeypatch.setattr(ac_paths, 'R11_CRL_LISTS_FILE', lists_path)
    default_store()
    feature_view('clip')
    original_embed = rt_data.embed

    def legacy_vectors(img_unit, text_unit):
        with monkeypatch.context() as patch:
            patch.setattr(rt_data, 'names', lambda split: [f'painting{i}' for i in range(len(img_unit))])
            patch.setattr(rt_crl, 'text_array', lambda aspect: text_unit)

            def embed(arm, split, aspect, seed=None):
                if arm == 'clip':
                    return img_unit
                return original_embed(arm, split, aspect, seed)

            patch.setattr(rt_data, 'embed', embed)
            return original_embed('crl', 'val', 'emotion')

    try:
        module = importlib.import_module('ac_crl')
    except ModuleNotFoundError as error:
        if error.name != 'ac_crl':
            raise  # A broken dependency inside real ac_crl must fail the suite.
        module = SimpleNamespace(vectors=legacy_vectors)
        monkeypatch.setitem(sys.modules, 'ac_crl', module)
        real = False
    else:
        real = True
    return SimpleNamespace(module=module, reference=legacy_vectors, real=real)


def crl_texts(encoder, wording, aspect='emotion', *, hashed=True):
    path = ac_paths.OUTPUTS / 'crl' / f'{encoder}_{wording}_{aspect}.npy'
    width = 768 if encoder == 'clip' else 2048
    # Nontrivial, signed cosine values; all these inputs are exactly representable.
    x = np.zeros((3, width), dtype=np.float32)
    x[0, :4] = [.5, .5, .5, .5]
    x[1, :4] = [-.5, .5, -.5, .5]
    x[2, 3] = 1
    texts = [f'a {wording} with the {aspect} of {value}' for value in ('joy', 'fear', 'calm')]
    side = {'texts': texts, 'encoder': encoder,
            'model_id': rt_crl.CLIP_ID if encoder == 'clip' else 'Qwen/Qwen3-VL-Embedding-2B',
            'instruction': None if encoder == 'clip' else "Represent the user's input.",
            'template': f'a {wording} with the {{aspect}} of {{value}}',
            'model_revision': rt_crl.CLIP_REVISION if encoder == 'clip' else rt_data.ct2_data.QWEN_REVISION}
    if hashed:
        side['array_sha256'] = rs_vh.array_sha(x)
    write_json(path.with_suffix('.json'), side)
    np.save(path, x)
    return path, side, x


@pytest.mark.parametrize('version', ac_paths.CRL_VERSIONS)
@pytest.mark.parametrize('split', ['val', 'test'])
def test_crl_arms_read_the_right_encoder_and_wording(crl_inputs, version, split):
    encoder, wording = version.split('_')
    crl_texts(encoder, wording)
    actual = ac_data.embed(f'crl_{version}', split)
    assert actual.dtype == np.float32
    # Worked examples test routing even when the real vector module is absent.
    if split == 'test':
        np.testing.assert_allclose(actual[0], [2**-.5, -(2**-.5), 0], rtol=0, atol=1e-7)
    elif encoder == 'clip':
        np.testing.assert_allclose(actual, [[2**-.5, -(2**-.5), 0], [2**-.5, 2**-.5, 0]], atol=1e-7)
    else:
        np.testing.assert_allclose(actual, [[1/6**.5, 1/6**.5, 2/6**.5],
                                            [2**-.5, -(2**-.5), 0]], atol=1e-7)


def test_real_crl_vectors_equal_r11_bit_for_bit(crl_inputs):
    if not crl_inputs.real:
        pytest.skip('ticket 05 ac_crl is absent; the loader stand-in cannot validate CRL numerical parity')
    rng = np.random.default_rng(20261030)
    img = rng.normal(size=(7, 768)).astype(np.float32)
    text = rng.normal(size=(5, 768)).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    actual = crl_inputs.module.vectors(img, text)
    expected = crl_inputs.reference(img, text)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('damage', ['texts', 'encoder', 'model_revision', 'array_sha256',
                                  'dtype', 'shape', 'nonunit', 'nonfinite'])
def test_crl_text_boundary_rejects_wrong_values_or_arrays(crl_inputs, damage):
    path, side, x = crl_texts('qwen', 'photo')
    if damage in ('encoder', 'model_revision', 'array_sha256'):
        side[damage] = 'wrong'
    elif damage == 'texts':
        side['texts'] = list(reversed(side['texts']))
    else:
        if damage == 'dtype':
            x = x.astype(np.float64)
        elif damage == 'shape':
            x = x[:, :768]
        elif damage == 'nonunit':
            x[0] = 0
        else:
            x[0, 0] = np.nan
        side['array_sha256'] = rs_vh.array_sha(x)
        np.save(path, x)
    write_json(path.with_suffix('.json'), side)
    with pytest.raises(AssertionError):
        ac_data.embed('crl_qwen_photo', 'val')


@pytest.mark.parametrize('encoder', ['clip', 'qwen'])
@pytest.mark.parametrize('field', ['model_id', 'instruction', 'template'])
def test_crl_text_metadata_rejects_wrong_encoder_or_prompt(crl_inputs, encoder, field):
    path, side, _ = crl_texts(encoder, 'photo')
    side[field] = 'wrong'
    write_json(path.with_suffix('.json'), side)
    with pytest.raises(AssertionError):
        ac_data.embed(f'crl_{encoder}_photo', 'val')


@pytest.mark.parametrize('field', ['texts', 'encoder', 'model_id', 'model_revision', 'instruction', 'template'])
def test_crl_text_requires_all_index_sidecar_fields(crl_inputs, field):
    path, side, _ = crl_texts('qwen', 'photo')
    write_json(path.with_suffix('.json'), {key: value for key, value in side.items() if key != field})
    with pytest.raises(AssertionError):
        ac_data.embed('crl_qwen_photo', 'val')


def test_crl_accepts_minimal_index_text_metadata(crl_inputs):
    crl_texts('clip', 'painting', hashed=False)
    assert ac_data.embed('crl_clip_painting', 'val').shape == (2, 3)


@pytest.mark.parametrize('split', ['val', 'held'])
@pytest.mark.parametrize('aspect', ac_paths.ASPECTS)
def test_artelingo_qwen_crl_uses_frozen_r11_plain_rows(crl_inputs, monkeypatch, split, aspect):
    _, _, text = crl_texts('qwen', 'painting', aspect)
    names = ['painting_a', 'painting_b']
    img = basis(2, 2048, np.float32)
    monkeypatch.setattr(rt_data, 'names', lambda part: names if part == split else [])

    def frozen_embed(arm, part, requested_aspect, seed=None):
        assert (arm, part, requested_aspect, seed) == ('plain', split, aspect, None)
        return img

    monkeypatch.setattr(rt_data, 'embed', frozen_embed)
    actual = ac_data.artelingo_crl_qwen(split, aspect)
    assert actual.shape == (len(names), 3) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, [[2**-.5, -(2**-.5), 0], [2**-.5, 2**-.5, 0]], atol=1e-7)
