"""Stage 2 row roles and extraction-ID/name joins."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import ct2_data as data


def sha(x):
    digest = hashlib.sha256()
    for start in range(0, len(x), 4096):
        digest.update(np.ascontiguousarray(x[start:start + 4096]).tobytes())
    return digest.hexdigest()


def test_sample_split_preserves_every_frozen_key_and_role():
    import pav_sample
    import itp_data

    with patch.object(pav_sample, 'SAMPLE', data.SAMPLE):
        frozen = pav_sample.load_frozen()
    _, fit, dev = itp_data.split(frozen)
    s = data.split2('sample')
    for key, value in frozen.items():
        assert np.array_equal(s[key], value), key
    assert np.array_equal(s['fit'], fit)
    assert np.array_equal(s['dev'], dev)
    assert np.array_equal(s['fit_pairs'], np.flatnonzero(np.isin(frozen['pid'], fit)))
    assert len(s['fit_pairs']) == 67327
    assert s['mode'] == 'sample'
    for role, rows in (('fit', fit), ('dev', dev), ('eval', s['subset_pid'])):
        names = s['all_painting_ids'][rows]
        assert s['names_sha256'][role] == hashlib.sha256('\n'.join(names).encode()).hexdigest()


def test_full_split_matches_q9_and_excludes_held():
    import ct_held

    s = data.split2('full')
    assert s['mode'] == 'full'
    assert (len(s['fit']), len(s['dev']), len(s['fit_pairs'])) == (42969, 6152, 216107)
    assert len(s['sample_ids']) - len(s['fit_pairs']) == 30872
    assert s['fit'].dtype == s['dev'].dtype == s['fit_pairs'].dtype == np.int64
    assert not len(s['subset_pid']) and not len(s['subset_image_indices'])
    assert not len(s['subset_painting_ids'])
    assert np.array_equal(np.concatenate([s['fit'], s['dev']]), np.arange(49121))
    assert s['names_sha256']['fit'] == '921b6b7c18ff2cf3c41bf06acf9b1f787d8b9f927ed728710944403e3e35f118'
    assert s['names_sha256']['dev'] == '1430d4b56e4a76dbe3e1c4d699e1db5c92967c5f2048b74b6cbcdf1589809e2e'
    assert not set(s['all_painting_ids']) & set(ct_held.held_names())
    assert not np.intersect1d(s['fit'], s['dev']).size
    assert np.array_equal(s['all_painting_ids'][s['pid']], s['painting_ids'])
    assert np.array_equal(s['painting_ids'][s['first']], s['all_painting_ids'])
    for role in ('fit', 'dev'):
        names = s['all_painting_ids'][s[role]].tolist()
        assert names == sorted(names)
    assert np.all(s['pid'][1:] >= s['pid'][:-1])
    same_painting = s['pid'][1:] == s['pid'][:-1]
    assert np.all(np.diff(s['sample_ids'])[same_painting] > 0)


def test_sample_default_images_match_stage1_bytes():
    import itp_data

    s = data.split2('sample')
    expected, _ = itp_data.default_images(s)
    actual = data.default_images(s)
    assert actual.dtype == np.float32
    assert sha(actual) == sha(expected)


def test_sample_default_captions_match_d2_graphs_and_percept_bytes():
    import itp_data

    s = data.split2('sample')
    expected = np.asarray(itp_data.prompted(s, 'default')[1], dtype=np.float32)
    actual = data.default_captions(s, per_caption=True)
    assert actual.dtype == np.float32
    assert sha(actual) == sha(expected)
    del actual
    expected_mean = itp_data.targets(s, 'default', txt=expected)
    assert sha(data.default_captions(s)) == sha(expected_mean)


@pytest.mark.parametrize('key', ['emotion_focus', 'style_word', 'genre_word'])
def test_sample_instructed_images_match_stage1_bytes(key):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '20261018_aspect_graph_sheet'))
    from ags_trainimg import aspect_images

    s = data.split2('sample')
    assert sha(data.instructed_images(s, key)) == sha(aspect_images(s, key))


@pytest.mark.parametrize('key', ['emotion_focus', 'style_word', 'genre_word'])
def test_sample_instructed_captions_match_graphs_and_percept_bytes(key):
    import itp_data

    s = data.split2('sample')
    expected = np.asarray(itp_data.prompted(s, key)[1], dtype=np.float32)
    actual = data.instructed_captions(s, key, per_caption=True)
    assert actual.dtype == np.float32
    assert sha(actual) == sha(expected)
    del actual, expected
    assert sha(data.instructed_captions(s, key)) == sha(itp_data.targets(s, key))


@pytest.mark.parametrize('key', ['emotion_focus', 'style_word', 'genre_word', 'roberta'])
def test_missing_ids_matches_q1_original_store_counts_and_hashes(key):
    stores = [data.ROBERTA] if key == 'roberta' else [data.ASPECT / key, data.ASPECT / f'{key}_trainimg']
    missing = data.missing_ids(stores)
    for role, counts, hashes in (
        ('train', (23080, 116108), (
            'd6e340cfb2ef4c28ebbfb62317d56d9f676057eaa03eb2728337ff3e97609b8f',
            '176b0928ddf1c9238cc97ff71c29cd9bf5fb5d9a4b8eed58fc904eaceaa8a8f6')),
        ('val', (6152, 30872), (
            '1430d4b56e4a76dbe3e1c4d699e1db5c92967c5f2048b74b6cbcdf1589809e2e',
            'a17d93bad7e2d1c7750e1c293cfe3938c425aa684b49cebb6940aef33c9f7462')),
    ):
        names, ids = missing[role]['painting_names'], missing[role]['sample_ids']
        assert (len(names), len(ids)) == counts
        assert names.tolist() == sorted(names.tolist())
        assert ids.dtype == np.int64 and np.all(np.diff(ids) > 0)
        assert hashlib.sha256('\n'.join(names).encode()).hexdigest() == hashes[0]
        assert hashlib.sha256('\n'.join(map(str, ids)).encode()).hexdigest() == hashes[1]
    assert len(missing['painting_names']) == 29232
    assert len(missing['sample_ids']) == 146980


def test_missing_ids_keeps_image_gaps_when_only_new_captions_exist(tmp_path, monkeypatch):
    s = data.split2('full')
    old_name, new_name = s['all_painting_ids'][:2]
    old_id, new_id = s['sample_ids'][s['first'][:2]]
    monkeypatch.setattr(data, 'ASPECT', tmp_path)
    base, extra = tmp_path / 'emotion_focus', tmp_path / 'emotion_focus_s2txt'
    base.mkdir()
    extra.mkdir()
    np.savez(base / 'ids.npz', image_painting_ids=[old_name], sample_ids=[old_id],
             caption_painting_ids=[old_name])
    np.savez(extra / 'ids.npz', sample_ids=[new_id], caption_painting_ids=[new_name])
    missing = data.missing_ids('emotion_focus')
    assert new_name in missing['painting_names']
    assert new_id not in missing['sample_ids']
    assert old_name not in missing['painting_names']
    assert old_id not in missing['sample_ids']
    by_path = data.missing_ids(base)
    assert np.array_equal(by_path['painting_names'], missing['painting_names'])


def vectors(rows):
    x = np.zeros((len(rows), 2048), dtype=np.float16)
    x[:, :3] = rows
    return x


def write_store(path, key, *, images=None, captions=None, new=False):
    path.mkdir(parents=True)
    ids = {}
    if images is not None:
        names, values = images
        ids['painting_names' if new else 'image_painting_ids'] = np.asarray(names)
        np.save(path / 'img.npy', vectors(values))
    if captions is not None:
        sample_ids, names, values = captions
        ids.update(sample_ids=np.asarray(sample_ids, dtype=np.int64), caption_painting_ids=names)
        np.save(path / 'txt.npy', vectors(values))
    np.savez(path / 'ids.npz', **ids)
    (path / 'done.json').write_text(json.dumps(dict(instruction=data.itp_data.PROMPTS[key])))


@pytest.fixture
def full_stores(tmp_path, monkeypatch):
    """Small full-mode rows; every store has a different order and coverage."""
    qwen, aspect = tmp_path / 'qwen', tmp_path / 'aspect'
    monkeypatch.setattr(data, 'QWEN', qwen)
    monkeypatch.setattr(data, 'ASPECT', aspect)
    qwen.mkdir()
    index = dict(images=[dict(painting=name) for name in ['c', 'a', 'b']],
                 texts=[dict(sample_id=sid, painting=name) for sid, name in
                        [(30, 'a'), (5, 'c'), (90, 'b'), (20, 'b')]])
    (qwen / 'index.json').write_text(json.dumps(index))
    np.save(qwen / 'img.npy', vectors([[0, 4, 0], [0, 0, 2], [3, 0, 0]]))
    np.save(qwen / 'txt.npy', vectors([[2, 0, 0], [0, 0, 2], [0, 4, 0], [6, 8, 0]]))
    old_images = (['c', 'b'], [[0, 3, 0], [2, 0, 0]])
    old_captions = ([90, 20], ['b', 'b'], [[0, 2, 0], [3, 4, 0]])
    write_store(aspect / 'default', 'default', images=old_images, captions=old_captions)
    for key in ('emotion_focus', 'style_word', 'genre_word'):
        write_store(aspect / key, key, images=old_images, captions=old_captions)
        write_store(aspect / f'{key}_trainimg', key, new=True, images=(['b'], [[2, 0.01, 0]]))
        write_store(aspect / f'{key}_s2img', key, new=True,
                    images=(['b', 'a', 'c'], [[2, 0.02, 0], [0, 0, 2], [0.01, 2, 0]]))
    write_store(aspect / 'emotion_focus_s2txt', 'emotion_focus',
                captions=([30, 5, 20], ['a', 'c', 'b'], [[2, 0, 0], [0, 0, 2], [6, 8, 0]]))
    return dict(mode='full', all_painting_ids=np.asarray(['b', 'c', 'a']),
                sample_ids=np.asarray([20, 90, 5, 30], dtype=np.int64),
                painting_ids=np.asarray(['b', 'b', 'c', 'a']), pid=np.asarray([0, 0, 1, 2]),
                first=np.asarray([0, 2, 3]), fit=np.asarray([0, 1]), dev=np.asarray([2]),
                subset_pid=np.empty(0, dtype=np.int64))


@pytest.mark.parametrize('kind', ['default', 'emotion_focus', 'style_word', 'genre_word'])
def test_full_images_join_names_and_keep_earlier_sources(full_stores, kind):
    x = data.default_images(full_stores) if kind == 'default' else data.instructed_images(full_stores, kind)
    assert x.dtype == np.float32
    assert np.array_equal(x, vectors([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32))


@pytest.mark.parametrize('kind', ['default', 'emotion_focus'])
def test_full_captions_join_extraction_ids_keep_raw_scale_and_mean_after_join(full_stores, kind):
    provider = data.default_captions if kind == 'default' else lambda s, **kw: data.instructed_captions(s, kind, **kw)
    x = provider(full_stores, per_caption=True)
    assert x.dtype == np.float32
    assert np.array_equal(x, vectors([[3, 4, 0], [0, 2, 0], [0, 0, 2], [2, 0, 0]]).astype(np.float32))
    expected = np.zeros((3, 2048), dtype=np.float32)
    expected[:, :3] = [[1 / np.sqrt(5), 2 / np.sqrt(5), 0], [0, 0, 1], [1, 0, 0]]
    np.testing.assert_allclose(provider(full_stores), expected, rtol=1e-6)


def test_default_image_provider_rejects_zero_vectors(full_stores):
    raw = np.load(data.QWEN / 'img.npy')
    raw[1] = 0
    np.save(data.QWEN / 'img.npy', raw)
    with pytest.raises(AssertionError, match='zero'):
        data.default_images(full_stores)


def test_default_image_reference_must_have_default_instruction(full_stores):
    (data.ASPECT / 'default/done.json').write_text(json.dumps(dict(instruction='wrong')))
    with pytest.raises(AssertionError, match='instruction'):
        data.default_images(full_stores)


def test_default_image_cosine_mean_is_over_whole_overlap(tmp_path, monkeypatch):
    qwen, aspect = tmp_path / 'qwen', tmp_path / 'aspect'
    monkeypatch.setattr(data, 'QWEN', qwen)
    monkeypatch.setattr(data, 'ASPECT', aspect)
    qwen.mkdir()
    names = [f'p{i}' for i in range(4097)]
    (qwen / 'index.json').write_text(json.dumps(dict(images=[dict(painting=n) for n in names])))
    raw = vectors([[1, 0, 0]] * len(names))
    np.save(qwen / 'img.npy', raw)
    reference = [[1, 0, 0]] * (len(names) - 1) + [[0.995, 0.1, 0]]
    write_store(aspect / 'default', 'default', images=(names, reference))
    assert np.array_equal(data.default_images(dict(all_painting_ids=np.asarray(names))), raw.astype(np.float32))


PROVIDERS = ('default_image', 'default_caption', 'instructed_image', 'instructed_caption')


def provide(s, kind):
    if kind == 'default_image':
        return data.default_images(s)
    if kind == 'instructed_image':
        return data.instructed_images(s, 'emotion_focus')
    if kind == 'default_caption':
        return data.default_captions(s, per_caption=True)
    return data.instructed_captions(s, 'emotion_focus', per_caption=True)


@pytest.mark.parametrize('kind', PROVIDERS)
def test_full_providers_require_coverage_of_every_row(full_stores, kind):
    if kind.endswith('image'):
        full_stores['all_painting_ids'][2] = 'x'
    else:
        full_stores['sample_ids'][2] = 999
    with pytest.raises(AssertionError, match='missing'):
        provide(full_stores, kind)


@pytest.mark.parametrize('kind', PROVIDERS)
def test_full_providers_gate_overlap_even_when_the_earlier_source_wins(full_stores, kind):
    path, row = {
        'default_image': (data.ASPECT / 'default/img.npy', 1),
        'default_caption': (data.QWEN / 'txt.npy', 3),
        'instructed_image': (data.ASPECT / 'emotion_focus_s2img/img.npy', 0),
        'instructed_caption': (data.ASPECT / 'emotion_focus_s2txt/txt.npy', 2),
    }[kind]
    raw = np.load(path)
    raw[row] = vectors([[0, 0, 1]])[0]
    np.save(path, raw)
    with pytest.raises(AssertionError, match='overlap cosine'):
        provide(full_stores, kind)


@pytest.mark.parametrize('kind', ['default_caption', 'instructed_caption'])
def test_caption_ids_must_join_to_the_same_painting(full_stores, kind):
    full_stores['painting_ids'][0] = 'x'
    with pytest.raises(AssertionError, match='caption painting'):
        provide(full_stores, kind)


@pytest.mark.parametrize('kind', PROVIDERS)
def test_full_providers_reject_nonfinite_features(full_stores, kind):
    path, row = {
        'default_image': (data.QWEN / 'img.npy', 1),
        'default_caption': (data.QWEN / 'txt.npy', 0),
        'instructed_image': (data.ASPECT / 'emotion_focus_s2img/img.npy', 1),
        'instructed_caption': (data.ASPECT / 'emotion_focus_s2txt/txt.npy', 0),
    }[kind]
    raw = np.load(path)
    raw[row] = np.nan
    np.save(path, raw)
    with pytest.raises(AssertionError):
        provide(full_stores, kind)


@pytest.mark.parametrize('key', ['style_word', 'genre_word'])
def test_full_style_genre_captions_cannot_fall_back_to_emotion(full_stores, key):
    with pytest.raises(AssertionError, match='missing captions'):
        data.instructed_captions(full_stores, key)
