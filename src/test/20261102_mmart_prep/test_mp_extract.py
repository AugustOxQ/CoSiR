"""CPU-only fixtures; no models, CUDA calls, GPU lock, or canonical data writes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_extract as E
import mp_stores as S


@pytest.fixture
def small(tmp_path, monkeypatch):
    monkeypatch.setenv('MP_STORE_ROOT', str(tmp_path / 'mmart'))
    monkeypatch.setenv('MP_OUTPUTS', str(tmp_path / 'outputs'))
    monkeypatch.setattr(E, 'N_NEW', 3)
    monkeypatch.setattr(E, 'N_PARITY', 2)
    monkeypatch.setattr(S, 'WIDTH', {v: 2 for v in S.VIEWS})
    groups = ['held', 'none', 'trainval', 'none', 'held', 'none', 'held']
    table = pd.DataFrame(dict(key=[f'style/k{i}.jpg' for i in range(7)],
                              painting=[f'k{i}' for i in range(7)], artelingo=groups))
    paths = []
    from PIL import Image
    for i in range(7):
        path = tmp_path / f'k{i}.jpg'
        Image.new('RGB', (2, 2), (i, 0, 0)).save(path)
        paths.append(str(path))
    table['image_path'] = paths
    index = tmp_path / 'index.parquet'
    table.to_parquet(index)
    monkeypatch.setattr(S, 'INDEX', index)
    art = tmp_path / 'art'
    monkeypatch.setattr(S, 'ART_ROOT', art)
    for view in S.VIEWS:
        folder = S.reference_folder(view)
        folder.mkdir(parents=True)
        if view in S.TEACHERS:
            names = ['k6', 'k4', 'k0']
            np.savez(folder / 'ids.npz', painting_names=np.asarray(names))
        else:
            names = ['k6', 'k2', 'k4', 'k0']
            (folder / 'index.json').write_text(json.dumps(dict(images=[dict(painting=n) for n in names])))
        vectors = {'k0': [1, 0], 'k2': [0, 1], 'k4': [-1, 0], 'k6': [0, -1]}
        np.save(folder / 'img.npy', np.asarray([vectors[n] for n in names], np.float16))
    return table, art


def store(view, which, keys, values, smoke=False):
    out = S.folder(view, which)
    pending = E.check_target(out)
    pending.mkdir(parents=True)
    x = np.asarray(values, np.float16)
    np.save(pending / 'img.npy', x)
    E.seal(pending, out, view, which, keys, x, dict(model_id='fake', model_revision='a' * 40), None, smoke, .1)
    return out


def populate():
    for view in S.VIEWS:
        store(view, 'new', E.target_keys(), [[1, 0], [0, 1], [-1, 0]])
        store(view, 'parity', E.target_keys('parity'), [[1, 0], [0, -1]])


def test_real_sized_key_sets():
    groups = ['held', 'none'] * 16782 + ['held'] * 10
    table = pd.DataFrame(dict(key=[f'k{i}' for i in range(len(groups))],
                              painting=[f'p{i}' for i in range(len(groups))], artelingo=groups))
    assert E.target_keys(index=table) == table.loc[table.artelingo == 'none', 'key'].tolist()
    held = table.loc[table.artelingo == 'held', 'key'].tolist()
    expected = [held[i] for i in np.linspace(0, len(held) - 1, 200).round().astype(int)]
    assert E.target_keys('parity', table) == expected
    with pytest.raises(AssertionError, match='expected'):
        E.target_keys(index=table.iloc[:-20])


def test_key_order_and_duplicates(small):
    assert E.target_keys() == ['style/k1.jpg', 'style/k3.jpg', 'style/k5.jpg']
    assert E.target_keys('parity') == ['style/k0.jpg', 'style/k6.jpg']
    assert E.pick(list(range(9)), 3) == [0, 4, 8]
    table, _ = small
    with pytest.raises(AssertionError, match='duplicate'):
        E.target_keys(index=pd.concat([table, table.iloc[:1]]))


def test_extract_fake_encoder_seal_refusals(small, monkeypatch):
    class Fake:
        def encode_images(self, images, batch_size):
            return np.tile([.6, .8], (len(images), 1))
    # The real r14 batching helper is imported read-only; it never calls CUDA.
    monkeypatch.setattr(E, 'load_encoder', lambda *a: (Fake(), dict(model_id='fake', model_revision='a' * 40), None))
    monkeypatch.setattr(E, 'BATCH', {**E.BATCH, 'siglip2': 2})
    meta = E.extract('siglip2', device='cpu')
    out = S.folder('siglip2')
    assert meta['n'] == 3 and meta['d'] == 2 and not meta['smoke']
    assert not out.with_name(out.name + '.pending').exists()
    assert E.check_store('siglip2', 'new')['ok']
    assert np.array_equal(np.load(out / 'img.npy')[0], np.asarray([.6, .8], np.float16))
    with pytest.raises(AssertionError, match='overwritten'):
        E.extract('siglip2', device='cpu')
    pending = S.folder('clip_l14').with_name('clip_l14.pending')
    pending.mkdir(parents=True)
    with pytest.raises(AssertionError, match='pending'):
        E.extract('clip_l14', device='cpu')
    with pytest.raises(AssertionError, match='pending'):
        E.extract_all()


def test_reader_join_mixed_sources(small):
    populate()
    keys = ['style/k2.jpg', 'style/k5.jpg', 'style/k0.jpg', 'style/k1.jpg', 'style/k6.jpg']
    result = S.load('qwen_default', keys)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[0, 1], [-1, 0], [1, 0], [1, 0], [0, -1]])
    np.testing.assert_array_equal(S.load('style_word', keys[1:]), result[1:])
    with pytest.raises(AssertionError, match='trainval'):
        S.load('genre_word', keys)
    with pytest.raises(AssertionError, match='unique'):
        S.load('siglip1', keys * 2)
    with pytest.raises(AssertionError, match='not found'):
        S.load('siglip1', ['absent'])
    folder = S.folder('qwen_default')
    (folder / 'keys.json').write_text(json.dumps(['x', 'x', 'y']))
    with pytest.raises(AssertionError, match='duplicate'):
        S.load('qwen_default', ['style/k1.jpg'])


def test_parity_pass_fail(small):
    populate()
    assert E.parity() == 0
    report = json.loads((E.outputs() / 'mp_parity.json').read_text())
    assert all(r['pass'] and r['n'] == 2 and r['min_cosine'] == 1 for r in report.values())
    np.save(S.folder('style_word', 'parity') / 'img.npy', np.asarray([[0, 1], [0, -1]], np.float16))
    assert E.parity() == 1
    assert not json.loads((E.outputs() / 'mp_parity.json').read_text())['style_word']['pass']


@pytest.mark.parametrize('damage', ['norm', 'nan', 'hash', 'keys', 'shape'])
def test_check_good_and_broken(small, damage):
    populate()
    assert E.check() == 0
    out = S.folder('siglip1')
    if damage == 'keys':
        (out / 'keys.json').write_text(json.dumps(E.target_keys()[::-1]))
    elif damage == 'hash':
        meta = json.loads((out / 'meta.json').read_text())
        meta['img_sha256'] = 'broken'
        (out / 'meta.json').write_text(json.dumps(meta))
    else:
        x = np.load(out / 'img.npy')
        if damage == 'shape':
            x = x[:, :1]
        else:
            x[0] = [0, 0] if damage == 'norm' else [np.nan, 0]
        np.save(out / 'img.npy', x)
    assert E.check() == 1
    assert not E.check_store('siglip1', 'new')['ok']


def test_nearest_chunking_inclusive_cut(small, tmp_path, monkeypatch):
    # Independent comparisons exercise both matrix chunk axes and nearest-name mapping.
    q = np.asarray([[1, 0], [0, 1], [.6, .8]], np.float32)
    ref = np.asarray([[0, 1], [1, 0], [-1, 0]], np.float32)
    maxima, ids = E.nearest_rows(q, ref, query_chunk=1, reference_chunk=1)
    np.testing.assert_allclose(maxima, [1, 1, .8])
    np.testing.assert_array_equal(ids, [1, 0, 0])
    boundary = np.asarray([[.85, np.sqrt(1 - .85 ** 2)]], np.float32)
    actual, _ = E.nearest_rows(np.asarray([[1, 0]], np.float32), boundary)
    assert actual[0] >= np.float32(E.NEAR_COPY_CUT)
    populate()
    held = tmp_path / 'held.json'
    held.write_text(json.dumps(dict(names=['k0', 'k4', 'k6'])))
    # Controlled maxima check inclusive reporting at exactly each threshold.
    monkeypatch.setattr(E, 'nearest_rows', lambda q, r: (np.asarray([.85, .95, .99], np.float32), np.zeros(3, int)))
    dest = tmp_path / 'nearcopy.parquet'
    assert E.nearcopy(held_list=held, parquet_path=dest) == 0
    report = json.loads((E.outputs() / 'mp_nearcopy.json').read_text())
    assert report['counts'] == {'0.85': 3, '0.90': 2, '0.95': 2, '0.99': 1}
    assert [r['key'] for r in report['matches']] == E.target_keys()[::-1]
    frame = pd.read_parquet(dest)
    assert frame.key.tolist() == E.target_keys()
    assert frame.nearest_painting.tolist() == ['k2'] * 3


def test_smoke_requires_scratch(small, monkeypatch):
    monkeypatch.delenv('MP_STORE_ROOT')
    with pytest.raises(AssertionError, match='MP_STORE_ROOT'):
        E.extract('qwen_default', smoke=1, device='cpu')
    with pytest.raises(AssertionError, match='MP_STORE_ROOT'):
        E.extract_all(smoke=1)
    monkeypatch.setenv('MP_STORE_ROOT', str(S.DEFAULT_ROOT))
    with pytest.raises(AssertionError, match='scratch'):
        E.smoke_guard(1)


def test_smoke_extract_check_and_skip(small, monkeypatch):
    class Fake:
        def encode_images(self, images, batch_size):
            return np.tile([1, 0], (len(images), 1))
    monkeypatch.setattr(E, 'load_encoder', lambda *a: (Fake(), dict(model_id='fake', model_revision='a' * 40), None))
    for which in ('new', 'parity'):
        for view in S.VIEWS:
            E.extract(view, which, smoke=1, device='cpu')
    assert E.check() == 0
    assert E.extract_all(smoke=1) == 0
    with pytest.raises(AssertionError, match='mismatch'):
        E.extract_all()


def test_held_instruction_and_revision_without_model(small, monkeypatch):
    # Exercise real instruction setup and actual Qwen prompt method with an uninitialised encoder.
    sys.path.insert(0, str(E.HERE.parents[2]))
    from src.data.feature_extract import Qwen3VLEmb
    sys.path.insert(0, str(E.HERE.parent / '20261017_prompted_aspect_views'))
    from pav_prompts import PROMPTS
    revision = '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'
    for view in S.TEACHERS:
        (S.reference_folder(view) / 'meta.json').write_text(json.dumps(dict(
            instruction=PROMPTS[view], model_id='fake-qwen', model_revision=revision)))
    def loader(view, device):
        enc = object.__new__(Qwen3VLEmb)
        enc.instruction = PROMPTS['default' if view == 'qwen_default' else view]
        enc.p = object()
        return enc, dict(model_id='fake-qwen', model_revision=revision), enc.instruction
    fake_bs = SimpleNamespace(load_encoder=loader, BATCH=E.BATCH)
    monkeypatch.setattr(E, 'r14', lambda: fake_bs)
    for view in ('qwen_default', *S.TEACHERS):
        enc, identity, instruction = E.load_encoder(view, 'cpu')
        assert enc.instruction == PROMPTS['default' if view == 'qwen_default' else view]
        assert identity['model_revision'] == revision
    meta_path = S.reference_folder('style_word') / 'meta.json'
    meta = json.loads(meta_path.read_text())
    meta['model_revision'] = 'b' * 40
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match='revision'):
        E.load_encoder('style_word', 'cpu')
    meta['instruction'] = 'incorrect'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match='pav_prompts'):
        E.held_identity('style_word')


def test_reader_missing_and_duplicate_reference_rows(small):
    populate()
    out = S.folder('qwen_default')
    (out / 'keys.json').write_text(json.dumps(['wrong', *E.target_keys()[1:]]))
    with pytest.raises(AssertionError, match='not found'):
        S.load('qwen_default', ['style/k1.jpg'])
    folder = S.reference_folder('style_word')
    np.savez(folder / 'ids.npz', painting_names=np.asarray(['k0', 'k0', 'k6']))
    with pytest.raises(AssertionError, match='duplicate'):
        S.load('style_word', ['style/k0.jpg'])


def test_seal_refuses_nonunit_output(small):
    pending = E.check_target(S.folder('clip_l14'))
    pending.mkdir(parents=True)
    with pytest.raises(AssertionError, match='finite/unit'):
        E.seal(pending, S.folder('clip_l14'), 'clip_l14', 'new', E.target_keys(),
               np.zeros((3, 2), np.float16), dict(model_id='fake', model_revision='a' * 40), None, False, 0)
    assert pending.exists() and not S.folder('clip_l14').exists()
