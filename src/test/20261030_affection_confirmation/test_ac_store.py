"""Synthetic-store tests for ac_store (CPU) and FI path resolution on real names (read-only)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import ac_paths as P
import ac_store as S

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import extract_features as E  # noqa: E402

D = 8


def unit(rng, n):
    x = rng.normal(size=(n, D)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def make_store(path, source, n_img, n_txt, seed, dataset, dropped=0):
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    np.save(path / 'img.npy', unit(rng, n_img))
    np.save(path / 'txt.npy', unit(rng, n_txt))
    images = [{'source': source, 'image_name': f'{source}_{i}.jpg', 'splits': ['train'], 'n_texts': 0}
              for i in range(n_img)]
    texts = [{'img': j % n_img, 'csv_row': j, 'emotion': 'awe', 'workerid': 'u', 'split': 'train'}
             for j in range(n_txt)]
    idx = {'dataset': dataset, 'backbone': 'b', 'smoke': False, 'dtype': 'float16', 'dim': D, 'n_images': n_img,
           'n_texts': n_txt, 'n_texts_truncated': 0, 'truncation_limit': 512, 'image_only': False,
           'extra': {'sources': [source], 'n_images_per_source': {source: n_img},
                     'n_texts_per_source': {source: n_txt}, 'n_empty_skipped': 0,
                     'n_rows_dropped_other_sources': dropped},
           'images': images, 'texts': texts}
    json.dump(idx, open(path / 'index.json', 'w'))


@pytest.fixture
def stores(tmp_path):
    make_store(tmp_path / 'old', 'coco_affective', 5, 12, 0, 'affection_local', dropped=7)
    make_store(tmp_path / 'fi', P.FI_SOURCE, 3, 7, 1, 'affection_fi')
    return tmp_path


def run_merge(t, **kw):
    return S.merge(t / 'old', t / 'fi', t / 'merged', t / 'hashes.json', t / 'stage.json',
                   expected_fi=(3, 7), **kw)


def test_record_refuses_overwrite(stores):
    S.record(stores / 'old', stores / 'hashes.json')
    with pytest.raises(SystemExit):
        S.record(stores / 'old', stores / 'hashes.json')


def test_merge_ok_and_swap(stores):
    t = stores
    S.record(t / 'old', t / 'hashes.json')
    old_img = np.load(t / 'old/img.npy')
    res = run_merge(t)
    assert res['ok'] and not res['swapped'] and (t / 'old').exists()
    assert json.load(open(t / 'stage.json'))['ok']
    m = np.load(t / 'merged/img.npy')
    assert m.shape == (8, D) and np.array_equal(m[:5], old_img)
    mi = json.load(open(t / 'merged/index.json'))
    assert mi['n_images'] == 8 and mi['n_texts'] == 19 and len(mi['texts']) == 19
    assert [x['img'] for x in mi['texts'][12:]] == [5 + j % 3 for j in range(7)]
    assert mi['extra']['sources'] == ['coco_affective', P.FI_SOURCE]
    assert mi['extra']['n_rows_dropped_other_sources'] == 0
    assert mi['extra']['n_images_per_source'][P.FI_SOURCE] == 3
    # swap on a second merge into a fresh folder
    import shutil
    shutil.rmtree(t / 'merged')
    res = run_merge(t, swap=True)
    assert res['swapped'] and (t / 'old_pre_fi').is_dir() and np.load(t / 'old/img.npy').shape == (8, D)
    assert np.array_equal(np.load(t / 'old_pre_fi/img.npy'), old_img)


def test_changed_old_row_fails(stores):
    t = stores
    S.record(t / 'old', t / 'hashes.json')
    a = np.load(t / 'old/img.npy')
    a[2, 0] += np.float16(0.01)
    np.save(t / 'old/img.npy', a)
    res = run_merge(t, swap=True)
    assert not res['ok'] and not res['swapped'] and not res['items']['old_store_unchanged_img_file']['ok']
    assert not (t / 'merged').exists() and not (t / 'old_pre_fi').exists()


def test_changed_old_index_fails(stores):
    t = stores
    S.record(t / 'old', t / 'hashes.json')
    idx = json.load(open(t / 'old/index.json'))
    idx['images'][1]['splits'] = ['val']
    json.dump(idx, open(t / 'old/index.json', 'w'))
    assert not run_merge(t)['ok']


def test_bad_fi_norm_fails(stores):
    t = stores
    S.record(t / 'old', t / 'hashes.json')
    a = np.load(t / 'fi/img.npy')
    a[1] *= 2
    np.save(t / 'fi/img.npy', a)
    res = run_merge(t)
    assert not res['ok'] and not res['items']['fi_extraction_gate']['ok'] and not (t / 'merged').exists()


def test_fi_path_resolution_real_names():
    spec = E.ADAPTERS['affection_fi']()
    assert len(spec.images) == P.N_FI['images'] and len(spec.texts) == P.N_FI['texts']
    m = spec.images[0]
    assert E._fi_path(E.AFFECTION_FI_ROOT, m['image_name']).parent.name == m['image_name'].rsplit('_', 1)[0]
    assert E._fi_path(E.AFFECTION_FI_ROOT, m['image_name']).is_file()
    with pytest.raises(SystemExit):
        E._fi_path(E.AFFECTION_FI_ROOT, 'nonexistent_0001.jpg')
    assert E.out_dir_for('affection_fi', 'qwen3vl_emb_2b', False) == P.STORE_FI
    assert 'affection_fi' in str(E.out_dir_for('affection_fi', 'qwen3vl_emb_2b', True))
