"""Tests of pp_stores (r19 ticket 01): the one reader of every old and new feature store, joined by key.

Fake-store tests write small stores in the formats of the index into tmp and point the reader at them. Real-store tests
compare the reader with the older rounds' readers on sampled keys in shuffled order (constitution C6).
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import pp_paths as PP  # noqa: F401  (first: sets the older rounds' output roots)
import pp_stores as S

PP_PROMPTS = 'qwen3vl_emb_2b_aspect_prompts'


# ------------------------------------------------------------------------------------------------ fake stores
def unit_rows(n, d, seed):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def write_keys_store(folder, keys, d, seed, meta=True, rows=None):
    folder.mkdir(parents=True, exist_ok=True)
    x = unit_rows(len(keys), d, seed) if rows is None else rows
    np.save(folder / 'img.npy', x)
    (folder / 'keys.json').write_text(json.dumps(list(keys)))
    if meta:
        (folder / 'meta.json').write_text(json.dumps(dict(
            n=len(keys), d=d, keys_sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
            img_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())))
    return x


def write_index_store(folder, dataset, keys, d, seed):
    """Index format: img.npy + index.json with the dataset's own key fields."""
    folder.mkdir(parents=True, exist_ok=True)
    x = unit_rows(len(keys), d, seed)
    np.save(folder / 'img.npy', x)
    if dataset == 'artelingo':
        images = [{'painting': k, 'image': f'Style/{k}.jpg'} for k in keys]
    elif dataset == 'affection':
        images = [{'source': k.split('/')[0], 'image_name': k.split('/')[1]} for k in keys]
    else:
        images = [{'image_file': k, 'split': 'train'} for k in keys]
    (folder / 'index.json').write_text(json.dumps({'images': images, 'texts': []}))
    return x


def write_ids_store(folder, names, d, seed):
    folder.mkdir(parents=True, exist_ok=True)
    x = unit_rows(len(names), d, seed)
    np.save(folder / 'img.npy', x)
    np.savez(folder / 'ids.npz', painting_names=np.asarray(names))
    return x


def f32unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    for name in ('ART_ROOT', 'AFF_ROOT', 'MMART_ROOT', 'SEMART_ROOT'):
        monkeypatch.setattr(S, name, tmp_path / name.lower())
    monkeypatch.setattr(S, 'NEW_STORE', {})
    monkeypatch.setattr(S, 'EXPECTED_ROWS', {})       # fake stores are small: the full-size counts are off
    S._CACHE.clear()
    return tmp_path


# ------------------------------------------------------------------------------------------------ new-format store
def test_new_store_rows_follow_the_requested_keys_not_the_store_order(roots):
    keys = [f'img{i:02d}.jpg' for i in range(12)]
    raw = write_keys_store(roots / 'sem_b32', keys, 512, 1)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'sem_b32'
    ask = [keys[i] for i in (7, 0, 11, 3, 5)]
    got = S.load('semart', 'clip_b32', ask)
    assert got.dtype == np.float32 and got.shape == (5, 512)
    for row, k in zip(got, ask):
        assert np.array_equal(row, f32unit(raw[keys.index(k)][None])[0])


def test_load_divides_by_the_norm_once(roots):
    keys = ['a.jpg', 'b.jpg', 'c.jpg']
    x = unit_rows(3, 512, 2) * np.float16(0.9)                      # stored rows are not exactly unit
    write_keys_store(roots / 's', keys, 512, 0, rows=x)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 's'
    got = S.load('semart', 'clip_b32', keys)
    assert np.abs(np.linalg.norm(got, axis=1) - 1).max() < 1e-6
    assert np.array_equal(got, f32unit(x))


def test_load_asserts_on_missing_duplicate_and_empty_keys(roots):
    keys = ['a.jpg', 'b.jpg']
    write_keys_store(roots / 's', keys, 512, 0)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 's'
    with pytest.raises(AssertionError, match='not found'):
        S.load('semart', 'clip_b32', ['a.jpg', 'zzz.jpg'])
    with pytest.raises(AssertionError):
        S.load('semart', 'clip_b32', ['a.jpg', 'a.jpg'])
    with pytest.raises(AssertionError):
        S.load('semart', 'clip_b32', [])


def test_load_asserts_on_wrong_width_and_nonfinite_and_zero_rows(roots):
    keys = ['a.jpg', 'b.jpg']
    write_keys_store(roots / 'w', keys, 256, 0)                       # clip_b32 is 512 wide
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'w'
    with pytest.raises(AssertionError, match='width'):
        S.load('semart', 'clip_b32', keys)
    bad = unit_rows(2, 512, 3)
    bad[1, 4] = np.nan
    write_keys_store(roots / 'n', keys, 512, 0, rows=bad)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'n'
    with pytest.raises(AssertionError, match='finite'):
        S.load('semart', 'clip_b32', keys)
    zero = unit_rows(2, 512, 3)
    zero[0] = 0
    write_keys_store(roots / 'z', keys, 512, 0, rows=zero)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'z'
    with pytest.raises(AssertionError, match='zero'):
        S.load('semart', 'clip_b32', keys)


def test_store_with_rows_not_matching_its_keys_asserts(roots):
    folder = roots / 'r'
    write_keys_store(folder, ['a.jpg', 'b.jpg', 'c.jpg'], 512, 0)
    (folder / 'keys.json').write_text(json.dumps(['a.jpg', 'b.jpg']))
    S.NEW_STORE[('semart', 'clip_b32')] = folder
    with pytest.raises(AssertionError, match='rows'):
        S.load('semart', 'clip_b32', ['a.jpg'])


def test_unknown_dataset_and_view_assert(roots):
    with pytest.raises(AssertionError):
        S.load('emoset', 'qwen', ['a'])
    with pytest.raises(AssertionError):
        S.load('affection', 'style_word', ['a/b'])                    # no Affection store of that view
    with pytest.raises(AssertionError):
        S.load('semart', 'siglip3', ['a.jpg'])


# ------------------------------------------------------------------------------------------------ index formats
def test_artelingo_default_view_by_painting_and_teacher_from_two_folders(roots):
    names = [f'artist_work-{i}' for i in range(10)]
    raw = write_index_store(roots / 'art_root' / 'clip_b32', 'artelingo', names, 512, 4)
    prompts = roots / 'art_root' / 'qwen3vl_emb_2b_aspect_prompts'
    s2img = write_ids_store(prompts / 'genre_word_s2img', names[:6], 2048, 5)
    held = write_ids_store(prompts / 'genre_word_held', names[6:], 2048, 6)
    ask = [names[i] for i in (9, 2, 6, 0)]
    got = S.load('artelingo', 'clip_b32', ask)
    assert np.array_equal(got, f32unit(raw[[9, 2, 6, 0]]))
    t = S.load('artelingo', 'genre_word', ask)                        # names 9 and 6 are in the held folder
    assert np.array_equal(t, f32unit(np.stack([held[3], s2img[2], held[0], s2img[0]])))


def test_a_key_in_two_folders_asserts(roots):
    prompts = roots / 'art_root' / 'qwen3vl_emb_2b_aspect_prompts'
    write_ids_store(prompts / 'genre_word_s2img', ['a', 'b'], 2048, 5)
    write_ids_store(prompts / 'genre_word_held', ['b', 'c'], 2048, 6)
    with pytest.raises(AssertionError, match='more than one'):
        S.load('artelingo', 'genre_word', ['a', 'b'])


def test_affection_key_is_source_slash_image_name_across_two_folders(roots):
    aff = roots / 'aff_root'
    a = write_keys_store(aff / 'clip_l14_valtest', ['coco/x.jpg', 'FI/y.jpg'], 768, 7)
    b = write_keys_store(aff / 'clip_l14_train', ['coco/z.jpg'], 768, 8)
    got = S.load('affection', 'clip_l14', ['coco/z.jpg', 'FI/y.jpg', 'coco/x.jpg'])
    assert np.array_equal(got, f32unit(np.stack([b[0], a[1], a[0]])))
    q = write_index_store(aff / 'qwen3vl_emb_2b', 'affection', ['coco/x.jpg', 'FI/y.jpg'], 2048, 9)
    assert np.array_equal(S.load('affection', 'qwen', ['FI/y.jpg']), f32unit(q[1:2]))
    assert np.array_equal(S.load('affection', 'qwen_default', ['FI/y.jpg']), f32unit(q[1:2]))   # r14's view name


def test_semart_old_index_format_is_keyed_by_image_file(roots):
    files = ['00001-a.jpg', '00002-b.jpg', '00003-c.jpg']
    x = write_index_store(roots / 'semart_root' / 'clip_b32', 'semart', files, 512, 10)
    got = S.load('semart', 'clip_b32', ['00003-c.jpg', '00001-a.jpg'])
    assert np.array_equal(got, f32unit(x[[2, 0]]))


# ------------------------------------------------------------------------------------------------ MMArt
@pytest.fixture
def mmart(roots):
    keys = [f'Style/p{i}.jpg' for i in range(9)]
    group = ['none'] * 3 + ['held'] * 3 + ['trainval'] * 3
    index = pd.DataFrame(dict(key=keys, painting=[f'p{i}' for i in range(9)], artelingo=group))
    art = roots / 'art_root'
    # ArtELingo side lists held and trainval paintings by painting name; row order differs from the index
    art_names = ['p8', 'p3', 'p6', 'p4', 'p5', 'p7']
    art_raw = write_index_store(art / 'clip_b32', 'artelingo', art_names, 512, 20)
    mm_raw = write_keys_store(roots / 'mm_new_b32', keys[:3], 512, 21)
    S.NEW_STORE[('mmart', 'clip_b32')] = roots / 'mm_new_b32'
    prompts = art / 'qwen3vl_emb_2b_aspect_prompts'
    s2 = write_ids_store(prompts / 'genre_word_s2img', ['p6', 'p7', 'p8'], 2048, 22)
    hd = write_ids_store(prompts / 'genre_word_held', ['p3', 'p4', 'p5'], 2048, 23)
    mm_t = write_keys_store(roots / 'mmart_root' / 'qwen3vl_emb_2b_aspect_prompts' / 'genre_word', keys[:3], 2048, 24)
    return dict(index=index, keys=keys, art_names=art_names, art_raw=art_raw, mm_raw=mm_raw, s2=s2, hd=hd, mm_t=mm_t)


def test_mmart_new_keys_from_mmart_store_others_from_artelingo_by_painting(mmart):
    m = mmart
    ask = [m['keys'][i] for i in (8, 1, 4, 0, 3, 7)]
    got = S.load('mmart', 'clip_b32', ask, index=m['index'])
    ai = m['art_names'].index
    want = np.stack([m['art_raw'][ai('p8')], m['mm_raw'][1], m['art_raw'][ai('p4')], m['mm_raw'][0],
                     m['art_raw'][ai('p3')], m['art_raw'][ai('p7')]])
    assert np.array_equal(got, f32unit(want))


def test_mmart_teacher_view_new_from_mmart_store_held_from_held_trainval_from_s2img(mmart):
    m = mmart
    ask = [m['keys'][i] for i in (7, 2, 3, 5, 0)]
    got = S.load('mmart', 'genre_word', ask, index=m['index'])
    want = np.stack([m['s2'][1], m['mm_t'][2], m['hd'][0], m['hd'][2], m['mm_t'][0]])
    assert np.array_equal(got, f32unit(want))


def test_mmart_key_missing_from_the_index_asserts(mmart):
    with pytest.raises(AssertionError, match='index'):
        S.load('mmart', 'clip_b32', ['Style/nope.jpg'], index=mmart['index'])


# ------------------------------------------------------------------------------------------------ check (gate (i))
def make_check_store(roots, n=40, **kw):
    keys = [f'f{i:03d}.jpg' for i in range(n)]
    write_keys_store(roots / 'chk', keys, 512, 30, **kw)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'chk'
    return keys


def test_check_passes_a_good_store(roots):
    make_check_store(roots)
    r = S.check('semart', 'clip_b32')
    assert r['ok'], r['problems']
    assert r['n'] == 40 and r['d'] == 512 and r['n_nonfinite'] == 0 and r['max_unit_dev'] < 1e-3


def test_check_flags_a_non_unit_row(roots):
    keys = [f'f{i:03d}.jpg' for i in range(10)]
    x = unit_rows(10, 512, 31)
    x[3] = (x[3].astype(np.float32) * 1.01).astype(np.float16)
    write_keys_store(roots / 'chk', keys, 512, 0, rows=x)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'chk'
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and r['max_unit_dev'] > 1e-3 and any('unit' in p for p in r['problems'])


def test_check_flags_non_finite_rows_and_does_not_raise(roots):
    keys = [f'f{i:03d}.jpg' for i in range(10)]
    x = unit_rows(10, 512, 32)
    x[2, 1] = np.inf
    write_keys_store(roots / 'chk', keys, 512, 0, rows=x)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'chk'
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and r['n_nonfinite'] >= 1


def test_check_flags_a_meta_hash_that_does_not_match(roots):
    make_check_store(roots)
    meta = json.loads((roots / 'chk' / 'meta.json').read_text())
    meta['img_sha256'] = '0' * 64
    (roots / 'chk' / 'meta.json').write_text(json.dumps(meta))
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and any('img_sha256' in p for p in r['problems'])
    meta['img_sha256'] = hashlib.sha256(np.ascontiguousarray(np.load(roots / 'chk' / 'img.npy')).tobytes()).hexdigest()
    meta['keys_sha256'] = '1' * 64
    (roots / 'chk' / 'meta.json').write_text(json.dumps(meta))
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and any('keys_sha256' in p for p in r['problems'])


def test_check_flags_duplicate_keys_and_a_row_count_mismatch(roots):
    keys = make_check_store(roots, n=6)
    (roots / 'chk' / 'keys.json').write_text(json.dumps(keys[:5] + [keys[0]]))
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and any('duplicate' in p for p in r['problems'])
    (roots / 'chk' / 'keys.json').write_text(json.dumps(keys[:5]))
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and any('rows' in p for p in r['problems'])


def test_check_flags_a_missing_folder(roots):
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'not_there'
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and any('missing' in p for p in r['problems'])


def test_check_new_store_row_count_and_key_set_against_the_reference(roots, monkeypatch):
    ref_keys = [f'f{i:03d}.jpg' for i in range(8)]
    write_index_store(roots / 'semart_root' / 'qwen3vl_emb_2b', 'semart', ref_keys, 2048, 40)
    write_keys_store(roots / 'new', ref_keys, 512, 41)
    S.NEW_STORE[('semart', 'clip_b32')] = roots / 'new'
    monkeypatch.setitem(S.EXPECTED_ROWS, 'semart', 8)
    assert S.check('semart', 'clip_b32')['ok']
    monkeypatch.setitem(S.EXPECTED_ROWS, 'semart', 9)
    assert not S.check('semart', 'clip_b32')['ok']
    monkeypatch.setitem(S.EXPECTED_ROWS, 'semart', 8)
    write_keys_store(roots / 'new', ref_keys[:7] + ['other.jpg'], 512, 41)       # right count, wrong key set
    r = S.check('semart', 'clip_b32')
    assert not r['ok'] and any('reference' in p for p in r['problems'])


# ------------------------------------------------------------------------------------------------ real stores
def shuffled(keys, n, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    return [keys[i] for i in pick]


@pytest.fixture(scope='module')
def art_index():
    return json.loads((PP.ART / 'clip_b32' / 'index.json').read_text())['images']


def test_real_artelingo_default_views_equal_rb_encoders_full_images(art_index):
    import eg_encoders
    import rb_encoders
    eg_encoders.extend_rb_maps()
    names = shuffled([m['painting'] for m in art_index], 150)
    for enc in PP.ENCODERS:
        want = rb_encoders.full_images(PP.ART_STORE[enc], names)
        assert np.array_equal(S.load('artelingo', enc, names), want), enc


def test_real_artelingo_equals_bs_stores_load_normalised(art_index):
    import bs_stores
    sample = shuffled(art_index, 150, seed=1)
    for view in ('qwen_default', 'clip_l14', 'siglip2', 'siglip1'):
        raw = bs_stores.load('artelingo', view, [m['image'] for m in sample])
        enc = 'qwen' if view == 'qwen_default' else view
        assert np.array_equal(S.load('artelingo', enc, [m['painting'] for m in sample]), f32unit(raw)), view


def test_real_artelingo_teachers_equal_mp_stores_join_of_s2img(art_index):
    import mp_stores
    names = shuffled(list(np.load(PP.ART / 'qwen3vl_emb_2b_aspect_prompts/genre_word_s2img/ids.npz')['painting_names']), 120)
    for view in ('emotion_focus', 'style_word', 'genre_word'):
        store = PP.ART / 'qwen3vl_emb_2b_aspect_prompts' / f'{view}_s2img'
        want = f32unit(mp_stores.join(store, names, view, teacher=True))
        assert np.array_equal(S.load('artelingo', view, names), want), view


def test_real_mmart_equals_mt_data_features_on_new_val_and_held_keys():
    import mt_data
    index = mt_data._index()
    keys = []
    for split in ('new', 'val', 'held'):
        keys += shuffled(index[index.split == split].key.tolist(), 70, seed=2)
    keys = shuffled(keys, len(keys), seed=3)
    for view, enc in (('qwen_default', 'qwen'), ('siglip2', 'siglip2'), ('siglip1', 'siglip1'),
                      ('clip_l14', 'clip_l14'), ('emotion_focus', 'emotion_focus'), ('style_word', 'style_word'),
                      ('genre_word', 'genre_word')):
        assert np.array_equal(S.load('mmart', enc, keys), mt_data.features(view, keys)), view


def test_real_affection_equals_bs_stores_load_normalised():
    import bs_stores
    qwen = [f"{m['source']}/{m['image_name']}"
            for m in json.loads((PP.AFF / 'qwen3vl_emb_2b' / 'index.json').read_text())['images']]
    keys = shuffled(qwen, 200, seed=4)
    for view, enc in (('qwen_default', 'qwen'), ('siglip2', 'siglip2'), ('siglip1', 'siglip1'),
                      ('clip_l14', 'clip_l14'), ('emotion_focus', 'emotion_focus'),
                      ('emotion_focus_photo', 'emotion_focus_photo')):
        ask = keys
        if view == 'emotion_focus_photo':                    # the photo view covers a subset of the 85,007 keys
            ask = shuffled(json.loads((PP.AFF / PP_PROMPTS / 'emotion_focus_photo' / 'keys.json').read_text()), 200, seed=8)
        want = f32unit(bs_stores.load('affection', view, ask))
        assert np.array_equal(S.load('affection', enc, ask), want), view


def test_real_affection_clip_l14_reads_both_folders():
    ks = {f: json.loads((PP.AFF / f / 'keys.json').read_text()) for f in ('clip_l14_valtest', 'clip_l14_train')}
    keys = shuffled(ks['clip_l14_valtest'], 20, seed=5) + shuffled(ks['clip_l14_train'], 20, seed=5)
    keys = shuffled(keys, 40, seed=6)
    got = S.load('affection', 'clip_l14', keys)
    for row, k in zip(got, keys):
        f = 'clip_l14_valtest' if k in set(ks['clip_l14_valtest']) else 'clip_l14_train'
        x = np.load(PP.AFF / f / 'img.npy', mmap_mode='r')[ks[f].index(k)]
        assert np.array_equal(row, f32unit(x[None])[0])


def test_real_semart_old_stores_equal_a_direct_read():
    for enc, folder in (('clip_b32', 'clip_b32'), ('qwen', 'qwen3vl_emb_2b'), ('siglip2', 'siglip2_so400m')):
        images = json.loads((PP.SEMART_PRE / folder / 'index.json').read_text())['images']
        files = [m['image_file'] for m in images]
        raw = np.load(PP.SEMART_PRE / folder / 'img.npy', mmap_mode='r')
        ask = shuffled(files, 100, seed=7)
        want = f32unit(np.stack([raw[files.index(k)] for k in ask]))
        assert np.array_equal(S.load('semart', enc, ask), want), enc


def test_real_check_passes_an_old_store_and_counts_its_rows():
    r = S.check('semart', 'clip_b32')
    assert r['ok'], r['problems']
    assert r['n'] == 21382 and r['d'] == 512
    r = S.check('affection', 'clip_l14')
    assert r['ok'], r['problems']
    assert r['n'] == 85007 and len(r['folders']) == 2
