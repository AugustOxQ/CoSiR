"""Tests of ns_photo (r17 ticket 04): key order and counts, labels, key-joined rows, the refusing joins. Real data
(r12 to r14 stores and exports, read only)."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as NP  # noqa: E402
import ns_photo as P  # noqa: E402
import bs_emoset  # noqa: E402
import bs_paths  # noqa: E402
import bs_stores  # noqa: E402
import rs_vh  # noqa: E402

N = NP.N_ROWS


# ------------------------------------------------------------------------------------------------ keys
@pytest.mark.parametrize('split', ['test', 'val'])
def test_affection_keys_are_python_sorted_with_the_counts(split):
    k = P.keys('affection', split)
    assert len(k) == N[('affection', split)] and len(set(k)) == len(k)
    assert k == sorted(k)
    assert all('/' in x for x in k)


@pytest.mark.parametrize('split', ['test', 'val'])
def test_emoset_keys_are_sha256_ordered_after_the_cut(split):
    k = P.keys('emoset', split)
    assert len(k) == N[('emoset', split)] and len(set(k)) == len(k)
    digest = [hashlib.sha256(x.encode()).hexdigest() for x in k]
    assert digest == sorted(digest)                       # D17's order, computed here without bs_emoset.order
    assert k != sorted(k)                                 # not the sorted-key order
    uncut = bs_emoset.keys(split, cut=False)
    assert len(uncut) > len(k) and set(k) < set(uncut)    # the near-copy cut removed rows
    removed = set(uncut) - set(k)
    rec = json.loads(bs_emoset.record_file().read_text())['splits'][split]
    assert removed == set(rec['removed_keys'])


def test_keys_refuse_other_sets_and_splits():
    with pytest.raises(AssertionError):
        P.keys('mmart', 'test')
    with pytest.raises(AssertionError):
        P.keys('affection', 'held')


# ------------------------------------------------------------------------------------------------ labels
def test_emoset_labels_follow_r14_class_order():
    assert bs_paths.EMOSET_CLASSES == ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear',
                                       'sadness')
    for split in ('test', 'val'):
        k, y = P.keys('emoset', split), P.labels('emoset', split)
        assert y.dtype == np.int64 and y.shape == (len(k),)
        assert sorted(set(y.tolist())) == list(range(8))
        names = bs_emoset.labels(k)                       # the class names of the split files, by key
        for i in (0, 1, 2, len(k) // 2, len(k) - 1):
            assert bs_paths.EMOSET_CLASSES[y[i]] == names[i]
        # the key's path names its class (image/<class>/...): independent of the split files' first field
        assert all(k[i].split('/')[1] == bs_paths.EMOSET_CLASSES[y[i]] for i in range(0, len(k), 97))


def test_affection_labels_are_nine_class_ints_in_key_order():
    import ac_data
    for split in ('test', 'val'):
        y = P.labels('affection', split)
        assert y.dtype == np.int64 and y.shape == (N[('affection', split)],) and y.min() == 0 and y.max() == 8
        assert np.array_equal(y, ac_data.labels(split))


def test_class_counts_equal_the_stage0_numbers():
    c = P.class_counts()
    assert c == {f'{s}/emotion': NP.CLASS_COUNT[(s, 'emotion')] for s in NP.PHOTO_SETS}


# ------------------------------------------------------------------------------------------------ the join
def test_join_rows_reorders_by_key():
    rows = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = P.join_rows(['d', 'a', 'c', 'b'], rows, ['a', 'b', 'c'])
    assert np.array_equal(out, rows[[1, 3, 2]])


@pytest.mark.parametrize('file_keys,want,kw', [
    (['a', 'b', 'c'], ['a', 'z'], {}),                      # a requested key the file lacks
    (['a', 'b', 'b'], ['a', 'b'], {}),                      # duplicate key in the file
    (['a', 'b', 'c'], ['a', 'a'], {}),                      # duplicate key in the request
    (['a', 'b', 'c'], ['a', 'b'], {'exact': {'a', 'b'}}),   # the file holds more keys than the split file
    (['a', 'b'], ['a', 'b'], {'exact': {'a', 'b', 'c'}}),   # the file lacks a key of the split file
])
def test_join_rows_refuses_a_key_mismatch(file_keys, want, kw):
    rows = np.zeros((len(file_keys), 2), np.float32)
    with pytest.raises(AssertionError):
        P.join_rows(file_keys, rows, want, **kw)


def test_join_rows_refuses_a_row_count_that_differs_from_the_key_count():
    with pytest.raises(AssertionError):
        P.join_rows(['a', 'b'], np.zeros((3, 2), np.float32), ['a'])


def test_photo_rows_are_the_export_rows_joined_by_key_not_position():
    split, seed = 'val', 2
    path = P.export_path('emoset', 'qwen', split, seed)
    with np.load(path) as z:
        fk, emb = z['keys'].tolist(), z['emb']
    assert len(fk) == len(bs_emoset.keys(split, cut=False)) > N[('emoset', split)]
    k = P.keys('emoset', split)
    x = P.photo('emoset', split, 'qwen', seed)
    assert x.dtype == np.float32 and x.shape == (len(k), 2048)
    assert fk[:50] != k[:50]                                # the file order is not the analysis order
    where = {key: i for i, key in enumerate(fk)}
    for i in list(range(0, len(k), 331)) + [len(k) - 1]:
        assert np.array_equal(x[i], emb[where[k[i]]].astype(np.float32))
    # r14's form: the float16 cast to float32, not divided by the norm
    stored = P.photo_stored('emoset', split, 'qwen', seed)
    assert np.array_equal(x, stored.astype(np.float32))
    norms = np.linalg.norm(x, axis=1)
    assert np.abs(norms - 1).max() > 1e-6                   # a renormalised array would be unit to float32 rounding
    assert np.abs(norms - 1).max() < NP.TOL_UNIT


def test_affection_photo_rows_follow_sorted_keys():
    split, seed = 'val', 0
    path = P.export_path('affection', 'siglip2', split, seed)
    with np.load(path) as z:
        fk, emb = z['keys'].tolist(), z['emb']
    k = P.keys('affection', split)
    assert set(fk) == set(k) and len(fk) == len(k)
    x = P.photo('affection', split, 'siglip2', seed)
    where = {key: i for i, key in enumerate(fk)}
    for i in range(0, len(k), 211):
        assert np.array_equal(x[i], emb[where[k[i]]].astype(np.float32))


def test_photo_refuses_an_export_whose_keys_do_not_match(tmp_path, monkeypatch):
    src = P.export_path('affection', 'clip_l14', 'val', 0)
    root = tmp_path / 'affection' / 'photo_clip_l14'
    root.mkdir(parents=True)
    with np.load(src) as z:
        fk, emb = z['keys'].tolist(), z['emb']
    fk[7] = fk[7] + '_renamed'                              # one key differs; sha and sidecar stay consistent
    dst = root / 'val__s0.npz'
    np.savez(dst, keys=np.asarray(fk), emb=emb)
    dst.with_suffix('.json').write_text(json.dumps({'sha256': rs_vh.file_sha(dst)}))
    monkeypatch.setattr(P, 'EXPORT_ROOT', tmp_path)
    with pytest.raises(AssertionError, match='not in the file|different key set'):
        P.photo('affection', 'val', 'clip_l14', 0)
    # the untouched copy joins: the refusal comes from the key, not from the copy
    fk[7] = fk[7][:-len('_renamed')]
    np.savez(dst, keys=np.asarray(fk), emb=emb)
    dst.with_suffix('.json').write_text(json.dumps({'sha256': rs_vh.file_sha(dst)}))
    assert np.array_equal(P.photo('affection', 'val', 'clip_l14', 0), P.photo_stored('affection', 'val', 'clip_l14', 0)
                          .astype(np.float32))


def test_photo_refuses_an_export_with_an_extra_key(tmp_path, monkeypatch):
    src = P.export_path('affection', 'clip_l14', 'val', 0)
    root = tmp_path / 'affection' / 'photo_clip_l14'
    root.mkdir(parents=True)
    with np.load(src) as z:
        fk, emb = z['keys'].tolist(), z['emb']
    dst = root / 'val__s0.npz'
    np.savez(dst, keys=np.asarray(fk + ['extra/not_in_the_split.jpg']), emb=np.concatenate([emb, emb[:1]]))
    dst.with_suffix('.json').write_text(json.dumps({'sha256': rs_vh.file_sha(dst)}))
    monkeypatch.setattr(P, 'EXPORT_ROOT', tmp_path)
    with pytest.raises(AssertionError, match='different key set'):
        P.photo('affection', 'val', 'clip_l14', 0)


def test_photo_refuses_a_file_that_differs_from_its_sidecar(tmp_path, monkeypatch):
    src = P.export_path('affection', 'clip_l14', 'val', 0)
    root = tmp_path / 'affection' / 'photo_clip_l14'
    root.mkdir(parents=True)
    shutil.copy(src, root / 'val__s0.npz')
    (root / 'val__s0.json').write_text(json.dumps({'sha256': '0' * 64}))
    monkeypatch.setattr(P, 'EXPORT_ROOT', tmp_path)
    with pytest.raises(AssertionError, match='sidecar'):
        P.photo('affection', 'val', 'clip_l14', 0)


def test_emoset_is_scored_for_qwen_only():
    for fn in (lambda: P.photo('emoset', 'val', 'siglip2', 0), lambda: P.plain('emoset', 'val', 'clip_l14'),
               lambda: P.teacher('emoset', 'val', 'siglip1')):
        with pytest.raises(AssertionError):
            fn()
    with pytest.raises(AssertionError):
        P.photo('affection', 'val', 'qwen', 5)


# ------------------------------------------------------------------------------------------------ plain and teacher
@pytest.mark.parametrize('set_name,enc', [('affection', 'clip_l14'), ('affection', 'siglip1'), ('emoset', 'qwen')])
def test_plain_rows_are_unit_float32_joined_by_key(set_name, enc):
    split = 'val'
    k = P.keys(set_name, split)
    x = P.plain(set_name, split, enc)
    assert x.dtype == np.float32 and x.shape == (len(k), NP.WIDTH[enc])
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-6
    # an independent load in the reversed key order gives the reversed rows
    raw = bs_stores.load(set_name, P.PLAIN_VIEW[enc], k[::-1])[::-1]
    i = np.arange(0, len(k), 53)
    want = raw[i] / np.linalg.norm(raw[i], axis=1, keepdims=True)
    assert np.abs(x[i] - want).max() < 1e-6


@pytest.mark.parametrize('set_name', ['affection', 'emoset'])
def test_teacher_is_the_qwen_artwork_image_for_every_encoder(set_name):
    split = 'val'
    k = P.keys(set_name, split)
    encs = NP.PHOTO_ENCODERS[set_name]
    t = P.teacher(set_name, split, encs[0])
    assert t.dtype == np.float32 and t.shape == (len(k), 2048)
    assert np.abs(np.linalg.norm(t, axis=1) - 1).max() < 1e-6
    for e in encs[1:]:
        assert np.array_equal(t, P.teacher(set_name, split, e))
    raw = bs_stores.load(set_name, 'emotion_focus', k[::-1])[::-1]       # the artwork wording, not the photo wording
    i = np.arange(0, len(k), 53)
    assert np.abs(t[i] - raw[i] / np.linalg.norm(raw[i], axis=1, keepdims=True)).max() < 1e-6
    photo_wording = bs_stores.load(set_name, 'emotion_focus_photo', k[:200])
    assert np.abs(t[:200] - photo_wording / np.linalg.norm(photo_wording, axis=1, keepdims=True)).max() > 1e-3


def test_teacher_wording_comes_from_the_earlier_records_and_must_be_artwork(tmp_path, monkeypatch):
    assert P.teacher_wording('affection') == P.teacher_wording('emoset') == 'emotion_focus'
    picks = json.loads(P.R12_PICKS.read_text())
    picks['affection']['teacher_wording']['clustering'] = 'photo'
    bad = tmp_path / 'picks.json'
    bad.write_text(json.dumps(picks))
    monkeypatch.setattr(P, 'R12_PICKS', bad)
    monkeypatch.setattr(P, '_WORDING', {})
    with pytest.raises(AssertionError):
        P.teacher_wording('affection')
