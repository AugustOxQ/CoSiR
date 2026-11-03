"""Tests of pp_register (r19 ticket 01): the new three in r14's and r15's tables, the store-reading functions on
pp_stores, idempotent, no frozen file edited."""
import importlib
import json

import numpy as np
import pytest

import pp_paths as PP
import pp_register as R
import pp_stores as S

import bs_affection
import bs_artclu
import bs_artret
import bs_paths
import bs_photo
import bs_stores
import mt_data
import mt_paths
import rb_encoders

NEW = ('dinov2_large', 'clip_b32', 'clip_b16')
WIDTH = {'dinov2_large': 1024, 'clip_b32': 512, 'clip_b16': 512}


def snapshot():
    """Every table pp_register touches, as comparable values (functions by identity)."""
    return dict(
        bs_paths=(repr(sorted(bs_paths.ENCODERS.items())), bs_paths.NEW_ENCODERS, repr(sorted(bs_paths.MATCHED.items())),
                  repr(sorted(bs_paths.FROZEN_ARM.items()))),
        bs_stores=(bs_stores.VIEWS, repr(sorted(bs_stores.WIDTH.items())), id(bs_stores.folders)),
        bs_photo=repr(sorted(bs_photo.VIEW.items())),
        bs_artclu=(repr(sorted(bs_artclu.ENC.items(), key=str)), id(bs_artclu.store_loader)),
        bs_artret=(repr(sorted(bs_artret.ENC.items())), id(bs_artret.frozen_embed)),
        bs_affection=(bs_affection.ENCODERS, repr(sorted(bs_affection.VIEW.items())), id(bs_affection.frozen_embed)),
        rb=repr(sorted(rb_encoders.WIDTH.items())),
        mt_paths=(mt_paths.ENCODERS, repr(sorted(mt_paths.DEFAULT_VIEW.items())), repr(sorted(mt_paths.WIDTH.items())),
                  repr(sorted(mt_paths.MATCHED.items())), repr(sorted(mt_paths.CSTAR_ROOT.items()))),
        mt_data=id(mt_data.features))


def test_every_table_lists_the_new_three():
    for e in NEW:
        assert bs_paths.ENCODERS[e]['width'] == WIDTH[e] and bs_paths.ENCODERS[e]['art'] == e
        assert bs_paths.ENCODERS[e]['aff'] == (e,)
        assert e in bs_paths.NEW_ENCODERS and e in mt_paths.ENCODERS
        assert e in bs_stores.VIEWS and bs_stores.WIDTH[e] == WIDTH[e]
        assert bs_photo.VIEW[e] == e and bs_affection.VIEW[e] == e and e in bs_affection.ENCODERS
        assert bs_artret.ENC[e] == dict(view=e, r11_arm=None)
        assert rb_encoders.WIDTH[e] == WIDTH[e]
        assert mt_paths.DEFAULT_VIEW[e] == e and mt_paths.WIDTH[e] == WIDTH[e]
        assert mt_paths.MATCHED[e] == PP.MATCHED[e] and mt_paths.CSTAR_ROOT[e] == PP.OUT / 'cstar' / e
    assert bs_artclu.ENC['clip_b32'] == dict(store='clip_b32', root=PP.RB_PART2, cand='K3', matched='M3', frozen='F3')
    assert bs_artclu.ENC['clip_b16'] == dict(store='clip_b16', root=PP.EG_PART2, cand='K6', matched='M6', frozen='F6')
    assert bs_artclu.ENC['dinov2_large'] == dict(store='dinov2_large', root=PP.M5_ROOT / 'part2', cand=None,
                                                 matched='M5', frozen=None)


def test_the_old_encoders_keep_their_entries():
    assert bs_artclu.ENC['siglip2'] == dict(store='siglip2_so400m', root=bs_artclu.RB_PART2, cand='K2', matched='M2',
                                            frozen='F2')
    assert mt_paths.CSTAR_ROOT['qwen'] == PP.R10_OUT / 'full'
    assert mt_paths.CSTAR_ROOT['clip_l14'] == PP.R14_OUT / 'cstar/clip_l14'
    assert bs_paths.ENCODERS['clip_l14']['aff'] == ('clip_l14_valtest', 'clip_l14_train')
    assert mt_paths.ENCODERS[:4] == ('qwen', 'siglip2', 'siglip1', 'clip_l14')
    assert bs_artret.ENC['clip_l14'] == dict(view='clip_l14', r11_arm='clip')


def test_registering_again_and_reloading_change_nothing():
    before = snapshot()
    R.register()
    R.register()
    assert snapshot() == before
    importlib.reload(R)
    assert snapshot() == before
    assert not hasattr(getattr(bs_stores.folders, '_pp_original', None), '_pp_original')   # wrapped once, not nested
    assert not hasattr(getattr(mt_data.features, '_pp_original', None), '_pp_original')


def test_check_passes_and_names_a_missing_entry():
    assert R.check()['entries'] > 30
    saved = bs_artclu.ENC.pop('clip_b16')
    try:
        with pytest.raises(AssertionError, match='clip_b16'):
            R.check()
    finally:
        bs_artclu.ENC['clip_b16'] = saved
    saved = mt_paths.WIDTH['dinov2_large']
    mt_paths.WIDTH['dinov2_large'] = 1
    try:
        with pytest.raises(AssertionError, match='dinov2_large'):
            R.check()
    finally:
        mt_paths.WIDTH['dinov2_large'] = saved
    R.check()


# ------------------------------------------------------------------------------------------------ code paths
@pytest.fixture(scope='module')
def art_names():
    images = json.loads((PP.ART / 'clip_b32' / 'index.json').read_text())['images']
    rng = np.random.default_rng(0)
    return [images[i]['painting'] for i in rng.choice(len(images), size=100, replace=False)]


@pytest.mark.parametrize('enc', NEW)
def test_artelingo_code_paths_return_the_rows_of_pp_stores(enc, art_names):
    want = S.load('artelingo', enc, art_names)
    assert want.shape == (100, WIDTH[enc]) and want.dtype == np.float32
    assert np.array_equal(bs_artclu.legacy_loader(enc, art_names), want)              # 10-25's route (rb_encoders)
    assert np.array_equal(bs_artclu.store_loader(enc, art_names), want)               # r14's bs_stores route, patched
    import rt_data
    val = rt_data.names('val')
    got = bs_artret.frozen_embed(enc, 'val')                                          # r14's retrieval route, patched
    assert got.shape == (len(val), WIDTH[enc])
    assert np.array_equal(got[:100], S.load('artelingo', enc, val[:100]))


@pytest.mark.parametrize('enc', NEW)
def test_mmart_code_path_on_val_and_held_keys_returns_the_rows_of_pp_stores(enc):
    index = mt_data._index()
    keys = index[index.split == 'held'].key.tolist()[:50] + index[index.split == 'val'].key.tolist()[:50]
    keys = [keys[i] for i in np.random.default_rng(1).permutation(len(keys))]
    got = mt_data.features(mt_paths.DEFAULT_VIEW[enc], keys)                           # r15's route, patched
    assert np.array_equal(got, S.load('mmart', enc, keys))
    paintings = index.set_index('key').loc[keys].painting.tolist()                      # the same rows by painting
    assert np.array_equal(got, S.load('artelingo', enc, paintings))


def test_old_views_still_go_through_the_older_readers():
    index = mt_data._index()
    keys = index[index.split == 'new'].key.tolist()[:20]
    assert np.array_equal(mt_data.features('siglip2', keys), S.load('mmart', 'siglip2', keys))
    assert np.array_equal(bs_artclu.store_loader('siglip1', ['vincent-van-gogh_young-woman-knitting-1881']),
                          S.load('artelingo', 'siglip1', ['vincent-van-gogh_young-woman-knitting-1881']))
    assert 'siglip2' not in R.NEW and callable(bs_artret.frozen_embed._pp_original)


def test_affection_code_paths_read_a_new_store_from_pp_stores(tmp_path, monkeypatch):
    keys = [f'coco/img{i}.jpg' for i in range(12)]
    rows = np.random.default_rng(2).normal(size=(12, 512)).astype(np.float16)
    folder = tmp_path / 'clip_b32'
    folder.mkdir()
    np.save(folder / 'img.npy', rows)
    (folder / 'keys.json').write_text(json.dumps(keys))
    monkeypatch.setitem(S.NEW_STORE, ('affection', 'clip_b32'), folder)
    S._CACHE.clear()
    ask = [keys[i] for i in (9, 1, 5, 0)]
    want = S.load('affection', 'clip_b32', ask)
    assert np.array_equal(want, (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(rows[[9, 1, 5, 0]].astype(np.float32)))
    assert np.array_equal(bs_affection.frozen_embed('clip_b32', ask), want)             # r14 Affection, patched
    assert np.array_equal(bs_photo.images('affection', 'clip_b32', ask), want)          # X photo model's input
    raw = bs_stores.load('affection', 'clip_b32', ask)                                  # the table route (folders)
    assert np.array_equal(raw, rows[[9, 1, 5, 0]].astype(np.float32))


def test_the_other_stores_do_not_resolve_for_the_new_views_without_a_store(monkeypatch):
    with pytest.raises(AssertionError):
        bs_stores.folders('emoset', 'clip_b32')                                         # EmoSet is out of scope (D7)
    monkeypatch.setitem(S.NEW_STORE, ('affection', 'clip_b32'), PP.AFF / 'not_extracted_yet')
    with pytest.raises(AssertionError, match='missing'):
        bs_affection.frozen_embed('clip_b32', ['coco/x.jpg'])


def test_wrapped_functions_for_the_new_views_do_not_call_bs_stores(monkeypatch, art_names, tmp_path):
    """The four wrapped functions answer a new view from pp_stores: a poisoned bs_stores.load is never reached."""
    def poisoned(*a, **k):
        raise RuntimeError('bs_stores.load was called')
    monkeypatch.setattr(bs_stores, 'load', poisoned)
    want = S.load('artelingo', 'clip_b32', art_names)
    assert np.array_equal(bs_artclu.store_loader('clip_b32', art_names), want)
    assert bs_artret.frozen_embed('clip_b32', 'val').shape == (6152, 512)
    keys = [f'coco/img{i}.jpg' for i in range(4)]
    folder = tmp_path / 'f'
    folder.mkdir()
    np.save(folder / 'img.npy', np.random.default_rng(3).normal(size=(4, 512)).astype(np.float16))
    (folder / 'keys.json').write_text(json.dumps(keys))
    monkeypatch.setitem(S.NEW_STORE, ('affection', 'clip_b32'), folder)
    S._CACHE.clear()
    assert np.array_equal(bs_affection.frozen_embed('clip_b32', keys[::-1]), S.load('affection', 'clip_b32', keys[::-1]))
