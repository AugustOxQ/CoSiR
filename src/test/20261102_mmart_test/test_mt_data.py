"""Tests of mt_data (ticket 01): cut order, thresholds, D10, sha256 order, labels, artist rule, feature routing, the
gate checkers; plus one real-data check of the two cut counts."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_data as D  # noqa: E402
import mt_paths as P  # noqa: E402


def sha(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def frame(rows):
    """Synthetic index and near-copy table from rows of (key, md5, cos, exact, split)."""
    idx = pd.DataFrame(dict(
        key=[r[0] for r in rows], painting=[Path(r[0]).stem for r in rows], md5=[r[1] for r in rows],
        split=[r[4] for r in rows], artelingo=['none' if r[4] == 'new' else 'trainval' for r in rows],
        exact_dup_trainval=[r[3] for r in rows], emotion=[''] * len(rows), style=['baroque'] * len(rows),
        genre=[-1] * len(rows)))
    new = [r for r in rows if r[4] == 'new']
    near = pd.DataFrame(dict(key=[r[0] for r in new], max_cos=np.asarray([r[2] for r in new], np.float32),
                             nearest_painting=['x'] * len(new)))
    return idx, near


def sha_first_is_lexicographically_last(prefix):
    """(first, second): the sha256-first key is the lexicographically LAST one."""
    for i in range(500):
        a, b = f'{prefix}/a{i}.jpg', f'{prefix}/b{i}.jpg'
        if sha(b) < sha(a):
            return b, a
    raise AssertionError


# ------------------------------------------------------------------------------------------------ order
def test_order_is_sha256_of_the_full_key():
    keys = [f'Baroque/artist-{i}_x.jpg' for i in range(30)] + [f'Cubism/artist-{i}_x.jpg' for i in range(30)]
    got = D.order(keys)
    assert got == sorted(keys, key=sha)
    assert got != sorted(keys)                       # not the lexicographic order (which groups style folders)
    assert D.order(reversed(keys)) == got            # independent of the input order


def test_order_refuses_duplicate_keys():
    with pytest.raises(AssertionError):
        D.order(['a/x.jpg', 'a/x.jpg'])


# ------------------------------------------------------------------------------------------------ cut
def test_cut_order_exact_below_threshold_still_leaves():
    idx, near = frame([('S/a_1.jpg', 'm1', 0.30, True, 'new'), ('S/a_2.jpg', 'm2', 0.30, False, 'new')])
    c = D.cut(0.95, index=idx, nearcopy=near)
    assert c['exact'] == ['S/a_1.jpg'] and c['near'] == [] and c['md5'] == [] and c['kept'] == ['S/a_2.jpg']


def test_cut_exact_copy_above_threshold_counts_as_exact_not_near():
    idx, near = frame([('S/a_1.jpg', 'm1', 1.0, True, 'new'), ('S/a_2.jpg', 'm2', 0.97, False, 'new')])
    c = D.cut(0.95, index=idx, nearcopy=near)
    assert c['exact'] == ['S/a_1.jpg'] and c['near'] == ['S/a_2.jpg'] and c['kept'] == []


def test_cut_near_copy_leaves_before_its_md5_group_is_counted():
    first, second = sha_first_is_lexicographically_last('S')
    # `first` (sha-first of the group) is a near copy: it leaves at the near step; `second` is then alone in its group
    idx, near = frame([(first, 'g', 0.99, False, 'new'), (second, 'g', 0.40, False, 'new')])
    c = D.cut(0.95, index=idx, nearcopy=near)
    assert c['near'] == [first] and c['md5'] == [] and c['kept'] == [second]


def test_cut_exact_copy_leaves_before_its_md5_group_is_counted():
    first, second = sha_first_is_lexicographically_last('S')
    idx, near = frame([(first, 'g', 0.40, True, 'new'), (second, 'g', 0.40, False, 'new')])
    c = D.cut(0.95, index=idx, nearcopy=near)
    assert c['exact'] == [first] and c['md5'] == [] and c['kept'] == [second]


def test_cut_inclusive_threshold():
    idx, near = frame([('S/a_1.jpg', 'm1', 0.75, False, 'new'), ('S/a_2.jpg', 'm2', 0.7499, False, 'new')])
    c = D.cut(0.75, index=idx, nearcopy=near)       # 0.75 is exact in float32
    assert c['near'] == ['S/a_1.jpg'] and c['kept'] == ['S/a_2.jpg']


def test_cut_d10_keeps_the_sha256_first_key_of_each_group():
    first, second = sha_first_is_lexicographically_last('S')
    idx, near = frame([(second, 'g', 0.1, False, 'new'), (first, 'g', 0.1, False, 'new'),
                       ('S/c_1.jpg', 'h', 0.1, False, 'new'), ('S/c_2.jpg', 'h', 0.1, False, 'new')])
    c = D.cut(0.95, index=idx, nearcopy=near)
    assert first in c['kept'] and second in c['md5']
    keep_h = min(['S/c_1.jpg', 'S/c_2.jpg'], key=sha)
    assert keep_h in c['kept'] and len(c['md5']) == 2
    assert c['kept'] == D.order(c['kept'])


def test_cut_group_of_three_keeps_one():
    keys = [f'S/t{i}_x.jpg' for i in range(3)]
    idx, near = frame([(k, 'g', 0.1, False, 'new') for k in keys])
    c = D.cut(0.95, index=idx, nearcopy=near)
    assert c['kept'] == [min(keys, key=sha)] and len(c['md5']) == 2


def test_cut_ignores_rows_outside_new():
    idx, near = frame([('S/a_1.jpg', 'm1', 0.1, False, 'new'), ('S/b_1.jpg', 'm1', 0.1, False, 'val'),
                       ('S/c_1.jpg', 'm1', 0.1, False, 'train')])
    c = D.cut(0.95, index=idx, nearcopy=near)
    assert c['kept'] == ['S/a_1.jpg'] and c['md5'] == []        # an md5 shared with a non-new row is not a D10 group


def test_cut_nearcopy_must_cover_every_new_key_once_and_be_finite():
    idx, near = frame([('S/a_1.jpg', 'm1', 0.1, False, 'new'), ('S/a_2.jpg', 'm2', 0.1, False, 'new')])
    with pytest.raises(AssertionError):
        D.cut(0.95, index=idx, nearcopy=near.iloc[:1])
    with pytest.raises(AssertionError):
        D.cut(0.95, index=idx, nearcopy=pd.concat([near, near.iloc[:1]]))
    bad = near.copy()
    bad.loc[0, 'max_cos'] = np.nan
    with pytest.raises(AssertionError):
        D.cut(0.95, index=idx, nearcopy=bad)
    extra = pd.concat([near, near.iloc[:1].assign(key='S/zz_9.jpg')])
    with pytest.raises(AssertionError):
        D.cut(0.95, index=idx, nearcopy=extra)


# ------------------------------------------------------------------------------------------------ labels
def label_frame():
    return pd.DataFrame(dict(
        key=['Baroque/a_1.jpg', 'Baroque/a_2.jpg', 'Cubism/b_1.jpg'], painting=['a_1', 'a_2', 'b_1'],
        emotion=['awe', '', 'something else'], style=['cubism', 'baroque', 'ukiyo_e'], genre=[4, -1, 9],
        split=['val'] * 3, artelingo=['trainval'] * 3, md5=['1', '2', '3'], exact_dup_trainval=[False] * 3))


def test_labels_missing_is_minus_one_and_class_order_follows_paths():
    idx = label_frame()
    keys = ['Cubism/b_1.jpg', 'Baroque/a_2.jpg', 'Baroque/a_1.jpg']
    e = D.labels(keys, 'emotion', index=idx)
    assert e.dtype == np.int64 and e.tolist() == [8, -1, P.CLASSES['emotion'].index('awe')]
    assert D.labels(keys, 'genre', index=idx).tolist() == [9, -1, 4]
    idx.loc[1, 'genre'] = -3
    assert D.labels(keys, 'genre', index=idx).tolist() == [9, -1, 4]
    s = D.labels(keys, 'style', index=idx)
    assert s.tolist() == [19, P.CLASSES['style'].index('baroque'), P.CLASSES['style'].index('cubism')]


def test_labels_come_from_the_column_not_from_the_key():
    idx = label_frame()                                   # key says Baroque, the style column says cubism
    assert D.labels(['Baroque/a_1.jpg'], 'style', index=idx).tolist() == [P.CLASSES['style'].index('cubism')]


def test_labels_missing_key_and_unknown_class_raise():
    idx = label_frame()
    with pytest.raises(AssertionError):
        D.labels(['Baroque/zzz.jpg'], 'emotion', index=idx)
    idx.loc[0, 'emotion'] = 'joy'
    with pytest.raises(AssertionError):
        D.labels(['Baroque/a_1.jpg'], 'emotion', index=idx)
    with pytest.raises(AssertionError):
        D.labels(['Baroque/a_1.jpg'], 'colour', index=label_frame())


# ------------------------------------------------------------------------------------------------ artist subset
def test_artist_is_the_file_name_before_the_first_underscore():
    assert D.artist_of('Baroque/rembrandt_the-night-watch.jpg') == 'rembrandt'
    assert D.artist_of('Abstract_Expressionism/a.y.-jackson_x_y.jpg') == 'a.y.-jackson'   # folder underscores ignored
    assert D.artist_of('Baroque/noartist.jpg') == 'noartist'
    assert D.artist_of('rembrandt_x') == 'rembrandt'                                       # a painting name too


def test_unseen_artist_keeps_test_keys_without_a_train_painting_by_the_artist():
    test = ['B/rembrandt_a.jpg', 'B/vermeer_b.jpg', 'B/hals_c.jpg', 'B/vermeer_d.jpg']
    train = ['rembrandt_night', 'hals-x_y']                # 'hals-x' is another artist than 'hals'
    assert D.unseen(test, train) == ['B/vermeer_b.jpg', 'B/hals_c.jpg', 'B/vermeer_d.jpg']


def test_artelingo_name_is_the_painting_stem_and_refuses_new_keys():
    idx, _ = frame([('S/a_1.jpg', 'm1', 0.1, False, 'new'), ('S/b_1.jpg', 'm2', 0.1, False, 'val')])
    assert D.artelingo_name('S/b_1.jpg', index=idx) == 'b_1'
    with pytest.raises(AssertionError):
        D.artelingo_name('S/a_1.jpg', index=idx)


# ------------------------------------------------------------------------------------------------ features
ASP = 'qwen3vl_emb_2b_aspect_prompts/emotion_focus'


def _store(path, names, x, how):
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'img.npy', x.astype(np.float16))
    if how == 'keys':
        (path / 'keys.json').write_text(json.dumps(names))
    elif how == 'index':
        (path / 'index.json').write_text(json.dumps(dict(images=[dict(painting=n) for n in names])))
    else:
        np.savez(path / 'ids.npz', painting_names=np.asarray(names))


def make_world(tmp_path):
    """new key n_1, val key v_1, held key h_1; each store has its own random rows."""
    idx = pd.DataFrame(dict(
        key=['S/n_1.jpg', 'S/v_1.jpg', 'S/h_1.jpg'], painting=['n_1', 'v_1', 'h_1'],
        artelingo=['none', 'trainval', 'held'], split=['new', 'val', 'held'], md5=['a', 'b', 'c'],
        exact_dup_trainval=[False] * 3, emotion=[''] * 3, style=['baroque'] * 3, genre=[0] * 3))
    root, art = tmp_path / 'mmart', tmp_path / 'art'
    rng = np.random.default_rng(0)
    r = lambda n: rng.normal(size=(n, 2048)) * 3 + 1          # noqa: E731  (not unit: normalisation must happen)
    _store(root / 'qwen3vl_emb_2b', ['S/n_1.jpg'], r(1), 'keys')
    _store(art / 'qwen3vl_emb_2b', ['other', 'h_1', 'v_1'], r(3), 'index')
    _store(root / ASP, ['S/n_1.jpg'], r(1), 'keys')
    _store(art / (ASP + '_s2img'), ['other', 'v_1'], r(2), 'ids')
    _store(art / (ASP + '_held'), ['h_1', 'other'], r(2), 'ids')
    return idx, root, art


def raw(path):
    return np.load(path / 'img.npy').astype(np.float32)


def unit(v):
    return v / np.linalg.norm(v)


def test_features_default_view_joins_by_key_and_normalises(tmp_path):
    idx, root, art = make_world(tmp_path)
    kw = dict(index=idx, root=root, art_root=art)
    x = D.features('qwen_default', ['S/h_1.jpg', 'S/n_1.jpg', 'S/v_1.jpg'], **kw)
    assert x.dtype == np.float32 and x.shape == (3, 2048)
    a, m = raw(art / 'qwen3vl_emb_2b'), raw(root / 'qwen3vl_emb_2b')
    assert np.allclose(x[0], unit(a[1]), atol=1e-6) and np.allclose(x[2], unit(a[2]), atol=1e-6)   # by name, not row
    assert np.allclose(x[1], unit(m[0]), atol=1e-6)
    assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)


def test_features_teacher_view_val_reads_s2img_held_reads_held_new_reads_mmart(tmp_path):
    idx, root, art = make_world(tmp_path)
    t = D.features('emotion_focus', ['S/v_1.jpg', 'S/h_1.jpg', 'S/n_1.jpg'], index=idx, root=root, art_root=art)
    assert np.allclose(t[0], unit(raw(art / (ASP + '_s2img'))[1]), atol=1e-6)
    assert np.allclose(t[1], unit(raw(art / (ASP + '_held'))[0]), atol=1e-6)
    assert np.allclose(t[2], unit(raw(root / ASP)[0]), atol=1e-6)


def test_features_missing_key_train_row_and_bad_view_raise(tmp_path):
    idx, root, art = make_world(tmp_path)
    kw = dict(index=idx, root=root, art_root=art)
    with pytest.raises(AssertionError):
        D.features('qwen_default', ['S/nope.jpg'], **kw)
    idx2 = idx.copy()
    idx2.loc[1, 'split'] = 'train'
    with pytest.raises(AssertionError):
        D.features('qwen_default', ['S/v_1.jpg'], **dict(kw, index=idx2))
    with pytest.raises(AssertionError):
        D.features('not_a_view', ['S/v_1.jpg'], **kw)
    with pytest.raises(AssertionError):
        D.features('emotion_focus', ['S/v_1.jpg', 'S/v_1.jpg'], **kw)          # duplicate keys


def test_features_val_key_absent_from_s2img_store_raises(tmp_path):
    idx, root, art = make_world(tmp_path)
    _store(art / (ASP + '_s2img'), ['other'], np.ones((1, 2048)), 'ids')
    with pytest.raises(AssertionError):
        D.features('emotion_focus', ['S/v_1.jpg'], index=idx, root=root, art_root=art)


def test_features_nonfinite_row_raises(tmp_path):
    idx, root, art = make_world(tmp_path)
    path = root / 'qwen3vl_emb_2b' / 'img.npy'
    x = np.load(path)
    x[0, 0] = np.nan
    np.save(path, x)
    with pytest.raises(AssertionError):
        D.features('qwen_default', ['S/n_1.jpg'], index=idx, root=root, art_root=art)


# ------------------------------------------------------------------------------------------------ gate checkers
def parity_doc(**over):
    doc = {v: dict(n=200, min_cosine=0.9995, mean_cosine=0.9999, **{'pass': True}) for v in P.VIEWS}
    for v, m in over.items():
        doc[v]['min_cosine'] = m
    return doc


def test_parity_check_needs_every_view_at_0999():
    assert D.check_parity(parity_doc())['pass']
    assert not D.check_parity(parity_doc(siglip1=0.9985))['pass']               # one view below the bar
    doc = parity_doc()
    del doc['clip_l14']
    assert not D.check_parity(doc)['pass']                                      # a view missing
    doc = parity_doc(siglip1=0.9985)
    doc['siglip1']['pass'] = True                                               # the stored flag is not trusted
    assert not D.check_parity(doc)['pass']
    assert D.check_parity(parity_doc(siglip1=0.999))['pass']                    # the bar is inclusive


def test_store_check_needs_14_of_14():
    good = dict(view='v', set='new', n=1, ok=True)
    assert D.check_stores(dict(ok=True, stores=[good] * 14))['pass']
    assert not D.check_stores(dict(ok=True, stores=[good] * 13))['pass']
    assert not D.check_stores(dict(ok=True, stores=[good] * 13 + [dict(view='w', ok=False)]))['pass']
    assert not D.check_stores(dict(ok=False, stores=[good] * 14))['pass']


def test_nearcopy_check_counts_and_reference():
    idx, near = frame([('S/a_1.jpg', 'm1', 0.99, False, 'new'), ('S/a_2.jpg', 'm2', 0.5, False, 'new')])
    ok = D.check_nearcopy(idx, near, reference=7, expect_reference=7)
    assert ok['pass'] and ok['n'] == 2
    assert not D.check_nearcopy(idx, near, reference=6, expect_reference=7)['pass']
    assert not D.check_nearcopy(idx, near.iloc[:1], reference=7, expect_reference=7)['pass']
    bad = near.copy()
    bad.loc[0, 'max_cos'] = np.inf
    assert not D.check_nearcopy(idx, bad, reference=7, expect_reference=7)['pass']


def test_unit_check_flags_nonunit_and_nonfinite_rows():
    x = np.eye(4, dtype=np.float32)
    assert D.check_unit(x)['pass']
    assert D.check_unit(x * 1.0005)['pass']
    assert not D.check_unit(x * 1.01)['pass']
    y = x.copy()
    y[0, 0] = np.nan
    assert not D.check_unit(y)['pass']


# ------------------------------------------------------------------------------------------------ real data
def test_real_cut_counts():
    assert len(D.cut(P.CUT_TEST)['kept']) == P.N_TEST == 16384
    assert len(D.cut(P.CUT_FAR)['kept']) == P.N_FAR == 10403
    assert len(D.cut(P.CUT_TEST)['exact']) == P.N_EXACT == 34
