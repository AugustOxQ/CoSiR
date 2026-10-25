"""Tests of rb_data on small fixtures; the real-data checks run with RB_SLOW=1."""
import hashlib
import json
import os

import numpy as np
import pytest

import rb_paths
import rb_data

slow = pytest.mark.skipif(not os.environ.get('RB_SLOW'), reason='real stores: set RB_SLOW=1')


def sha(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


# ------------------------------------------------------------------ the split
def test_p_split_is_sorted_train_val_minus_sample():
    train, val = ['d', 'a', 'f', 'c'], ['b', 'e']
    names, counts = rb_data.p_split(train, val, sample_names=['a', 'e'])
    assert names == ['b', 'c', 'd', 'f']
    assert counts == {'train': 3, 'val': 1}


def test_p_split_rejects_sample_outside_train_val():
    with pytest.raises(AssertionError):
        rb_data.p_split(['a', 'b'], ['c'], sample_names=['a', 'z'])


def test_p_split_rejects_overlapping_roles_and_duplicates():
    with pytest.raises(AssertionError):
        rb_data.p_split(['a', 'b'], ['b'], sample_names=[])
    with pytest.raises(AssertionError):
        rb_data.p_split(['a', 'a'], ['c'], sample_names=[])


def test_hash_is_newline_joined_utf8_without_trailing_newline():
    names = ['a', 'bé', 'c']
    assert rb_data.names_sha256(names) == hashlib.sha256('a\nbé\nc'.encode('utf-8')).hexdigest()
    assert rb_data.names_sha256(names) != sha(names + [''])


def test_p_json_roundtrip_and_tamper(tmp_path):
    names = ['a', 'b', 'c']
    path = rb_data.write_p_json(tmp_path, names, {'train': 2, 'val': 1})
    assert path == tmp_path / 'part1' / 'P.json'
    value = json.loads(path.read_text())
    assert set(value) == {'names', 'sha256', 'counts'} and value['names'] == names
    assert value['sha256'] == sha(names) and value['counts'] == {'train': 2, 'val': 1}
    assert rb_data.read_p_json(tmp_path, expect=names) == names
    with pytest.raises(AssertionError):
        rb_data.read_p_json(tmp_path, expect=['a', 'b', 'd'])
    value['names'] = ['a', 'c', 'b']
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        rb_data.read_p_json(tmp_path, expect=None)


def test_p_json_is_rewritten_only_when_identical(tmp_path):
    rb_data.write_p_json(tmp_path, ['a', 'b'], {'train': 1, 'val': 1})
    rb_data.write_p_json(tmp_path, ['a', 'b'], {'train': 1, 'val': 1})
    with pytest.raises(AssertionError):
        rb_data.write_p_json(tmp_path, ['a', 'c'], {'train': 1, 'val': 1})


# ------------------------------------------------------------------ name joins
def test_select_by_name_follows_wanted_order_not_store_order():
    store_names = np.array(['c', 'a', 'b', 'd'])
    x = np.eye(4, dtype=np.float32)
    out = rb_data.select_by_name(store_names, x, ['b', 'd', 'a'])
    assert np.array_equal(out, x[[2, 3, 1]])
    assert out.dtype == np.float32


def test_select_by_name_rejects_missing_and_duplicates():
    with pytest.raises(AssertionError):
        rb_data.select_by_name(np.array(['a', 'b']), np.eye(2, dtype=np.float32), ['a', 'z'])
    with pytest.raises(AssertionError):
        rb_data.select_by_name(np.array(['a', 'a']), np.eye(2, dtype=np.float32), ['a'])


def test_unit_norm_check():
    good = np.array([[1, 0], [0.6, 0.8]], dtype=np.float32)
    rb_data.check_unit(good, 2)
    with pytest.raises(AssertionError):
        rb_data.check_unit(good * 1.01, 2)
    with pytest.raises(AssertionError):
        rb_data.check_unit(good.astype(np.float64), 2)
    with pytest.raises(AssertionError):
        rb_data.check_unit(good, 3)


# ------------------------------------------------------------------ labels
def fixture_split():
    """Four paintings, captions deliberately out of painting order. Emotion codes 0..8; -1 genre = no WikiArt label."""
    names = np.array(['p0', 'p1', 'p2', 'p3'])
    pid = np.array([2, 0, 1, 0, 0, 2, 1, 3, 3])
    emo = np.array([8, 5, 1, 5, 2, 8, 1, 4, 3])      # p0: 5,5,2 -> 5; p1: 1,1 -> 1; p2: 8,8 -> 8; p3: 4,3 tie -> 3
    sty = np.array([7, 3, 4, 3, 3, 7, 4, 9, 9])
    gen = np.array([-1, 2, 0, 2, 2, -1, 0, 5, 5])
    return dict(all_painting_ids=names, pid=pid, first=np.array([1, 0, 2, 7]), emo=emo, sty=sty, gen=gen)


def test_labels_plurality_ties_masking_and_name_order():
    s = fixture_split()
    out = rb_data.labels_by_name(s, ['p3', 'p0', 'p2', 'p1'])
    assert set(out) == {'emotion', 'style', 'genre'}
    assert out['emotion'].tolist() == [3, 5, 8, 1]       # p3 tie 3 vs 4 -> smallest
    assert out['style'].tolist() == [9, 3, 7, 4]
    assert out['genre'].tolist() == [5, 2, -1, 0]        # p2 has no genre -> -1
    assert all(v.dtype == np.int64 for v in out.values())


def test_labels_something_else_counts_in_plurality():
    s = fixture_split()
    s['emo'] = np.array([8, 8, 1, 8, 2, 8, 1, 4, 3])    # p0: 8,8,2 -> 8 (code 8 = something else, a real class)
    assert rb_data.labels_by_name(s, ['p0'])['emotion'].tolist() == [8]


def test_style_and_genre_constant_per_painting_so_caption_order_cannot_matter():
    s = fixture_split()
    rb_data.check_constant_per_painting(s)
    # the label does not move when the captions are reordered
    perm = np.array([8, 3, 0, 5, 1, 7, 2, 6, 4])
    shuffled = dict(s, pid=s['pid'][perm], emo=s['emo'][perm], sty=s['sty'][perm], gen=s['gen'][perm])
    names = ['p0', 'p1', 'p2', 'p3']
    a, b = rb_data.labels_by_name(s, names), rb_data.labels_by_name(shuffled, names)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    bad = dict(s, sty=s['sty'].copy())
    bad['sty'][0] = 0
    with pytest.raises(AssertionError):
        rb_data.check_constant_per_painting(bad)
    bad = dict(s, gen=s['gen'].copy())
    bad['gen'][1] = 7
    with pytest.raises(AssertionError):
        rb_data.check_constant_per_painting(bad)


def test_labels_reject_unknown_name():
    with pytest.raises(AssertionError):
        rb_data.labels_by_name(fixture_split(), ['p0', 'nope'])


def test_genre_label_count_counts_labelled_paintings():
    labels = {'emotion': np.array([1, 2, 3]), 'style': np.array([1, 2, 3]), 'genre': np.array([4, -1, 0])}
    assert rb_data.count_labelled(labels, 'genre') == 2
    assert rb_data.count_labelled(labels, 'emotion') == 3


# ------------------------------------------------------------------ old 5,000 and fit/stop
@slow
def test_real_roles_partition_the_sample_and_match_the_trained_split():
    import itp_data
    roles = rb_data.sample_roles()
    assert [len(roles[k]) for k in ('old5000', 'fit', 'stop')] == [5000, 13400, 1489]
    union = np.concatenate([roles['old5000'], roles['fit'], roles['stop']])
    assert len(np.unique(union)) == 19889 == len(union)
    s = rb_data._sample()
    _, fit, dev = itp_data.split(s)
    assert np.array_equal(roles['fit'], fit) and np.array_equal(roles['stop'], dev)
    assert np.array_equal(roles['old5000'], s['subset_pid'])
    names = rb_data.sample_names()
    assert rb_data.old5000_names() == [names[i] for i in roles['old5000']]
    assert sha(rb_data.old5000_names()) == rb_data.names_sha256(rb_data.old5000_names())


@slow
def test_real_old5000_labels_equal_ab_score_labels():
    import ab_score
    s = ab_score.load_sample()
    ref = ab_score.label_vectors(s, s['subset_pid'])
    mine = rb_data.old5000_labels()
    for a in rb_paths.RB_ASPECTS:
        assert np.array_equal(mine[a], ref[a]), a
    assert int((mine['genre'] >= 0).sum()) == 4053     # the spec's count on the old 5,000


# ------------------------------------------------------------------ real data
@slow
def test_real_p_counts_and_hash_hold():
    names = rb_data.p_names()
    assert len(names) == 29232 == len(set(names)) and names == sorted(names)
    assert rb_data.p_counts() == {'train': 23080, 'val': 6152}
    assert rb_data.p_sha256() == sha(names)
    sample = rb_data.sample_names()
    assert len(sample) == 19889 and not set(sample) & set(names)


@slow
def test_real_labels_agree_with_the_ct_helper_and_are_constant_per_painting():
    import ct_held
    names = rb_data.p_names()
    mine = rb_data.p_labels()
    theirs = ct_held.painting_labels(names)      # all annotation captions, annotation order
    for a in rb_paths.RB_ASPECTS:
        assert mine[a].shape == (29232,) and mine[a].dtype == np.int64
        assert np.array_equal(mine[a], theirs[a]), a
    rb_data.check_constant_per_painting(rb_data._full())
    assert rb_data.genre_label_count() == int((mine['genre'] >= 0).sum()) > 0


@slow
def test_real_features_are_unit_float32_and_name_joined():
    n = 29232
    for x in (rb_data.p_default_images(), rb_data.p_default_captions(),
              *[rb_data.p_instructed_images(k) for k in rb_data.INSTRUCTED_KEYS]):
        assert x.shape == (n, 2048) and x.dtype == np.float32
        assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-3


@slow
def test_real_p_instructed_rows_come_from_the_s2_stores_stage_2_read():
    import ct2_data
    names = rb_data.p_names()
    for key in rb_data.INSTRUCTED_KEYS:
        store, raw = ct2_data._image_store(ct2_data.ASPECT / f'{key}_s2img')
        assert set(names) == set(store.tolist()), key           # P is exactly the s2img store's paintings
        want = rb_data.p_instructed_images(key)
        got = rb_data.select_by_name(store, np.asarray(raw, dtype=np.float32), names)
        got = got / np.linalg.norm(got, axis=1, keepdims=True)
        assert np.abs(np.sum(got * want, axis=1) - 1).max() < 1e-4, key
