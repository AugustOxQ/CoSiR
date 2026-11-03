"""Tests of pp_semart (r19 ticket 03): the SemArt set's rule, md5 step, order, labels, D14 subsets, inner pairs.

Synthetic frames test the logic; the `real` tests read sc_nearcopy.parquet, the CSVs, the store indexes and the image
files (gate (vi), spec §3.5)."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import pp_paths
import pp_semart as S


def sha(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------------------------- the rule
def frame():
    f32 = np.float32
    rows = [  # image_file, author, max_cos, tier, expected in the set
        ('a.jpg', 'REMBRANDT', f32(0.40), 'none', True),
        ('b.jpg', 'REMBRANDT', f32(0.9499), 'none', True),
        ('c.jpg', 'REMBRANDT', f32(0.95), 'none', False),              # strict: 0.95 itself is a near copy
        ('d.jpg', 'REMBRANDT', f32(0.99), 'none', False),
        ('e.jpg', 'SMITH, John', f32(0.5), 'exact', False),            # each name tier excludes
        ('f.jpg', 'SMITH, John', f32(0.5), 'given_surname', False),
        ('g.jpg', 'SMITH, John', f32(0.5), 'surname_only', False),
        ('h.jpg', 'UNKNOWN MASTER', f32(0.5), 'none', False),
        ('i.jpg', 'UNKNOWN PAINTER, ICON', f32(0.5), 'none', False),   # not "equals UNKNOWN MASTER"
        ('j.jpg', 'UNKNOWN', f32(0.5), 'none', False),
        ('k.jpg', 'Unknown master', f32(0.5), 'none', False),          # case-insensitive, as sc_check
        ('l.jpg', 'KNOWN UNKNOWN, Mr', f32(0.5), 'none', True),        # only a prefix counts
    ]
    return pd.DataFrame(rows, columns=['image_file', 'author', 'max_cos', 'artist_tier', 'want'])


def test_rule_selects_exactly_the_wanted_rows():
    f = frame()
    got = S.new_artist_mask(f.max_cos.to_numpy(np.float32), f.artist_tier.to_numpy(), f.author.to_numpy())
    assert got.dtype == bool
    assert got.tolist() == f.want.tolist()


def test_unknown_rule_is_a_prefix_not_equality():
    assert S.unknown_author('UNKNOWN PAINTER, ICON') and S.unknown_author('unknown master')
    assert not S.unknown_author('SMITH, UNKNOWN')


def test_recompute_tiers_uses_sc_check_tiers():
    train = {'peter-paul-rubens', 'titian'}
    got = S.recompute_tiers(['RUBENS, Peter Paul', 'TIZIANO Vecellio', 'ZZZZ, Nobody'], train)
    assert got == ['exact', 'exact', 'none']          # the second through sc_check's alias table


# ---------------------------------------------------------------------------------------------- order, md5
def test_sha_order_is_not_alphabetical_and_matches_hashlib():
    keys = [f'{i:05d}-x.jpg' for i in range(50)]
    got = S.sha_order(keys)
    assert got == sorted(keys, key=sha)
    assert got != sorted(keys)
    assert sorted(got) == keys


def test_sha_order_rejects_duplicates():
    with pytest.raises(AssertionError):
        S.sha_order(['a.jpg', 'a.jpg'])


def test_md5_step_keeps_first_in_sha_order_whatever_the_input_order():
    keys = [f'{i}-p.jpg' for i in range(30)]
    md5 = {k: f'u{i}' for i, k in enumerate(keys)}
    groups = [keys[1:4], keys[10:12]]                    # one group of three, one of two
    for g in groups:
        for k in g:
            md5[k] = 'dup' + g[0]
    kept, recorded = S.md5_step(list(reversed(keys)), md5)
    dropped = {k for g in groups for k in g if k != min(g, key=sha)}
    assert kept == [k for k in sorted(keys, key=sha) if k not in dropped]
    assert len(kept) == 30 - 2 - 1
    assert kept == sorted(kept, key=sha)
    by_first = {r['keep']: r for r in recorded}
    for g in groups:
        first = min(g, key=sha)
        assert by_first[first]['dropped'] == sorted((k for k in g if k != first), key=sha)
    assert len(recorded) == 2


def test_md5_step_without_duplicates_changes_nothing():
    keys = ['a', 'b', 'c']
    kept, recorded = S.md5_step(keys, {'a': '1', 'b': '2', 'c': '3'})
    assert kept == S.sha_order(keys) and recorded == []


def test_md5_step_needs_an_md5_for_every_key():
    with pytest.raises(AssertionError):
        S.md5_step(['a', 'b'], {'a': '1'})


# ---------------------------------------------------------------------------------------------- D14 subsets
def test_split_close_far_threshold_inclusive_and_sha_order():
    keys = [f'{i}-q.jpg' for i in range(40)]
    cos = {k: np.float32(0.5) for k in keys}
    cos[keys[3]] = np.float32(0.93)                      # exactly at the cut: close
    cos[keys[4]] = np.float32(0.9299)
    cos[keys[5]] = np.float32(0.949)
    close, far = S.split_close(keys, cos)
    assert set(close) == {keys[3], keys[5]}
    assert set(far) == set(keys) - set(close)
    assert close == sorted(close, key=sha) and far == sorted(far, key=sha)


# ---------------------------------------------------------------------------------------------- labels
def test_type_labels_index_in_semart_types_and_ignore_the_key_text():
    type_of = {'religious-1.jpg': 'landscape', 'landscape-2.jpg': 'study', 'x.jpg': 'genre'}
    got = S.type_labels(['religious-1.jpg', 'landscape-2.jpg', 'x.jpg'], type_of)
    assert got.dtype == np.int64
    assert got.tolist() == [pp_paths.SEMART_TYPES.index(t) for t in ('landscape', 'study', 'genre')]


def test_type_labels_reject_unknown_key_or_type():
    with pytest.raises(AssertionError):
        S.type_labels(['nope.jpg'], {'x.jpg': 'genre'})
    with pytest.raises(AssertionError):
        S.type_labels(['x.jpg'], {'x.jpg': 'sculpture'})


def test_check_types_flags_a_disagreement():
    S.check_types(['a', 'b'], {'a': 'genre', 'b': 'study'}, {'a': 'genre', 'b': 'study'})
    with pytest.raises(AssertionError):
        S.check_types(['a', 'b'], {'a': 'genre', 'b': 'study'}, {'a': 'genre', 'b': 'other'})


# ---------------------------------------------------------------------------------------------- inner pairs
def test_inner_pairs_count_upper_triangle_at_cut():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 16)).astype(np.float32)
    x[1] = x[0] + 1e-3 * rng.normal(size=16)             # a pair
    x[4] = x[3] + 1e-3 * rng.normal(size=16)             # a pair
    x[5] = x[3] + 1e-3 * rng.normal(size=16)             # a triangle with 3, 4
    n_pairs, members = S.inner_pairs(x, 0.95, block=2)
    assert (n_pairs, members) == (1 + 3, 5)
    n_pairs, members = S.inner_pairs(x, 0.95, block=4096)
    assert (n_pairs, members) == (4, 5)


def test_inner_pairs_none_when_orthogonal():
    assert S.inner_pairs(np.eye(5, dtype=np.float32), 0.95) == (0, 0)


# ---------------------------------------------------------------------------------------------- real data
@pytest.fixture(scope='module')
def gate():
    return S.gate_vi()


def test_real_gate_vi_passes(gate):
    sc = json.loads(pp_paths.SC_CHECK.read_text())
    assert gate['pass'] is True
    assert gate['paintings'] == 11162 and gate['artists'] == 2717
    assert gate['types'] == sc['new_artist_types']
    assert gate['tiers_match_parquet'] is True
    assert gate['n_set'] == 11159 == pp_paths.N_SEMART
    assert len(gate['md5_groups']) == 3
    pairs = {frozenset([g['keep'], *g['dropped']]) for g in gate['md5_groups']}
    assert pairs == {frozenset(['28700-flowers3.jpg', '28701-flowers3.jpg']),
                     frozenset(['22768-blindman.jpg', '22767-blindman.jpg']),
                     frozenset(['03597-1trinity.jpg', '03598-1trinity.jpg'])}
    assert {g['keep'] for g in gate['md5_groups']} == {'28700-flowers3.jpg', '22768-blindman.jpg',
                                                       '03597-1trinity.jpg'}


def test_real_keys_sets_and_order():
    g, s, c, f = (S.keys(n) for n in ('gate', 'set', 'close', 'far'))
    assert len(g) == 11162 and len(s) == 11159 and len(set(s)) == 11159
    assert set(s) < set(g)
    for ks in (g, s, c, f):
        assert ks == sorted(ks, key=sha)
    assert set(c) | set(f) == set(s) and not set(c) & set(f)
    assert S.keys('gate') == g and S.keys('gate') is not S.keys('gate')     # a copy: callers cannot corrupt it
    with pytest.raises(AssertionError):
        S.keys('other')


def test_real_close_count_before_md5_is_217():
    rec = S.record()
    assert rec['close_before_md5'] == 217
    assert rec['close'] + rec['far'] == 11159
    assert rec['close'] == len(S.keys('close'))


def test_real_labels_match_csv_and_counts():
    s = S.keys('set')
    y = S.labels(s)
    assert y.dtype == np.int64 and y.shape == (11159,) and y.min() >= 0 and y.max() < 10
    assert set(np.unique(y)) == set(range(10))                        # all 10 classes present (D10)
    assert S.labels(s[:5]).tolist() == y[:5].tolist()
    csv = pd.concat([pd.read_csv(pp_paths.SEMART_DATA / f'semart_{p}.csv', sep='\t', encoding='latin-1')
                     for p in ('train', 'val', 'test')]).set_index('IMAGE_FILE')
    assert [pp_paths.SEMART_TYPES[i] for i in y[:200]] == csv.loc[s[:200], 'TYPE'].tolist()


def test_real_every_key_is_in_every_existing_store_index():
    rec = S.record()
    assert rec['stores_checked'], 'no SemArt store index found'
    assert {'qwen3vl_emb_2b', 'siglip2_so400m', 'clip_b32'} <= set(rec['stores_checked'])


def test_real_record_is_json_and_has_inner_pairs():
    rec = S.record()
    json.dumps(rec)
    assert rec['inner_pairs_ge_0.95']['pairs'] >= 0
    assert rec['n_gate'] == 11162 and rec['n_set'] == 11159
    assert rec['md5_groups'] and rec['gate']['pass'] is True
