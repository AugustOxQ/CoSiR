"""Tests of ns_data (r17 ticket 03). Pure-logic tests use fakes; the others read the real stores (CPU, seconds each).

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_new_scores/test_ns_data.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ns_data  # noqa: E402
import rt_data  # noqa: E402
import mt_data  # noqa: E402
import ac_crl  # noqa: E402
import rt_crl  # noqa: E402

SQRT3 = math.sqrt(3.0)


def unit_rows(n, d, seed):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------ keys and labels
def test_keys_artelingo_sorted_and_counts():
    for split in ('held', 'val'):
        k = ns_data.keys('artelingo', split)
        assert k == sorted(k) and len(k) == len(set(k)) == ns_paths.N_ROWS[('artelingo', split)]
        assert k == rt_data.names(split)


def test_keys_mmart_sha256_order_and_counts():
    for split in ('test', 'val'):
        k = ns_data.keys('mmart', split)
        assert len(k) == len(set(k)) == ns_paths.N_ROWS[('mmart', split)]
        assert k == mt_data.order(k) and k != sorted(k)         # sha256 order, not sorted order
    assert ns_data.keys('mmart', 'test') == mt_data.keys('test')


def test_keys_refuse_unknown_set_or_split():
    for bad in (('artelingo', 'test'), ('mmart', 'held'), ('affection', 'test'), ('mmart', 'train')):
        with pytest.raises(AssertionError):
            ns_data.keys(*bad)


def test_keys_returns_a_copy():
    k = ns_data.keys('artelingo', 'val')
    k.reverse()
    assert ns_data.keys('artelingo', 'val') == sorted(ns_data.keys('artelingo', 'val'))


LABELLED = {  # inventory section a: labelled rows (row total minus unlabelled)
    ('artelingo', 'held', 'emotion'): 12281, ('artelingo', 'held', 'style'): 12281, ('artelingo', 'held', 'genre'): 9949,
    ('artelingo', 'val', 'emotion'): 6152, ('artelingo', 'val', 'style'): 6152, ('artelingo', 'val', 'genre'): 4960,
    ('mmart', 'test', 'emotion'): 16218, ('mmart', 'test', 'style'): 16384, ('mmart', 'test', 'genre'): 13420,
    ('mmart', 'val', 'emotion'): 5720, ('mmart', 'val', 'style'): 5769, ('mmart', 'val', 'genre'): 4679,
}


@pytest.mark.parametrize('cell', sorted(LABELLED))
def test_labels_minus_one_marks_unlabelled(cell):
    set_, split, aspect = cell
    y = ns_data.labels(set_, split, aspect)
    n = ns_paths.N_ROWS[(set_, split)]
    assert y.dtype == np.int64 and y.shape == (n,)
    assert int((y >= 0).sum()) == LABELLED[cell]            # -1 exactly on the unlabelled rows
    assert int((y < 0).sum()) == n - LABELLED[cell] and set(y[y < 0].tolist()) <= {-1}
    assert y.max() < ns_paths.CLASS_COUNT[(set_, aspect)]


def test_labels_follow_the_keys_not_the_stores_order():
    """Labels are read per key: MMArt from the index row of the key (independent read), ArtELingo genre from r11's
    sorted-name codes, whose unlabelled rows are the sorted names absent from the genre lists."""
    import pandas as pd
    idx = pd.read_parquet('/data/SSD/mmart/mmart_index.parquet').set_index('key')
    classes = {a: list(c) for a, c in __import__('mt_paths').CLASSES.items()}
    keys = ns_data.keys('mmart', 'test')
    for a in ('emotion', 'style'):
        y = ns_data.labels('mmart', 'test', a)
        for i in (0, 1, 2, 3, 500, 9000, 16383):
            v = idx.loc[keys[i], a]
            assert y[i] == (classes[a].index(v) if v != '' else -1), (a, i, v, y[i])
    g = ns_data.labels('mmart', 'test', 'genre')
    for i in (0, 1, 2, 3, 500, 9000, 16383):
        name = idx.loc[keys[i], 'genre_name']
        assert g[i] == (classes['genre'].index(name) if name != '' else -1)


# ------------------------------------------------------------------------------------------------ CRL
@pytest.mark.parametrize('aspect', ns_paths.ASPECTS)
def test_crl_equals_r11_on_artelingo_held(aspect):
    raw = ns_data.crl('artelingo', 'held', aspect, raw=True)
    ref = rt_data.raw('crl', 'held', aspect)
    assert raw.dtype == np.float64 and raw.shape == ref.shape == (12281, 50)
    assert np.abs(raw - ref).max() <= 1e-12
    x = ns_data.crl('artelingo', 'held', aspect)
    assert x.dtype == np.float32 and x.shape == (12281, 50)
    assert np.array_equal(x, rt_data.embed('crl', 'held', aspect))      # r11's vector, bit for bit, never re-normalised
    assert np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max() < 1e-6


def test_crl_mmart_uses_the_same_texts_and_rows_by_key():
    keys = ns_data.keys('mmart', 'val')
    x = ns_data.crl('mmart', 'val', 'style')
    assert x.dtype == np.float32 and x.shape == (len(keys), 50)
    want = ac_crl.vectors(mt_data.features('clip_l14', keys), rt_crl.text_array('style'))
    assert np.array_equal(x, want)
    # the same painting in ArtELingo validation has the same CRL row (the image store is shared; join by name)
    pos = {n: i for i, n in enumerate(ns_data.keys('artelingo', 'val'))}
    xa = ns_data.crl('artelingo', 'val', 'style')
    for i in (0, 17, 1234, 5000):
        assert np.array_equal(x[i], xa[pos[mt_data.artelingo_name(keys[i])]])


# ------------------------------------------------------------------------------------------------ rows
@pytest.mark.parametrize('cell', [('artelingo', 'val'), ('mmart', 'val')])
def test_plain_teacher_cstar_shapes_units_and_order(cell):
    set_, split = cell
    n = ns_paths.N_ROWS[cell]
    for enc in ns_paths.ENCODERS:
        x = ns_data.plain(set_, split, enc)
        assert x.dtype == np.float32 and x.shape == (n, ns_paths.WIDTH[enc])
        assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-5
    t = ns_data.teacher(set_, split, 'style')
    assert t.dtype == np.float32 and t.shape == (n, 2048) and np.abs(np.linalg.norm(t, axis=1) - 1).max() < 1e-5
    c = ns_data.cstar(set_, split, 'clip_l14', 'genre', 2)
    assert c.dtype == np.float32 and c.shape == (n, 2048) and np.abs(np.linalg.norm(c, axis=1) - 1).max() < 1e-5
    s = ns_data.cstar(set_, split, 'clip_l14', 'genre', 2, unit=False)      # the clustering form: stored values as float32
    assert s.dtype == np.float32 and not np.array_equal(s, c) and np.abs(s - c).max() < 2e-3


def test_mmart_rows_are_joined_by_key_across_stores():
    """MMArt validation plain and c* rows of a painting equal the ArtELingo validation rows (plain) / the export row of the key."""
    keys = ns_data.keys('mmart', 'val')
    x = ns_data.plain('mmart', 'val', 'siglip1')
    xa = ns_data.plain('artelingo', 'val', 'siglip1')
    pos = {n: i for i, n in enumerate(ns_data.keys('artelingo', 'val'))}
    for i in (3, 99, 4000):
        assert np.array_equal(x[i], xa[pos[mt_data.artelingo_name(keys[i])]])
    c = ns_data.cstar('mmart', 'val', 'qwen', 'emotion', 1, unit=False)
    with np.load(ns_paths.R15_OUT / 'exports/val/qwen/cstar/emotion__s1.npz') as z:
        have, emb = z['keys'].tolist(), z['emb']
    where = {k: i for i, k in enumerate(have)}
    for i in (0, 5, 5768):
        assert np.array_equal(c[i], emb[where[keys[i]]].astype(np.float32))


def test_cstar_artelingo_unit_form_is_r11_embed():
    got = ns_data.cstar('artelingo', 'val', 'qwen', 'style', 3)
    assert np.array_equal(got, rt_data.embed('cstar', 'val', 'style', 3))


# ------------------------------------------------------------------------------------------------ PercepT
def test_percept_held_and_test_stored():
    p = ns_data.percept_probs('artelingo', 'held', 'siglip2', 1)
    assert p.dtype == np.float32 and p.shape == (12281, 40) and p.min() >= 0 and p.max() <= 1
    q = ns_data.percept_probs('mmart', 'test', 'clip_l14', 0)
    assert q.dtype == np.float32 and q.shape == (16384, 40) and q.min() >= 0 and q.max() <= 1
    keys = ns_data.keys('mmart', 'test')
    with np.load(ns_paths.R15_OUT / 'exports/new/clip_l14/matched/probs__s0.npz') as z:
        have, probs = z['keys'].tolist(), z['probs']
    where = {k: i for i, k in enumerate(have)}
    for i in (0, 77, 16383):
        assert np.array_equal(q[i], probs[where[keys[i]]])


def fake_val_file(root, enc, seed, perm_seed=0, mutate=None):
    """A ticket-05 style file for ArtELingo validation: rows in a shuffled order, names saying which row is which."""
    names = np.asarray(rt_data.names('val'))
    order = np.random.default_rng(perm_seed).permutation(len(names))
    probs = np.random.default_rng(1).random((len(names), 40)).astype(np.float32)
    by_name = dict(zip(names.tolist(), probs))                  # the truth, by name
    arm = ns_paths.MATCHED[enc]
    path = Path(root) / 'percept_val' / enc / f'{arm}__s{seed}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    stored_names, stored = names[order], probs[order]
    if mutate:
        stored_names, stored = mutate(stored_names, stored)
    np.savez(path, names=stored_names, probs=stored)
    import hashlib
    side = dict(mapper=str(path), mapper_sha256='0' * 64, input_store='fake', n=len(stored_names),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(side))
    return by_name


def test_percept_val_joined_by_painting_name(tmp_path, monkeypatch):
    monkeypatch.setattr(ns_paths, 'OUT', tmp_path)
    by_name = fake_val_file(tmp_path, 'siglip2', 3)
    p = ns_data.percept_probs('artelingo', 'val', 'siglip2', 3)
    keys = ns_data.keys('artelingo', 'val')
    assert p.dtype == np.float32 and p.shape == (6152, 40)
    assert np.array_equal(p, np.stack([by_name[k] for k in keys]))      # rows follow keys, not the file's order
    q = ns_data.percept_probs('mmart', 'val', 'siglip2', 3)             # MMArt validation: 5,769 of the 6,152, by name
    mk = ns_data.keys('mmart', 'val')
    assert q.shape == (5769, 40)
    for i in (0, 10, 5768):
        assert np.array_equal(q[i], by_name[mt_data.artelingo_name(mk[i])])


def test_percept_val_refuses_bad_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ns_paths, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        ns_data.percept_probs('artelingo', 'val', 'siglip1', 0)           # no file yet
    fake_val_file(tmp_path, 'siglip1', 0, mutate=lambda n, p: (n[:-1], p[:-1]))     # a name missing
    with pytest.raises(AssertionError):
        ns_data.percept_probs('artelingo', 'val', 'siglip1', 0)
    fake_val_file(tmp_path, 'clip_l14', 0, mutate=lambda n, p: (n, p * 2 + 1))       # probabilities outside [0, 1]
    with pytest.raises(AssertionError):
        ns_data.percept_probs('artelingo', 'val', 'clip_l14', 0)
    fake_val_file(tmp_path, 'qwen', 0)
    side = tmp_path / 'percept_val/qwen/P3__s0.json'
    doc = json.loads(side.read_text())
    doc['sha256'] = '1' * 64                                              # sidecar no longer names the npz
    side.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        ns_data.percept_probs('artelingo', 'val', 'qwen', 0)


# ------------------------------------------------------------------------------------------------ untold
def heads(n=50, d=16, norm_dev=0.0):
    hs = [unit_rows(n, d, s) * (1 + norm_dev) for s in (1, 2, 3)]
    return hs


def test_join_heads_scaling_norm_and_order():
    hs = heads()
    ids = ['k'] * 3
    u = ns_data.join_heads(hs, ids)
    assert u.dtype == np.float32 and u.shape == (50, 48)
    assert np.abs(np.linalg.norm(u.astype(np.float64), axis=1) - 1).max() < 1e-6
    for i, h in enumerate(hs):
        assert np.allclose(u[:, 16 * i:16 * (i + 1)], h / SQRT3, atol=1e-7)       # emotion, style, genre blocks, / sqrt 3
    assert not np.allclose(ns_data.join_heads(hs[::-1], ids), u)                 # block order is part of the contract


def test_join_heads_renormalises_each_head_before_joining():
    hs = [h * (1 + 4e-4) for h in heads()]                                       # stored norms inside TOL_UNIT
    u = ns_data.join_heads(hs, ['k'] * 3)
    assert np.abs(np.linalg.norm(u[:, :16].astype(np.float64), axis=1) - 1 / SQRT3).max() < 1e-6


def test_join_heads_refuses_heads_that_differ():
    hs, ids = heads(), [['a', 'b'], ['a', 'b'], ['a', 'b']]
    ns_data.join_heads(hs, ids)
    with pytest.raises(AssertionError):
        ns_data.join_heads(hs, [['a', 'b'], ['b', 'a'], ['a', 'b']])             # shuffled keys in one head
    with pytest.raises(AssertionError):
        ns_data.join_heads(hs, ['x', 'x', 'y'])                                  # different list hashes
    with pytest.raises(AssertionError):
        ns_data.join_heads([hs[0], hs[1], hs[2][:-1]], ['k'] * 3)                # row counts differ
    with pytest.raises(AssertionError):
        ns_data.join_heads([hs[0], hs[1], hs[2] * 1.01], ['k'] * 3)              # rows not unit within TOL_UNIT
    with pytest.raises(AssertionError):
        ns_data.join_heads(hs[:2], ['k'] * 2)                                    # three heads, no fewer


def test_untold_refuses_a_head_with_shuffled_keys(monkeypatch):
    real = ns_data._head_id

    def shuffled(set_, split, enc, aspect, seed):
        ids = real(set_, split, enc, aspect, seed)
        return list(reversed(ids)) if aspect == 'style' else ids
    monkeypatch.setattr(ns_data, '_head_id', shuffled)
    with pytest.raises(AssertionError):
        ns_data.untold('mmart', 'val', 'clip_l14', 0)
    monkeypatch.setattr(ns_data, '_head_id', real)
    assert ns_data.untold('mmart', 'val', 'clip_l14', 0).shape == (5769, 6144)


@pytest.mark.parametrize('cell', [('artelingo', 'val'), ('mmart', 'val')])
def test_untold_real_rows(cell):
    set_, split = cell
    u = ns_data.untold(set_, split, 'siglip2', 4)
    n = ns_paths.N_ROWS[cell]
    assert u.dtype == np.float32 and u.shape == (n, ns_paths.UNTOLD_WIDTH)
    assert np.abs(np.linalg.norm(u.astype(np.float64), axis=1) - 1).max() < 1e-5
    for i, a in enumerate(ns_paths.ASPECTS):
        c = ns_data.cstar(set_, split, 'siglip2', a, 4) / np.float32(SQRT3)
        assert np.allclose(u[:, 2048 * i:2048 * (i + 1)], c, atol=1e-7)


# ------------------------------------------------------------------------------------------------ gates and counts
def test_check_ids_equal():
    assert ns_data.ids_equal([['a'], ['a'], ['a']]) is True
    assert ns_data.ids_equal([['a'], ['b'], ['a']]) is False
    assert ns_data.ids_equal(['h', 'h', 'h']) is True and ns_data.ids_equal(['h', 'g', 'h']) is False


def test_gate_v_one_encoder_real_data():
    g = ns_data.gate_v(encoders=('clip_l14',), seeds=(0,))
    assert g['clip_l14'] == 'ok', g['evidence']
    ev = g['evidence']['clip_l14']
    assert ev['mmart_keys_equal'] and ev['artelingo_held_list_sha256_equal'] and ev['artelingo_val_rows'] == 6152
    assert ev['max_unit_deviation'] < ns_paths.TOL_UNIT


def test_class_counts_equal_the_spec():
    got = ns_data.class_counts()
    assert got == {f'{s}/{a}': n for (s, a), n in ns_paths.CLASS_COUNT.items() if s in ns_paths.PAINTING_SETS}
    by_split = ns_data.class_counts_by_split()
    for (s, a), n in ns_paths.CLASS_COUNT.items():
        if s in ns_paths.PAINTING_SETS:
            for split in (ns_paths.HELD_SPLIT[s], 'val'):
                assert by_split[f'{s}/{split}/{a}'] == n
