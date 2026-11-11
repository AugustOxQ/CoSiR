"""D16 unit tests; no fixture files, pytest cache or bytecode writes needed.

Run with the requested Python/environment and `pytest -q -p no:cacheprovider`.
The small real-slice replay deliberately uses one thread and needs no queue.
"""
import copy
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import r21_d16 as d


@pytest.mark.parametrize('lo,hi,expected', [
    (0.01, 0.2, 'raises'), (-0.2, -0.01, 'lowers'),
    (0, 0.2, 'no clear change'), (-0.2, 0, 'no clear change'),
    (-0.2, 0.2, 'no clear change'), (0, 0, 'no clear change'),
    (np.nextafter(0., 1.), 0.2, 'raises'),
    (-0.2, np.nextafter(0., -1.), 'lowers')])
def test_word_edges(lo, hi, expected):
    assert d.word(lo, hi) == expected


def test_invalid_interval():
    for lo, hi in [(1, -1), (float('nan'), 1), (0, float('inf'))]:
        with pytest.raises(ValueError):
            d.word(lo, hi)


def test_pair_by_seed_not_dictionary_order():
    arm = {s: float(10 * s + 1) for s in reversed(d.SEEDS)}
    base = {s: float(10 * s) for s in d.SEEDS}
    r = d.paired_values(arm, base)
    assert r['arm_seeds'] == [1, 11, 21, 31, 41]
    assert r['cstar_seeds'] == [0, 10, 20, 30, 40]
    assert r['d'] == 1 and r['arm_mean'] == 21 and r['cstar_mean'] == 20
    with pytest.raises(ValueError, match='matching seeds'):
        d.paired_values({**arm, 5: 1}, base)
    wrong = dict(base)
    wrong[5] = wrong.pop(4)
    with pytest.raises(ValueError):
        d.paired_values(arm, wrong)


def test_ami_hand_partition_and_label_mask():
    y = np.array([0, 0, 1, 1, -1])
    perfect = np.tile([9, 9, 2, 2, 3], (3, 1))
    crossing = np.tile([0, 1, 0, 1, 100], (3, 1))
    assert d.ami_value(perfect, y) == pytest.approx(1, abs=1e-15)
    assert d.ami_value(crossing, y) == pytest.approx(-0.5, abs=1e-15)
    assert d.ami_value(np.stack([perfect[0], crossing[0], perfect[0]]), y) == pytest.approx(0.5)


def test_kmeans_keeps_every_row(monkeypatch):
    seen = []

    class KMeans:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def fit(self, x):
            assert x.shape == (5, 2) and x.dtype == np.float32
            self.labels_ = np.arange(5) % 2
            return self

    monkeypatch.setattr(d, 'MiniBatchKMeans', KMeans)
    x = d.unit(np.ones((5, 2)))
    assert d.partitions(x, 2, threads=1).shape == (3, 5)
    assert seen == [dict(n_clusters=2, random_state=s, n_init=3, batch_size=4096) for s in (0, 1, 2)]


def test_map_hand_case_stable_ties_self_and_unlabelled():
    # Identical embeddings: gallery order alone determines the ranks.
    # Labelled sequence A,B,A,B: AP = 1/2,1/3,1,1/2.
    x = np.tile(np.array([1., 0.], dtype=np.float32), (5, 1))
    y = np.array([0, 1, -1, 0, 1])
    ids = np.array(['painting-z', 'painting-a', 'unlabelled', 'painting-b', 'painting-c'])
    r = d.label_ap(x, y, ids, workers=1)
    np.testing.assert_array_equal(r['query_ids'], ids[[0, 1, 3, 4]])
    np.testing.assert_allclose(r['ap'], [1/2, 1/3, 1, 1/2], rtol=0, atol=1e-15)
    assert r['ap'].mean() == pytest.approx(7/12)
    assert r['dropped'] == 0


def test_map_singleton_stays_in_gallery():
    x = np.tile(np.array([1., 0.], dtype=np.float32), (3, 1))
    r = d.label_ap(x, np.array([0, 1, 0]), np.array(['a', 'b', 'c']), workers=1)
    np.testing.assert_allclose(r['ap'], [.5, 1.])
    assert r['query_ids'].tolist() == ['a', 'c'] and r['dropped'] == 1
    with pytest.raises(ValueError, match='no queries'):
        d.label_ap(x, np.arange(3), np.array(['a', 'b', 'c']), workers=1)


def test_map_multiple_relevant_items():
    x = np.tile(np.array([1., 0.], dtype=np.float32), (4, 1))
    r = d.label_ap(x, np.array([0, 0, 1, 0]), np.array(['a', 'b', 'c', 'd']), workers=1)
    np.testing.assert_allclose(r['ap'], [5/6, 5/6, 1], rtol=0, atol=1e-15)
    assert r['dropped'] == 1


def identity(names):
    return dict(kind='held', arm='arm', aspect='emotion', seed=0, scale='full',
                rows=len(names), width=2048, dtype='float16', list_sha256=d.names_sha(names))


def test_c6_rejects_reordered_ids():
    names = np.array(['extracted-42', 'extracted-9', 'extracted-103'])
    meta = identity(names)
    d.check_identity(meta, names, 'arm', 'emotion', 0)
    with pytest.raises(ValueError, match='list_sha256'):
        d.check_identity(meta, names[::-1], 'arm', 'emotion', 0)
    with pytest.raises(ValueError, match='actual painting IDs'):
        d.check_identity(meta, np.arange(3), 'arm', 'emotion', 0)
    with pytest.raises(ValueError, match='duplicate'):
        d.check_identity(meta, np.array(['same', 'same', 'same']), 'arm', 'emotion', 0)
    for field, value in [('seed', 1), ('rows', 4), ('arm', 'other')]:
        with pytest.raises(ValueError, match=field):
            d.check_identity(dict(meta, **{field: value}), names, 'arm', 'emotion', 0)


def test_loader_rejects_bad_array_hash_without_files(monkeypatch):
    names = np.array(['id-23', 'id-51'])
    x = np.zeros((2, 2048), dtype=np.float16)
    x[:, 0] = 1
    meta = dict(identity(names), sha256=d.array_sha(x),
                arrays=dict(img=dict(shape=list(x.shape), dtype='float16', sha256=d.array_sha(x))))

    class Store(dict):
        files = ['img']

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(d.Path, 'read_bytes', lambda self: json.dumps(meta).encode())
    monkeypatch.setattr(d, 'sha256', lambda path: 'file-sha')
    monkeypatch.setattr(d.np, 'load', lambda *args, **kwargs: Store(img=x))
    hashes = {}
    assert d.load_export('/unused', 'arm', 'emotion', 0, names, hashes).dtype == np.float32
    assert len(hashes) == 2
    meta['sha256'] = 'corrupt'
    with pytest.raises(ValueError, match='array hash mismatch'):
        d.load_export('/unused', 'arm', 'emotion', 0, names, {})


def test_default_picks_and_pending_picks():
    for field in ('pick', 'arm'):
        picks = dict(stage1b={f: {field: f'{f}-chosen'} for f in ('plain', 'start', 'teacher')})
        assert d.chosen_arms(picks) == ['dec-off', 'plain-chosen', 'start-chosen', 'teacher-chosen']
    assert d.chosen_arms(dict(stage1b=dict(plain='p', start='s', teacher='t'))) == ['dec-off', 'p', 's', 't']
    for bad in [None, {}, dict(plain='p', start='s'), dict(plain='same', start='same', teacher='t')]:
        with pytest.raises(ValueError):
            d.chosen_arms(dict(stage1b=bad))


def test_stacking_checks_query_identity():
    records = {s: dict(ap=np.array([.3, .5]), query_ids=np.array(['p1', 'p2']), dropped=0) for s in d.SEEDS}
    assert d.stacked_ap(records)['ap'].shape == (5, 2)
    records[3] = dict(records[3], query_ids=np.array(['p2', 'p1']))
    with pytest.raises(ValueError, match='query painting order'):
        d.stacked_ap(records)


def test_output_guard():
    with pytest.raises(ValueError, match='outputs must be'):
        d.write_json(d.HERE / 'forbidden.json', {})


def test_small_real_slice_equals_frozen_paths(monkeypatch):
    d.older_paths()
    import ns_metrics
    import ns_paths
    import r20_ret
    # One core throughout this quick test, including the frozen scorer.
    monkeypatch.setitem(ns_paths.KMEANS_THREADS, 'artelingo', 1)
    with threadpool_limits(limits=1):
        names, labels = d.held_data(('emotion',))
        y, ids = labels['emotion'][:96], names[:96]
        ours, frozen, aps, frozen_aps, part_seeds = {}, {}, {}, {}, {}
        for s in d.SEEDS:
            x = d.load_export(d.CSTAR_ROOT, d.CSTAR, 'emotion', s, names, {})[:96]
            parts = d.partitions(x, 9, threads=1)
            part_seeds[s] = parts
            ref = ns_metrics.partitions(x, 9, 'artelingo')
            np.testing.assert_array_equal(parts, ref)
            ours[s] = d.ami_value(parts, y)
            frozen[s] = ns_metrics.score_partitions('ami', y, ref)
            aps[s] = d.label_ap(d.unit(x), y, ids, workers=1)
            frozen_aps[s] = r20_ret.score_rows(r20_ret.unit(x), y, ids, workers=1)
            np.testing.assert_array_equal(aps[s]['query_ids'], frozen_aps[s]['query_ids'])
            np.testing.assert_allclose(aps[s]['ap'], frozen_aps[s]['ap'], rtol=0, atol=1e-15)
            assert aps[s]['ap'].mean() == pytest.approx(frozen_aps[s]['ap'].mean(), abs=1e-15)
        np.testing.assert_allclose(list(ours.values()), list(frozen.values()), rtol=0, atol=1e-15)
        # Also exercise both permitted bootstrap adapters on our partitions/APs.
        interval = d.ami_interval(part_seeds, part_seeds, y, draws=2, workers=1)
        assert interval['lo'] == interval['hi'] == interval['effect'] == 0
        np.testing.assert_allclose([interval['arm_seed_ami'][str(s)] for s in d.SEEDS],
                                   [ours[s] for s in d.SEEDS], rtol=0, atol=1e-12)
        ap = d.stacked_ap(aps)
        assert r20_ret.paired(ap, copy.deepcopy(ap), draws=2) == dict(d=0, lo=0, hi=0)
