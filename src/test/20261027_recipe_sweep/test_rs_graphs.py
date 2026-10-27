"""Ticket 03: graph variants. CPU only, synthetic data; the checks on the real arrays are the gates (rs_graphs check)."""
import hashlib
import inspect
import json
from contextlib import ExitStack

import numpy as np
import pytest

import ab_graphs
import ct2_graphs
import rs_graphs as G

N, D = 300, 16


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope='module')
def spaces():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(N, D))
    # correlated views, so the mutual graphs have edges
    return {key: unit(base + w * rng.normal(size=(N, D)))
            for key, w in zip(ct2_graphs.KEYS, (.5, .6, .7, .8, .9))}


NODES = np.arange(N, dtype=np.int64)


def test_original_partition_source_is_pinned():
    # `_partition_aspect` is a copy of this function: if the CT's changes, the copy needs a review.
    source = inspect.getsource(ct2_graphs._partition)
    assert hashlib.sha256(source.encode()).hexdigest() == G.CT_PARTITION_SHA256


def test_mutual_equals_ct2_at_20_and_follows_k(spaces):
    x = np.ascontiguousarray(spaces['E'])
    assert G._mutual(x, 20) == ct2_graphs._mutual(x)
    for k in (3, 7):
        pairs = G._mutual(x, k)
        scores = x @ x.T
        np.fill_diagonal(scores, -np.inf)
        top = [set(np.argsort(-row)[:k]) for row in scores]   # random floats: no ties
        expected = {(a, b) for a in range(N) for b in top[a] if a < b and a in top[b]}
        assert pairs == expected


def test_aspect_partition_equals_ct2_at_20(spaces):
    mine, original = G._partition_aspect(spaces, NODES, 20), ct2_graphs._partition(spaces, NODES)
    assert set(mine) == set(original) == set(G.ASPECTS)
    assert all(mine[a] == original[a] for a in G.ASPECTS)
    assert all(len(mine[a]) > 0 for a in G.ASPECTS)


def test_aspect_partition_k_changes_every_graph_and_keeps_lambda_graph_in_step(spaces):
    small, large = G._partition_aspect(spaces, NODES, 5), G._partition_aspect(spaces, NODES, 30)
    for aspect in G.ASPECTS:
        assert len(small[aspect]) < len(large[aspect])
        degree = np.bincount(ags_pairs(small[aspect]).ravel(), minlength=N)
        assert degree.max() <= 5    # reciprocal top-5: at most 5 mutual neighbours per node


def ags_pairs(pairs):
    return ct2_graphs.ags.pair_array(pairs)


def test_uncorrected_partition_is_the_plain_mutual_graphs(spaces):
    for k in (20, 11):
        out = G._partition_k(spaces, NODES, k=k, variant='uncorrected')
        assert out['emotion'] == G._mutual(spaces['E'], k)
        assert out['style'] == G._mutual(spaces['style.img'], k)
        assert out['genre'] == G._mutual(spaces['genre.img'], k)
        assert out['genre'] == G._partition_aspect(spaces, NODES, k)['genre']   # genre's graph is the same in both
        assert out['emotion'] != G._partition_aspect(spaces, NODES, k)['emotion']   # and the correction changes the rest


def test_default_view_partition_is_dview_for_all_aspects(spaces):
    out = G._partition_k(spaces, NODES, k=11, variant='default_view')
    pairs, _ = ab_graphs.dview_pairs(spaces['D'], spaces['I'], 11)
    assert out['emotion'] == out['style'] == out['genre'] == set(map(tuple, pairs.tolist()))
    # the content score of the CT: mean of the standardized cosines (the D2 ablation's Dview), not either view alone
    assert out['emotion'] != G._mutual(spaces['D'], 11)


def build(spaces, k, variant, force=False):
    with ExitStack() as stack:
        G.enter_patches(stack, k, variant, force=force)
        return ct2_graphs.build(spaces, NODES[:200], NODES[200:])


def test_patches_leave_the_ct_graphs_alone_and_restore(spaces):
    original = (ct2_graphs._partition, ct2_graphs.ROWS, ct2_graphs.METHOD, ct2_graphs._check_sample)
    with ExitStack() as stack:
        G.enter_patches(stack, 20, 'aspect')
        assert ct2_graphs._partition is original[0] and ct2_graphs.METHOD == original[2]
    graphs, manifest = build(spaces, 20, 'aspect')
    assert manifest['identity']['method'] == ct2_graphs.METHOD
    for variant, k in (('aspect', 20), ('aspect', 11), ('uncorrected', 11), ('default_view', 20)):
        build(spaces, k, variant, force=True)
        assert (ct2_graphs._partition, ct2_graphs.ROWS, ct2_graphs.METHOD, ct2_graphs._check_sample) == original


def test_forced_build_at_20_equals_the_ct_build_bit_for_bit(spaces):
    ct_graphs, ct_manifest = build(spaces, 20, 'aspect')
    graphs, manifest = build(spaces, 20, 'aspect', force=True)
    assert manifest['identity'] == ct_manifest['identity']
    assert manifest['archives'] == ct_manifest['archives']
    assert manifest['array_sha256'] == ct_manifest['array_sha256']


def test_variant_builds_name_their_rows_and_differ_in_method_and_content(spaces):
    base, base_manifest = build(spaces, 20, 'aspect')
    seen = {base_manifest['identity']['method']}
    for variant, k in (('aspect', 11), ('uncorrected', 20), ('default_view', 20), ('default_view', 11)):
        graphs, manifest = build(spaces, k, variant)
        for aspect, row in G.ROWS[variant].items():
            assert set(graphs[aspect]['rows']) == {row}
            pairs, weights = graphs[aspect]['rows'][row]['fit']
            assert pairs.dtype == np.int64 and weights.dtype == np.float32 and (weights == 1).all()
            assert (pairs[:, 0] < pairs[:, 1]).all() and pairs.max() < 200
        method = manifest['identity']['method']
        assert method not in seen and f'k={k}' in method and f'variant={variant}' in method
        seen.add(method)
        assert manifest['archives'] != base_manifest['archives']
    assert G.row_names('default_view') == dict.fromkeys(G.ASPECTS, 'Dview')


def test_k_replaces_20_in_the_lambda_graph_too(spaces):
    # emotion's graph at k = 5 is the guided graph with lambda from the k = 5 mutual graphs: rebuilding it with the
    # lambda of k = 20 gives another graph
    out5 = G._partition_aspect(spaces, NODES, 5)['emotion']
    calls = []
    original = G._mutual

    def spy(x, k):
        calls.append(k)
        return original(x, k)
    try:
        G._mutual = spy
        G._partition_aspect(spaces, NODES, 5)
    finally:
        G._mutual = original
    assert calls == [5, 5, 5, 5] and out5


def test_compare_arrays_detects_each_kind_of_difference(tmp_path):
    a = dict(x=np.arange(6, dtype=np.int64).reshape(3, 2), w=np.ones(3, dtype=np.float32), only_frozen=np.zeros(1))
    np.savez(tmp_path / 'a.npz', **a)
    b = dict(a, x=a['x'].copy(), w=a['w'].astype(np.float64))
    b.pop('only_frozen')
    np.savez(tmp_path / 'b.npz', **b)
    out = G.compare_arrays(tmp_path / 'a.npz', tmp_path / 'b.npz', ['x', 'w', 'only_frozen', 'absent'])
    assert out['x']['equal'] and out['x']['frozen_sha256'] == out['x']['rebuilt_sha256']
    assert not out['w']['equal']                                   # dtype differs
    assert out['only_frozen'] == dict(equal=False, reason='missing', in_frozen=True, in_rebuilt=False)
    assert not out['absent']['equal']
    b['x'][2, 1] += 1                                              # one value
    np.savez(tmp_path / 'c.npz', **b)
    assert not G.compare_arrays(tmp_path / 'a.npz', tmp_path / 'c.npz', ['x'])['x']['equal']


def write_gate(outputs, name, passed, items=None, code=None):
    path = G.gate_path(outputs, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'pass': passed, 'items': items or {}, 'code_sha256': code or G.code_sha256()}))


def test_builds_refuse_without_their_gate(tmp_path):
    for scale, k, variant, gate in (('sample', 11, 'aspect', 'k11_sample'), ('full', 40, 'aspect', 'k11_full'),
                                    ('sample', 20, 'uncorrected', 'ablation_graphs')):
        with pytest.raises(G.GraphRefusal, match='missing'):
            G.require_gates(tmp_path, scale, k, variant)
        write_gate(tmp_path, gate, False, items={variant: {'pass': False}})
        with pytest.raises(G.GraphRefusal, match='failed'):
            G.require_gates(tmp_path, scale, k, variant)
        write_gate(tmp_path, gate, True, items={variant: {'pass': True}}, code='0' * 64)
        with pytest.raises(G.GraphRefusal, match='other graph code'):
            G.require_gates(tmp_path, scale, k, variant)
        write_gate(tmp_path, gate, True, items={variant: {'pass': True}})
        G.require_gates(tmp_path, scale, k, variant)


def test_gates_are_per_scale_and_per_variant(tmp_path):
    write_gate(tmp_path, 'k11_sample', True)
    G.require_gates(tmp_path, 'sample', 30, 'aspect')
    with pytest.raises(G.GraphRefusal):
        G.require_gates(tmp_path, 'full', 30, 'aspect')            # V4c is the full scale's own
    write_gate(tmp_path, 'ablation_graphs', True, items={'default_view': {'pass': True}, 'uncorrected': {'pass': False}})
    G.require_gates(tmp_path, 'full', 20, 'default_view')
    with pytest.raises(G.GraphRefusal):
        G.require_gates(tmp_path, 'full', 20, 'uncorrected')
    G.require_gates(tmp_path, 'full', 20, 'aspect')                # the CT's graphs need no gate


def test_the_loaders_refuse_before_any_work(tmp_path):
    with pytest.raises(G.GraphRefusal, match='missing'):
        G.sample_store(11, outputs=tmp_path, s={})
    with pytest.raises(G.GraphRefusal, match='missing'):
        G.full_graphs(30, 'aspect', outputs=tmp_path)
    with pytest.raises(G.GraphRefusal, match='missing'):
        G.full_graphs(20, 'default_view', outputs=tmp_path)


def test_store_layout_and_frozen_inputs():
    assert G.store_root('/o', 'sample', 11, 'uncorrected').as_posix() == '/o/graphs/sample/uncorrected__k11'
    assert G.REFERENCE['aspect'][1] == {'emotion': 'Gres', 'style': 'Fres05', 'genre': 'R0o'}
    assert G.REFERENCE['uncorrected'][1] == {'emotion': 'R0', 'style': 'R0', 'genre': 'R0o'}
    assert G.REFERENCE['default_view'][1] == dict.fromkeys(G.ASPECTS, 'Dview')
    assert G.FROZEN == (20, 'aspect')


def test_full_check_passes_on_equal_stores_and_fails_on_one_changed_value(spaces, tmp_path, monkeypatch):
    graphs, manifest = build(spaces, 20, 'aspect')
    ct_full, outputs = tmp_path / 'ct_full', tmp_path / 'out'
    ct2_graphs.save(graphs, manifest, ct_full / 'd2/graphs')
    monkeypatch.setattr(G.rs_paths, 'CT_FULL', ct_full)
    rebuilt = tmp_path / 'rebuilt'

    def fake_materialize(scale, k, variant, outputs, *, force=False):
        assert (scale, k, variant, force) == ('full', 20, 'aspect', True)
        ct2_graphs.save(*build(spaces, 20, 'aspect', force=True), rebuilt / 'd2/graphs')
        return rebuilt, None
    monkeypatch.setattr(G, '_materialize', fake_materialize)
    gate = G.check_k11_full(outputs)
    assert gate['pass'] and gate['name'] == 'k11_full'
    assert json.loads(G.gate_path(outputs, 'k11_full').read_text())['pass']
    G.require_gate(outputs, 'k11_full')
    # one changed value in the rebuilt emotion pairs
    arrays = dict(np.load(rebuilt / 'd2/graphs/emotion.npz'))
    arrays['Gres__fit'] = arrays['Gres__fit'].copy()
    arrays['Gres__fit'][0, 1] += 1
    np.savez(rebuilt / 'd2/graphs/emotion.npz', **arrays)
    monkeypatch.setattr(G, '_materialize', lambda *args, **kwargs: (rebuilt, None))
    gate = G.check_k11_full(outputs)
    assert not gate['pass'] and not gate['items']['arrays']['emotion']['Gres__fit']['equal']
    with pytest.raises(G.GraphRefusal, match='failed'):
        G.require_gate(outputs, 'k11_full')
