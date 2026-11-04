"""Tests of om_gate_iii (ticket 07): the comparison and pass logic, and the (b) partition-file path, on synthetic data.

The real replays read the older rounds' stores and take minutes; they are run by the CLI, not here.

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_outside_methods/test_om_gate_iii.py
"""
import copy
import json
import zlib

import numpy as np
import pytest

import om_data as od
import om_gate_iii as g
import om_paths as op

N = 60
K = op.K20


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Fake keys (60 rows per split) so the partition files can be written without the real stores; one bootstrap process."""
    ks = [f'p{i:03d}' for i in range(N)]
    monkeypatch.setattr(od, 'keys', lambda s, sp: list(ks))
    g.ns_clu.WORKERS = 1
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, N).astype(np.int64)
    y[:4] = -1
    cand = rng.integers(0, K, (5, 3, N)).astype(np.int32)
    cand[:, :, :] = np.where(rng.random((5, 3, N)) < 0.5, y.clip(0) * 3, cand)       # some signal
    comp = rng.integers(0, K, (3, N)).astype(np.int32)
    return dict(keys=ks, y=y, cand=cand, comp=comp, out=tmp_path / 'out')


def stored_from(got):
    return dict(candidate_value=got['cand'], comparator_value=got['comp'], d=got['d'], lo=got['lo'], hi=got['hi'])


# ------------------------------------------------------------------------------------------------ comparison and pass
def test_a_perturbed_expected_value_of_1e8_fails_and_1e10_passes():
    got = dict(cand=0.3, comp=0.2, d=0.1, lo=0.05, hi=0.15)
    stored = stored_from(got)
    ok = g.finish(g.cell_items('t', 'src', got, stored, True), 2000, 2000)
    assert ok['pass'] and ok['max_abs_diff'] == 0.0
    for key in ('candidate_value', 'comparator_value', 'd', 'lo', 'hi'):
        bad = dict(stored, **{key: stored[key] + 1e-8})
        res = g.finish(g.cell_items('t', 'src', got, bad, True), 2000, 2000)
        assert not res['pass'] and not res['wiring_ok'], key
        assert res['max_abs_diff'] == pytest.approx(1e-8, rel=1e-3)
        near = dict(stored, **{key: stored[key] + 1e-10})
        assert g.finish(g.cell_items('t', 'src', got, near, True), 2000, 2000)['pass'], key


def test_pass_needs_full_draws_and_every_cell():
    got = dict(cand=0.3, comp=0.2, d=0.1, lo=0.05, hi=0.15)
    items = g.cell_items('t', 'src', got, stored_from(got), True)
    assert not g.finish(items, 100, 2000)['pass']                       # fewer draws: values only
    assert g.finish(items, 100, 2000)['wiring_ok']
    assert not g.finish(items, 2000, 2000, complete=False)['pass']      # a cell or aspect left out
    assert len(g.cell_items('t', 'src', got, stored_from(got), False)) == 3   # no interval below the full draws


def test_a_partition_difference_counts_entries_and_fails():
    a = np.arange(10, dtype=np.int32)
    same = g.partition_item('p', 'src', a, a.copy())
    assert same['ok'] and same['abs_diff'] == 0
    b = a.copy()
    b[3] = 99
    diff = g.partition_item('p', 'src', a, b)
    assert not diff['ok'] and diff['abs_diff'] == 1
    with pytest.raises(AssertionError):
        g.partition_item('p', 'src', a, a[:5])


def test_gate_file_merges_parts_and_passes_only_with_all_three(tmp_path):
    path = tmp_path / 'stage0' / 'gate_iii.json'
    part = lambda ok: {'pass': ok, 'code_sha256': g.code_sha256()}          # noqa: E731
    assert g.write_gate(path, 'a', part(True))['pass'] is False
    assert g.write_gate(path, 'b', part(True))['pass'] is False
    res = g.write_gate(path, 'c', part(True))
    assert res['pass'] is True and set(res) >= {'a', 'b', 'c', 'pass'}
    assert json.loads(path.read_text())['pass'] is True
    assert g.write_gate(path, 'b', part(False))['pass'] is False             # a rerun replaces its part
    res = g.write_gate(path, 'b', {'pass': True, 'code_sha256': 'older code'})
    assert res['pass'] is False                                              # a part from other code does not count


# ------------------------------------------------------------------------------------------------ (b) files
def test_b_files_give_the_same_cell_as_memory(fake):
    mem = g.clu_cell(fake['cand'], fake['comp'], fake['y'], 40)
    got, back, _ = g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 40, fake['out'])
    assert np.array_equal(back, fake['comp'])
    for key in ('cand', 'comp', 'd', 'lo', 'hi'):
        assert abs(got[key] - mem[key]) <= op.TOL_REPLAY
    side = json.loads((fake['out'] / 'parts/replay/artelingo/qwen/plain/k20__start1.json').read_text())
    assert side['method'] == 'replay' and side['K'] == K and side['start'] == 1
    assert (fake['out'] / 'parts/replay/artelingo/qwen/cstar/emotion__s4__k20.npz').exists()


def test_b_c_star_round_trip_is_state_zero_of_each_seed(fake):
    g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 20, fake['out'], aspect='style')
    back = od.turtle_parts('artelingo', 'qwen', 'cstar', K, aspect='style', method='replay', out=fake['out'])
    assert back.shape == (5, 1, N) and np.array_equal(back[:, 0], fake['cand'][:, 0])


def test_b_refuses_partitions_in_a_wrong_order(fake):
    g.write_replay_parts('artelingo', 'qwen', fake['cand'], fake['comp'], 'emotion', fake['out'])
    path = fake['out'] / 'parts/replay/artelingo/qwen/plain/k20__start2.npz'
    with np.load(path) as z:
        ks, y = z['keys'], z['labels']
    perm = np.random.default_rng(1).permutation(N)
    np.savez(path, keys=ks[perm], labels=y[perm])                             # keys and labels permuted together
    side = path.with_suffix('.json')
    meta = json.loads(side.read_text())
    meta['sha256'] = op.sha256_file(path)
    side.write_text(json.dumps(meta))                                          # a consistent sidecar: only the order is wrong
    with pytest.raises(AssertionError, match='keys differ'):
        g.read_replay_parts('artelingo', 'qwen', fake['out'])


def test_b_rows_in_a_wrong_order_change_the_values(fake):
    mem = g.clu_cell(fake['cand'], fake['comp'], fake['y'], 40)
    perm = np.random.default_rng(2).permutation(N)
    got, _, _ = g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'][:, perm], fake['y'], 40, fake['out'])
    res = g.finish(g.cell_items('t', 'src', got, stored_from(mem), True), 2000, 2000)
    assert not res['wiring_ok']


@pytest.mark.parametrize('arm', ['plain', 'cstar'])
def test_b_refuses_a_file_that_is_not_what_was_written(fake, monkeypatch, arm):
    real = od.turtle_parts

    def stale(set_, enc, arm_, *a, **kw):
        y = real(set_, enc, arm_, *a, **kw).copy()
        if arm_ == arm:
            y[0, 0] = (y[0, 0] + 1) % K
        return y
    monkeypatch.setattr(od, 'turtle_parts', stale)
    with pytest.raises(AssertionError, match='read back'):
        g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 20, fake['out'])


def test_b_second_run_overwrites_its_own_replay_files(fake):
    for _ in range(2):
        g.replay_b_cell('artelingo', 'qwen', fake['cand'], fake['comp'], fake['y'], 20, fake['out'])


# ------------------------------------------------------------------------------------------------ (c) picks
def test_c_refuses_a_pick_that_differs_from_the_recorded_one():
    g.check_pick('center+k10', 'center+k10', 'c*')
    with pytest.raises(AssertionError, match='pick'):
        g.check_pick('center+k10', 'none+k10', 'c*')


# ------------------------------------------------------------------------------------------------ the replays end to end
# Loaders, r10's reader, the stored records and the retrieval arms are replaced by synthetic ones; the replay code, r17's
# clustering and retrieval cell code and the (b) partition files are the real ones. The stored values are the cells the
# replays give on the same synthetic data, so a consistent world passes with a difference of 0 and every guard can be hit
# by changing one stored value, one pick or one argument.
DR = 10
NQ = 50
_STORED = {}                     # the stored records, computed once (a clustering cell takes a fraction of a second)


def _rng(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


@pytest.fixture
def world(monkeypatch, tmp_path):
    ks = [f'p{i:03d}' for i in range(N)]
    monkeypatch.setattr(od, 'keys', lambda s, sp: list(ks))
    monkeypatch.setattr(g.ns_clu, 'WORKERS', 1)
    monkeypatch.setattr(op, 'OUT', tmp_path / 'real_out')                 # nothing of a test may reach the real folder
    monkeypatch.setattr(g, 'FULL', dict(clu=DR, ret=DR))
    w = dict(out=tmp_path / 'real_out', r10={}, calls=[])

    def y_of(set_, aspect):
        y = _rng('y', set_, aspect).integers(0, 5, N).astype(np.int64)
        y[:4] = -1
        return y

    def cstar(set_, split, enc, aspect, s, unit=True):
        y = y_of(set_, aspect)
        x = _rng('c', set_, enc, aspect, s).integers(0, K, (3, N))
        return np.where(_rng('m', set_, enc, aspect, s).random((3, N)) < 0.5, y.clip(0) * 3, x).astype(np.int32)

    def plain(set_, split, enc):
        return _rng('p', set_, enc).integers(0, K, (3, N)).astype(np.int32)

    monkeypatch.setattr(od, 'labels', lambda set_, split, aspect: y_of(set_, aspect))
    monkeypatch.setattr(od, 'cstar', cstar)
    monkeypatch.setattr(od, 'plain', plain)
    monkeypatch.setattr(g.ns_metrics, 'partitions', lambda x, k, set_: np.asarray(x, dtype=np.int32))   # rows are partitions

    def frozen(path, n_rows):                                             # r10's file: c* seed s of aspect a, "stored"
        a, s = path.name.split('.')[0].split('__')[:2]
        s = int(s[1:])
        return w['r10'].setdefault((a, s), cstar('artelingo', 'held', 'qwen', a, s)).copy()
    monkeypatch.setattr(g.rg_held, 'read_frozen_partitions', frozen)
    w['frozen'] = frozen

    class Rows:                                                           # ns_ret._Rows: per-query AP of an arm
        def __init__(self, set_, enc):
            self.set, self.enc = set_, enc

        def ap(self, arm, aspect, pick, y, workers):
            w['calls'].append((self.set, self.enc, arm, aspect, pick))
            ap = _rng('ap', self.set, self.enc, arm, aspect).random((5, NQ) if arm == 'cstar' else NQ)
            return dict(ap=ap, rows=np.arange(NQ, dtype=np.int64))
    monkeypatch.setattr(g.ns_ret, '_Rows', Rows)
    monkeypatch.setattr(g.ns_ret, 'reused_pick', lambda arm, set_, enc: f'{arm}-pick')

    if not _STORED:
        def cell(c):                                                      # a stored cell from a computed one
            return dict(candidate_value=c['cand'], comparator_value=c['comp'], d=c['d'], lo=c['lo'], hi=c['hi'], n_draws=DR)
        cache, store = {}, {}
        for round_, set_, enc in g.CLU_CELLS:
            for a in g.ASPECTS:
                got = g.clu_cell(g.cand_parts(set_, enc, a, cache), g.plain_parts(set_, enc, cache), y_of(set_, a), DR)
                store[(round_, 'clu', 'SC1', enc, a, g.DATA_TAG[set_])] = cell(got)
        for round_, set_, enc in g.RET_CELLS:
            rows = Rows(set_, enc)
            for a in g.ASPECTS:
                got = g.ret_cell(rows.ap('cstar', a, None, None, 1), rows.ap('plain', a, None, None, 1), DR)
                store[(round_, 'ret', 'SC2', enc, a, g.DATA_TAG[set_])] = dict(
                    cell(got), candidate_readout='cstar-pick', comparator_readout='plain-pick', n_queries=NQ)
        rows = Rows('artelingo', 'qwen')
        got = g.ret_cell(rows.ap('cstar', 'emotion', None, None, 1), rows.ap('plain', 'emotion', None, None, 1), DR)
        _STORED.update(store=store, verdict=dict(
            draws=DR, readouts={'cstar': 'cstar-pick', 'plain': 'plain-pick'},
            sc={'SC10': {'comparisons': [dict(aspect='emotion', candidate=['cstar', 'emotion'],
                                              comparator=['plain', 'emotion'], d=got['d'], interval=[got['lo'], got['hi']])]}},
            map={'cstar': {'emotion': {'mean': got['cand']}}, 'plain': {'emotion': {'mean': got['comp']}}},
            n_queries={'cstar/emotion/emotion': [NQ]}))
    w.update(copy.deepcopy(_STORED))
    w['calls'].clear()
    monkeypatch.setattr(g, 'stored_cell', lambda round_, task, sc, enc, aspect, data:
                        (w['store'][(round_, task, sc, enc, aspect, data)], f'stored/{round_}'))
    monkeypatch.setattr(g, '_json', lambda path: w['verdict'])
    return w


def run(part, **kw):
    kw.setdefault('draws', DR)
    if part == 'a':
        return g.replay_a(**kw)
    if part == 'b':
        return g.replay_b(**kw)
    return g.replay_c(workers=1, **kw)


def failing(res):
    return sorted(i['name'] for i in res['items'] if not i['ok'])


@pytest.mark.parametrize('part', ['a', 'b', 'c'])
def test_replays_of_a_consistent_world_pass_with_zero_difference(world, part):
    res = run(part)
    assert res['pass'] and res['complete'] and res['wiring_ok'] and res['max_abs_diff'] <= 1e-12
    names = [i['name'] for i in res['items']]
    assert len(names) == len(set(names)) > 0
    if part == 'a':
        assert sum(n.startswith('r10/') for n in names) == len(g.ASPECTS) * len(g.SEEDS)
        assert len(names) == len(g.ASPECTS) * len(g.SEEDS) + 3 * len(g.ASPECTS) * 5
    if part == 'b':
        assert any(n.endswith('/cstar_read_back') for n in names) and any(n.endswith('/plain_read_back') for n in names)
        assert (world['out'] / 'parts/replay/artelingo/siglip2/plain/k20__start0.npz').exists()   # under the patched OUT
    if part == 'c':
        assert any(n.startswith('r11/') for n in names)
        assert ('artelingo', 'qwen', 'cstar', 'emotion', 'cstar-pick') in world['calls']      # r11 at its recorded pick
        assert ('mmart', 'clip_l14', 'plain', 'genre', 'plain-pick') in world['calls']


LEFT_OUT = [('a', dict(aspects=g.ASPECTS[:1])), ('a', dict(cells=tuple(c for c in g.ALL_CELLS if c != 'r10'))),
            ('a', dict(cells=g.ALL_CELLS[1:])), ('b', dict(aspects=g.ASPECTS[:1])), ('b', dict(cells=g.ALL_CELLS[1:3])),
            ('c', dict(aspects=g.ASPECTS[:1])), ('c', dict(cells=tuple(c for c in g.ALL_CELLS if c != 'r11'))),
            ('c', dict(cells=g.ALL_CELLS[1:]))]


@pytest.mark.parametrize('part, kw', LEFT_OUT, ids=[f'{p}-{sorted(kw)[0]}-{i}' for i, (p, kw) in enumerate(LEFT_OUT)])
def test_a_part_with_a_cell_or_aspect_left_out_never_passes(world, part, kw):
    res = run(part, **kw)
    assert res['wiring_ok'] and not res['complete'] and not res['pass']


def test_b_needs_neither_r10_nor_r11_to_be_complete(world):
    assert run('b', cells=g.ALL_CELLS[:3])['pass']


def test_a_stored_values_off_by_1e8_fail_whichever_value_and_cell(world):
    s = world['store']
    s[('r15', 'clu', 'SC1', 'qwen', 'style', 'mmart_test')]['candidate_value'] += 1e-8
    s[('r15', 'clu', 'SC1', 'clip_l14', 'genre', 'mmart_test')]['comparator_value'] += 1e-8
    s[('r14', 'clu', 'SC1', 'siglip2', 'emotion', 'artelingo_held')]['d'] += 1e-8
    s[('r15', 'clu', 'SC1', 'qwen', 'emotion', 'mmart_test')]['lo'] += 1e-8
    s[('r15', 'clu', 'SC1', 'qwen', 'genre', 'mmart_test')]['hi'] += 1e-8
    bad = failing(run('a'))
    assert bad == sorted(['r15/qwen/mmart/style/SC1/candidate_value', 'r15/clip_l14/mmart/genre/SC1/comparator_value', 'r14/siglip2/artelingo/emotion/SC1/d',
                          'r15/qwen/mmart/emotion/SC1/lo', 'r15/qwen/mmart/genre/SC1/hi'])
    res = run('a')
    assert not res['wiring_ok'] and not res['pass'] and res['max_abs_diff'] == pytest.approx(1e-8, rel=1e-3)


def test_b_stored_values_off_by_1e8_fail_so_b_does_not_compare_a_cell_to_itself(world):
    s = world['store']
    s[('r15', 'clu', 'SC1', 'qwen', 'style', 'mmart_test')]['candidate_value'] += 1e-8
    s[('r15', 'clu', 'SC1', 'clip_l14', 'genre', 'mmart_test')]['comparator_value'] += 1e-8
    s[('r14', 'clu', 'SC1', 'siglip2', 'emotion', 'artelingo_held')]['d'] += 1e-8
    s[('r15', 'clu', 'SC1', 'qwen', 'emotion', 'mmart_test')]['lo'] += 1e-8
    s[('r15', 'clu', 'SC1', 'qwen', 'genre', 'mmart_test')]['hi'] += 1e-8
    res = run('b')
    assert failing(res) == sorted(['r15/qwen/mmart/style/SC1/files/candidate_value', 'r15/clip_l14/mmart/genre/SC1/files/comparator_value',
                                   'r14/siglip2/artelingo/emotion/SC1/files/d', 'r15/qwen/mmart/emotion/SC1/files/lo',
                                   'r15/qwen/mmart/genre/SC1/files/hi'])
    assert not res['wiring_ok'] and not res['pass']


def test_c_stored_values_off_by_1e8_fail_whichever_value_and_cell(world):
    s = world['store']
    s[('r15', 'ret', 'SC2', 'qwen', 'style', 'mmart_test')]['candidate_value'] += 1e-8
    s[('r15', 'ret', 'SC2', 'clip_l14', 'genre', 'mmart_test')]['comparator_value'] += 1e-8
    s[('r15', 'ret', 'SC2', 'qwen', 'emotion', 'mmart_test')]['d'] += 1e-8
    s[('r15', 'ret', 'SC2', 'qwen', 'genre', 'mmart_test')]['lo'] += 1e-8
    s[('r15', 'ret', 'SC2', 'clip_l14', 'style', 'mmart_test')]['hi'] += 1e-8
    s[('r15', 'ret', 'SC2', 'clip_l14', 'emotion', 'mmart_test')]['n_queries'] += 1
    res = run('c')
    assert failing(res) == sorted(['r15/qwen/mmart/style/SC2/candidate_value', 'r15/clip_l14/mmart/genre/SC2/comparator_value', 'r15/qwen/mmart/emotion/SC2/d',
                                   'r15/qwen/mmart/genre/SC2/lo', 'r15/clip_l14/mmart/style/SC2/hi',
                                   'r15/clip_l14/mmart/emotion/SC2/n_queries'])
    assert not res['wiring_ok'] and not res['pass']


@pytest.mark.parametrize('key', ['candidate_value', 'comparator_value', 'd', 'lo', 'hi', 'n_queries'])
def test_c_the_r11_stored_value_off_by_1e8_fails(world, key):
    v = world['verdict']
    if key == 'candidate_value':
        v['map']['cstar']['emotion']['mean'] += 1e-8
    elif key == 'comparator_value':
        v['map']['plain']['emotion']['mean'] += 1e-8
    elif key == 'd':
        v['sc']['SC10']['comparisons'][0]['d'] += 1e-8
    elif key == 'n_queries':
        v['n_queries']['cstar/emotion/emotion'][0] += 1
    else:
        v['sc']['SC10']['comparisons'][0]['interval'][key == 'hi'] += 1e-8
    res = run('c', cells=('r11',))
    assert not res['wiring_ok'] and not res['pass']
    assert len(failing(res)) == 1 and failing(res)[0].startswith('r11/')


def test_a_r10_partitions_differ_in_any_k_means_state(world):
    only = dict(cells=('r10',), aspects=g.ASPECTS)
    assert run('a', **only)['wiring_ok']                                   # fills r10's stored partitions
    r10 = world['r10']
    for state, seed, aspect in ((0, 0, 'emotion'), (1, 2, 'style'), (2, 4, 'genre')):       # one entry in state 0, 1, 2
        r10[(aspect, seed)][state, 7] = (r10[(aspect, seed)][state, 7] + 1) % K
    res = run('a', **only)
    assert failing(res) == sorted(f'r10/qwen/artelingo/{a}/s{s}/partitions' for _, s, a in
                                  ((0, 0, 'emotion'), (1, 2, 'style'), (2, 4, 'genre')))
    assert all(i['abs_diff'] == 1 for i in res['items'] if not i['ok']) and not res['wiring_ok'] and not res['pass']


@pytest.mark.parametrize('field', ['candidate_readout', 'comparator_readout'])
def test_c_refuses_a_cell_whose_recorded_pick_differs(world, field):
    world['store'][('r15', 'ret', 'SC2', 'clip_l14', 'emotion', 'mmart_test')][field] = 'none+k10'
    with pytest.raises(AssertionError, match='pick'):
        run('c', cells=('r15:mmart:clip_l14',), aspects=('emotion',))


@pytest.mark.parametrize('arm', ['cstar', 'plain'])
def test_c_refuses_an_r11_cell_whose_recorded_pick_differs(world, arm):
    world['verdict']['readouts'][arm] = 'none+k10'
    with pytest.raises(AssertionError, match='pick'):
        run('c', cells=('r11',))


def test_b_reads_the_files_back_and_refuses_files_that_differ(world, monkeypatch):
    real = od.turtle_parts

    def other_plain(set_, enc, arm, *a, **kw):                             # the files hold other partitions than written
        y = real(set_, enc, arm, *a, **kw)
        return (y + 1) % K if arm == 'plain' else y
    monkeypatch.setattr(od, 'turtle_parts', other_plain)
    with pytest.raises(AssertionError, match='read back'):
        run('b', cells=g.ALL_CELLS[:1], aspects=('emotion',))


# ------------------------------------------------------------------------------------------------ the code hash
def test_code_hash_covers_the_loaders_and_the_scorer():
    names = {f.name for f in g.code_files()}
    assert names == {'om_gate_iii.py', 'om_data.py', 'om_paths.py', 'ns_clu.py', 'ns_metrics.py', 'ns_ret.py', 'ns_data.py'}
    assert all(f.exists() for f in g.code_files())


def test_code_hash_changes_when_any_covered_file_changes(tmp_path, monkeypatch):
    files = []
    for name in ('a.py', 'b.py', 'c.py'):
        f = tmp_path / name
        f.write_text(name)
        files.append(f)
    monkeypatch.setattr(g, 'code_files', lambda: files)
    base = g.code_sha256()
    assert g.code_sha256() == base
    for f in files:
        f.write_text('changed')
        assert g.code_sha256() != base, f.name
        f.write_text(f.name)
    assert g.code_sha256() == base
