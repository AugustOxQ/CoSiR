"""CPU hand cases for C13; artifacts stay in its authorized smoke directory."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location('independent_c13', Path(__file__).with_name('tcl_c13.py'))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def test_ap_masks_unlabelled_after_ranking():
    order = torch.tensor([[0, 1, 2, 3, 4], [4, 0, 1, 2, 3], [1, 0, 2, 3, 4]])
    z = c.metric_arrays(order, [7, 8, 9], [0, 7, 1, 7, 4], {'emotion': (np.array([2, -1, 6]), np.array([-1, 2, 3, 2, -1]))})
    assert z['ap__emotion'][0] == pytest.approx((1 + 2 / 3) / 2)
    assert np.isnan(z['ap__emotion'][1:]).all()
    np.testing.assert_equal(z['first_rank'], [2, np.nan, np.nan])


def test_first_rank_any_caption_and_pair_no_hit():
    z = c.metric_arrays(torch.tensor([[2, 1, 0], [0, 1, 2]]), [4, 8], [4, 4, 9], {})
    np.testing.assert_equal(z['first_rank'], [2, np.nan])
    assert c.pair_summary(z['first_rank']) == dict(hits=1, mrr=0.25)


def test_text_self_row_removed_other_caption_remains_and_ties():
    order = c.rank_order(torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 0.]]), torch.tensor([0, 1]))
    np.testing.assert_array_equal(order, [[1, 2, 3], [0, 2, 3]])
    z = c.metric_arrays(order, [4, 4], [4, 4, 5, 6], {'emotion': ([1, 1], [1, 1, 0, 0])})
    np.testing.assert_equal(z['first_rank'], [1, 1])
    np.testing.assert_equal(z['ap__emotion'], [1, 1])


def test_mm_text_self_has_no_pair_hit():
    order = c.rank_order(torch.eye(3), torch.arange(3))
    z = c.metric_arrays(order, [0, 1, 2], [0, 1, 2], {})
    assert np.isnan(z['first_rank']).all()
    assert c.pair_summary(z['first_rank']) == dict(hits=0, mrr=0)


def test_expansion_hand_sum_and_top_ten():
    q = torch.tensor([[1., 0.]])
    g = torch.tensor([[0., 1.]] * 10 + [[99., 99.]])
    actual = c.expand(q, g, torch.arange(11)[None])
    np.testing.assert_allclose(actual.numpy(), [[1 / np.sqrt(101), 10 / np.sqrt(101)]], atol=1e-7)


def test_centering_and_expansion_cpu():
    t = np.array([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
    fr = dict(tp=np.arange(3), ip=np.arange(3), labels={'emotion': {'t': np.array([1, 1, 0]), 'i': np.array([1, 1, 0])}})
    result = c.evaluate(t, t, fr, 'emotion', 't2t', list(c.READOUTS), c.fit_means(t, t), 'cpu')
    for z in result.values():
        assert np.isnan(z['first_rank']).all()
    np.testing.assert_equal(result['r0']['ap__emotion'], [1, 1, np.nan])


def test_mean_float64_then_float32():
    x = np.array([[1e8, 1], [1, 2], [-1e8, 3]], dtype=np.float32)
    z = c.fit_means(x, x)
    np.testing.assert_array_equal(z['t'], np.array([1 / 3, 2], dtype=np.float32))
    assert z['n_t'] == 3


def test_replicate_hand_weighted_queries():
    # Six paintings, unequal query counts; repeated paintings affect both sums and counts.
    sums = np.array([[1, 4, 3, 8, 5, 12], [2, 8, 6, 16, 10, 24]], dtype=float)
    baseline = np.array([0, 1, 1, 2, 2, 3], dtype=float)
    counts = np.array([1, 2, 1, 2, 1, 3], dtype=float)
    # I gives weights [2,0,1,2,0,1], denominator 10; seed sums 33 and 66.
    value = c.replicate(sums, baseline, counts, [0, 0, 2, 3, 3, 5], [1, 1, 0, 1, 0])
    assert value == pytest.approx((3 * 66 + 2 * 33) / 5 / 10 - 8 / 10)


def test_interval_draw_order_and_units(monkeypatch):
    calls = []
    class RNG:
        def integers(self, low, high, size):
            calls.append(('I', low, high, size))
            return np.array([0, 0]) if len(calls) == 1 else np.array([1, 1])
        def choice(self, seeds, size, replace):
            calls.append(('S', seeds, size, replace))
            return np.array([4, 3, 2, 1, 0])
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: RNG() if seed == 0 else None)
    candidate = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7], [0.4, 0.6, 0.8], [0.5, 0.7, 0.9]])
    result = c.painting_interval(candidate, [0, 0, 0], [9, 2, 9], draws=2)
    assert [x[0] for x in calls] == ['I', 'S', 'I', 'S']
    assert calls[0] == ('I', 0, 2, 2)
    assert calls[1] == ('S', [0, 1, 2, 3, 4], 5, True)
    # Ascending painting 2 has mean .5; painting 9 also has mean .5.
    assert result == pytest.approx(dict(point=.5, lower=.5, upper=.5, n_query=3, n_units=2))


def test_interval_real_rng_hand_first_draw():
    values = np.arange(15, dtype=float).reshape(5, 3) / 20
    rng = np.random.default_rng(0)
    I = rng.integers(0, 3, size=3)
    S = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
    hand = values[S][:, I].mean() - np.array([.1, .2, .3])[I].mean()
    result = c.painting_interval(values, [.1, .2, .3], [0, 1, 2], draws=1)
    assert result['lower'] == pytest.approx(hand, abs=1e-15)
    assert result['upper'] == result['lower']


def test_interval_rejects_seed_mask_change():
    candidate = np.ones((5, 2))
    candidate[3, 1] = np.nan
    with pytest.raises(AssertionError, match='masks'):
        c.painting_interval(candidate, [1, 1], [0, 1])


@pytest.mark.parametrize('lower,sr,word', [(1e-12, .1, 'WIN'), (0, .1, 'LEVEL'), (-.1 + 1e-12, .1, 'LEVEL'), (-.1, .1, 'BELOW'), (-.2, .1, 'BELOW'), (0, 0, 'BELOW')])
def test_strict_outcome(lower, sr, word):
    assert c.outcome(lower, sr) == word


@pytest.mark.parametrize('P,T,Q,expected', [(8, 9, None, 'fails'), (9, 2, None, 'fails'), (9, 3, None, 'qualified'), (9, 4, None, 'qualified'), (9, 5, None, 'holds'), (9, 9, False, 'fails'), (9, 3, True, 'qualified')])
def test_verdict_edges(P, T, Q, expected):
    assert c.verdict(P, T, Q) == expected


def test_pick_ties_and_head_order():
    assert c.pick(dict(r0=.5, center=.5, center_k10=.4)) == 'r0'
    assert c.pick(dict(emotion=17, style=17, genre=16), c.ASPECTS) == 'emotion'


def cell():
    return dict(sc='SC1', cell='label', candidate='cstar', comparator='plain', aspect='emotion', direction='t2i',
                point=.2, lower=.1, upper=.3, seed_range=.01, outcome='WIN', head='emotion',
                candidate_readout='r0', comparator_readout='center', n_query=3, n_units=2, claim_voided=False)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'word'])
def test_compare_rejects_cells(mutation):
    expected = [cell()]
    actual = [cell()]
    if mutation == 'missing':
        actual = []
    elif mutation == 'extra':
        actual.append(dict(cell(), direction='i2t'))
    else:
        actual[0]['outcome'] = 'LEVEL'
    audit = c.Audit()
    c.compare_cells(audit, expected, actual)
    assert not audit.result()['pass']


@pytest.mark.parametrize('mutation', ['ap', 'mask', 'rank', 'missing'])
def test_compare_rejects_score_changes(mutation):
    expected = dict(q_pid=np.array([0, 1]), first_rank=np.array([1., np.nan], dtype=np.float32), ap__emotion=np.array([.5, np.nan], dtype=np.float32))
    actual = {k: v.copy() for k, v in expected.items()}
    if mutation == 'ap':
        actual['ap__emotion'][0] += 2e-6
    elif mutation == 'mask':
        actual['ap__emotion'][1] = 0
    elif mutation == 'rank':
        actual['first_rank'][0] = 2
    else:
        actual.pop('first_rank')
    audit = c.Audit()
    c.compare_score(audit, 'test', expected, actual)
    assert audit.failed_total == 1


def test_compare_accepts_tolerance_and_counts_bit_changes():
    audit = c.Audit()
    assert audit.check('rows', 'x', np.array([1.], dtype=np.float32), np.array([1.0000005], dtype=np.float32))
    assert audit.categories['rows']['not_bit_identical'] == 1


def test_compare_stale_result_is_retired_before_failure(monkeypatch, tmp_path):
    out = tmp_path
    result = out / 'compare_all.json'
    result.write_text('{"pass":true}')
    other = out / 'compare_val.json'
    other.write_text('{"pass":true}')
    monkeypatch.setattr(c, 'output_root', lambda path: out)
    # Prevent even the allowed round imports in this CPU test.
    monkeypatch.setitem(sys.modules, 'tcl_paths', None)
    code = c.compare(argparse.Namespace(c13_out=out, round_out=out, records=out, part='all'))
    assert code == 1
    assert json.loads(result.read_text())['pass'] is False
    archives = list(out.glob('compare_all.json.previous.*'))
    assert len(archives) == 1 and json.loads(archives[0].read_text())['pass'] is True
    assert not other.exists()
    assert len(list(out.glob('compare_val.json.previous.*'))) == 1


def test_failure_list_cap_keeps_total():
    audit = c.Audit()
    for i in range(205):
        audit.check('x', i, 1, 2, exact=True)
    assert len(audit.failures) == 200 and audit.failed_total == 205


def test_atomic_complete_and_join(tmp_path):
    out = tmp_path
    p = out / 'ranking.npz'
    c.atomic_npz(p, q_pid=np.array([1]), first_rank=np.array([np.nan]))
    assert c.complete(p, ('q_pid', 'first_rank'))
    assert not c.complete(p, ('ap__emotion',))
    np.testing.assert_array_equal(c.gather(['z', 'x', 'y'], ['y', 'z']), [2, 0])
    with pytest.raises(KeyError):
        c.gather(['x'], ['missing'])


def test_interval_units_are_sorted_and_unscored_paintings_excluded(monkeypatch):
    class RNG:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            return np.array([0, 0])
        def choice(self, seeds, size, replace):
            return np.arange(5)
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: RNG())
    # Painting 2 precedes 9 despite query order. Painting 1 has no scored query.
    candidate = np.tile([.1, .9, .3, np.nan], (5, 1))
    result = c.painting_interval(candidate, [0, 0, 0, np.nan], [9, 2, 9, 1], draws=1)
    assert result['lower'] == pytest.approx(.9)
    assert result['point'] == pytest.approx(1.3 / 3)
    assert (result['n_query'], result['n_units']) == (3, 2)


def test_derive_module_boundary_rejects_round_module(monkeypatch):
    inputs = object.__new__(c.Inputs)
    inputs.allowed = {'rt_data', 'mt_data'}
    # Do not import any round implementation in this test.
    monkeypatch.setitem(sys.modules, 'tcl_metrics', object())
    with pytest.raises(AssertionError, match='tcl_metrics'):
        inputs.assert_independent()


def tiny_stage(group):
    kind = c.kind_of(group)
    picks = {r: {a: {d: 'r0' for d in c.DIRECTIONS} for a in c.ASPECTS} for r in c.ROWS}
    values = {r: {a: {d: {ro: .5 for ro in c.READOUTS} for d in c.DIRECTIONS} for a in c.ASPECTS} for r in c.ROWS}
    ranges = {r: {a: {d: .04 for d in c.DIRECTIONS} for a in c.ASPECTS} for r in ('cstar', 'late')}
    pair = {d: dict(late=dict(head='emotion', readout='center'), plain=dict(readout='center')) for d in c.DIRECTIONS[:2]}
    return dict(group=group, picks={kind: picks}, values={kind: values}, ranges={kind: ranges}, pair_picks={kind: pair})


def all_cells():
    cells = []
    for group in ('art', 'mm'):
        for row in ('cstar', 'late'):
            sc = {('art', 'cstar'): 'SC1', ('mm', 'cstar'): 'SC2', ('art', 'late'): 'SC3', ('mm', 'late'): 'SC4'}[group, row]
            for comp in ('plain', 'teacher'):
                for a in c.ASPECTS:
                    for d in c.DIRECTIONS:
                        cells.append(dict(cell(), sc=sc, candidate=row, comparator=comp, aspect=a, head=a,
                                          direction=d, group=group, kind=c.kind_of(group)))
            if row == 'late':
                for d in c.DIRECTIONS[:2]:
                    cells.append(dict(cell(), sc=sc, cell='pair', candidate=row, aspect=None, head='emotion', direction=d,
                                      group=group, kind=c.kind_of(group), seed_range=None, outcome=None, above_limit=True))
    assert len(cells) == 76
    return cells


@pytest.fixture
def comparison(monkeypatch, tmp_path):
    """Real compare orchestration, tiny independent files and CPU-only round stubs."""
    own, other = tmp_path / 'c13', tmp_path / 'round'
    own.mkdir()
    records = other / 'records'
    records.mkdir(parents=True)
    configs = [('cstar', 'emotion', 0), ('late', 'emotion', 0), ('plain', 'all', 'x'), ('teacher', 'emotion', 'x')]
    monkeypatch.setattr(c, 'configurations', lambda: iter(configs))
    monkeypatch.setattr(c, 'output_root', lambda path: own)
    P = types.ModuleType('tcl_paths')
    P.SMOKE, P.STORE_ROOT = 2, None
    P.score_path = lambda n, k, row, h, s, d, r, out=None: c.score_path(Path(out), n, row, h, s, d, r)
    P.means_path = lambda n, k, row, h, s, out=None: c.vector_path(Path(out), 'means', n, row, h, s)
    forward = types.ModuleType('tcl_forward')
    def rows(n, k, row, h, s, device):
        with np.load(c.vector_path(other, 'rows', n, row, h, s)) as z:
            return z['t'], z['i']
    forward.rows = rows
    monkeypatch.setitem(sys.modules, 'tcl_paths', P)
    monkeypatch.setitem(sys.modules, 'tcl_forward', forward)
    stages = {g: tiny_stage(g) for g in ('art', 'mm')}
    score_files, row_files, means_files = [], [], []
    for g in ('art', 'mm'):
        stage = stages[g]
        c.atomic_json(own / f'stage1_{g}.json', stage)
        c.atomic_json(records / f'stage1_{g}.json', stage)
        for n in (g + '_val', 'art_held' if g == 'art' else 'mm_test'):
            for row, h, s in configs:
                rp = c.vector_path(own, 'rows', n, row, h, s)
                pr = c.vector_path(other, 'rows', n, row, h, s)
                for p in (rp, pr):
                    c.atomic_npz(p, t=np.array([[1., 0.], [0., 1.]], dtype=np.float32), i=np.array([[0., 1.], [1., 0.]], dtype=np.float32))
                row_files.append((rp, pr))
                if n.endswith('_val'):
                    mp = c.vector_path(own, 'means', n, row, h, s)
                    mr = c.vector_path(other, 'means', n, row, h, s)
                    for p in (mp, mr):
                        c.atomic_npz(p, t=np.array([.5, .5], dtype=np.float32), i=np.array([.5, .5], dtype=np.float32), n_t=2, n_i=2)
                    means_files.append((mp, mr))
                # Independent plan, not needed(): the fixture must detect missing comparisons.
                plan = [(d, r) for d in c.DIRECTIONS for r in c.READOUTS] if n.endswith('_val') else [(d, 'r0') for d in c.DIRECTIONS]
                if not n.endswith('_val') and row in ('plain', 'late'):
                    plan += [(d, 'center') for d in c.DIRECTIONS[:2]]
                for d, r in plan:
                    a = c.score_path(own, n, row, h, s, d, r)
                    b = c.score_path(other, n, row, h, s, d, r)
                    z = dict(q_pid=np.array([0, 1]), first_rank=np.array([1, 11], dtype=np.float32))
                    z.update({'ap__' + aspect: np.array([.25, .75], dtype=np.float32) for aspect in (c.ASPECTS if h == 'all' else (h,))})
                    for p in (a, b):
                        c.atomic_npz(p, **z)
                    score_files.append((a, b))
    c.atomic_json(own / 'inputs.json', dict(version=2, derive_sha256=c.derive_sha256(), smoke=2, stores=None))
    cells = all_cells()
    outcomes = dict(verdicts={sc: dict(P=9, T=5, Q=True if sc in ('SC3', 'SC4') else None, verdict='holds')
                              for sc in ('SC1', 'SC2', 'SC3', 'SC4')}, voided=[])
    for p in (own / 'cells.json', other / 'stage2/cells.json'):
        c.atomic_json(p, cells)
    for p in (own / 'outcomes.json', other / 'stage2/outcomes.json'):
        c.atomic_json(p, outcomes)
    status = dict(gate_i={}, gate_ii=dict(art=True, mm=True), gate_iii=True, gate_iv=dict(art=True, mm=True),
                  stage1=dict(art=True, mm=True), c13_void=[])
    c.atomic_json(records / 'status.json', status)
    def run(part='all'):
        rc = c.compare(argparse.Namespace(part=part, c13_out=own, round_out=other, records=records))
        result = c.read_json(own / f'compare_{part}.json')
        assert not result['categories'].get('exception', {}).get('failed'), result['failures']
        return rc, result
    return types.SimpleNamespace(own=own, other=other, records=records, P=P, status=status,
                                 scores=score_files, rows=row_files, means=means_files, stages=stages, run=run)


def change_npz(path, field, value):
    with np.load(path) as z:
        values = {k: z[k] for k in z.files}
    values[field] = value
    c.atomic_npz(path, **values)


def test_compare_checks_every_ranking(comparison):
    rc, result = comparison.run()
    assert rc == 0 and result['pass']
    assert result['categories']['q_pid']['items'] == 104
    assert result['categories']['ap']['items'] == 160
    assert result['categories']['cells']['items'] == 1448
    assert not result['void_differences']


@pytest.mark.parametrize('fault,category', [
    ('missing_score', 'scores'), ('missing_row', 'rows'), ('row', 'rows'), ('means', 'means'),
    ('stage_value', 'stage1'), ('stage_pick', 'stage1'), ('verdict', 'verdicts'),
    ('P', 'verdicts'), ('T', 'verdicts'), ('Q', 'verdicts'), ('extra_cell', 'cell_set'),
    ('missing_cell', 'cell_set'), ('later_ranking', 'ap'), ('smoke', 'inputs'),
])
def test_compare_faults(comparison, fault, category):
    f = comparison
    if fault in ('missing_score', 'missing_row'):
        # Retain evidence without deleting anything.
        p = f.scores[-1][1] if fault == 'missing_score' else f.rows[-1][0]
        p.rename(p.with_suffix('.saved'))
    elif fault == 'row':
        change_npz(f.rows[0][1], 't', np.array([[.9, 0], [0, 1]], dtype=np.float32))
    elif fault == 'means':
        change_npz(f.means[0][1], 't', np.array([.6, .5], dtype=np.float32))
    elif fault.startswith('stage_'):
        p = f.records / 'stage1_art.json'
        stage = c.read_json(p)
        if fault == 'stage_value':
            stage['values']['caption']['cstar']['emotion']['t2i']['r0'] = .6
        else:
            stage['picks']['caption']['cstar']['emotion']['t2i'] = 'center'
        c.atomic_json(p, stage)
    elif fault in ('verdict', 'P', 'T', 'Q'):
        p = f.other / 'stage2/outcomes.json'
        data = c.read_json(p)
        data['verdicts']['SC3'][fault] = {'verdict': 'fails', 'P': 8, 'T': 4, 'Q': False}[fault]
        c.atomic_json(p, data)
    elif fault in ('extra_cell', 'missing_cell'):
        p = f.other / 'stage2/cells.json'
        cells = c.read_json(p)
        if fault == 'missing_cell':
            cells.pop()
        else:
            cells.append(dict(cells[0], direction='other'))
        c.atomic_json(p, cells)
    elif fault == 'later_ranking':
        # Deliberately outside the first sorted ranking of a held row (M39).
        change_npz(f.scores[-1][1], 'ap__emotion', np.array([.26, .75], dtype=np.float32))
    else:
        f.P.SMOKE = 3
    rc, result = f.run()
    assert rc == 1 and result['categories'][category]['failed'] > 0


def test_compare_derived_and_round_both_short(comparison):
    f = comparison
    for p in (f.own / 'cells.json', f.other / 'stage2/cells.json'):
        cells = c.read_json(p)
        c.atomic_json(p, cells[:-1])
    rc, result = f.run()
    assert rc == 1 and result['categories']['cell_count']['failed'] == 1


def declare_void(f, sc='SC1'):
    p = f.other / 'stage2/outcomes.json'
    data = c.read_json(p)
    data['verdicts'][sc] = dict(P=None, T=None, Q=None, verdict='inconclusive')
    data['voided'].append(dict(what='verdict', key=sc, why='review test'))
    c.atomic_json(p, data)
    p = f.other / 'stage2/cells.json'
    cells = c.read_json(p)
    for cell_ in cells:
        if cell_['sc'] == sc:
            cell_['claim_voided'] = True
    c.atomic_json(p, cells)


@pytest.mark.parametrize('cause', ['c13', 'gate_ii', 'gate_iii', 'gate_iv', 'stage1', 'gate_i', 'none'])
def test_compare_voided_verdict(comparison, cause):
    f = comparison
    declare_void(f)
    if cause == 'c13':
        f.status['c13_void'] = ['SC1']
    elif cause == 'gate_i':
        f.status['gate_i']['art_held/caption/emotion_focus/teacher'] = False
    elif cause == 'gate_iii':
        f.status['gate_iii'] = False
    elif cause != 'none':
        f.status[cause]['art'] = False
    c.atomic_json(f.records / 'status.json', f.status)
    rc, result = f.run()
    assert rc == (1 if cause == 'none' else 0)
    assert result['categories']['verdicts']['failed'] == 0
    assert len(result['voided']) == 1
    if cause == 'c13':
        assert result['void_differences'] == []  # null P/T/Q genuinely match


@pytest.mark.parametrize('kind,voids,passes', [
    ('candidate', [], False), ('candidate', ['SC1'], True),
    ('shared', ['SC1'], False), ('shared', ['SC1', 'SC3'], True),
    ('missing_cell', [], False), ('missing_cell', ['SC1'], True),
    ('cell_value', [], False), ('cell_value', ['SC1'], True),
    ('missing_score', [], False), ('missing_score', ['SC1'], True),
    ('pick', [], False), ('pick', ['SC1'], True),
])
def test_compare_void_differences(comparison, kind, voids, passes):
    f = comparison
    f.status['c13_void'] = voids
    c.atomic_json(f.records / 'status.json', f.status)
    for sc in voids:
        declare_void(f, sc)
    if kind in ('candidate', 'shared'):
        row = 'cstar' if kind == 'candidate' else 'plain'
        p = next(b for a, b in f.rows if '/art_val/' in str(b) and '/' + row + '/' in str(b))
        change_npz(p, 't', np.array([[.9, 0], [0, 1]], dtype=np.float32))
    elif kind in ('missing_cell', 'cell_value'):
        p = f.other / 'stage2/cells.json'
        cells = c.read_json(p)
        if kind == 'missing_cell':
            cells.pop(0)
        else:
            cells[0]['lower'] = .7
        c.atomic_json(p, cells)
    elif kind == 'pick':
        p = f.records / 'stage1_art.json'
        stage = c.read_json(p)
        stage['picks']['caption']['cstar']['emotion']['t2i'] = 'center'
        c.atomic_json(p, stage)
    else:
        p = next(b for a, b in f.scores if '/art_held/' in str(b) and '/cstar/' in str(b))
        p.rename(p.with_suffix('.saved'))
    rc, result = f.run()
    assert rc == (0 if passes else 1)
    assert bool(result['void_differences']) is passes
    if passes:
        assert not result['failures'] and result['failed_total'] == 0
        assert all(set(item['scs']) <= set(voids) for item in result['void_differences'])


def test_compare_type_fault_continues(comparison):
    f = comparison
    p = f.other / 'stage2/cells.json'
    cells = c.read_json(p)
    cells[0]['lower'] = None
    cells[1]['outcome'] = 'BELOW'
    c.atomic_json(p, cells)
    change_npz(f.means[0][1], 't', np.array(['bad', 'type']))
    rc, result = f.run()
    assert rc == 1
    assert result['categories']['cells']['failed'] == 2
    assert result['categories']['means']['failed'] == 1
    assert result['categories']['verdicts']['items'] == 16
    assert result['categories']['inputs']['items'] >= 4


@pytest.mark.parametrize('setting', ['mismatch', 'round_without_smoke', 'manifest_without_smoke', 'identity'])
def test_compare_input_identity_and_stores(comparison, setting):
    f = comparison
    p = f.own / 'inputs.json'
    manifest = c.read_json(p)
    if setting == 'mismatch':
        f.P.STORE_ROOT = f.own / 'other_stores'
    elif setting == 'round_without_smoke':
        f.P.SMOKE = 0
        f.P.STORE_ROOT = f.own / 'stores'
    elif setting == 'manifest_without_smoke':
        manifest.update(smoke=0, stores=str(f.own / 'stores'))
        f.P.SMOKE = 0
        f.P.STORE_ROOT = f.own / 'stores'
    else:
        manifest['derive_sha256'] = 'stale'
    c.atomic_json(p, manifest)
    rc, result = f.run()
    assert rc == 1 and result['categories']['inputs']['failed'] > 0


def test_pair_summary_rank_ten_boundary():
    assert c.pair_summary([10, 11, np.nan]) == dict(hits=1, mrr=pytest.approx((.1 + 1 / 11) / 3))


def test_make_stage1_pair_ties_mrr_and_picked_range(tmp_path):
    out = tmp_path
    for row, head, seed in c.configurations():
        for d in c.DIRECTIONS:
            for r in c.READOUTS:
                # r0 range .4, center range .04. Label picks center (and earlier than a tie).
                ap = .1 + .1 * seed if r == 'r0' and row in ('cstar', 'late') else (.7 + .01 * seed if row in ('cstar', 'late') else .7)
                # r0: MRR .5 but one hit; center: MRR .1 but two hits. Must pick by MRR.
                ranks = [1, np.nan] if r == 'r0' else [10, 10]
                # Emotion/style tie on hits, despite different MRR: choose emotion, not last head.
                if head == 'style' and r == 'r0':
                    ranks = [2, np.nan]
                if head == 'genre':
                    ranks = [np.nan, np.nan]
                arrays = dict(q_pid=np.array([0, 1]), first_rank=np.array(ranks, dtype=np.float32))
                arrays.update({'ap__' + a: np.array([ap, ap], dtype=np.float32) for a in (c.ASPECTS if head == 'all' else (head,))})
                c.atomic_npz(c.score_path(out, 'art_val', row, head, seed, d, r), **arrays)
    stage = c.make_stage1(out, 'art')
    assert stage['picks']['caption']['cstar']['emotion']['t2i'] == 'center'
    assert stage['ranges']['caption']['cstar']['emotion']['t2i'] == pytest.approx(.04, abs=1e-7)
    for d in c.DIRECTIONS[:2]:
        assert stage['pair_picks']['caption'][d] == dict(late=dict(head='emotion', readout='r0'), plain=dict(readout='r0'))


def test_needed_keeps_pair_only_readouts():
    stage = tiny_stage('art')
    base = {(d, 'r0') for d in c.DIRECTIONS}
    assert c.needed(stage, 'cstar', 'emotion') == base
    assert c.needed(stage, 'late', 'style') == base
    assert c.needed(stage, 'late', 'emotion') == base | {('t2i', 'center'), ('i2t', 'center')}
    assert c.needed(stage, 'plain', 'all') == base | {('t2i', 'center'), ('i2t', 'center')}


def test_build_cells_own_picks_pairs_and_verdicts(monkeypatch, tmp_path):
    stage = tiny_stage('art')
    for row in ('plain', 'teacher'):
        for a in c.ASPECTS:
            for d in c.DIRECTIONS:
                stage['picks']['caption'][row][a][d] = 'center'
    c.atomic_json(tmp_path / 'stage1_art.json', stage)
    for row, head, seed in c.configurations():
        for d in c.DIRECTIONS:
            for r in c.READOUTS:
                seeded = row in ('cstar', 'late')
                ap = .8 if seeded else (.3 if r == 'center' else .99)
                rank = [10, 11] if seeded else [10, 10]
                arrays = dict(q_pid=np.array([0, 1]), first_rank=np.array(rank, dtype=np.float32))
                arrays.update({'ap__' + a: np.array([ap, ap], dtype=np.float32) for a in (c.ASPECTS if head == 'all' else (head,))})
                c.atomic_npz(c.score_path(tmp_path, 'art_held', row, head, seed, d, r), **arrays)
    def interval(candidate, baseline, pid):
        candidate = np.asarray(candidate)
        point = float(candidate.mean() - baseline.mean())
        # Directional pair bounds straddle the strict -.02 bar. Candidate rank 11 must be a miss.
        if set(np.unique(candidate)) <= {0, 1}:
            assert candidate.mean() == .5
            pair_bounds = iter_bounds
            lower = next(pair_bounds)
        else:
            assert baseline.mean() == pytest.approx(.3)
            lower = point - .01
        return dict(point=point, lower=lower, upper=point + .01, n_query=2, n_units=2)
    iter_bounds = iter([-.019, -.02])
    monkeypatch.setattr(c, 'painting_interval', interval)
    cells, outcomes = c.build_cells(tmp_path, ('art',))
    assert len(cells) == 38
    label = cells[0]
    assert label['point'] == pytest.approx(.5)
    assert (label['candidate_readout'], label['comparator_readout'], label['seed_range']) == ('r0', 'center', .04)
    assert label['outcome'] == 'WIN'
    pairs = [x for x in cells if x['cell'] == 'pair']
    assert [x['above_limit'] for x in pairs] == [True, False]
    assert outcomes['verdicts'] == dict(SC1=dict(P=9, T=9, Q=None, verdict='holds'), SC3=dict(P=9, T=9, Q=False, verdict='fails'))


def fake_inputs(monkeypatch, smoke=2):
    obj = object.__new__(c.Inputs)
    obj.smoke, obj.stores = smoke, None
    obj.art_names = lambda split: ['p0', 'p1']
    obj.art_labels = lambda split: dict(emotion=np.array([8, 8]), style=np.array([3, 4]), genre=np.array([-1, 2]))
    obj.mm_keys = lambda split: ['k0', 'k1']
    obj.mm_labels = lambda names, aspect: np.zeros(len(names), dtype=np.int64)
    annotations = [dict(painting='p0' if j == 0 else 'p1', emotion=f'e{j}') for j in range(9)]
    index = dict(texts=[dict(sample_id=1, painting='p1'), dict(sample_id=0, painting='p0')])
    def read(path):
        if str(path).endswith('index.json'):
            return index
        assert str(path).endswith('artelingo_train.json')
        return annotations
    monkeypatch.setattr(c, 'read_json', read)
    monkeypatch.setattr(c.pd, 'read_parquet', lambda *a, **kw: c.pd.DataFrame(dict(key=['k0', 'k1'], painting=['p0', 'p1'], artelingo=['none', 'none'])))
    return obj


def test_inputs_caption_emotion_and_id_join(monkeypatch):
    inputs = fake_inputs(monkeypatch)
    frame = inputs.frame('art_val')
    np.testing.assert_array_equal(frame['ids'], [0, 1])
    np.testing.assert_array_equal(frame['tp'], [0, 1])
    np.testing.assert_array_equal(frame['labels']['emotion']['t'], [0, 1])
    np.testing.assert_array_equal(frame['labels']['emotion']['i'], [8, 8])
    np.testing.assert_array_equal(frame['labels']['genre']['t'], [-1, 2])


@pytest.mark.parametrize('name', ['art_val', 'art_held', 'mm_val', 'mm_test'])
def test_inputs_full_count_guard(monkeypatch, name):
    inputs = fake_inputs(monkeypatch, smoke=0)
    with pytest.raises(AssertionError, match=name):
        inputs.frame(name)


def test_inputs_fusion_weights_and_sides(monkeypatch):
    obj = object.__new__(c.Inputs)
    plain = (np.array([[1., 0.]], dtype=np.float32), np.array([[0., 1.]], dtype=np.float32))
    obj.plain = lambda name: plain
    state = {}
    for side, bias in (('cap', [3., 4.]), ('img', [0., 2.])):
        state.update({f'{side}.trunk.0.weight': torch.zeros(2, 2), f'{side}.trunk.0.bias': torch.zeros(2),
                      f'{side}.output.weight': torch.zeros(2, 2), f'{side}.output.bias': torch.tensor(bias)})
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: state)
    t, i = obj.rows('art_val', 'late', 'emotion', 0, device='cpu')
    np.testing.assert_allclose(t, [[np.sqrt(.3) * .6, np.sqrt(.3) * .8, np.sqrt(.7), 0]], atol=1e-7)
    np.testing.assert_allclose(i, [[0, np.sqrt(.3), 0, np.sqrt(.7)]], atol=1e-7)


@pytest.fixture
def derive_stub(monkeypatch, tmp_path):
    calls = types.SimpleNamespace(independent=0, means=[], roots=[])
    class TinyInputs:
        def __init__(self, smoke, stores):
            pass
        def frame(self, name):
            return {}
        def rows(self, *args):
            return np.array([[1., 0.], [0., 1.]], dtype=np.float32), np.array([[0., 1.], [1., 0.]], dtype=np.float32)
        def assert_independent(self):
            calls.independent += 1
    def root(path):
        calls.roots.append(Path(path))
        return tmp_path
    def evaluate(t, i, frame, head, direction, readouts, means):
        calls.means.append({k: v.copy() for k, v in means.items()})
        return {r: dict(q_pid=np.array([0, 1]), first_rank=np.array([1., 2.]),
                        **{'ap__' + a: np.array([.5, .5]) for a in c.ASPECTS}) for r in readouts}
    monkeypatch.setattr(c, 'Inputs', TinyInputs)
    monkeypatch.setattr(c, 'output_root', root)
    monkeypatch.setattr(c, 'configurations', lambda: iter([('plain', 'all', 'x')]))
    monkeypatch.setattr(c, 'evaluate', evaluate)
    monkeypatch.setattr(c, 'make_stage1', lambda out, group: tiny_stage(group))
    for g in ('art', 'mm'):
        c.atomic_json(tmp_path / f'stage1_{g}.json', tiny_stage(g))
        c.atomic_npz(c.vector_path(tmp_path, 'means', g + '_val', 'plain', 'all', 'x'),
                     t=np.array([.1, .2], dtype=np.float32), i=np.array([.3, .4], dtype=np.float32), n_t=2, n_i=2)
    calls.out = tmp_path
    return calls


def derive_args(**kwargs):
    args = dict(part='held', group='art', smoke=2, stores=None, out=None)
    args.update(kwargs)
    return argparse.Namespace(**args)


def test_derive_held_uses_validation_means(derive_stub):
    c.derive(derive_args())
    assert len(derive_stub.means) == 3
    for means in derive_stub.means:
        np.testing.assert_array_equal(means['t'], np.array([.1, .2], dtype=np.float32))
        np.testing.assert_array_equal(means['i'], np.array([.3, .4], dtype=np.float32))


def test_derive_calls_independence_at_end(derive_stub):
    c.derive(derive_args())
    assert derive_stub.independent == 1


def test_derive_default_smoke_output(derive_stub):
    c.derive(derive_args())
    assert derive_stub.roots == [c.SMOKE_OUT]


def test_derive_identity_refuses_old_cache(derive_stub, monkeypatch):
    c.derive(derive_args())
    manifest = c.read_json(derive_stub.out / 'inputs.json')
    assert manifest['derive_sha256'] == c.derive_sha256()
    monkeypatch.setattr(c, 'derive_sha256', lambda: 'changed-derive-code')
    with pytest.raises(AssertionError, match='fresh --out'):
        c.derive(derive_args())
    assert derive_stub.independent == 1  # rejected before reading any cached row


def test_derive_identity_excludes_compare():
    source = Path(c.__file__).read_text()
    assert c.derive_sha256(source) == c.derive_sha256(source.replace('def compare(args):', 'def compare_changed(args):'))
    assert c.derive_sha256(source) != c.derive_sha256(source.replace('def fit_means(t, i):', 'def fit_means_changed(t, i):'))


@pytest.mark.parametrize('group', ['art', 'mm', 'all'])
def test_derive_publishes_cells_only_for_all(derive_stub, monkeypatch, group):
    called = []
    def cells(out, groups):
        called.append(groups)
        return [], dict(verdicts={})
    monkeypatch.setattr(c, 'build_cells', cells)
    c.derive(derive_args(group=group))
    assert (derive_stub.out / 'cells.json').exists() is (group == 'all')
    assert (derive_stub.out / 'outcomes.json').exists() is (group == 'all')
    assert bool(called) is (group == 'all')


def test_stores_without_smoke_rejected_before_any_read(monkeypatch):
    monkeypatch.setattr(c, 'output_root', lambda path: pytest.fail('must reject before output access'))
    with pytest.raises(AssertionError, match='requires --smoke'):
        c.derive(derive_args(smoke=0, stores=Path('/unused')))
    with pytest.raises(AssertionError, match='requires --smoke'):
        c.Inputs(smoke=0, stores='/unused')


@pytest.mark.parametrize('group,row,expected', [('art', 'cstar', {'SC1'}), ('art', 'late', {'SC3'}),
                                               ('art', 'teacher', {'SC1', 'SC3'}), ('mm', 'plain', {'SC2', 'SC4'})])
def test_audit_void_reader_scopes(group, row, expected):
    for category in ('rows', 'means', 'scores', 'q_pid', 'first_rank', 'ap'):
        key = f'{group}_held/{c.kind_of(group)}/{row}/emotion/0/t2i/r0'
        assert c.readers_of(category, key) == expected
    for section in ('picks', 'values', 'ranges'):
        assert c.readers_of('stage1', f'{group}/{section}/{c.kind_of(group)}/{row}/emotion/t2i') == expected
    assert c.readers_of('stage1', f'{group}/pair_picks/{c.kind_of(group)}/t2i/{row}/readout') == expected


def test_compare_val_retires_all_result(comparison):
    f = comparison
    old = f.own / 'compare_all.json'
    c.atomic_json(old, {'pass': True})
    rc, result = f.run(part='val')
    assert rc == 0
    assert result['categories']['q_pid']['items'] == 72
    assert 'cells' not in result['categories']
    assert not old.exists()
    assert len(list(f.own.glob('compare_all.json.previous.*'))) == 1


@pytest.mark.parametrize('cause', ['other_group', 'diagnostic_store'])
def test_compare_unrelated_gate_cannot_authorize_void(comparison, cause):
    f = comparison
    declare_void(f)
    if cause == 'other_group':
        f.status['gate_ii']['mm'] = False
    else:
        f.status['gate_i']['art_held/caption/none/clip_l14'] = False
    c.atomic_json(f.records / 'status.json', f.status)
    rc, result = f.run()
    assert rc == 1
    assert result['categories']['voided']['failed'] == 1
