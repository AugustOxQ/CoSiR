"""Tests of r29_text_cells (ticket 07): the nine ArtELingo text cells of a new model against P1 and Q1 by r23's protocol.

Small synthetic score files in r23's layout; `tcl_paths.SMOKE` is patched to the painting count (r23's own smoke switch for
`tcl_outcome._load`'s shape checks). No real data.
"""
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_text_cells as TC  # noqa: E402

P23, X = TC.P23, TC.X
N_I = 6
PID_T = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 5], dtype=np.int64)    # caption queries: painting 0 has five
PID_I = np.arange(N_I, dtype=np.int64)
ASPECT = 'emotion'


@pytest.fixture(autouse=True)
def small_sets():
    with mock.patch.object(P23, 'SMOKE', N_I):
        yield


def pid_of(direction):
    return PID_I if direction == 'i2t' else PID_T


def write_unit(root, row, head, seed, direction, readout, ap, pid=None, aspect=ASPECT):
    """`ap`: one array (column ap__<aspect>) or a dict {aspect: array} (a one-vector row's file holds every aspect)."""
    path = Path(root) / 'scores' / 'art_held' / 'caption' / row / f'{head}__s{seed}__{direction}__{readout}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = pid_of(direction) if pid is None else pid
    cols = ap if isinstance(ap, dict) else {aspect: ap}
    np.savez(path, q_pid=pid, **{f'ap__{a}': np.asarray(v, dtype=np.float32) for a, v in cols.items()})
    return path


def make_arm(root, row, head, seeds, direction, readout, rng, mask=None, aspect=ASPECT):
    """One ranking set: per seed random AP in (0.05, 0.95); `mask` False entries are NaN. Returns the float32 values [seeds, n]."""
    n = len(pid_of(direction))
    vals = rng.uniform(0.05, 0.95, size=(len(seeds), n)).astype(np.float32)
    if mask is not None:
        vals[:, ~mask] = np.nan
    for s, v in zip(seeds, vals):
        write_unit(root, row, head, s, direction, readout, v, aspect=aspect)
    return vals


def build(tmp_path, direction='t2i', cand_readout='r0', comp_readout='center', seed=0):
    """Candidate (row cstar, five seeds), a one-vector comparator (row siglip_so400m, head all) in other folders."""
    rng = np.random.default_rng(seed)
    cand = make_arm(tmp_path / 'cand', 'cstar', ASPECT, range(5), direction, cand_readout, rng)
    comp = make_arm(tmp_path / 'p1', 'siglip_so400m', 'all', ['x'], direction, comp_readout, rng)[0]
    return cand, comp


def reference_interval(cand, comp, pid, draws):
    """Independent painting-unit bootstrap (r23 §3.4): draw paintings with replacement, keep every query of each drawn
    painting, draw seeds with replacement (RNG 0, paintings then seeds per draw)."""
    units = np.unique(pid)
    m, k = len(units), cand.shape[0]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(draws):
        paintings = rng.integers(0, m, size=m)
        seeds = rng.choice(list(range(k)), size=k, replace=True)
        idx = np.concatenate([np.flatnonzero(pid == units[i]) for i in paintings])
        boots.append(cand[:, idx].mean(axis=1)[seeds].mean() - comp[idx].mean())
    return np.percentile(boots, [2.5, 97.5])


# ------------------------------------------------------------------------------------------------ painting unit
def test_painting_draws_carry_all_queries_of_a_painting(tmp_path):
    cand, comp = build(tmp_path, 't2i')
    got = TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=60)
    lo, hi = reference_interval(cand.astype(np.float64), comp.astype(np.float64), PID_T, 60)
    assert got['lo'] == pytest.approx(lo, abs=1e-12) and got['hi'] == pytest.approx(hi, abs=1e-12)
    assert got['n_units'] == N_I and got['n_query'] == len(PID_T)
    # a query-level bootstrap (every query its own unit) gives another interval: the test is able to tell them apart
    naive = reference_interval(cand.astype(np.float64), comp.astype(np.float64), np.arange(len(PID_T)), 60)
    assert abs(naive[0] - lo) > 1e-6


def test_value_and_difference_are_the_means_over_seeds_and_scored_queries(tmp_path):
    cand, comp = build(tmp_path, 't2i')
    got = TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=20)
    assert got['value'] == pytest.approx(float(cand.astype(np.float64).mean()), abs=1e-15)
    assert got['comparator_value'] == pytest.approx(float(comp.astype(np.float64).mean()), abs=1e-15)
    assert got['d'] == pytest.approx(got['value'] - got['comparator_value'], abs=1e-15)
    assert (got['candidate_readout'], got['comparator_readout']) == ('r0', 'center')


def test_unscored_queries_leave_both_arms(tmp_path):
    rng = np.random.default_rng(3)
    mask = np.ones(len(PID_T), bool)
    mask[[1, 7]] = False
    cand = make_arm(tmp_path / 'cand', 'cstar', ASPECT, range(5), 't2i', 'r0', rng, mask)
    comp = make_arm(tmp_path / 'p1', 'siglip_so400m', 'all', ['x'], 't2i', 'center', rng, mask)[0]
    got = TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=20)
    assert got['n_query'] == mask.sum()
    assert got['value'] == pytest.approx(float(cand[:, mask].astype(np.float64).mean()), abs=1e-15)


# ------------------------------------------------------------------------------------------------ outcome edges
@pytest.mark.parametrize('lower, r, want', [
    (1e-12, 0.01, 'WIN'),
    (0.0, 0.01, 'LEVEL'),                 # not strictly above 0
    (-0.01 + 1e-12, 0.01, 'LEVEL'),
    (-0.01, 0.01, 'BELOW'),               # not strictly above -r
    (-0.5, 0.01, 'BELOW'),
    (0.0, 0.0, 'BELOW'),                  # r = 0: LEVEL is empty
])
def test_outcome_edges(tmp_path, lower, r, want):
    build(tmp_path, 't2i')
    real = TC.O23.painting_interval

    def edge(cand, comp, pid, draws):                                # the real point and unit count, the lower bound at the edge
        point, _, _, n = real(cand, comp, pid, draws=draws)
        return point, lower, lower + 0.1, n

    with mock.patch.object(TC.O23, 'painting_interval', edge):
        got = TC.paired(tmp_path / 'cand', 'r0', r, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)
    assert got['outcome'] == want and got['lo'] == lower


def test_nan_lower_bound_raises(tmp_path):
    build(tmp_path, 't2i')
    with mock.patch.object(TC.O23, 'painting_interval', return_value=(0.0, float('nan'), 0.1, N_I)):
        with pytest.raises(AssertionError):
            TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)


# ------------------------------------------------------------------------------------------------ agreement checks
def test_other_queries_raise(tmp_path):
    rng = np.random.default_rng(4)
    make_arm(tmp_path / 'cand', 'cstar', ASPECT, range(5), 't2i', 'r0', rng)
    other = PID_T.copy()
    other[0], other[5] = other[5], other[0]                    # same multiset, another order
    write_unit(tmp_path / 'p1', 'siglip_so400m', 'all', 'x', 't2i', 'center', rng.uniform(0.1, 0.9, len(PID_T)), pid=other)
    with pytest.raises(AssertionError, match='arms differ in queries'):
        TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)


def test_other_scored_rows_raise(tmp_path):
    rng = np.random.default_rng(5)
    make_arm(tmp_path / 'cand', 'cstar', ASPECT, range(5), 't2i', 'r0', rng)
    mask = np.ones(len(PID_T), bool)
    mask[3] = False
    make_arm(tmp_path / 'p1', 'siglip_so400m', 'all', ['x'], 't2i', 'center', rng, mask)
    with pytest.raises(AssertionError, match='arms differ in scored queries'):
        TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)


def test_one_seed_with_other_scored_rows_raises(tmp_path):
    rng = np.random.default_rng(6)
    cand = make_arm(tmp_path / 'cand', 'cstar', ASPECT, range(5), 't2i', 'r0', rng)
    bad = cand[3].copy()
    bad[2] = np.nan
    write_unit(tmp_path / 'cand', 'cstar', ASPECT, 3, 't2i', 'r0', bad)
    make_arm(tmp_path / 'p1', 'siglip_so400m', 'all', ['x'], 't2i', 'center', rng)
    with pytest.raises(AssertionError, match='arms differ in scored queries'):
        TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)


def test_missing_score_file_raises(tmp_path):
    rng = np.random.default_rng(7)
    make_arm(tmp_path / 'cand', 'cstar', ASPECT, range(5), 't2i', 'r0', rng)
    with pytest.raises(Exception, match='siglip_so400m'):
        TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)


def test_the_comparator_is_never_read_from_the_candidates_folder(tmp_path):
    """Routing: row cstar -> the candidate's folder, the comparator row -> its own: a file only in the wrong place is a miss."""
    rng = np.random.default_rng(8)
    make_arm(tmp_path / 'cand', 'cstar', ASPECT, range(5), 't2i', 'r0', rng)
    make_arm(tmp_path / 'cand', 'siglip_so400m', 'all', ['x'], 't2i', 'center', rng)     # only here
    with pytest.raises(Exception, match='siglip_so400m'):
        TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)


def test_routing_restores_the_score_path(tmp_path):
    before = P23.score_path
    build(tmp_path, 't2i')
    TC.paired(tmp_path / 'cand', 'r0', 0.01, 'siglip_so400m', tmp_path / 'p1', 'center', ASPECT, 't2i', draws=5)
    assert P23.score_path is before and TC.O23._interval.__module__ == 'tcl_outcome'


# ------------------------------------------------------------------------------------------------ nine cells x two comparators
def full_tree(tmp_path, readout_c='r0'):
    """Candidate five seeds, P1 (siglip_so400m, all) and Q1 (teacher, per aspect) for every aspect and direction, at fixed
    read-outs; returns the three spec dicts for `score_cells`."""
    rng = np.random.default_rng(11)
    picks_c, picks_p, picks_q, ranges = {}, {}, {}, {}
    for d in TC.DIRECTIONS:                                           # a one-vector row's file holds all three aspects
        n = len(pid_of(d))
        write_unit(tmp_path / 'p1', 'siglip_so400m', 'all', 'x', d, 'center', {a: rng.uniform(0.05, 0.95, n) for a in TC.ASPECTS})
    for a in TC.ASPECTS:
        picks_c[a], picks_p[a], picks_q[a] = {}, {}, {}
        for d in TC.DIRECTIONS:
            make_arm(tmp_path / 'cand', 'cstar', a, range(5), d, readout_c, rng, aspect=a)
            make_arm(tmp_path / 'q1', 'teacher', a, ['x'], d, 'center_k10', rng, aspect=a)
            picks_c[a][d] = dict(pick=readout_c, mean=0.5, range=0.004)
            picks_p[a][d], picks_q[a][d] = 'center', 'center_k10'
    cand = dict(root=tmp_path / 'cand', picks=picks_c)
    p1 = dict(row='siglip_so400m', root=tmp_path / 'p1', picks=picks_p, form='plain')
    q1 = dict(row='teacher', root=tmp_path / 'q1', picks=picks_q)
    return cand, p1, q1


def test_score_cells_nine_by_two_with_ids_and_fields(tmp_path):
    cand, p1, q1 = full_tree(tmp_path)
    cells = TC.score_cells('siglip1', 'siglip1', cand, p1, q1, draws=30)
    assert len(cells) == 18
    ids = [c['id'] for c in cells]
    assert len(set(ids)) == 18
    for a in TC.ASPECTS:
        for d in TC.DIRECTIONS:
            for comp in ('p1', 'q1'):
                assert f'artelingo/siglip1/siglip1/{a}/{d}/{comp}' in ids
    c = [c for c in cells if c['id'] == 'artelingo/siglip1/siglip1/genre/i2t/q1'][0]
    for key in ('type', 'protocol', 'comparator', 'aspect', 'arm', 'value', 'd', 'lo', 'hi', 'r', 'outcome'):
        assert key in c
    assert (c['type'], c['comparator'], c['aspect'], c['arm'], c['direction'], c['r']) == ('text', 'q1', 'genre', 'siglip1', 'i2t', 0.004)
    assert c['comparator_row'] == 'teacher' and c['candidate_readout'] == 'r0' and c['comparator_readout'] == 'center_k10'
    assert c['n_draws'] == 30
    p = [c for c in cells if c['comparator'] == 'p1'][0]
    assert p['comparator_row'] == 'siglip_so400m' and p['p1_form'] == 'plain'


def test_score_cells_uses_the_candidates_own_range_not_the_comparators(tmp_path):
    cand, p1, q1 = full_tree(tmp_path)
    cand['picks']['style']['t2t']['range'] = 123.0
    cells = TC.score_cells('siglip1', 'siglip1', cand, p1, q1, draws=10)
    for c in cells:
        assert c['r'] == (123.0 if (c['aspect'], c['direction']) == ('style', 't2t') else 0.004)
        if (c['aspect'], c['direction']) == ('style', 't2t'):
            assert c['outcome'] != 'BELOW'                           # r huge: never BELOW


def test_score_cells_reads_each_comparator_at_its_own_pick(tmp_path):
    cand, p1, q1 = full_tree(tmp_path)
    p1['picks']['emotion']['t2i'] = 'r0'                           # no P1 file at r0: a read at the wrong pick is a miss
    with pytest.raises(Exception, match='siglip_so400m'):
        TC.score_cells('siglip1', 'siglip1', cand, p1, q1, draws=5)


# ------------------------------------------------------------------------------------------------ P1's form (D20), K*
def write_prefix_pick(path, **kw):
    doc = dict(kstar='siglip1_prefix', kstar_form='prefixed', p1_form='plain', flag=None)
    doc.update(kw)
    Path(path).write_text(json.dumps(doc))
    return path


def test_p1_form_is_read_and_asserted(tmp_path):
    f = write_prefix_pick(tmp_path / 'prefix_pick.json', p1_form='prefixed')
    assert TC.read_p1_form(f, 'siglip1_prefix') == 'prefixed'
    f = write_prefix_pick(tmp_path / 'p2.json', p1_form='plain')
    assert TC.read_p1_form(f, 'siglip1_prefix') == 'plain'
    f = write_prefix_pick(tmp_path / 'p3.json', kstar='siglip1', kstar_form='plain', p1_form='plain')
    assert TC.read_p1_form(f, 'siglip1') == 'plain'


def test_p1_form_must_fit_the_arm_and_the_vocabulary(tmp_path):
    with pytest.raises(AssertionError):                            # K1 won: D20 never picks a prefixed P1
        TC.read_p1_form(write_prefix_pick(tmp_path / 'a.json', kstar='siglip1', p1_form='prefixed'), 'siglip1')
    with pytest.raises(AssertionError):
        TC.read_p1_form(write_prefix_pick(tmp_path / 'b.json', p1_form='both'), 'siglip1_prefix')
    with pytest.raises(AssertionError):
        TC.read_p1_form(write_prefix_pick(tmp_path / 'b2.json', kstar_form='both'), 'siglip1_prefix')
    with pytest.raises(AssertionError):
        TC.read_p1_form(write_prefix_pick(tmp_path / 'b3.json', kstar='qwen'), 'siglip1_prefix')
    doc = dict(kstar='siglip1_prefix', kstar_form='prefixed', p1_form='plain', flag=None)
    for key in doc:                                                # every key of the index's shared interface is asserted
        (tmp_path / f'no_{key}.json').write_text(json.dumps({k: v for k, v in doc.items() if k != key}))
        with pytest.raises(AssertionError, match=key):
            TC.read_p1_form(tmp_path / f'no_{key}.json', 'siglip1_prefix')
    with pytest.raises(RuntimeError, match='prefix_pick'):
        TC.read_p1_form(tmp_path / 'missing.json', 'siglip1')


def test_a_losing_arm_is_refused(tmp_path):
    """Review must-fix 2: `prefix_pick.json` names K*; the other arm gets no test cells."""
    f = write_prefix_pick(tmp_path / 'a.json', kstar='siglip1', kstar_form='plain', p1_form='plain')
    with pytest.raises(RuntimeError, match='K\\* is'):
        TC.read_p1_form(f, 'siglip1_prefix')
    f = write_prefix_pick(tmp_path / 'b.json', kstar='siglip1_prefix')
    with pytest.raises(RuntimeError, match='K\\* is'):
        TC.read_p1_form(f, 'siglip1')


def test_p1_spec_plain_is_r23s_side_row_prefixed_is_tickets_04_rows(tmp_path):
    r23 = {'picks': {'caption': {'siglip_so400m': {a: {d: 'center' for d in TC.DIRECTIONS} for a in TC.ASPECTS}}}}
    plain = TC.p1_spec('plain', r23_stage1=r23, p1_pick_record=None)
    assert plain['row'] == 'siglip_so400m' and plain['root'] == TC.R23_OUT and plain['picks'] == r23['picks']['caption']['siglip_so400m']
    rec = {'picks': {a: {d: dict(pick='r0', mean=0.1, range=None) for d in TC.DIRECTIONS} for a in TC.ASPECTS}}
    pre = TC.p1_spec('prefixed', r23_stage1=r23, p1_pick_record=rec)
    assert pre['row'] == 'siglip_so400m' and pre['root'] == X.score_root('p1_siglip1_prefix')
    assert pre['picks'] == {a: {d: 'r0' for d in TC.DIRECTIONS} for a in TC.ASPECTS}
    with pytest.raises(AssertionError):
        TC.p1_spec('prefixed', r23_stage1=r23, p1_pick_record=None)


def test_q1_spec_is_r23s_teacher_at_r23s_pick():
    r23 = {'picks': {'caption': {'teacher': {a: {d: 'center_k10' for d in TC.DIRECTIONS} for a in TC.ASPECTS}}}}
    q1 = TC.q1_spec(r23)
    assert q1['row'] == 'teacher' and q1['root'] == TC.R23_OUT and q1['picks']['genre']['t2i'] == 'center_k10'


# ------------------------------------------------------------------------------------------------ refusal (order rule)
def records_dir(tmp_path, pass_rule=True):
    d = tmp_path / 'records'
    d.mkdir()
    if pass_rule:
        (d / 'pass_rule.json').write_text(json.dumps(dict(status='fail')))      # D9: present, not its status, is checked
    return d


def test_cells_refuse_without_the_pick_record(tmp_path):
    with mock.patch.object(X, 'RECORDS_DIR', records_dir(tmp_path)), mock.patch.object(X, 'SMOKE', True):
        with pytest.raises(RuntimeError, match='no pick record'):
            TC.cells('siglip1', draws=5)
    assert not list(tmp_path.glob('**/text_cells_*.json'))


def test_cells_refuse_a_record_not_committed_unchanged(tmp_path):
    """The pick record is read through ticket 04's `require_pick_record`: its git check (outside smoke) is the refusal."""
    with mock.patch.object(X, 'RECORDS_DIR', records_dir(tmp_path)), mock.patch.object(X, 'SMOKE', True):
        with mock.patch.object(X, 'require_pick_record', side_effect=RuntimeError('not committed unchanged in HEAD')) as req:
            with pytest.raises(RuntimeError, match='not committed'):
                TC.cells('siglip1', draws=5)
    req.assert_called_once_with('siglip1')


def test_cells_refuse_an_unknown_arm():
    with pytest.raises(AssertionError):
        TC.cells('qwen', draws=5)


def test_cells_refuse_without_the_held_record(tmp_path):
    rec = {'picks': {a: {d: dict(pick='r0', mean=0.1, range=0.01) for d in TC.DIRECTIONS} for a in TC.ASPECTS}}
    with mock.patch.object(X, 'RECORDS_DIR', records_dir(tmp_path)), mock.patch.object(X, 'SMOKE', True):
        with mock.patch.object(X, 'require_pick_record', return_value=rec):
            with pytest.raises(RuntimeError, match='held record'):
                TC.cells('siglip1', draws=5)


def test_held_record_picks_must_equal_the_pick_record(tmp_path):
    pick = {'picks': {a: {d: dict(pick='r0', mean=0.1, range=0.01) for d in TC.DIRECTIONS} for a in TC.ASPECTS}}
    held = {'picks': {a: {d: 'r0' for d in TC.DIRECTIONS} for a in TC.ASPECTS}, 'cells': {}}
    TC.check_held_record(pick, held)
    held['picks']['genre']['i2t'] = 'center'
    with pytest.raises(AssertionError, match='genre'):
        TC.check_held_record(pick, held)


# ------------------------------------------------------------------------------------------------ cross-checks
def test_check_value_tolerance():
    TC.check_value('x', 0.5, 0.5 + 0.9e-9)
    with pytest.raises(AssertionError, match='x'):
        TC.check_value('x', 0.5, 0.5 + 2e-9)
    with pytest.raises(AssertionError):
        TC.check_value('x', float('nan'), 0.5)


def test_cross_check_values_against_the_records(tmp_path):
    cand, p1, q1 = full_tree(tmp_path)
    cells = TC.score_cells('siglip1', 'siglip1', cand, p1, q1, draws=5)
    held = {'cells': {a: {d: dict(mean=[c for c in cells if (c['aspect'], c['direction'], c['comparator']) == (a, d, 'p1')][0]['value'])
                          for d in TC.DIRECTIONS} for a in TC.ASPECTS}}
    TC.check_candidate_values(cells, held)
    held['cells']['style']['t2t']['mean'] += 1e-6
    with pytest.raises(AssertionError, match='style'):
        TC.check_candidate_values(cells, held)


def test_check_comparator_values(tmp_path):
    cand, p1, q1 = full_tree(tmp_path)
    cells = TC.score_cells('siglip1', 'siglip1', cand, p1, q1, draws=5)
    refs = {k: {a: {d: [c for c in cells if (c['aspect'], c['direction'], c['comparator']) == (a, d, k)][0]['comparator_value']
                    for d in TC.DIRECTIONS} for a in TC.ASPECTS} for k in ('p1', 'q1')}
    TC.check_comparator_values(cells, refs)
    refs['q1']['genre']['t2i'] += 1e-6
    with pytest.raises(AssertionError, match='genre/t2i/q1'):
        TC.check_comparator_values(cells, refs)


# ------------------------------------------------------------------------------------------------ gate (v) text
def rec_cell(c, **over):
    r = dict(sc='SC1', comparator=c['comparator_row'], aspect=c['aspect'], direction=c['direction'],
             point=c['d'], lower=c['lo'], upper=c['hi'], outcome=c['outcome'], seed_range=c['r'],
             candidate_readout=c['candidate_readout'], comparator_readout=c['comparator_readout'])
    r.update(over)
    return r


def test_compare_to_r23_tolerance_and_outcome(tmp_path):
    cand, p1, q1 = full_tree(tmp_path)
    cells = TC.score_cells('siglip1', 'siglip1', cand, p1, q1, draws=20)
    recs = [rec_cell(c) for c in cells]
    items = TC.compare_to_r23(cells, recs, check_interval=True)
    assert len(items) == 18 and all(i['ok'] for i in items) and max(i['max_abs_diff'] for i in items) == 0.0
    recs[0] = rec_cell(cells[0], lower=cells[0]['lo'] + 0.9e-9)
    assert all(i['ok'] for i in TC.compare_to_r23(cells, recs, check_interval=True))
    recs[0] = rec_cell(cells[0], lower=cells[0]['lo'] + 2e-9)
    assert not TC.compare_to_r23(cells, recs, check_interval=True)[0]['ok']
    recs[0] = rec_cell(cells[0], outcome='WIN' if cells[0]['outcome'] != 'WIN' else 'BELOW')
    assert not TC.compare_to_r23(cells, recs, check_interval=True)[0]['ok']
    assert TC.compare_to_r23(cells, recs, check_interval=False)[0]['ok']          # a reduced replay checks no interval or outcome


def test_compare_to_r23_needs_every_cell_once(tmp_path):
    cand, p1, q1 = full_tree(tmp_path)
    cells = TC.score_cells('siglip1', 'siglip1', cand, p1, q1, draws=5)
    recs = [rec_cell(c) for c in cells]
    with pytest.raises(AssertionError):
        TC.compare_to_r23(cells, recs[1:], check_interval=True)
    with pytest.raises(AssertionError):
        TC.compare_to_r23(cells, recs + [recs[0]], check_interval=True)


# ------------------------------------------------------------------------------------------------ `cells` end to end
def file_mean(root, row, head, seeds, direction, readout, aspect):
    vals = [np.load(Path(root) / 'scores' / 'art_held' / 'caption' / row / f'{head}__s{sd}__{direction}__{readout}.npz')
            [f'ap__{aspect}'].astype(np.float64) for sd in seeds]
    return float(np.mean(vals))


class World:
    """r23's folder (rows cstar, teacher, plain, siglip_so400m), a second root for the prefixed P1, r23's stage 1 and r28's
    text.json, with the means r28 would hold, on small synthetic files. `patch()` points the module at it."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.root, self.pre = tmp_path / 'r23', tmp_path / 'p1pre'
        rng = np.random.default_rng(21)
        picks = {k: {a: {d: v for d in TC.DIRECTIONS} for a in TC.ASPECTS}
                 for k, v in (('cstar', 'r0'), ('teacher', 'center_k10'), ('plain', 'center'), ('siglip_so400m', 'center'))}
        self.stage1 = dict(group='art', picks={'caption': picks},
                           ranges={'caption': {'cstar': {a: {d: 0.004 for d in TC.DIRECTIONS} for a in TC.ASPECTS}}})
        for root, rows in ((self.root, ('plain', 'siglip_so400m')), (self.pre, ('siglip_so400m',))):
            for row in rows:
                for d in TC.DIRECTIONS:
                    n = len(pid_of(d))
                    write_unit(root, row, 'all', 'x', d, 'center', {a: rng.uniform(0.05, 0.95, n) for a in TC.ASPECTS})
        for a in TC.ASPECTS:
            for d in TC.DIRECTIONS:
                make_arm(self.root, 'cstar', a, range(5), d, 'r0', rng, aspect=a)
                make_arm(self.root, 'teacher', a, ['x'], d, 'center_k10', rng, aspect=a)
        self.r28 = {key: self.means(root, row, head, seeds, rd) for key, root, row, head, seeds, rd in (
            ('ours_qwen3', self.root, 'cstar', None, range(5), 'r0'), ('instructed_qwen3', self.root, 'teacher', None, ['x'], 'center_k10'),
            ('plain_qwen3', self.root, 'plain', 'all', ['x'], 'center'), ('plain_siglip1', self.root, 'siglip_so400m', 'all', ['x'], 'center'),
            ('p1_prefix', self.pre, 'siglip_so400m', 'all', ['x'], 'center'))}
        (tmp_path / 'stage1.json').write_text(json.dumps(self.stage1))
        self.write_r28()

    def means(self, root, row, head, seeds, readout):
        return {a: {d: file_mean(root, row, head or a, seeds, d, readout, a) for d in TC.DIRECTIONS} for a in TC.ASPECTS}

    def write_r28(self):
        keys = ('ours_qwen3', 'instructed_qwen3', 'plain_qwen3', 'plain_siglip1')
        (self.tmp / 'r28_text.json').write_text(json.dumps({k: {'cells': {a: {d: {'mean': self.r28[k][a][d]} for d in TC.DIRECTIONS}
                                                                          for a in TC.ASPECTS}} for k in keys}))

    def pick_record(self, readout='r0'):
        return {'picks': {a: {d: dict(pick=readout, mean=0.5, range=0.004) for d in TC.DIRECTIONS} for a in TC.ASPECTS}}

    def held_record(self, means, readout='r0'):
        return {'picks': {a: {d: readout for d in TC.DIRECTIONS} for a in TC.ASPECTS},
                'cells': {a: {d: dict(mean=means[a][d]) for d in TC.DIRECTIONS} for a in TC.ASPECTS}}

    def patch(self, stack, records):
        stack.enter_context(mock.patch.object(TC, 'R23_OUT', self.root))
        stack.enter_context(mock.patch.object(TC, 'R23_STAGE1', self.tmp / 'stage1.json'))
        stack.enter_context(mock.patch.object(TC, 'R28_TEXT', self.tmp / 'r28_text.json'))
        stack.enter_context(mock.patch.object(X, 'RECORDS_DIR', records))
        stack.enter_context(mock.patch.object(X, 'SMOKE', True))
        stack.enter_context(mock.patch.object(X, 'score_root', lambda arm: {'siglip1': self.root, 'siglip1_prefix': self.root,
                                                                           'p1_siglip1_prefix': self.pre}[arm]))
        stack.enter_context(mock.patch.object(X, 'require_pick_record', lambda arm: self.pick_record('center' if arm == 'p1_siglip1_prefix' else 'r0')))


@pytest.fixture
def world(tmp_path):
    from contextlib import ExitStack
    w = World(tmp_path)
    w.records = records_dir(tmp_path)
    w.set_pick = lambda **kw: write_prefix_pick(w.records / 'prefix_pick.json', **kw)
    w.set_held = lambda arm, means: (w.records / f'text_{arm}_held.json').write_text(
        json.dumps(w.held_record(means, 'center' if arm == 'p1_siglip1_prefix' else 'r0')))
    for arm in ('siglip1', 'siglip1_prefix', 'p1_siglip1_prefix'):     # the files `require_pick_record` (patched) reads
        (w.records / f'text_{arm}.json').write_text(json.dumps(w.pick_record()))
    with ExitStack() as stack:
        w.patch(stack, w.records)
        yield w


def test_cells_end_to_end_plain_p1(world):
    world.set_pick(kstar='siglip1', kstar_form='plain', p1_form='plain')
    world.set_held('siglip1', world.r28['ours_qwen3'])
    path, rec = TC.cells('siglip1', draws=20)
    assert path == world.records / 'text_cells_siglip1.json' and json.loads(path.read_text())['arm'] == 'siglip1'
    assert len(rec['cells']) == 18 and rec['p1_form'] == 'plain'
    p1 = [c for c in rec['cells'] if c['comparator'] == 'p1']
    assert len(p1) == 9 and all(c['comparator_row'] == 'siglip_so400m' and c['p1_form'] == 'plain' for c in p1)
    assert all(c['comparator_row'] == 'teacher' for c in rec['cells'] if c['comparator'] == 'q1')
    assert all(rec[k] for k in ('pick_record', 'held_record', 'prefix_pick'))


def test_cells_end_to_end_prefixed_p1(world):
    world.set_pick(kstar='siglip1_prefix', kstar_form='prefixed', p1_form='prefixed')
    world.set_held('siglip1_prefix', world.r28['ours_qwen3'])
    world.set_held('p1_siglip1_prefix', world.r28['p1_prefix'])
    path, rec = TC.cells('siglip1_prefix', draws=20)
    p1 = [c for c in rec['cells'] if c['comparator'] == 'p1']
    assert rec['p1_form'] == 'prefixed' and all(c['p1_form'] == 'prefixed' for c in p1)
    for c in p1:                                                    # read from ticket 04's root, checked against its held record
        assert c['comparator_value'] == pytest.approx(world.r28['p1_prefix'][c['aspect']][c['direction']], abs=1e-9)


def test_cells_refuse_the_losing_arm(world):
    world.set_pick(kstar='siglip1', kstar_form='plain', p1_form='plain')
    world.set_held('siglip1_prefix', world.r28['ours_qwen3'])
    with pytest.raises(RuntimeError, match='K\\* is'):
        TC.cells('siglip1_prefix', draws=5)
    assert not (world.records / 'text_cells_siglip1_prefix.json').exists()


def test_cells_refuse_without_pass_rule_record(world):
    world.set_pick(kstar='siglip1', kstar_form='plain', p1_form='plain')
    world.set_held('siglip1', world.r28['ours_qwen3'])
    (world.records / 'pass_rule.json').unlink()
    with pytest.raises(RuntimeError, match='pass_rule'):
        TC.cells('siglip1', draws=5)
    assert not (world.records / 'text_cells_siglip1.json').exists()


def test_cells_check_the_candidate_values(world):
    world.set_pick(kstar='siglip1', kstar_form='plain', p1_form='plain')
    means = {a: dict(v) for a, v in world.r28['ours_qwen3'].items()}
    means['genre']['t2t'] += 1e-6
    world.set_held('siglip1', means)
    with pytest.raises(AssertionError, match='candidate value'):
        TC.cells('siglip1', draws=5)
    assert not (world.records / 'text_cells_siglip1.json').exists()


@pytest.mark.parametrize('key, match', [('instructed_qwen3', r'/q1 comparator value'), ('plain_siglip1', r'/p1 comparator value')])
def test_cells_check_the_comparator_values(world, key, match):
    world.set_pick(kstar='siglip1', kstar_form='plain', p1_form='plain')
    world.set_held('siglip1', world.r28['ours_qwen3'])
    world.r28[key]['style']['i2t'] += 1e-6
    world.write_r28()
    with pytest.raises(AssertionError, match=match):
        TC.cells('siglip1', draws=5)
    assert not (world.records / 'text_cells_siglip1.json').exists()


def test_cells_check_the_prefixed_p1_values_against_its_held_record(world):
    world.set_pick(kstar='siglip1_prefix', kstar_form='prefixed', p1_form='prefixed')
    world.set_held('siglip1_prefix', world.r28['ours_qwen3'])
    bad = {a: dict(v) for a, v in world.r28['p1_prefix'].items()}
    bad['emotion']['t2i'] += 1e-6
    world.set_held('p1_siglip1_prefix', bad)
    with pytest.raises(AssertionError, match='comparator value'):
        TC.cells('siglip1_prefix', draws=5)


# ------------------------------------------------------------------------------------------------ order rule with real git
def git_repo(tmp_path):
    import subprocess
    repo = tmp_path / 'repo'
    (repo / 'records').mkdir(parents=True)
    run = lambda *a: subprocess.run(['git', '-C', str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run('init', '-q')
    run('config', 'user.email', 'a@b.c')
    run('config', 'user.name', 'x')
    return repo, run


def test_require_committed_with_git(tmp_path):
    repo, run = git_repo(tmp_path)
    f = repo / 'records' / 'a.json'
    f.write_text('{"a": 1}')
    with mock.patch.object(TC.P29, 'REPO', repo), mock.patch.object(X, 'SMOKE', False):
        with pytest.raises(RuntimeError, match='not committed'):
            TC.require_committed(f)                                  # untracked
        run('add', 'records/a.json')
        run('commit', '-q', '-m', 'a')
        TC.require_committed(f)                                      # committed unchanged
        f.write_text('{"a": 2}')
        with pytest.raises(RuntimeError, match='not committed'):
            TC.require_committed(f)                                  # changed since
        with pytest.raises(RuntimeError, match='no record'):
            TC.require_committed(repo / 'records' / 'none.json')


@pytest.mark.parametrize('commit_pass_rule, commit_prefix, match', [
    (False, True, 'pass_rule.json is not committed'),
    (True, False, 'prefix_pick.json is not committed'),
])
def test_cells_refuse_records_not_committed(world, tmp_path, commit_pass_rule, commit_prefix, match):
    """`cells` names both records in HEAD: an uncommitted pass_rule.json or prefix_pick.json is a refusal, before any number."""
    repo, run = git_repo(tmp_path)
    rec = repo / 'records'
    world.set_pick(kstar='siglip1', kstar_form='plain', p1_form='plain')
    world.set_held('siglip1', world.r28['ours_qwen3'])
    for name in ('pass_rule.json', 'prefix_pick.json', 'text_siglip1_held.json'):
        (rec / name).write_text((world.records / name).read_text())
    for name, yes in (('pass_rule.json', commit_pass_rule), ('prefix_pick.json', commit_prefix)):
        if yes:
            run('add', f'records/{name}')
    if commit_pass_rule or commit_prefix:
        run('commit', '-q', '-m', 'x')
    with mock.patch.object(TC.P29, 'REPO', repo), mock.patch.object(X, 'RECORDS_DIR', rec), mock.patch.object(X, 'SMOKE', False):
        with pytest.raises(RuntimeError, match=match):
            TC.cells('siglip1', draws=5)
    assert not (rec / 'text_cells_siglip1.json').exists()


# ------------------------------------------------------------------------------------------------ gate (v) text, end to end
GATE_DRAWS = 20


@pytest.fixture
def gate_world(world):
    """The gate on the synthetic world, with r23's `stage2/cells.json` made from the same code (so every cell agrees) at a
    reduced draw count, which the test makes the module's N_DRAWS (the gate then checks intervals)."""
    r23 = world.stage1
    cand_picks = {a: {d: dict(pick='r0', range=0.004) for d in TC.DIRECTIONS} for a in TC.ASPECTS}
    plain = dict(row='plain', root=world.root, picks=r23['picks']['caption']['plain'], form='x')
    cells = TC.score_cells('r23_cstar', 'qwen', dict(root=world.root, picks=cand_picks), plain, TC.q1_spec(r23), GATE_DRAWS)
    (world.root / 'stage2').mkdir()
    (world.root / 'stage2' / 'cells.json').write_text(json.dumps([rec_cell(c, group='art', cell='label') for c in cells]))
    with mock.patch.object(TC, 'N_DRAWS', GATE_DRAWS):
        yield world


def test_gate_v_text_passes_and_stores_the_values(gate_world):
    ok, det = TC.gate_v_text(draws=GATE_DRAWS)
    assert ok and det['intervals_checked'] and det['n_cells'] == 18 and det['max_abs_diff'] <= 1e-12
    c = det['cells'][0]
    assert set(c['replayed']) == {'d', 'r', 'lo', 'hi'} and set(c['recorded']) == {'d', 'r', 'lo', 'hi'}
    assert set(c['replayed_means']) == {'value', 'comparator_value'} == set(c['r28_means'])
    assert c['replayed']['lo'] == c['recorded']['lo']
    p1 = det['p1_plain_check']
    assert p1['n_cells'] == 9 and len(p1['cells']) == 9 and sum(p1['outcomes'].values()) == 9
    assert all(i['id'].endswith('/p1') and i['comparator_row'] == 'siglip_so400m' for i in p1['cells'])
    i = p1['cells'][0]
    assert {'value', 'comparator_value', 'r28_value', 'r28_comparator_value', 'd', 'lo', 'hi', 'r', 'outcome'} <= set(i)
    assert i['comparator_value'] == pytest.approx(gate_world.r28['plain_siglip1'][i['id'].split('/')[3]][i['id'].split('/')[4]], abs=1e-12)


@pytest.mark.parametrize('key', ['ours_qwen3', 'instructed_qwen3', 'plain_qwen3', 'plain_siglip1'])
def test_gate_v_text_fails_when_a_cell_mean_differs_from_r28(gate_world, key):
    """Each of r28's four rows is checked: the candidate, Q1, the one-vector row, and the P1 plain route."""
    gate_world.r28[key]['genre']['i2t'] += 1e-6
    gate_world.write_r28()
    ok, det = TC.gate_v_text(draws=GATE_DRAWS)
    assert not ok


def test_gate_v_text_fails_when_r23_s_cell_differs(gate_world):
    path = gate_world.root / 'stage2' / 'cells.json'
    recs = json.loads(path.read_text())
    recs[3]['lower'] += 1e-6
    path.write_text(json.dumps(recs))
    ok, det = TC.gate_v_text(draws=GATE_DRAWS)
    assert not ok


def test_gate_v_text_checks_the_cell_means_with_a_reduced_run_too(gate_world):
    gate_world.r28['ours_qwen3']['style']['t2t'] += 1e-6
    gate_world.write_r28()
    ok, det = TC.gate_v_text(aspects=('style',), draws=5)
    assert not ok and not det['intervals_checked']


# ------------------------------------------------------------------------------------------------ the gate record
def det(aspects=None, intervals=True):
    return dict(aspects=list(aspects or TC.ASPECTS), intervals_checked=intervals)


def test_a_full_run_writes_the_gate_record_and_counts_tries(tmp_path):
    with mock.patch.object(X, 'RECORDS_DIR', tmp_path / 'records'):
        path, rec = TC.write_gate(True, det(), partial_dir=tmp_path / 'partial')
        assert path == tmp_path / 'records' / 'gates' / 'v_text.json' and rec['status'] == 'pass' and rec['tries'] == 1
        _, rec = TC.write_gate(False, det(), partial_dir=tmp_path / 'partial')
        assert rec['status'] == 'fail' and rec['tries'] == 2
    assert not (tmp_path / 'partial').exists()


@pytest.mark.parametrize('d', [det(['style']), det(intervals=False), det(['emotion', 'genre'], intervals=False)])
def test_a_reduced_run_never_writes_the_gate_record(tmp_path, d):
    with mock.patch.object(X, 'RECORDS_DIR', tmp_path / 'records'):
        path, rec = TC.write_gate(True, d, partial_dir=tmp_path / 'partial')
        assert path == tmp_path / 'partial' / 'v_text.json' and rec['status'] == 'partial-pass'
        assert TC.write_gate(False, d, partial_dir=tmp_path / 'partial')[1]['status'] == 'partial-fail'
    assert not (tmp_path / 'records' / 'gates' / 'v_text.json').exists()


def test_a_third_failure_parks_and_a_fourth_try_raises(tmp_path):
    """Final review fix A, ruling 3: v_text goes through r29_captions.write_gate_record."""
    with mock.patch.object(X, 'RECORDS_DIR', tmp_path / 'records'):
        got = [TC.write_gate(False, det(), partial_dir=tmp_path / 'partial')[1] for _ in range(3)]
        assert [(r['status'], r['tries']) for r in got] == [('fail', 1), ('fail', 2), ('parked', 3)]
        assert set(got[2]) == {'gate', 'status', 'tries', 'details', 'written'}
        with pytest.raises(AssertionError, match='fourth'):
            TC.write_gate(False, det(), partial_dir=tmp_path / 'partial')
