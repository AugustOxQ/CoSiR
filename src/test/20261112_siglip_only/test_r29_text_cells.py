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


# ------------------------------------------------------------------------------------------------ P1's form (D20)
def write_prefix_pick(path, **kw):
    Path(path).write_text(json.dumps(kw))
    return path


def test_p1_form_is_read_and_asserted(tmp_path):
    f = write_prefix_pick(tmp_path / 'prefix_pick.json', p1_form='prefixed')
    assert TC.read_p1_form(f, 'siglip1_prefix') == 'prefixed'
    f = write_prefix_pick(tmp_path / 'p2.json', p1_form='plain')
    assert TC.read_p1_form(f, 'siglip1_prefix') == 'plain' and TC.read_p1_form(f, 'siglip1') == 'plain'


def test_p1_form_must_fit_the_arm_and_the_vocabulary(tmp_path):
    with pytest.raises(AssertionError):                            # K1 won: D20 never picks a prefixed P1
        TC.read_p1_form(write_prefix_pick(tmp_path / 'a.json', p1_form='prefixed'), 'siglip1')
    with pytest.raises(AssertionError):
        TC.read_p1_form(write_prefix_pick(tmp_path / 'b.json', p1_form='both'), 'siglip1_prefix')
    with pytest.raises(AssertionError):
        TC.read_p1_form(write_prefix_pick(tmp_path / 'c.json', other=1), 'siglip1_prefix')
    with pytest.raises(RuntimeError, match='prefix_pick'):
        TC.read_p1_form(tmp_path / 'missing.json', 'siglip1')


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
def test_cells_refuse_without_the_pick_record(tmp_path):
    with mock.patch.object(X, 'RECORDS_DIR', tmp_path):
        with pytest.raises(RuntimeError, match='no pick record'):
            TC.cells('siglip1', draws=5)
    assert not list(tmp_path.glob('text_cells_*.json'))


def test_cells_refuse_a_record_not_committed_unchanged(tmp_path):
    """The pick record is read through ticket 04's `require_pick_record`: its git check (outside smoke) is the refusal."""
    with mock.patch.object(X, 'require_pick_record', side_effect=RuntimeError('not committed unchanged in HEAD')) as req:
        with pytest.raises(RuntimeError, match='not committed'):
            TC.cells('siglip1', draws=5)
    req.assert_called_once_with('siglip1')


def test_cells_refuse_an_unknown_arm():
    with pytest.raises(AssertionError):
        TC.cells('qwen', draws=5)


def test_cells_refuse_without_the_held_record(tmp_path):
    rec = {'picks': {a: {d: dict(pick='r0', mean=0.1, range=0.01) for d in TC.DIRECTIONS} for a in TC.ASPECTS}}
    with mock.patch.object(X, 'RECORDS_DIR', tmp_path), mock.patch.object(X, 'require_pick_record', return_value=rec):
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
