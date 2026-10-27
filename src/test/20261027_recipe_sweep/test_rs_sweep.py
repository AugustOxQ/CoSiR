"""Stages 1 and 2 (rule §3.2, §4.2): the gains, the guard, knob gains, the grid, the pool and TOP on synthetic A values,
the score-file layer on a tmp_path, and the scorer's refusals. No k-means runs here (V1a does that on frozen exports)."""
import json
from pathlib import Path

import numpy as np
import pytest

import rs_paths
import rs_sheet as H
import rs_sweep as S

A0 = dict(emotion=0.10, style=0.40, genre=0.50)
R = H.RANGES


def row(gains=(0.0, 0.0, 0.0), status='ok'):
    """A score row whose gains over A0 are `gains` in seed ranges (emotion, style, genre); status 'failed' has no A."""
    if status == 'failed':
        return dict(status='failed', cause='nonfinite loss', A=None, best_epoch=None)
    return dict(status='ok', cause=None, A={a: A0[a] + g * R[a] for a, g in zip(H.ASPECTS, gains)},
                best_epoch=dict(emotion=3, style=2, genre=4))


def one(k, v, *gains, **kw):
    return H.name(H.config({k: v})), row(gains, **kw)


def table(*items):
    """{'G': row at A0} plus the (name, row) items."""
    t = {'G': row()}
    t.update(dict(items))
    return t


def judged(*items):
    return S.judge_all(table(*items), A0)


# ------------------------------------------------------------------ the rule's numbers
def test_ranges_are_the_rules():
    assert S.RANGES == dict(emotion=0.0046, style=0.0082, genre=0.0128) == H.RANGES


def test_mean_gain_is_in_seed_ranges():
    r = S.RANGES
    assert S.mean_gain(dict(emotion=r['emotion'], style=r['style'], genre=r['genre'])) == pytest.approx(1.0)
    assert S.mean_gain(dict(emotion=3 * r['emotion'], style=0.0, genre=0.0)) == pytest.approx(1.0)
    assert S.mean_gain(dict(emotion=-r['emotion'], style=0.0, genre=2 * r['genre'])) == pytest.approx(1 / 3)


def test_gain_is_run_minus_reference():
    g = S.gain_of(dict(emotion=0.11, style=0.39, genre=0.5), A0)
    assert g == dict(emotion=pytest.approx(0.01), style=pytest.approx(-0.01), genre=0.0)


# ------------------------------------------------------------------ the guard
@pytest.mark.parametrize('aspect', H.ASPECTS)
def test_guard_edges(aspect):
    base = dict(emotion=0.0, style=0.0, genre=0.0)
    edge = -S.RANGES[aspect] / 2
    assert S.guard_ok(dict(base, **{aspect: edge}))                    # exactly half a seed range below: passes
    assert not S.guard_ok(dict(base, **{aspect: np.nextafter(edge, -1)}))
    assert S.guard_ok(dict(base, **{aspect: np.nextafter(edge, 0)}))
    assert S.guard_ok(dict(base, **{aspect: 1.0}))


def test_guard_needs_every_aspect():
    big = dict(emotion=1.0, style=1.0, genre=1.0)
    for a in H.ASPECTS:
        assert not S.guard_ok(dict(big, **{a: -S.RANGES[a]}))


def test_judge_marks_guard_and_counting():
    j = S.judge(row((4.0, 4.0, -0.6)), A0)                              # genre 0.6 r below: guard fails
    assert j['M'] == pytest.approx((4 + 4 - 0.6) / 3) and not j['guard'] and not j['counting']
    j = S.judge(row((4.0, 4.0, -0.4)), A0)
    assert j['guard'] and j['counting']


def test_failed_run_does_not_count_and_has_no_gain():
    j = S.judge(row(status='failed'), A0)
    assert not j['counting'] and j['M'] is None and j['gain'] is None and j['cause'] == 'nonfinite loss'


# ------------------------------------------------------------------ knob gains
def test_knob_gain_is_the_best_counting_m():
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k6', 5e-4, 2, 2, 2), one('k6', 2e-3, 9, 9, -3))   # the last fails the guard
    assert S.knob_gain(j, 'k6') == pytest.approx(2.0)


def test_knob_with_no_counting_value_has_no_gain():
    j = judged(one('k6', 3e-4, status='failed'), one('k6', 5e-4, 9, 9, -3), one('k6', 2e-3, status='failed'))
    assert S.knob_gain(j, 'k6') is None
    assert S.knob_gain(j, 'k7') is None                                 # no run at all


def test_knob_gain_can_be_negative():
    j = judged(one('k6', 3e-4, -0.2, -0.2, -0.2), one('k6', 5e-4, -0.4, -0.3, -0.3), one('k6', 2e-3, -2, -1, -1))   # the last: guard
    assert S.knob_gain(j, 'k6') == pytest.approx(-0.2)


def test_curve_lists_values_in_sheet_order_with_gains():
    j = judged(one('k6', 2e-3, 1, 0, 0), one('k6', 3e-4, 0, 3, 0))
    curve = S.knob_curve(j, 'k6')
    assert [c['value'] for c in curve] == list(H.VALUES['k6'])
    assert curve[0]['M'] == pytest.approx(1.0) and curve[0]['gain']['style'] == pytest.approx(3 * R['style'])
    assert curve[1]['name'] == 'k6=0.0005' and curve[1]['M'] is None    # not run


# ------------------------------------------------------------------ grid knobs, outcome
def test_grid_knobs_are_positive_only():
    g = {k: None for k in H.KNOBS}
    g.update(k2=0.0, k6=0.3, k10=-1.0)
    assert S.grid_knobs(g) == ['k6']
    assert S.stage1_outcome(['k6']) == 'GRID-1'


def test_no_positive_knob_gain_is_sweep_none():
    g = {k: None for k in H.KNOBS}
    g.update(k2=0.0, k6=-0.3)
    assert S.grid_knobs(g) == [] and S.stage1_outcome([]) == 'SWEEP-NONE'


def test_more_than_four_knobs_keep_the_four_largest():
    g = {k: None for k in H.KNOBS}
    g.update(k1=0.5, k3=0.9, k4=0.1, k7=0.7, k9=0.6, k12=0.2)
    assert S.grid_knobs(g) == ['k3', 'k7', 'k9', 'k1']                  # ordered by gain; k4, k12 dropped
    assert S.stage1_outcome(S.grid_knobs(g)) == 'GRID-4'


def test_ties_at_the_cut_go_to_the_sheets_order():
    g = {k: None for k in H.KNOBS}
    g.update(k9=0.5, k2=0.5, k5=0.5, k11=0.5, k1=0.5, k13=0.9)
    assert S.grid_knobs(g) == ['k13', 'k1', 'k2', 'k5']                 # k1 first among the tied; k9, k11 dropped


# ------------------------------------------------------------------ levels
def test_levels_are_g_plus_counting_positive_values():
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k6', 5e-4, -1, -1, -1), one('k6', 2e-3, 3, 3, 3))
    assert S.levels(j, 'k6') == [H.G['k6'], 3e-4, 2e-3]                 # G first, then the sheet's order


def test_levels_keep_the_two_best_and_ties_go_to_listed_order():
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k6', 5e-4, 2, 2, 2), one('k6', 2e-3, 2, 2, 2))
    assert S.levels(j, 'k6') == [H.G['k6'], 5e-4, 2e-3]
    j = judged(one('k6', 3e-4, 2, 2, 2), one('k6', 5e-4, 2, 2, 2), one('k6', 2e-3, 2, 2, 2))
    assert S.levels(j, 'k6') == [H.G['k6'], 3e-4, 5e-4]
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k6', 5e-4, 2, 2, 2), one('k6', 2e-3, 3, 3, 3))
    assert S.levels(j, 'k6') == [H.G['k6'], 5e-4, 2e-3]


def test_levels_with_fewer_than_two_positive_values():
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k6', 5e-4, 0, 0, 0), one('k6', 2e-3, status='failed'))   # M = 0 is not positive
    assert S.levels(j, 'k6') == [H.G['k6'], 3e-4]


def test_a_positive_m_that_fails_the_guard_is_no_level():
    j = judged(one('k6', 3e-4, 5, 5, -1), one('k6', 5e-4, 1, 1, 1))
    assert S.levels(j, 'k6') == [H.G['k6'], 5e-4]


# ------------------------------------------------------------------ the grid
def lv(**by_knob):
    return {k: [H.G[k], *v] for k, v in by_knob.items()}


def test_grid_of_two_knobs_reuses_stage_1_points():
    ks = ['k2', 'k6']
    pts = S.grid_points(ks, lv(k2=(2, 8), k6=(3e-4, 5e-4)))
    assert len(pts) == 9
    split = S.split_points(pts)
    assert split['g0'] == ['G']
    assert sorted(split['reused']) == sorted(['k2=2', 'k2=8', 'k6=0.0003', 'k6=0.0005'])
    assert len(split['new']) == 4 and all(n.count('=') == 2 for n in split['new'])
    assert [H.name(p) for p in pts] == split['order']
    assert len(set(split['order'])) == 9


def test_grid_with_a_single_level_knob_is_its_own_stage_1_point():
    pts = S.grid_points(['k6'], lv(k6=(3e-4,)))
    assert [H.name(p) for p in pts] == ['G', 'k6=0.0003']
    assert S.split_points(pts)['new'] == []


def test_four_knobs_at_three_levels_have_81_points_72_new():
    ks = ['k1', 'k2', 'k3', 'k4']
    pts = S.grid_points(ks, lv(k1=(2, 8), k2=(2, 8), k3=(4, 12), k4=(0.015, 0.025)))
    split = S.split_points(pts)
    assert len(pts) == 81 and len(split['g0']) == 1 and len(split['reused']) == 8 and len(split['new']) == 72
    for n in split['order']:
        assert H.name(H.parse(n)) == n


def test_grid_points_change_only_grid_knobs():
    pts = S.grid_points(['k2', 'k6'], lv(k2=(2,), k6=(3e-4,)))
    for p in pts:
        assert set(H.changed(p)) <= {'k2', 'k6'} and p['remove'] == ()


# ------------------------------------------------------------------ the pool and TOP
def test_pool_excludes_failed_guarded_and_nonpositive():
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k6', 5e-4, 0, 0, 0), one('k6', 2e-3, 9, 9, -3),
               one('k7', 0.0, status='failed'), one('k10', 0.2, 0.1, 0.1, 0.1))
    assert S.pool(j) == ['k6=0.0003', 'k10=0.2']


def test_pool_excludes_g0_and_g0prime_even_with_a_positive_m():
    j = S.judge_all({'G': row((5, 5, 5)), 'G-repeat': row((5, 5, 5)), 'k6=0.0003': row((1, 1, 1))}, A0)
    assert j['G']['M'] > 0 and j['G-repeat']['counting'] and S.pool(j) == ['k6=0.0003']


def test_pool_counts_a_reused_point_once():
    stage1 = judged(one('k6', 3e-4, 1, 1, 1))
    both = dict(stage1, **S.judge_all({'k6=0.0003': row((1, 1, 1)), 'k2=2+k6=0.0003': row((2, 2, 2))}, A0))
    assert S.pool(both) == ['k6=0.0003', 'k2=2+k6=0.0003']


def test_top_takes_the_three_largest_m():
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k6', 5e-4, 4, 4, 4), one('k7', 0.0, 3, 3, 3),
               one('k10', 0.2, 2, 2, 2), one('k10', 0.3, 0.5, 0.5, 0.5))
    assert S.top(j, S.pool(j)) == ['k6=0.0005', 'k7=0.0', 'k10=0.2']


def test_top_has_all_when_fewer_than_three():
    j = judged(one('k6', 3e-4, 1, 1, 1), one('k7', 0.0, 2, 2, 2))
    assert S.top(j, S.pool(j)) == ['k7=0.0', 'k6=0.0003']


def test_top_tie_break_fewer_changed_knobs_first():
    two = ('k2=8+k6=0.0003', row((1, 1, 1)))
    j = judged(two, one('k13', 30, 1, 1, 1), one('k9', 512, 1, 1, 1), one('k1', 2, 1, 1, 1))
    assert 'k2=8+k6=0.0003' not in S.top(j, S.pool(j))                  # the three one-knob runs fill TOP
    assert S.top(j, S.pool(j), n=4)[-1] == 'k2=8+k6=0.0003'


def test_top_tie_break_knob_numbers_compared_lexicographically():
    a = ('k2=2+k3=4', row((1, 1, 1)))                                   # numbers (2, 3), positions (0, 0)
    b = ('k1=16+k4=0.035', row((1, 1, 1)))                              # numbers (1, 4), positions (2, 2): numbers decide first
    c = ('k1=2+k13=30', row((1, 1, 1)))                                 # numbers (1, 13), positions (0, 2)
    j = judged(a, b, c)
    assert S.top(j, S.pool(j)) == ['k1=16+k4=0.035', 'k1=2+k13=30', 'k2=2+k3=4']
    assert H.knob_numbers(H.parse('k1=2+k13=30')) == (1, 13)


def test_top_tie_break_value_positions_last():
    j = judged(one('k2', 16, 1, 1, 1), one('k2', 2, 1, 1, 1), one('k2', 8, 1, 1, 1))
    assert S.top(j, S.pool(j)) == ['k2=2', 'k2=8', 'k2=16']
    a = ('k2=8+k6=0.0003', row((1, 1, 1)))                              # positions (1, 0)
    b = ('k2=2+k6=0.0005', row((1, 1, 1)))                              # positions (0, 1)
    j = judged(a, b)
    assert S.top(j, S.pool(j)) == ['k2=2+k6=0.0005', 'k2=8+k6=0.0003']


def test_top_larger_m_beats_every_tie_break():
    j = judged(one('k2', 2, 1, 1, 1), ('k2=8+k6=0.0003', row((1.001, 1.001, 1.001))))
    assert S.top(j, S.pool(j)) == ['k2=8+k6=0.0003', 'k2=2']


def test_top_none_when_the_pool_is_empty():
    j = judged(one('k6', 3e-4, 0, 0, 0), one('k7', 0.0, status='failed'))
    assert S.pool(j) == [] and S.top(j, []) == [] and S.stage2_outcome([]) == 'TOP-NONE'
    assert S.stage2_outcome(['k6=0.0003']) == 'TOP'


# ------------------------------------------------------------------ the whole stages on score files
def put_score(outputs, machine, name, rowx, seed=0, source='rs_trainer', **extra):
    path = S.score_path(outputs, machine, name, seed)
    rec = dict(stage='score', name=name, seed=seed, machine=machine, smoke=False, source=source, p_sha256=rs_paths.P_SHA256,
               code_sha256='x', **rowx, **extra)
    S._json_dump(path, json.loads(json.dumps(rec)))


def put_stage1_runs(outputs, machine, gains_by_name, skip=()):
    put_score(outputs, machine, 'G', row(), source=S.REF_SOURCE[machine])          # G₀: 12-27's run locally, the trainer's on the node
    for cfg in H.stage1():
        n = H.name(cfg)
        if n not in skip:
            put_score(outputs, machine, n, row(gains_by_name.get(n, (0.0, 0.0, 0.0))))


def test_stage1_refuses_when_a_run_is_missing(tmp_path):
    put_score(tmp_path, 'local', 'G', row())
    put_score(tmp_path, 'local', 'k1=2', row())
    with pytest.raises(AssertionError, match='missing'):
        S.run_stage1(tmp_path, 'local')


def test_stage1_writes_curves_gains_grid(tmp_path):
    put_stage1_runs(tmp_path, 'local', {'k6=0.0003': (1, 1, 1), 'k6=0.0005': (2, 2, 2), 'k2=8': (0.5, 0.5, 0.5),
                                        'k10=0.2': (3, 3, -1.0)})                            # the last fails the guard
    rec = S.run_stage1(tmp_path, 'local')
    assert rec['outcome'] == 'GRID-2' and rec['grid_knobs'] == ['k6', 'k2']
    assert rec['knob_gains']['k6'] == pytest.approx(2.0) and rec['knob_gains']['k10'] == pytest.approx(0.0)
    assert rec['levels'] == {'k6': [1e-3, 3e-4, 5e-4], 'k2': [0, 8]}
    assert len(rec['grid']['points']) == 6
    assert rec['grid']['new'] == sorted(rec['grid']['new'], key=rec['grid']['order'].index)
    assert set(rec['grid']['new']) == {'k2=8+k6=0.0003', 'k2=8+k6=0.0005'}
    assert rec['curves']['k10'][1]['guard'] is False
    assert rec['reference']['name'] == 'G' and rec['machine'] == 'local'
    assert json.loads((tmp_path / 'sweep/local/stage1.json').read_text()) == rec
    assert S.run_stage1(tmp_path, 'local') == rec                                           # rerun keeps the file


def test_stage1_sweep_none_lists_failed_runs(tmp_path):
    put_stage1_runs(tmp_path, 'local', {})
    put_score(tmp_path, 'local', 'k5=0.5', row(status='failed'))
    rec = S.run_stage1(tmp_path, 'local')
    assert rec['outcome'] == 'SWEEP-NONE' and rec['grid']['new'] == []
    assert rec['failed'] == [dict(name='k5=0.5', cause='nonfinite loss')]
    with pytest.raises(AssertionError, match='SWEEP-NONE'):
        S.run_stage2(tmp_path, 'local')


def test_machines_do_not_mix(tmp_path):
    put_stage1_runs(tmp_path, 'node402', {'k6=0.0003': (1, 1, 1)})
    with pytest.raises(AssertionError, match='missing'):
        S.run_stage1(tmp_path, 'local')
    assert S.run_stage1(tmp_path, 'node402')['outcome'] == 'GRID-1'


def test_a_score_file_of_another_machine_is_refused(tmp_path):
    put_stage1_runs(tmp_path, 'local', {})
    path = S.score_path(tmp_path, 'local', 'k1=2', 0)
    rec = json.loads(path.read_text())
    rec['machine'] = 'node402'
    path.write_text(json.dumps(rec))
    with pytest.raises(AssertionError, match='machine'):
        S.run_stage1(tmp_path, 'local')


def stage2_setup(tmp_path):
    put_stage1_runs(tmp_path, 'local', {'k6=0.0003': (1, 1, 1), 'k6=0.0005': (2, 2, 2), 'k2=8': (0.5, 0.5, 0.5)})
    return S.run_stage1(tmp_path, 'local')


def test_stage2_refuses_when_a_new_point_is_missing(tmp_path):
    stage2_setup(tmp_path)
    put_score(tmp_path, 'local', 'k2=8+k6=0.0003', row((3, 3, 3)))
    with pytest.raises(AssertionError, match='k2=8\\+k6=0.0005'):
        S.run_stage2(tmp_path, 'local')


def test_stage2_pool_top_and_outcome(tmp_path):
    stage2_setup(tmp_path)
    put_score(tmp_path, 'local', 'G-repeat', row((9, 9, 9)))                                # never in the pool
    put_score(tmp_path, 'local', 'k2=8+k6=0.0003', row((3, 3, 3)))
    put_score(tmp_path, 'local', 'k2=8+k6=0.0005', row(status='failed'))
    rec = S.run_stage2(tmp_path, 'local')
    assert rec['outcome'] == 'TOP'
    assert rec['pool'] == ['k2=8', 'k6=0.0003', 'k6=0.0005', 'k2=8+k6=0.0003']
    assert [t['name'] for t in rec['top']] == ['k2=8+k6=0.0003', 'k6=0.0005', 'k6=0.0003']
    assert rec['failed'] == [dict(name='k2=8+k6=0.0005', cause='nonfinite loss')]
    assert 'G-repeat' not in rec['pool'] and 'G' not in rec['pool']
    assert rec['stage1_sha256'] == S._file_sha256(tmp_path / 'sweep/local/stage1.json')
    assert json.loads((tmp_path / 'sweep/local/stage2.json').read_text()) == rec


def test_stage2_with_no_new_points(tmp_path):
    put_stage1_runs(tmp_path, 'local', {'k6=0.0003': (1, 1, 1)})
    S.run_stage1(tmp_path, 'local')
    rec = S.run_stage2(tmp_path, 'local')                                                   # no new points: GRID-1
    assert rec['pool'] == ['k6=0.0003'] and rec['outcome'] == 'TOP'


def test_stage2_refuses_stale_stage1(tmp_path):
    stage2_setup(tmp_path)
    put_score(tmp_path, 'local', 'k2=8+k6=0.0003', row((3, 3, 3)))
    put_score(tmp_path, 'local', 'k2=8+k6=0.0005', row((3, 3, 3)))
    path = S.score_path(tmp_path, 'local', 'k6=0.0003', 0)
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **row((0, 0, 0)))))      # a run changed after stage 1
    with pytest.raises(AssertionError, match='stale'):
        S.run_stage2(tmp_path, 'local')


# ------------------------------------------------------------------ N3 (pure part)
def test_n3_edges():
    r = S.RANGES
    a = dict(emotion=0.1, style=0.4, genre=0.5)
    ok = S.n3_items(a, {k: a[k] + r[k] / 2 * 0.999 for k in a})
    assert all(i['pass'] for i in ok.values())
    bad = S.n3_items(a, dict(a, genre=a['genre'] + r['genre'] / 2 * 1.001))
    assert not bad['genre']['pass'] and bad['emotion']['pass']
    assert S.n3_items(a, a)['style']['diff'] == 0.0


@pytest.mark.parametrize('aspect', ['emotion', 'style', 'genre'])
def test_n3_a_difference_of_exactly_r_a_over_2_passes(aspect):
    """Rule §4.2: `<= r_a / 2`, the edge passes. Base 0.0, so the float difference is exactly r_a / 2."""
    r = S.RANGES
    base = dict(emotion=0.0, style=0.0, genre=0.0)
    item = S.n3_items(base, dict(base, **{aspect: r[aspect] / 2}))[aspect]
    assert item['diff'] == item['limit'] == r[aspect] / 2 and item['pass'] is True
    item = S.n3_items(base, dict(base, **{aspect: -r[aspect] / 2}))[aspect]            # and downward
    assert item['diff'] == item['limit'] and item['pass'] is True
    over = S.n3_items(base, dict(base, **{aspect: np.nextafter(r[aspect] / 2, 1.0)}))[aspect]
    assert over['diff'] > over['limit'] and over['pass'] is False


# ------------------------------------------------------------------ the scorer's refusals
def write_run(folder, **run):
    folder.mkdir(parents=True, exist_ok=True)
    cfg = H.config({'k6': 3e-4})
    rec = dict(name='k6=0.0003', config=json.loads(json.dumps(cfg)), seed=0, status='ok', cause=None, machine='local',
               p_sha256=rs_paths.P_SHA256, best_epoch=dict(emotion=3, style=2, genre=4), smoke=False)
    rec.update(run)
    (folder / 'run.json').write_text(json.dumps(rec))
    return rec


def test_v1b_problems_flag_each_failure():
    good = np.zeros((rs_paths.N_P, rs_paths.WIDTH), np.float16)
    good[:, 0] = 1
    assert S.v1b_problems({a: good for a in H.ASPECTS}) == {}
    bad = good.copy()
    bad[5, :] = np.nan
    assert set(S.v1b_problems({'emotion': good, 'style': bad, 'genre': good[:10]})) == {'style', 'genre'}
    assert 'norm' in S.v1b_problems({'emotion': good * 2})['emotion']


def test_score_refuses_a_run_that_was_not_trained(tmp_path):
    with pytest.raises(AssertionError, match='run.json'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)


def test_score_refuses_another_p(tmp_path):
    write_run(rs_paths.run_dir(tmp_path, 'k6=0.0003', 0), p_sha256='0' * 64)
    with pytest.raises(AssertionError, match='P'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)


def test_score_refuses_a_run_of_another_machine_or_config(tmp_path):
    write_run(rs_paths.run_dir(tmp_path, 'k6=0.0003', 0), machine='node402')
    with pytest.raises(AssertionError, match='machine'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)
    write_run(rs_paths.run_dir(tmp_path, 'k6=0.0003', 0), config=json.loads(json.dumps(H.config({'k6': 5e-4}))))
    with pytest.raises(AssertionError, match='config'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)


def test_score_refuses_exports_that_fail_v1b(tmp_path):
    folder = rs_paths.run_dir(tmp_path, 'k6=0.0003', 0)
    write_run(folder)
    for a in H.ASPECTS:
        np.savez(folder / f'{a}__P.npz', img=np.zeros((10, 2048), np.float16))
    with pytest.raises(AssertionError, match='V1b'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)


def test_score_records_a_failed_run_without_reading_exports(tmp_path):
    folder = rs_paths.run_dir(tmp_path, 'k6=0.0003', 0)
    write_run(folder, status='failed', cause='nonfinite loss at epoch 2 (emotion)', failure_kind='nonfinite_loss')
    rec = S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)['k6=0.0003']
    assert rec['status'] == 'failed' and rec['A'] is None and rec['cause'].startswith('nonfinite')
    assert S.read_score(tmp_path, 'local', 'k6=0.0003', 0) == rec


def test_score_refuses_a_smoke_run_outside_a_smoke(tmp_path):
    write_run(rs_paths.run_dir(tmp_path, 'k6=0.0003', 0), smoke=True)
    with pytest.raises(AssertionError, match='smoke'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)


def test_frozen_g0_only_for_g_at_seed_0_locally(tmp_path):
    with pytest.raises(AssertionError):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0, frozen_g0=True)
    with pytest.raises(AssertionError):
        S.run_score(tmp_path, 'node402', ['G'], 0, frozen_g0=True)


def test_a_failed_run_needs_a_known_failure_kind_and_a_va3_failure_its_rerun(tmp_path):
    folder = rs_paths.run_dir(tmp_path, 'k6=0.0003', 0)
    write_run(folder, status='failed', cause='x')
    with pytest.raises(AssertionError, match='failure_kind'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)
    write_run(folder, status='failed', cause='x', failure_kind='va3')
    with pytest.raises(AssertionError, match='rerun'):
        S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)
    write_run(folder, status='failed', cause='x', failure_kind='va3', rerun_of='/somewhere')
    rec = S.run_score(tmp_path, 'local', ['k6=0.0003'], 0)['k6=0.0003']
    assert rec['status'] == 'failed' and rec['rerun_of'] == '/somewhere'


# ------------------------------------------------------------------ G₀'s source; the k11 gate
def test_g0_must_be_12_27s_run_locally_and_the_trainers_on_the_node(tmp_path):
    put_stage1_runs(tmp_path, 'local', {})
    put_score(tmp_path, 'local', 'G', row(), source='rs_trainer')
    with pytest.raises(AssertionError, match='12-27'):
        S.run_stage1(tmp_path, 'local')
    put_stage1_runs(tmp_path, 'node402', {})
    put_score(tmp_path, 'node402', 'G', row(), source='12-27')
    with pytest.raises(AssertionError, match='rs_trainer'):
        S.run_stage1(tmp_path, 'node402')


def put_k11_gate(outputs, passed, **extra):
    S._json_dump(Path(outputs) / 'gates' / 'k11_sample.json', dict(**{'pass': passed}, **extra))


K11 = ['k11=10', 'k11=30', 'k11=40']


def test_failed_k11_gate_enters_k11_runs_as_failed_with_the_cause(tmp_path):
    put_stage1_runs(tmp_path, 'local', {'k6=0.0003': (1, 1, 1)}, skip=K11)
    put_k11_gate(tmp_path, False, cause='graph arrays differ at k = 20')
    rec = S.run_stage1(tmp_path, 'local')
    assert rec['outcome'] == 'GRID-1'
    assert [f['name'] for f in rec['failed']] == K11 and all('graph arrays differ' in f['cause'] for f in rec['failed'])
    assert rec['knob_gains']['k11'] is None and all(not c['counting'] for c in rec['curves']['k11'])
    assert rec['inputs']['gates/k11_sample.json'] == S._file_sha256(Path(tmp_path) / 'gates/k11_sample.json')
    assert S.run_stage2(tmp_path, 'local')['outcome'] == 'TOP'                      # stage 2 reads the same table


def test_k11_runs_stay_missing_without_a_failed_gate(tmp_path):
    put_stage1_runs(tmp_path, 'local', {}, skip=K11)
    with pytest.raises(AssertionError, match='k11=10'):
        S.run_stage1(tmp_path, 'local')                                             # no gate file
    put_k11_gate(tmp_path, True)
    with pytest.raises(AssertionError, match='k11=10'):
        S.run_stage1(tmp_path, 'local')                                             # the gate passed: the runs must exist


def test_failed_k11_gate_refuses_k11_scores(tmp_path):
    put_stage1_runs(tmp_path, 'local', {})
    put_k11_gate(tmp_path, False)
    with pytest.raises(AssertionError, match='k11'):
        S.run_stage1(tmp_path, 'local')


def test_a_record_rewritten_with_equal_content_keeps_the_file_whatever_the_time_written(tmp_path):
    path = tmp_path / 'x' / 'verdict.json'
    first = S._keep_or_write(path, dict(value=1, written='2026-10-28 09:15', host='a', code_commit='c1', code_dirty=False))
    again = S._keep_or_write(path, dict(value=1, written='2026-10-28 11:40', host='b', code_commit='c2', code_dirty=True))
    assert again == first and json.loads(path.read_text())['written'] == '2026-10-28 09:15'
    with pytest.raises(AssertionError, match='other inputs'):
        S._keep_or_write(path, dict(value=2, written='2026-10-28 09:15'))
