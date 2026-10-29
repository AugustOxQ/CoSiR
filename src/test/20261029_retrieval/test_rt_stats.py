"""Tests for rt_stats (r11 ticket 03): verdict code, written test first."""
import numpy as np
import pytest

import rt_stats


# ---- outcome ---------------------------------------------------------------------------------------------------

def test_outcome_boundaries():
    r = 0.01
    assert rt_stats.outcome(1e-12, r) == 'WIN'
    assert rt_stats.outcome(0.0, r) == 'LEVEL'            # lower = 0 is not a WIN (strict)
    assert rt_stats.outcome(-0.005, r) == 'LEVEL'
    assert rt_stats.outcome(-r, r) == 'BELOW'             # lower = -r is not a LEVEL (strict)
    assert rt_stats.outcome(-r - 1e-12, r) == 'BELOW'
    assert rt_stats.outcome(-1.0, r) == 'BELOW'
    assert rt_stats.outcome(0.3, r) == 'WIN'


# ---- seed range and chance -------------------------------------------------------------------------------------

def test_seed_range_is_max_minus_min_of_five():
    assert rt_stats.seed_range([0.30, 0.34, 0.31, 0.29, 0.33]) == pytest.approx(0.05)
    assert rt_stats.seed_range(np.array([1.0, 1.0, 1.0, 1.0, 1.0])) == 0.0
    with pytest.raises(AssertionError):
        rt_stats.seed_range([0.1, 0.2, 0.3])


def test_chance_is_sum_of_squared_class_shares_over_labelled_rows():
    # classes 0,0,1,2 -> shares .5,.25,.25 -> .25+.0625+.0625
    assert rt_stats.chance(np.array([0, 0, 1, 2])) == pytest.approx(0.375)
    # genre-style: -1 rows are unlabelled and not counted
    assert rt_stats.chance(np.array([-1, 0, -1, 0, 1, 1])) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        rt_stats.chance(np.array([-1, -1]))


# ---- paired bootstrap ------------------------------------------------------------------------------------------

def hand_loop(cand, comp, draws):
    """The plain per-draw loop of the rule: I first, then S only when either side is seeded, shared by both."""
    rng = np.random.default_rng(0)
    n = cand.shape[-1]
    seeded = cand.ndim == 2 or comp.ndim == 2
    out = []
    for _ in range(draws):
        I = rng.integers(0, n, size=n)
        S = rng.choice([0, 1, 2, 3, 4], size=5, replace=True) if seeded else None

        def side(x):
            return float(np.mean([x[s][I].mean() for s in S])) if x.ndim == 2 else float(x[I].mean())
        out.append(side(cand) - side(comp))
    return np.array(out)


def _case(kind, n=7, seed=3):
    g = np.random.default_rng(seed)
    cand = g.random((5, n)) if kind[0] == 's' else g.random(n)
    comp = g.random((5, n)) if kind[1] == 's' else g.random(n)
    return cand, comp


@pytest.mark.parametrize('kind', ['uu', 'su', 'us', 'ss'])
def test_paired_matches_hand_loop_draw_order(kind):
    cand, comp = _case(kind)
    got = rt_stats.paired(cand, comp, draws=60)
    np.testing.assert_allclose(got['boot'], hand_loop(cand, comp, 60), rtol=0, atol=1e-12)
    assert got['boot'].shape == (60,)


def test_paired_draw_order_survives_chunking():
    # more draws than one chunk: the rng stream must run on across chunks
    cand, comp = _case('ss', n=5)
    got = rt_stats.paired(cand, comp, draws=1234)
    np.testing.assert_allclose(got['boot'], hand_loop(cand, comp, 1234), rtol=0, atol=1e-12)


def test_paired_point_value_and_interval():
    for kind in ['uu', 'su', 'us', 'ss']:
        cand, comp = _case(kind)
        got = rt_stats.paired(cand, comp, draws=200)
        assert got['d'] == pytest.approx(cand.mean() - comp.mean(), abs=1e-14)
        np.testing.assert_allclose(got['interval'], np.percentile(got['boot'], [2.5, 97.5]), rtol=0, atol=0)
        assert len(got['interval']) == 2


def test_paired_is_deterministic_per_call():
    cand, comp = _case('ss')
    a = rt_stats.paired(cand, comp, draws=50)
    b = rt_stats.paired(cand, comp, draws=50)
    np.testing.assert_array_equal(a['boot'], b['boot'])


def test_paired_seed_resample_is_shared_by_both_sides():
    # Seeds differ in level: candidate seed s has constant s, comparator constant 0 for every seed. Then every draw's
    # d_b is mean(S) (one S for both sides). If the sides had their own S, comparator's term would be 0 anyway, so
    # use comparator constant s too: d_b must be exactly 0 only under a shared S.
    n = 6
    cand = np.repeat(np.arange(5.0)[:, None], n, axis=1)
    comp = cand.copy()
    got = rt_stats.paired(cand, comp, draws=100)
    np.testing.assert_array_equal(got['boot'], np.zeros(100))


def test_paired_rejects_bad_shapes():
    g = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        rt_stats.paired(g.random(5), g.random(6), draws=3)             # different n
    with pytest.raises(AssertionError):
        rt_stats.paired(g.random((4, 5)), g.random(5), draws=3)        # not five seeds
    with pytest.raises(AssertionError):
        rt_stats.paired(g.random(5).astype(np.float32), g.random(5), draws=3)   # dtype


# ---- the SC table ----------------------------------------------------------------------------------------------

ASPECTS = ('emotion', 'style', 'genre')


def _cmp(sc_id):
    return [(c.candidate, c.comparator, c.bar) for c in rt_stats.SC[sc_id].comparisons]


def test_sc_table_has_fifteen_ids_in_order():
    assert list(rt_stats.SC) == [f'SC{i}' for i in range(1, 16)]


@pytest.mark.parametrize('i,aspect', [(1, 'emotion'), (2, 'style'), (3, 'genre')])
def test_sc1_to_sc3_cstar_vs_plain_win(i, aspect):
    sc = rt_stats.SC[f'SC{i}']
    assert (sc.stage, sc.aspect, sc.readout, sc.metric) == (1, aspect, 'R0', 'map')
    assert _cmp(f'SC{i}') == [(('cstar', aspect), ('plain', None), 'WIN')]


@pytest.mark.parametrize('i,aspect', [(4, 'emotion'), (5, 'style'), (6, 'genre')])
def test_sc4_to_sc6_cstar_vs_instructed_level_or_win(i, aspect):
    sc = rt_stats.SC[f'SC{i}']
    assert (sc.stage, sc.aspect, sc.readout) == (1, aspect, 'R0')
    assert _cmp(f'SC{i}') == [(('cstar', aspect), ('instructed', aspect), 'LEVEL_OR_WIN')]


@pytest.mark.parametrize('i,a', [(7, 'emotion'), (8, 'style'), (9, 'genre')])
def test_sc7_to_sc9_head_a_against_each_other_head_in_column_a(i, a):
    sc = rt_stats.SC[f'SC{i}']
    assert (sc.stage, sc.aspect, sc.readout) == (2, a, 'R0')
    others = [h for h in ASPECTS if h != a]
    assert _cmp(f'SC{i}') == [(('cstar', a), ('cstar', h), 'WIN') for h in others]
    assert len(sc.comparisons) == 2


@pytest.mark.parametrize('i', range(10, 16))
def test_sc10_to_sc15_are_sc1_to_sc6_on_stage_5_with_picks(i):
    sc, base = rt_stats.SC[f'SC{i}'], rt_stats.SC[f'SC{i - 9}']
    assert (sc.stage, sc.readout) == (5, 'pick')
    assert (sc.aspect, sc.comparisons, sc.metric) == (base.aspect, base.comparisons, base.metric)


def test_sc_rows_are_frozen_data():
    with pytest.raises(AttributeError):
        rt_stats.SC['SC1'].aspect = 'style'


# ---- pass rules ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('o,ok', [('WIN', True), ('LEVEL', False), ('BELOW', False)])
def test_win_bar_needs_win(o, ok):
    for sc in ('SC1', 'SC2', 'SC3', 'SC10', 'SC11', 'SC12'):
        assert rt_stats.sc_pass(sc, [o]) is ok
        assert rt_stats.sc_pass(sc, o) is ok                 # one comparison: a bare outcome is accepted


@pytest.mark.parametrize('o,ok', [('WIN', True), ('LEVEL', True), ('BELOW', False)])
def test_level_or_win_bar(o, ok):
    for sc in ('SC4', 'SC5', 'SC6', 'SC13', 'SC14', 'SC15'):
        assert rt_stats.sc_pass(sc, [o]) is ok


@pytest.mark.parametrize('a,b,ok', [('WIN', 'WIN', True), ('WIN', 'LEVEL', False), ('LEVEL', 'WIN', False),
                                    ('WIN', 'BELOW', False), ('BELOW', 'WIN', False), ('LEVEL', 'LEVEL', False)])
def test_switch_sc_needs_win_against_both(a, b, ok):
    for sc in ('SC7', 'SC8', 'SC9'):
        assert rt_stats.sc_pass(sc, [a, b]) is ok


def test_sc_pass_checks_the_number_of_outcomes():
    with pytest.raises(AssertionError):
        rt_stats.sc_pass('SC7', ['WIN'])
    with pytest.raises(AssertionError):
        rt_stats.sc_pass('SC7', 'WIN')
    with pytest.raises(AssertionError):
        rt_stats.sc_pass('SC1', ['WIN', 'WIN'])
    with pytest.raises(AssertionError):
        rt_stats.sc_pass('SC1', ['GOOD'])
    with pytest.raises(KeyError):
        rt_stats.sc_pass('SC16', ['WIN'])


# ---- GO logic --------------------------------------------------------------------------------------------------

def _passing(ids):
    return {i: (['WIN', 'WIN'] if i in ('SC7', 'SC8', 'SC9') else ['WIN']) for i in ids}


STAGE1, STAGE5 = [f'SC{i}' for i in range(1, 7)], [f'SC{i}' for i in range(10, 16)]


@pytest.mark.parametrize('stage,ids', [(1, STAGE1), (5, STAGE5)])
def test_table_go_needs_all_six(stage, ids):
    out = _passing(ids)
    assert rt_stats.table_go(out, stage) is True
    for fail in ids:
        bad = dict(out)
        bad[fail] = ['BELOW']
        assert rt_stats.table_go(bad, stage) is False, fail
    # LEVEL passes the instructed bar but not the plain bar
    lv = dict(out)
    lv[ids[3]] = ['LEVEL']
    assert rt_stats.table_go(lv, stage) is True
    lv[ids[0]] = ['LEVEL']
    assert rt_stats.table_go(lv, stage) is False


def test_table_go_refuses_a_partial_or_wrong_table():
    out = _passing(STAGE1)
    del out['SC6']
    with pytest.raises(AssertionError):
        rt_stats.table_go(out, 1)
    with pytest.raises(AssertionError):
        rt_stats.table_go(_passing(STAGE1), 5)
    with pytest.raises(AssertionError):
        rt_stats.table_go(_passing(STAGE1), 2)
    extra = _passing(STAGE1 + ['SC10'])
    with pytest.raises(AssertionError):
        rt_stats.table_go(extra, 1)


def test_switch_go_needs_sc7_to_sc9():
    out = _passing(['SC7', 'SC8', 'SC9'])
    assert rt_stats.switch_go(out) is True
    for fail in out:
        bad = dict(out)
        bad[fail] = ['WIN', 'LEVEL']
        assert rt_stats.switch_go(bad) is False, fail
    del out['SC9']
    with pytest.raises(AssertionError):
        rt_stats.switch_go(out)


# ---- stage 3 ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('o1', ['WIN', 'LEVEL', 'BELOW'])
@pytest.mark.parametrize('o4', ['WIN', 'LEVEL', 'BELOW'])
def test_stage3_holds_only_when_both_nDCG_outcomes_equal_sc1_and_sc4(o1, o4):
    assert rt_stats.stage3_reading(nd_plain=o1, nd_instructed=o4, sc1=o1, sc4=o4) == 'holds'
    for n1 in ('WIN', 'LEVEL', 'BELOW'):
        for n4 in ('WIN', 'LEVEL', 'BELOW'):
            want = 'holds' if (n1, n4) == (o1, o4) else 'qualified'
            assert rt_stats.stage3_reading(n1, n4, o1, o4) == want
