"""Stage iii scorer: the rule's conditions (§4.3) on small synthetic inputs."""
import numpy as np

import eg_paths  # noqa: F401  (sys.path)
import eg_score3 as s3
import eg_variants as V

FIRST = V.FIRST


def res(a_genre=None, e_emotion=0.0, e_style=0.0, e_genre=0.0, status='ok'):
    """One variant's scored record, as `condition_record` builds it from the effects."""
    if status == 'failed':
        return s3.failed_record('non-finite loss')
    return s3.condition_record({'emotion': dict(A=0.1 + e_emotion, e=e_emotion, lo=0.0, hi=0.0),
                                'style': dict(A=0.4 + e_style, e=e_style, lo=0.0, hi=0.0),
                                'genre': dict(A=a_genre, e=e_genre, lo=0.0, hi=0.0)})


# ------------------------------------------------------------------ guard
def test_guard_boundaries():
    assert s3.is_guarded(-0.0046, -0.0082)                        # exactly the ranges: holds (>=)
    assert not s3.is_guarded(np.nextafter(-0.0046, -1), 0.0)
    assert not s3.is_guarded(0.0, np.nextafter(-0.0082, -1))
    assert s3.is_guarded(0.01, 0.01)
    assert not s3.is_guarded(-0.0047, 0.02) and not s3.is_guarded(0.02, -0.0083)


def test_failed_record_is_not_guarded_and_has_no_genre():
    f = res(status='failed')
    assert f['status'] == 'failed' and f['A_genre'] is None and f['guarded'] is False and f['cause'] == 'non-finite loss'


def test_condition_record_applies_the_guard_to_the_point_estimates():
    assert res(.5, e_emotion=-0.0046, e_style=-0.0082)['guarded'] is True
    assert res(.5, e_emotion=-0.0047)['guarded'] is False


# ------------------------------------------------------------------ closest method and extras
def first_results(unguarded=(), **a):
    """Five first variants from {name: A_genre or 'failed'}; unguarded ones lose 0.02 on emotion."""
    return {v: res(status='failed') if a[v] == 'failed' else res(a[v], e_emotion=-0.02 if v in unguarded else 0.0)
            for v in FIRST}


def test_closest_is_the_guarded_largest():
    r = first_results(unguarded=('GR-2',), **{'GT-2': .50, 'GT-4': .49, 'GR-0.5': .48, 'GR-2': .60, 'GA-1': .47})
    assert s3.closest_variant(r) == 'GT-2'


def test_closest_without_a_guarded_variant_takes_the_largest_of_the_five():
    r = first_results(unguarded=FIRST, **{'GT-2': .50, 'GT-4': .49, 'GR-0.5': .48, 'GR-2': .60, 'GA-1': .47})
    assert s3.closest_variant(r) == 'GR-2'


def test_closest_tie_goes_to_the_first_in_order():
    r = first_results(**{'GT-2': .4, 'GT-4': .5, 'GR-0.5': .5, 'GR-2': .5, 'GA-1': .5})
    assert s3.closest_variant(r) == 'GT-4'
    r = first_results(**{'GT-2': .4, 'GT-4': .4, 'GR-0.5': .4, 'GR-2': .5, 'GA-1': .5})
    assert s3.closest_variant(r) == 'GR-2'


def test_closest_skips_failed_runs_and_is_none_when_all_fail():
    r = first_results(**{'GT-2': 'failed', 'GT-4': .3, 'GR-0.5': 'failed', 'GR-2': .2, 'GA-1': 'failed'})
    assert s3.closest_variant(r) == 'GT-4'
    assert s3.closest_variant({v: res(status='failed') for v in FIRST}) is None


def test_extras_largest_rung_guarded_gives_the_next_two_above():
    assert s3.extras_of('GT-4', True) == ['GT-6', 'GT-8']
    assert s3.extras_of('GR-2', True) == ['GR-4', 'GR-8']
    assert s3.extras_of('GA-1', True) == ['GA-2', 'GA-4']


def test_extras_largest_rung_but_unguarded_gives_one_each_side():
    assert s3.extras_of('GT-4', False) == ['GT-3', 'GT-6']
    assert s3.extras_of('GR-2', False) == ['GR-1', 'GR-4']
    assert s3.extras_of('GA-1', False) == ['GA-0.5', 'GA-2']


def test_extras_not_the_largest_rung_gives_the_nearest_unrun_each_side():
    assert s3.extras_of('GT-2', True) == ['GT-1.5', 'GT-3']
    assert s3.extras_of('GT-2', False) == ['GT-1.5', 'GT-3']
    assert s3.extras_of('GR-0.5', True) == ['GR-0.25', 'GR-1']


def test_extras_skip_rungs_already_run(monkeypatch):
    monkeypatch.setitem(V.LADDERS, 'GT', (1, 2, 3, 4, 6))
    assert s3.extras_of('GT-2', True, run=('GT-2', 'GT-3', 'GT-6')) == ['GT-1', 'GT-4']
    assert s3.extras_of('GT-3', True, run=('GT-2', 'GT-3', 'GT-4')) == ['GT-1', 'GT-6']


def test_extras_at_a_ladder_end_give_none_on_that_side(monkeypatch):
    monkeypatch.setitem(V.LADDERS, 'GT', (2, 4))
    assert s3.extras_of('GT-4', True, run=('GT-2', 'GT-4')) == []
    assert s3.extras_of('GT-4', False, run=('GT-2', 'GT-4')) == []
    monkeypatch.setitem(V.LADDERS, 'GT', (1.5, 2, 4, 6))
    assert s3.extras_of('GT-4', True, run=('GT-2', 'GT-4')) == ['GT-6']          # one rung above only
    assert s3.extras_of('GT-2', True, run=('GT-2', 'GT-6')) == ['GT-1.5', 'GT-4']
    assert s3.extras_of('GT-6', True, run=('GT-2', 'GT-6')) == []                # largest rung, guarded, ladder end
    assert s3.extras_of('GT-6', False, run=('GT-2', 'GT-6')) == ['GT-4']         # none above
    monkeypatch.setitem(V.LADDERS, 'GT', (2, 3, 4))
    assert s3.extras_of('GT-2', True, run=('GT-2', 'GT-4')) == ['GT-3']          # none below


def test_decide_first_counts_failed_runs_as_run():
    r = first_results(**{'GT-2': .3, 'GT-4': 'failed', 'GR-0.5': .2, 'GR-2': .2, 'GA-1': .2})
    d = s3.decide_first(r)
    assert d['closest'] == 'GT-2' and d['extras'] == ['GT-1.5', 'GT-3']
    r = first_results(**{'GT-2': 'failed', 'GT-4': .3, 'GR-0.5': .2, 'GR-2': .2, 'GA-1': .2})
    d = s3.decide_first(r)
    assert d['closest'] == 'GT-4' and d['extras'] == ['GT-6', 'GT-8']             # GT-4 guarded and the largest run
    d = s3.decide_first({v: res(status='failed') for v in FIRST})
    assert d['closest'] is None and d['extras'] == []


def test_decide_first_unguarded_closest_gives_both_sides():
    r = first_results(unguarded=FIRST, **{'GT-2': .3, 'GT-4': .4, 'GR-0.5': .2, 'GR-2': .2, 'GA-1': .2})
    d = s3.decide_first(r)
    assert d['closest'] == 'GT-4' and d['guarded_pool'] is False and d['extras'] == ['GT-3', 'GT-6']


# ------------------------------------------------------------------ winner, outcomes
def all_results(first, extras):
    r = first_results(**first)
    for name, a in extras.items():
        r[name] = res(a)
    return r


def test_winner_is_the_guarded_largest_over_all_seven():
    first = {'GT-2': .50, 'GT-4': .51, 'GR-0.5': .48, 'GR-2': .47, 'GA-1': .46}
    r = all_results(first, {'GT-6': .52, 'GT-8': .53})
    assert s3.pick_winner(r, ['GT-6', 'GT-8']) == 'GT-8'
    r['GT-8'] = res(.53, e_style=-0.01)
    assert s3.pick_winner(r, ['GT-6', 'GT-8']) == 'GT-6'


def test_winner_ties_first_then_extras_in_ladder_order():
    first = {'GT-2': .5, 'GT-4': .5, 'GR-0.5': .4, 'GR-2': .4, 'GA-1': .4}
    r = all_results(first, {'GT-6': .5, 'GT-8': .5})
    assert s3.pick_winner(r, ['GT-6', 'GT-8']) == 'GT-2'
    first = {'GT-2': .4, 'GT-4': .4, 'GR-0.5': .4, 'GR-2': .4, 'GA-1': .4}
    r = all_results(first, {'GT-6': .5, 'GT-8': .5})
    assert s3.pick_winner(r, ['GT-6', 'GT-8']) == 'GT-6'


def test_winner_none_without_a_guarded_variant_and_a_failed_run_never_wins():
    r = first_results(unguarded=FIRST, **{'GT-2': .5, 'GT-4': .5, 'GR-0.5': .4, 'GR-2': .4, 'GA-1': .4})
    assert s3.pick_winner(r, []) is None
    assert s3.pick_winner({v: res(status='failed') for v in FIRST}, []) is None
    r = first_results(**{'GT-2': 'failed', 'GT-4': .3, 'GR-0.5': .2, 'GR-2': .2, 'GA-1': .2})
    assert s3.pick_winner(r, []) == 'GT-4'


def test_outcome_levels_at_the_unrounded_thresholds():
    t, low = V.T_G, V.T_G - V.RANGES['genre']
    assert s3.outcome('GT-2', t) == 'III-GO'
    assert s3.outcome('GT-2', np.nextafter(t, 0)) == 'III-PROM'
    assert s3.outcome('GT-2', low) == 'III-PROM'
    assert s3.outcome('GT-2', np.nextafter(low, 0)) == 'III-NONE'
    assert s3.outcome('GT-2', 0.4989) == 'III-PROM'                    # the rounded 0.4989 is below T_g
    assert s3.outcome('GT-2', 0.4861) == 'III-NONE'                    # the rounded 0.4861 is below T_g - r
    assert s3.outcome(None, 0.9) == 'III-NONE'


def test_fresh_seeds_trigger():
    rg = V.RANGES['genre']
    assert s3.fresh_trigger('GT-2', rg) is True
    assert s3.fresh_trigger('GT-2', np.nextafter(rg, 0)) is False
    assert s3.fresh_trigger(None, 1.0) is False


def test_kept_weights_split_at_t_g_minus_r():
    low = V.T_G - V.RANGES['genre']
    r = {'a': res(low), 'b': res(np.nextafter(low, 0)), 'c': res(status='failed'), 'd': res(0.6)}
    kept, rest = s3.split_weights(r)
    assert kept == ['a', 'd'] and rest == ['b', 'c']


# ------------------------------------------------------------------ VG4
def test_vg4_refuses_any_variant_set_but_first_plus_the_extras():
    s3.check_vg4(list(FIRST) + ['GT-6', 'GT-8'], ['GT-6', 'GT-8'])
    for bad in (list(FIRST), list(FIRST) + ['GT-6'], list(FIRST) + ['GT-6', 'GT-8', 'GA-2'],
                list(FIRST[:4]) + ['GT-6', 'GT-8'], list(FIRST) + ['GT-6', 'GR-4']):
        try:
            s3.check_vg4(bad, ['GT-6', 'GT-8'])
        except AssertionError:
            continue
        raise AssertionError(f'VG4 accepted {bad}')
