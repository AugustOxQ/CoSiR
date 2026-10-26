"""Stage A scorer (ticket 03): the guard, the climb, v*_A, the A outcomes, VA4 -- on synthetic records."""
import pytest

import rg_score as S


def rec(a_genre, e_emotion=0.0, e_style=0.0):
    """A scored arm's record as `condition_record` builds it, from the three numbers the conditions read."""
    return S.condition_record({'emotion': dict(A=0.1, e=e_emotion, lo=0, hi=0), 'style': dict(A=0.4, e=e_style, lo=0, hi=0),
                               'genre': dict(A=a_genre, e=a_genre - 0.4674, lo=0, hi=0)})


def test_guard_boundaries_are_inclusive():
    assert S.is_guarded(-0.0046, -0.0082)                     # exactly at both limits: guarded
    assert not S.is_guarded(-0.0046 - 1e-9, 0.0)
    assert not S.is_guarded(0.0, -0.0082 - 1e-9)
    assert S.is_guarded(0.01, 0.01)


def ladder(*values, **special):
    """{rung: record} for GR-8, GR-16, ... from A_genre values; `special` maps a rung name with '_' for '-' to a record."""
    names = S.V.LADDER[:len(values)]
    out = {n: rec(v) for n, v in zip(names, values)}
    out.update({k.replace('_', '-'): r for k, r in special.items()})
    return out


@pytest.mark.parametrize('values, special, nxt, reason', [
    ((0.5018, 0.505, 0.507), {}, 'GR-64', 'rises'),
    ((0.5018, 0.505, 0.505), {}, None, 'tied'),                       # equal to the best guarded rung: stops
    ((0.5018, 0.505, 0.503), {}, None, 'lower'),
    ((0.5018, 0.505, 0.507), dict(GR_32=rec(0.507, e_emotion=-0.0047)), None, 'unguarded'),
    ((0.5018, 0.505, 0.507), dict(GR_32=S.failed_record('nonfinite loss')), None, 'failed'),
    ((0.5018, 0.505, 0.507), dict(GR_16=rec(0.520, e_style=-0.0083)), 'GR-64', 'rises'),   # unguarded GR-16 is not a rival
    ((0.5018, 0.505, 0.507), dict(GR_16=S.failed_record('x')), 'GR-64', 'rises'),          # a failed rung is not a rival
    ((0.5018, 0.505, 0.507), dict(GR_8=rec(0.508)), None, 'lower'),                         # GR-8 is a rival
    ((0.5018, 0.505, 0.507, 0.509), {}, 'GR-128', 'rises'),
    ((0.5018, 0.505, 0.507, 0.506), {}, None, 'lower'),
    ((0.5018, 0.505, 0.507, 0.509, 0.510), {}, 'GR-256', 'rises'),
    ((0.5018, 0.505, 0.507, 0.509, 0.510, 0.512), {}, None, 'cap'),   # nothing after GR-256
])
def test_climb_branches(values, special, nxt, reason):
    d = S.climb_decision(ladder(*values, **special))
    assert (d['next'], d['reason']) == (nxt, reason)


def test_climb_reports_a_rung_still_rising_at_the_cap():
    assert S.climb_decision(ladder(0.5018, 0.505, 0.507, 0.509, 0.510, 0.512))['still_rising']
    d = S.climb_decision(ladder(0.5018, 0.505, 0.507, 0.509, 0.510, 0.500))
    assert d['reason'] == 'cap' and not d['still_rising']


def arms(**kw):
    """{arm: record} from keyword A_genre values (a record from `rec`) or ready records; '_' stands for '-' in names."""
    return {k.replace('_', '-').replace('GS-0-', 'GS-0.'): (v if isinstance(v, dict) else rec(v)) for k, v in kw.items()}


def test_winner_is_the_guarded_arm_with_the_largest_a_genre():
    r = arms(GR_8=0.5018, GR_16=0.505, GR_32=0.507, GN_8=0.504, GS_0_02=0.5069, GS_0_01=0.5, GQ_8=0.5)
    assert S.pick_winner(r, eq_pass=True) == 'GR-32'


def test_winner_skips_unguarded_and_failed_arms():
    r = arms(GR_8=0.5018, GR_16=rec(0.52, e_emotion=-0.005), GR_32=S.failed_record('nonfinite'), GN_8=0.503)
    assert S.pick_winner(r, eq_pass=True) == 'GN-8'


@pytest.mark.parametrize('values, expect', [
    (dict(GR_8=0.5, GR_16=0.5, GR_32=0.5), 'GR-8'),                      # GR-8 first on equal values
    (dict(GR_16=0.5, GN_8=0.5, GS_0_02=0.5, GQ_8=0.5), 'GR-16'),
    (dict(GR_32=0.5, GN_8=0.5), 'GR-32'),                                 # ladder before GN-8
    (dict(GN_8=0.5, GS_0_02=0.5, GS_0_01=0.5, GQ_8=0.5), 'GN-8'),
    (dict(GS_0_02=0.5, GS_0_01=0.5, GQ_8=0.5), 'GS-0.02'),
    (dict(GS_0_01=0.5, GQ_8=0.5), 'GS-0.01'),
    (dict(GR_64=0.5, GR_128=0.5), 'GR-64'),
])
def test_winner_ties_follow_the_tie_order(values, expect):
    full = dict(GR_8=0.49, GR_16=0.4, GR_32=0.4)
    full.update(values)
    assert S.pick_winner(arms(**full), eq_pass=True) == expect


def test_gq8_is_out_of_the_pool_when_eq_fails_but_stays_scored():
    r = arms(GR_8=0.5018, GR_16=0.505, GQ_8=0.51)
    assert S.pick_winner(r, eq_pass=True) == 'GQ-8'
    assert S.pick_winner(r, eq_pass=False) == 'GR-16'
    assert 'GQ-8' in S.split_weights(r)[0]           # its A_genre still decides its weights


def test_outcomes_levels_and_weights():
    assert S.a_outcome('GR-8') == 'A-SAME' and S.a_outcome('GR-16') == 'A-NEW' and S.a_outcome('GQ-8') == 'A-NEW'
    g = 0.4989161752444698
    assert S.level(g) == 'target' and S.level(g - 1e-12) == 'promising'
    assert S.level(g - 0.0128) == 'promising' and S.level(g - 0.0128 - 1e-12) == 'neither'
    assert S.level(None) is None
    r = arms(GR_8=0.5018, GR_16=g - 0.0128, GR_32=g - 0.0128 - 1e-9, GN_8=S.failed_record('x'))
    kept, rest = S.split_weights(r)
    assert kept == ['GR-8', 'GR-16'] and rest == ['GR-32', 'GN-8']


def test_va4_refuses_any_arm_set_but_the_climbs_record_plus_the_others():
    ok = ['GR-8', 'GR-16', 'GR-32'] + list(S.V.A_OTHER)
    S.check_va4(ok, ['GR-8', 'GR-16', 'GR-32'])
    S.check_va4(ok + ['GR-64'], ['GR-8', 'GR-16', 'GR-32', 'GR-64'])
    with pytest.raises(AssertionError, match='VA4'):
        S.check_va4(ok + ['GR-2'], ['GR-8', 'GR-16', 'GR-32'])           # a stage B arm
    with pytest.raises(AssertionError, match='VA4'):
        S.check_va4(ok + ['GR-64'], ['GR-8', 'GR-16', 'GR-32'])          # a rung the climb did not allow
    with pytest.raises(AssertionError, match='VA4'):
        S.check_va4(ok[:-1], ['GR-8', 'GR-16', 'GR-32'])                 # GQ-8 missing
    with pytest.raises(AssertionError, match='VA4'):
        S.check_va4(ok[1:], ['GR-8', 'GR-16', 'GR-32'])                  # GR-8 missing


def entry(*values, **special):
    """A climb.json entry as `climb_entry` builds it from synthetic rung records."""
    return S.climb_entry(ladder(*values, **special))


def test_climb_log_follows_each_decision_in_order():
    first = entry(0.5018, 0.505, 0.507)
    assert first['rungs_run'] == ['GR-8', 'GR-16', 'GR-32'] and first['next'] == 'GR-64'
    log = S.extend_climb_log([], first)
    assert S.extend_climb_log(log, first) == log                       # the same decision again: nothing appended
    second = entry(0.5018, 0.505, 0.507, 0.506)
    log2 = S.extend_climb_log(log, second)
    assert [e['newest'] for e in log2] == ['GR-32', 'GR-64'] and log2[-1]['next'] is None
    assert S.climb_rungs(log2) == ['GR-8', 'GR-16', 'GR-32', 'GR-64']


def test_climb_log_refuses_a_skipped_rung_a_changed_value_and_a_rung_after_the_stop():
    first = entry(0.5018, 0.505, 0.507)
    with pytest.raises(AssertionError, match='GR-32'):                  # a first decision that skips ahead
        S.extend_climb_log([], entry(0.5018, 0.505, 0.507, 0.509))
    log = S.extend_climb_log([], first)
    with pytest.raises(AssertionError, match='differs'):                # the same rungs, another value
        S.extend_climb_log(log, entry(0.5018, 0.505, 0.5071))
    with pytest.raises(AssertionError, match='GR-64'):                  # GR-128 before GR-64's decision
        S.extend_climb_log(log, entry(0.5018, 0.505, 0.507, 0.509, 0.51))
    stopped = S.extend_climb_log([], entry(0.5018, 0.505, 0.504))
    with pytest.raises(AssertionError, match='stopped'):
        S.extend_climb_log(stopped, entry(0.5018, 0.505, 0.504, 0.52))


def test_climb_rungs_needs_a_finished_climb():
    with pytest.raises(AssertionError, match='no climb'):
        S.climb_rungs([])
    with pytest.raises(AssertionError, match='not finished'):
        S.climb_rungs(S.extend_climb_log([], entry(0.5018, 0.505, 0.507)))
    assert S.climb_rungs(S.extend_climb_log([], entry(0.5018, 0.505, 0.504))) == ['GR-8', 'GR-16', 'GR-32']


def test_climb_entry_carries_every_rungs_numbers():
    e = entry(0.5018, 0.505, 0.507, GR_16=rec(0.505, e_emotion=-0.003, e_style=-0.002))
    assert set(e['rungs']) == {'GR-8', 'GR-16', 'GR-32'}
    assert e['rungs']['GR-16'] == dict(status='ok', cause=None, A_genre=0.505, e_emotion=-0.003, e_style=-0.002,
                                       e_genre=pytest.approx(0.505 - 0.4674), guarded=True)
    f = S.climb_entry(ladder(0.5018, 0.505, 0.507, GR_32=S.failed_record('nonfinite loss')))
    assert f['rungs']['GR-32']['status'] == 'failed' and f['rungs']['GR-32']['cause'] == 'nonfinite loss'
    assert f['rungs']['GR-32']['A_genre'] is None and f['rungs']['GR-32']['guarded'] is False and f['reason'] == 'failed'


def full_set(**over):
    """GR-8, GR-16, GR-32 and A_OTHER, all guarded at 0.49, with `over` on top."""
    base = dict(GR_8=0.5018, GR_16=0.49, GR_32=0.49, GN_8=0.49, GS_0_02=0.49, GS_0_01=0.49, GQ_8=0.49)
    base.update(over)
    return arms(**base)


def test_decide_a_gives_winner_outcome_levels_weights():
    r = full_set(GR_16=0.505, GQ_8=S.failed_record('nonfinite loss'), GN_8=0.40)
    d = S.decide_a(r, eq_pass=True, rungs=['GR-8', 'GR-16', 'GR-32'])
    assert (d['winner'], d['outcome'], d['A_genre_winner']) == ('GR-16', 'A-NEW', 0.505)
    assert d['levels']['GQ-8'] is None and d['levels']['GN-8'] == 'neither' and d['levels']['GR-16'] == 'target'
    assert d['kept_weights'] == ['GR-8', 'GR-16', 'GR-32', 'GS-0.02', 'GS-0.01'] and d['pending_deletions'] == ['GN-8', 'GQ-8']
    assert d['trains_fresh'] == ['GR-16']
    assert S.decide_a(full_set(), True, ['GR-8', 'GR-16', 'GR-32'])['outcome'] == 'A-SAME'
    assert S.decide_a(full_set(), True, ['GR-8', 'GR-16', 'GR-32'])['trains_fresh'] == []


def test_decide_a_refuses_another_arm_set():
    r = full_set()
    del r['GQ-8']
    with pytest.raises(AssertionError, match='VA4'):
        S.decide_a(r, True, ['GR-8', 'GR-16', 'GR-32'])
    with pytest.raises(AssertionError, match='VA4'):
        S.decide_a(full_set(), True, ['GR-8', 'GR-16'])


def test_climb_refuses_before_gr16_and_gr32_exist_and_on_gaps():
    with pytest.raises(AssertionError, match='GR-16'):
        S.climb_decision({'GR-8': rec(0.5018)})
    with pytest.raises(AssertionError, match='GR-32'):
        S.climb_decision({'GR-8': rec(0.5018), 'GR-16': rec(0.505)})
    gap = ladder(0.5018, 0.505, 0.507)
    gap['GR-128'] = rec(0.52)
    with pytest.raises(AssertionError, match='contiguous'):
        S.climb_decision(gap)
    with pytest.raises(AssertionError, match='GR-8'):
        S.climb_decision({k: v for k, v in ladder(0.5018, 0.505, 0.507).items() if k != 'GR-8'})
