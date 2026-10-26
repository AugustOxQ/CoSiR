"""Stage A scorer (ticket 03): the guard, the climb, v*_A, the A outcomes, VA4 -- on synthetic records."""
from pathlib import Path

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


# ------------------------------------------------------------------ file boundaries (review fixes)
import json

import numpy as np


def write_run(tmp_path, arm='GR-16', **over):
    """A seed-0 run.json of a new arm under tmp_path, failed unless `over` says otherwise; returns tmp_path."""
    run = dict(variant=arm, seed=0, switches=S.V.switches(arm), status='failed', cause='x', failure_kind='nonfinite_loss',
               p_sha256=S.rg_paths.P_SHA256)
    va1 = write_va1(tmp_path, smoke=over.get('smoke', False))
    run['rg_code_sha256'] = va1['rg_code_sha256']
    run.update(over)
    path = S.rg_paths.run_dir(tmp_path, arm, 0) / 'run.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({k: v for k, v in run.items() if v is not None}))
    return tmp_path


def read_failed(tmp_path, **over):
    return S.read_run(write_run(tmp_path, **over), 'GR-16', S.rg_paths.P_SHA256)


def test_read_run_failed_kinds(tmp_path):
    assert read_failed(tmp_path)['status'] == 'failed'                                    # a non-finite loss
    other = tmp_path / 'b'
    rec = read_failed(other, failure_kind='va3', rerun_of='first.json', cause='VA3 after rerun')
    assert rec['status'] == 'failed' and rec['rerun_of'] == 'first.json'                 # a VA3 failure after its rerun


def test_read_run_refuses_an_unknown_failure_kind(tmp_path):
    with pytest.raises(AssertionError, match='failure_kind'):
        read_failed(tmp_path, failure_kind='vg3', rerun_of='first.json')                 # 12-25's spelling is not this round's
    with pytest.raises(AssertionError, match='failure_kind'):
        read_failed(tmp_path / 'b', failure_kind=None)


def test_read_run_refuses_a_va3_failure_that_was_not_rerun(tmp_path):
    with pytest.raises(AssertionError, match='rerun it once first'):
        read_failed(tmp_path, failure_kind='va3')


def test_new_arm_must_carry_the_rg_switch_form_but_gr8_the_frozen_one(tmp_path):
    eg_form = S.V.eg_form(S.V.ARMS['GR-16'])
    with pytest.raises(AssertionError, match='switches'):
        read_failed(tmp_path, switches=eg_form)                                           # 12-25's {m, lam, w} on a new arm
    assert read_failed(tmp_path / 'b', switches=S.V.switches('GR-16'))['status'] == 'failed'
    assert S.V.eg_form(S.V.ARMS['GR-8']) == dict(m=1, lam=8, w=0)                         # GR-8 is read in 12-25's form only


def test_read_run_carries_rerun_of_only_when_present(tmp_path):
    assert 'rerun_of' not in read_failed(tmp_path)
    assert read_failed(tmp_path / 'b', failure_kind='va3', rerun_of='r.json')['rerun_of'] == 'r.json'


def write_eq(outputs, **over):
    from rg_train import eq_code_sha256
    eq = {'pass': True, 'smoke': False, 'eq_code_sha256': eq_code_sha256(), **over}
    path = outputs / 'regression' / 'eq.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(eq))


def test_read_eq_checks_the_mode_and_hashes_the_file(tmp_path):
    write_eq(tmp_path)
    got = S.read_eq(tmp_path, smoke=False)
    assert got['pass'] is True and len(got['sha256']) == 64
    with pytest.raises(AssertionError, match='smoke'):
        S.read_eq(tmp_path, smoke=True)
    write_eq(tmp_path, smoke=True)
    with pytest.raises(AssertionError, match='smoke'):
        S.read_eq(tmp_path, smoke=False)
    assert S.read_eq(tmp_path, smoke=True)['sha256'] != got['sha256']
    write_eq(tmp_path, smoke=False, **{'pass': False})
    assert S.read_eq(tmp_path, smoke=False)['pass'] is False and S.read_eq(tmp_path, smoke=False)['sha256'] != got['sha256']


def fake_run_a(monkeypatch, tmp_path, eq_pass, rerun_of=None, on_score=lambda: None):
    """run_a on synthetic climb.json, eq.json and scored records (no data, no training)."""
    write_va1(tmp_path, smoke=True)
    climb = dict(stage='climb', smoke=True, decisions=S.extend_climb_log([], entry(0.5018, 0.505, 0.504)))
    (tmp_path / 'stage_a').mkdir(exist_ok=True)
    (tmp_path / 'stage_a' / 'climb.json').write_text(json.dumps(climb))
    write_eq(tmp_path, smoke=True, **{'pass': eq_pass})
    results = full_set(GR_16=0.505, GR_32=0.504)
    if rerun_of:
        results['GR-32'] = dict(results['GR-32'], rerun_of=rerun_of)
    labels = {a: np.array([0, 1, -1]) for a in S.ASPECTS}
    monkeypatch.setattr(S.E, 'load_context', lambda: dict(p_sha='p', labels=labels))

    def score_arms(*a, **k):
        on_score()
        return dict(base={x: 0.1 for x in S.ASPECTS}, results=results, identity=dict(arms=list(results)))
    monkeypatch.setattr(S, 'score_arms', score_arms)
    return S.run_a(tmp_path, 20, 1, smoke=True)


def test_verdict_identity_holds_the_eq_result_and_a_changed_eq_stops_the_rewrite(monkeypatch, tmp_path):
    first = fake_run_a(monkeypatch, tmp_path, True)
    assert first['identity']['eq']['pass'] is True and len(first['identity']['eq']['sha256']) == 64
    fake_run_a(monkeypatch, tmp_path, True)                                              # same eq.json: kept
    with pytest.raises(AssertionError, match='other inputs'):
        fake_run_a(monkeypatch, tmp_path, False)                                         # eq.json changed: stops


def test_rerun_of_reaches_verdict_and_climb_records(monkeypatch, tmp_path):
    v = fake_run_a(monkeypatch, tmp_path, True, rerun_of='first.json')
    assert v['results']['GR-32']['rerun_of'] == 'first.json' and 'rerun_of' not in v['results']['GR-16']
    recs = ladder(0.5018, 0.505, 0.507)
    recs['GR-32'] = dict(recs['GR-32'], rerun_of='first.json')
    e = S.climb_entry(recs)
    assert e['rungs']['GR-32']['rerun_of'] == 'first.json' and 'rerun_of' not in e['rungs']['GR-16']


# ------------------------------------------------------------------ review fixes: an old check file, and code_state taken once
def counting_code_state(monkeypatch, fixed=False):
    """Replace code_state by a counter: call n gives code 'h<n>' (or always 'h1' when `fixed`). Returns the call list."""
    calls = []

    def fake():
        calls.append(1)
        n = 1 if fixed else len(calls)
        return dict(code_commit=f'c{n}', code_dirty=False, code_sha256=f'h{n}')
    monkeypatch.setattr(S, 'code_state', fake)
    return calls


def check_record(code='h1', identity=None, value=1.0, ok=True):
    return dict(stage='va1_scorer', smoke=False, **{'pass': ok}, code_sha256=code, identity=identity or {'x': 1},
                items={'k': dict(value=value)})


def test_keep_check_writes_keeps_and_refuses_without_touching_the_old_file(tmp_path):
    path = tmp_path / 'regression' / 'va1_scorer.json'
    assert S.keep_check(path, check_record()) == check_record()                       # no file: written
    before = path.read_bytes()
    assert S.keep_check(path, check_record()) == check_record()                       # same code, inputs, values: kept
    for new, match in [(check_record(code='h2'), 'code_sha256.*move.*aside.*rerun'),
                       (check_record(identity={'x': 2}), 'other inputs'),
                       (check_record(value=1.1), 'other values'),
                       (check_record(ok=False), 'other values')]:
        with pytest.raises(AssertionError, match=match):
            S.keep_check(path, new)
        assert path.read_bytes() == before                                           # never overwritten


def test_run_a_and_climb_take_code_state_once_at_the_start(monkeypatch, tmp_path):
    calls = counting_code_state(monkeypatch)
    v = fake_run_a(monkeypatch, tmp_path, True, on_score=lambda: calls.append('scoring') if len(calls) != 1 else None)
    assert calls == [1] and (v['code_commit'], v['code_sha256']) == ('c1', 'h1')
    calls.clear()
    labels = {a: np.array([0, 1, -1]) for a in S.ASPECTS}
    monkeypatch.setattr(S.E, 'load_context', lambda: dict(p_sha='p', labels=labels))

    def ladder_records(*a, **k):
        calls.append('scoring') if len(calls) != 1 else None            # the code was read before the scoring started
        return ladder(0.5018, 0.505, 0.507), dict(identity={'x': 1})
    monkeypatch.setattr(S, 'ladder_records', ladder_records)
    write_va1(tmp_path / 'c', smoke=True)
    e = S.run_climb(tmp_path / 'c', 1, smoke=True)
    assert calls == [1] and (e['code_commit'], e['code_sha256']) == ('c1', 'h1')


# Final review A: gate currency at the entry points and file boundary.
def write_va1(outputs, smoke=False, **over):
    folder = outputs / 'regression'
    folder.mkdir(parents=True, exist_ok=True)
    scorer = dict(stage='va1_scorer', smoke=smoke, code_sha256=S.scorer_code_sha256(), **{'pass': True})
    training = dict(smoke=smoke, rg_code_sha256={'rg_train.py': 'train', 'rg_loss.py': 'loss', 'rg_variants.py': 'variants'},
                    **{'pass': not smoke})
    scorer.update(over.get('scorer', {}))
    training.update(over.get('training', {}))
    (folder / 'va1_scorer.json').write_text(json.dumps(scorer))
    (folder / 'va1.json').write_text(json.dumps(training))
    return training


@pytest.mark.parametrize('smoke', [False, True])
def test_read_va1_accepts_both_halves_including_the_smoke_training_failure(tmp_path, smoke):
    write_va1(tmp_path, smoke)
    assert S.read_va1(tmp_path, smoke)['smoke'] is smoke


@pytest.mark.parametrize('half, change', [
    ('scorer', {'stage': 'vb1'}), ('scorer', {'smoke': True}),
    ('scorer', {'pass': False}), ('scorer', {'code_sha256': 'stale'}),
    ('training', {'pass': False}), ('training', {'smoke': True}),
])
def test_read_va1_refuses_invalid_halves(tmp_path, half, change):
    write_va1(tmp_path, **{half: change})
    with pytest.raises(AssertionError, match='VA1'):
        S.read_va1(tmp_path, False)


@pytest.mark.parametrize('half', ['va1', 'va1_scorer'])
def test_read_va1_requires_both_files(tmp_path, half):
    write_va1(tmp_path)
    (tmp_path / 'regression' / f'{half}.json').rename(tmp_path / f'{half}.saved')
    with pytest.raises(AssertionError, match='VA1'):
        S.read_va1(tmp_path, False)


@pytest.mark.parametrize('mode', ['a', 'climb'])
def test_stage_a_checks_va1_before_loading_data(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(S.E, 'load_context', lambda: pytest.fail('data loaded before VA1'))
    with pytest.raises(AssertionError, match='VA1'):
        if mode == 'a':
            S.run_a(tmp_path, 20, 1)
        else:
            S.run_climb(tmp_path, 1)


@pytest.mark.parametrize('smoke', [False, True])
@pytest.mark.parametrize('hashes', [None, {'rg_train.py': 'other'}])
def test_seed0_runs_must_use_the_code_va1_checked(tmp_path, smoke, hashes):
    write_run(tmp_path, smoke=smoke, rg_code_sha256=hashes)
    with pytest.raises(AssertionError, match='trained by other code than VA1 checked'):
        S.read_run(tmp_path, 'GR-16', S.rg_paths.P_SHA256, smoke=smoke)


@pytest.mark.parametrize('hashes', [None, {'rg_loss.py': 'stale'}])
def test_read_eq_refuses_missing_or_stale_code_hashes(tmp_path, hashes):
    write_eq(tmp_path, eq_code_sha256=hashes)
    with pytest.raises(AssertionError, match='EQ.*code'):
        S.read_eq(tmp_path)


def test_run_code_requires_a_complete_va1_hash_manifest(tmp_path):
    write_run(tmp_path, rg_code_sha256={})
    write_va1(tmp_path, training={'rg_code_sha256': {}})
    with pytest.raises(AssertionError, match='VA1.*rg_code_sha256'):
        S.read_run(tmp_path, 'GR-16', S.rg_paths.P_SHA256)


@pytest.mark.parametrize('problem', ['missing_record', 'ok', 'nonfinite_loss', 'wrong_identity', 'unlinked', 'wrong_link', 'extra_arm'])
def test_va4_refuses_replaced_seed0_runs_except_linked_va3_failures(tmp_path, problem):
    arm = 'GN-8' if problem == 'extra_arm' else 'GR-16'
    origin = tmp_path / 'replaced' / arm / 's0-previous'
    origin.mkdir(parents=True)
    old = dict(variant=arm, seed=0, status='failed', failure_kind='va3')
    if problem == 'ok':
        old['status'] = 'ok'
    if problem == 'nonfinite_loss':
        old['failure_kind'] = 'nonfinite_loss'
    if problem == 'wrong_identity':
        old['seed'] = 1
    if problem != 'missing_record':
        (origin / 'run.json').write_text(json.dumps(old))
    write_run(tmp_path, arm=arm, rerun_of=str(tmp_path / 'wrong') if problem == 'wrong_link' else
              None if problem == 'unlinked' else str(origin))
    with pytest.raises(AssertionError, match='VA4'):
        S.check_arm_dirs(tmp_path, ['GR-16'])


def test_va4_accepts_the_linked_va3_failure_and_ignores_replaced_fresh_seeds(tmp_path):
    origin = tmp_path / 'replaced/GR-16/s0-previous'
    origin.mkdir(parents=True)
    (origin / 'run.json').write_text(json.dumps(dict(variant='GR-16', seed=0, status='failed', failure_kind='va3')))
    write_run(tmp_path, rerun_of=str(origin))
    (tmp_path / 'replaced/GN-8/s1-other').mkdir(parents=True)
    S.check_arm_dirs(tmp_path, ['GR-16'])


def test_a_checks_replaced_runs_before_scoring(tmp_path, monkeypatch):
    (tmp_path / 'replaced/GR-16/s0-unrecorded').mkdir(parents=True)
    with pytest.raises(AssertionError, match='VA4'):
        fake_run_a(monkeypatch, tmp_path, True)



def test_va4_refuses_an_origin_with_no_current_run(tmp_path):
    origin = tmp_path / 'replaced/GR-16/s0-origin'
    origin.mkdir(parents=True)
    (origin / 'run.json').write_text(json.dumps(dict(variant='GR-16', seed=0, status='failed', failure_kind='va3')))
    with pytest.raises(AssertionError, match='VA4.*no current run'):
        S.check_arm_dirs(tmp_path, ['GR-16'])


def test_va4_refuses_replaced_arm_outside_the_climb(tmp_path):
    origin = tmp_path / 'replaced/GN-8/s0-origin'
    origin.mkdir(parents=True)
    (origin / 'run.json').write_text(json.dumps(dict(variant='GN-8', seed=0, status='failed', failure_kind='va3')))
    # No live GN-8 folder: only the replaced-folder scan can spot this arm.
    with pytest.raises(AssertionError, match='VA4.*unapproved'):
        S.check_arm_dirs(tmp_path, ['GR-16'])


@pytest.mark.parametrize('smoke', [False, True])
def test_queue_a_scorer_failure_stops_real_runs_and_notes_smokes(tmp_path, smoke):
    import subprocess
    script = (Path(S.__file__).parent / 'rg_queue_a.sh').read_text().split('# --- EQ')[0]
    # Exercise the queue's actual gate block; substitute commands, never run a trainer or GPU.
    stub = tmp_path / 'common.sh'
    stub.write_text('''outputs=$1
workers=1
device=cpu
smoke=()
here=/unused
softstep() { [[ "$1" != va1_scorer ]]; }
pending_add() { :; }
regression_gate() { return 0; }  # stale files still say pass
note() { echo "$*"; }
stop() { echo "$*"; exit 17; }
''')
    source = next(line for line in script.splitlines() if line.startswith('source '))
    script = script.replace(source, 'source "$2"') + '\necho AFTER_VA1\n'
    env = dict(__import__('os').environ, RG_SMOKE='1' if smoke else '0')
    got = subprocess.run(['bash', '-c', script, 'queue-test', str(tmp_path), str(stub)], env=env,
                         capture_output=True, text=True, check=False)
    if smoke:
        assert got.returncode == 0 and 'smoke:' in got.stdout and 'AFTER_VA1' in got.stdout
    else:
        assert got.returncode == 17 and "VA1's scorer half failed or is stale: nothing runs" in got.stdout
        assert 'AFTER_VA1' not in got.stdout
