"""Stage B scorer (ticket 04): the paired draws, the confirmation, the pick, failed runs, VB1 and VB4 -- on synthetic data."""
import json

import numpy as np
import pytest

import rg_score as S

R = S.RANGES


def eff(e_genre=0.02, lo=0.005, e_emotion=0.0, e_style=0.0, abar=0.5):
    """One arm's fresh-seed effects as `fresh_aspect` returns them (only the fields the conditions read)."""
    return {'emotion': dict(e=e_emotion, lo=-1, hi=1), 'style': dict(e=e_style, lo=-1, hi=1),
            'genre': dict(e=e_genre, lo=lo, hi=1, Abar={'x': abar})}


# ---------------------------------------------------------------- the paired draws (rule §3.1: its own default_rng(0))
def small_case(seed=7, n=60, n_draws=9):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, size=n)
    parts = {name: {s: rng.integers(0, 4, size=(3, n)).astype(np.int64) for s in (1, 2, 3)} for name in ('v', 'w', 'r')}
    return labels, parts, n_draws


def reference_fresh(labels, parts, arm, ref, n_draws):
    """The rule's text: one default_rng(0) per aspect; per draw b first I_b = integers(0, n, n), then S_b = choice of 3 seeds."""
    from rb_ami import score
    L = np.flatnonzero(labels >= 0)
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(n_draws):
        I = L[rng.integers(0, len(L), size=len(L))]
        sb = rng.choice([1, 2, 3], size=3, replace=True)
        gains.append(np.mean([score(labels, parts[arm][s], I) - score(labels, parts[ref][s], I) for s in sb]))
    return np.percentile(gains, [2.5, 97.5])


def test_each_arm_draws_i_then_s_from_its_own_rng_and_shares_them_with_the_reference():
    labels, parts, n_draws = small_case()
    for arm in ('v', 'w'):
        got = S.fresh_effect(labels, {arm: parts[arm], 'r': parts['r']}, arm, 'r', n_draws, workers=1)
        lo, hi = reference_fresh(labels, parts, arm, 'r', n_draws)
        assert (got['lo'], got['hi']) == pytest.approx((lo, hi), abs=1e-12)
        assert got['e'] == pytest.approx(got['Abar'][arm] - got['Abar']['r'])
    # scoring v beside w, or w first, changes nothing for v: the draws belong to the arm's own rng
    again = S.fresh_effect(labels, {'v': parts['v'], 'r': parts['r']}, 'v', 'r', n_draws, workers=1)
    assert (again['lo'], again['hi']) == pytest.approx(tuple(reference_fresh(labels, parts, 'v', 'r', n_draws)), abs=1e-12)


# ---------------------------------------------------------------- confirmation (rule §4.2)
@pytest.mark.parametrize('kw, expect', [
    (dict(e_genre=R['genre'], lo=1e-9), 'C-HOLDS'),                          # exactly r_genre, lower bound above 0
    (dict(e_genre=R['genre'] - 1e-9, lo=0.01), 'C-QUAL'),                    # just short of r_genre
    (dict(e_genre=0.03, lo=0.0), 'C-QUAL'),                                  # lower bound must be > 0
    (dict(e_genre=0.03, lo=-1e-9), 'C-QUAL'),
    (dict(e_genre=0.03, lo=0.01, e_emotion=-R['emotion']), 'C-HOLDS'),       # guards are >= -r
    (dict(e_genre=0.03, lo=0.01, e_emotion=-R['emotion'] - 1e-9), 'C-FAILS'),
    (dict(e_genre=0.03, lo=0.01, e_style=-R['style']), 'C-HOLDS'),
    (dict(e_genre=0.03, lo=0.01, e_style=-R['style'] - 1e-9), 'C-FAILS'),
    (dict(e_genre=0.0, lo=-0.01), 'C-FAILS'),
    (dict(e_genre=1e-9, lo=-0.01), 'C-QUAL'),
])
def test_c_levels_and_the_confirmation_boundaries(kw, expect):
    assert S.c_level(eff(**kw)) == expect
    assert S.b_record(eff(**kw), 'x')['confirmed'] == (expect == 'C-HOLDS')


# ---------------------------------------------------------------- the pick (rule §4.2)
def rec(abar, confirmed=True):
    return S.b_record(eff(abar=abar, **({} if confirmed else dict(e_genre=0.0, lo=-1))), 'x')


def test_pick_is_the_largest_unrounded_abar_among_the_confirmed():
    r = {'GR-8': rec(0.5018), 'GR-2': rec(0.5040), 'GR-4': rec(0.5039)}
    d = S.decide_b(r, None)
    assert (d['outcome'], d['v_held'], d['v_held_switches']) == ('B-PICK', 'GR-2', S.V.switches('GR-2'))
    r['GR-2'] = rec(0.9, confirmed=False)                                    # not confirmed: out, however high
    assert S.decide_b(r, None)['v_held'] == 'GR-4'


@pytest.mark.parametrize('g8, g2, g4, vs, expect', [
    (0.5, 0.5, 0.5, 0.5, 'GR-8'),                                            # a four-way tie: GR-8 first
    (0.4, 0.5, 0.5, 0.5, 'GR-2'),                                            # then GR-2
    (0.4, 0.4, 0.5, 0.5, 'GR-4'),                                            # then GR-4
    (0.4, 0.4, 0.4, 0.5, 'GR-16'),                                           # v*_A last
    (0.4, 0.5, 0.4, 0.4, 'GR-2'),
])
def test_pick_ties_go_by_b_tie_order_then_vstar(g8, g2, g4, vs, expect):
    r = {'GR-8': rec(g8), 'GR-2': rec(g2), 'GR-4': rec(g4), 'GR-16': rec(vs)}
    assert S.decide_b(r, 'GR-16')['v_held'] == expect


def test_b_none_when_no_arm_is_confirmed_and_the_held_fields_are_null():
    r = {a: rec(0.5, confirmed=False) for a in ('GR-8', 'GR-2', 'GR-4')}
    d = S.decide_b(r, None)
    assert (d['outcome'], d['v_held'], d['v_held_switches'], d['confirmed'], d['within_one_range']) == \
        ('B-NONE', None, None, [], [])


def test_a_failed_run_leaves_its_arm_not_confirmed_and_never_the_pick():
    r = {'GR-8': rec(0.5018), 'GR-2': S.failed_b_record('nonfinite loss at seed 2'), 'GR-4': rec(0.5)}
    assert r['GR-2']['confirmed'] is False and r['GR-2']['c_outcome'] is None and r['GR-2']['Abar_genre'] is None
    d = S.decide_b(r, None)
    assert d['v_held'] == 'GR-8' and d['confirmed'] == ['GR-8', 'GR-4'] and d['levels']['GR-2'] is None
    r['GR-8'] = S.failed_b_record('x')
    assert S.decide_b(r, None)['v_held'] == 'GR-4'
    r['GR-4'] = S.failed_b_record('y')
    assert S.decide_b(r, None)['outcome'] == 'B-NONE'


def test_arms_within_one_genre_range_of_the_pick_are_listed():
    r = {'GR-8': rec(0.5100), 'GR-2': rec(0.5100 - R['genre']), 'GR-4': rec(0.5100 - R['genre'] - 1e-6),
         'GR-16': rec(0.5, confirmed=False)}
    d = S.decide_b(r, 'GR-16')
    assert d['v_held'] == 'GR-8' and d['within_one_range'] == ['GR-2']      # not GR-4 (outside), not GR-16 (unconfirmed)
    assert S.decide_b({'GR-8': rec(0.51), 'GR-2': rec(0.40), 'GR-4': rec(0.41)}, None)['within_one_range'] == []


# ---------------------------------------------------------------- VB4
def test_vb4_refuses_other_arms_and_seeds():
    S.check_vb4(['GR-8', 'GR-2', 'GR-4'], None, (1, 2, 3))
    S.check_vb4(['GR-8', 'GR-2', 'GR-4', 'GR-16'], 'GR-16', (1, 2, 3))
    for arms, vs in [(['GR-8', 'GR-2'], None), (['GR-8', 'GR-2', 'GR-4', 'GR-16'], None),
                     (['GR-8', 'GR-2', 'GR-4', 'GN-8'], 'GR-16'), (['GR-8', 'GR-2', 'GR-4'], 'GR-16')]:
        with pytest.raises(AssertionError, match='VB4'):
            S.check_vb4(arms, vs, (1, 2, 3))
    with pytest.raises(AssertionError, match='VB4'):
        S.check_vb4(['GR-8', 'GR-2', 'GR-4'], None, (0, 1, 2, 3))
    with pytest.raises(AssertionError, match='VB4'):
        S.check_vb4(['GR-8', 'GR-2', 'GR-4'], None, (1, 2))


def test_vb4_refuses_run_folders_of_other_arms_on_the_fresh_seeds(tmp_path):
    allowed = ['GR-2', 'GR-4', 'GR-16']
    for a, s in [('GR-2', 1), ('GR-4', 3), ('GR-16', 2), ('GN-8', 0), ('GR-32', 0)]:     # seed 0 folders are stage A's
        (tmp_path / 'train' / a / f's{s}').mkdir(parents=True)
    S.check_fresh_dirs(tmp_path, allowed)
    (tmp_path / 'train' / 'GN-8' / 's2').mkdir()
    with pytest.raises(AssertionError, match='VB4.*GN-8'):
        S.check_fresh_dirs(tmp_path, allowed)
    S.check_fresh_dirs(tmp_path / 'nothing', allowed)                                    # no train folder: nothing to refuse


def test_the_stage_b_arm_set_follows_the_a_outcome():
    assert S.b_arms(dict(outcome='A-SAME', winner='GR-8')) == ('GR-8', 'GR-2', 'GR-4')
    assert S.b_arms(dict(outcome='A-NEW', winner='GS-0.02')) == ('GR-8', 'GR-2', 'GR-4', 'GS-0.02')


# ---------------------------------------------------------------- the gates
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_b_refuses_without_a_passing_vb1_for_this_code_and_without_stage_a(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'scorer_code_sha256', lambda: 'code-1')
    vb1, va = tmp_path / 'stage_b' / 'vb1.json', tmp_path / 'stage_a' / 'verdict.json'
    with pytest.raises(AssertionError, match='vb1'):
        S.read_vb1(tmp_path, smoke=False)
    write(vb1, {'stage': 'vb1', 'pass': False, 'smoke': False, 'code_sha256': 'code-1'})
    with pytest.raises(AssertionError, match='VB1.*not pass'):
        S.read_vb1(tmp_path, smoke=False)
    write(vb1, {'stage': 'vb1', 'pass': True, 'smoke': True, 'code_sha256': 'code-1'})
    with pytest.raises(AssertionError, match='smoke'):
        S.read_vb1(tmp_path, smoke=False)
    write(vb1, {'stage': 'vb1', 'pass': True, 'smoke': False, 'code_sha256': 'code-0'})
    with pytest.raises(AssertionError, match='code_sha256'):
        S.read_vb1(tmp_path, smoke=False)
    write(vb1, {'stage': 'vb1', 'pass': True, 'smoke': False, 'code_sha256': 'code-1'})
    assert S.read_vb1(tmp_path, smoke=False)['pass'] is True
    with pytest.raises(AssertionError, match='verdict'):
        S.read_stage_a(tmp_path, smoke=False)
    write(va, {'stage': 'a', 'smoke': False, 'outcome': 'A-SAME', 'winner': 'GR-8'})
    assert S.read_stage_a(tmp_path, smoke=False)['winner'] == 'GR-8'
    with pytest.raises(AssertionError, match='smoke'):
        S.read_stage_a(tmp_path, smoke=True)


# ---------------------------------------------------------------- VB1 comparison
REF_IMG = {'GR-8': {'1': {'emotion': 'aa', 'style': 'bb', 'genre': 'cc'}}, 'R6-5': {'1': {'emotion': 'dd', 'style': 'ee', 'genre': 'ff'}}}


def test_vb1_hashes_must_equal_the_frozen_identity_per_arm_seed_aspect():
    S.check_frozen_hashes(REF_IMG, REF_IMG)
    bad = json.loads(json.dumps(REF_IMG))
    bad['R6-5']['1']['style'] = 'xx'
    with pytest.raises(AssertionError, match='VB1.*R6-5.*style'):
        S.check_frozen_hashes(bad, REF_IMG)
    with pytest.raises(AssertionError, match='VB1'):
        S.check_frozen_hashes({'GR-8': REF_IMG['GR-8']}, REF_IMG)


def test_vb1_effects_agree_within_1e_6_and_report_each_item():
    ref = {'genre': {'Abar': {'GR-8': 0.5, 'R6-5': 0.4}, 'e': 0.1, 'lo': 0.05, 'hi': 0.15}}
    got = json.loads(json.dumps(ref))
    items = S.compare_effects(got, ref, ('genre',), False)
    assert all(i['pass'] for i in items.values()) and len(items) == 5            # Abar of both arms, e, lo, hi
    got['genre']['hi'] += 2e-6
    items = S.compare_effects(got, ref, ('genre',), False)
    assert [k for k, i in items.items() if not i['pass']] == ['genre_hi']
    got['genre']['hi'] -= 1.5e-6                                                   # 0.5e-6 off: within tolerance
    assert S.compare_effects(got, ref, ('genre',), False)['genre_hi']['pass']
    assert set(S.compare_effects(got, ref, ('genre',), True)) == {'genre_Abar_GR-8', 'genre_Abar_R6-5', 'genre_e'}


# ---------------------------------------------------------------- reading a fresh-seed run (failed runs, VB3)
def make_run(root, arm, seed, status='ok', cause=None, smoke=False):
    """A run folder under `root` as the trainer leaves it; its exports are too small to pass VB3."""
    folder = S.rg_paths.run_dir(root, arm, seed)
    folder.mkdir(parents=True)
    run = dict(variant=arm, seed=seed, switches=S.V.switches(arm), status=status, cause=cause, best_epoch={}, smoke=smoke,
               p_sha256=S.rg_paths.P_SHA256)
    (folder / 'run.json').write_text(json.dumps(run))
    for a in S.ASPECTS:
        np.savez(folder / f'{a}__P.npz', img=np.zeros((4, 2048), np.float16))
    return folder


def test_a_failed_fresh_run_is_read_without_its_exports_and_fails_its_arm(tmp_path):
    make_run(tmp_path, 'GR-2', 1)
    make_run(tmp_path, 'GR-2', 2, status='failed', cause='nonfinite loss')
    r2 = S.read_fresh_run(tmp_path, 'GR-2', 2, S.rg_paths.P_SHA256, {}, False)
    assert (r2['status'], r2['cause'], r2['img']) == ('failed', 'nonfinite loss', None)
    assert S.arm_run_status({1: dict(status='ok', cause=None), 2: r2}) == ('failed', 'seed 2: nonfinite loss')
    assert S.arm_run_status({1: dict(status='ok', cause=None)}) == ('ok', None)
    with pytest.raises(AssertionError, match='VB3'):                         # exports that fail VB3, not declared
        S.read_fresh_run(tmp_path, 'GR-2', 1, S.rg_paths.P_SHA256, {}, False)
    r1 = S.read_fresh_run(tmp_path, 'GR-2', 1, S.rg_paths.P_SHA256, {'GR-2': 'VB3 after rerun'}, False)
    assert (r1['status'], r1['cause']) == ('failed', 'VB3 after rerun') and 'genre' in r1['vg3_problem']


def test_a_declared_failure_needs_exports_that_fail_vb3_and_a_run_folder_must_match(tmp_path, monkeypatch):
    monkeypatch.setattr(S.rb_pick, 'vp2_problem', lambda x: None)           # every export passes VB3
    make_run(tmp_path, 'GR-4', 3)
    with pytest.raises(AssertionError, match='declared it'):
        S.read_fresh_run(tmp_path, 'GR-4', 3, S.rg_paths.P_SHA256, {'GR-4': 'x'}, False)
    assert S.read_fresh_run(tmp_path, 'GR-4', 3, S.rg_paths.P_SHA256, {}, False)['status'] == 'ok'
    make_run(tmp_path, 'GR-2', 1, smoke=True)
    with pytest.raises(AssertionError, match='smoke'):
        S.read_fresh_run(tmp_path, 'GR-2', 1, S.rg_paths.P_SHA256, {}, False)
    with pytest.raises(AssertionError, match='missing'):                     # a run that has not finished
        S.read_fresh_run(tmp_path, 'GR-4', 2, S.rg_paths.P_SHA256, {}, False)
