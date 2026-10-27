"""Stage 3 (rule §3.3, §4.3): the qualifiers, PICK or KEEP-G, the joint bootstrap's draw order and statistics on a tiny
case against a brute-force loop, and the verdict file's refusals. No export of P is read here (the V3a gate does that on
the frozen files); the file layer runs on a tmp_path with failed runs, which carry no exports."""
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score as ami

import rs_paths
import rs_sheet as H
import rs_stage3 as T
import rs_sweep as S

R = H.RANGES
ASP = H.ASPECTS


def rec(e, lo, hi=None, M=None, M_lo=1.0, M_hi=2.0, status='ok'):
    """A configuration's scored record: per-aspect e, lo, hi; M defaults to the rule's mean of e / r."""
    if status == 'failed':
        return dict(status='failed', cause='seed 2: nonfinite loss')
    hi = hi if hi is not None else {a: lo[a] + 0.02 for a in ASP}
    eff = {a: dict(e=e[a], lo=lo[a], hi=hi[a]) for a in ASP}
    point = sum(e[a] / R[a] for a in ASP) / 3 if M is None else M
    return dict(status='ok', cause=None, effects=eff, M=dict(M=point, lo=M_lo, hi=M_hi))


def good(M=None, **kw):
    """A qualifying record: every lower bound well above -r_a, M's lower bound above 0."""
    return rec(dict(emotion=0.0, style=0.01, genre=0.01), {a: -R[a] / 2 for a in ASP}, M=M, **kw)


# ------------------------------------------------------------------ the rule's worked example (§4.3)
def test_worked_example_mean_gain_and_qualifier():
    e = dict(emotion=-0.0010, style=0.0100, genre=0.0080)
    lo = dict(emotion=-0.0040, style=-0.0081, genre=-0.0127)             # just inside style's and genre's limits
    r = rec(e, lo, hi=dict(emotion=0.0020, style=0.03, genre=0.03))
    assert T.mean_gain_bar(e) == pytest.approx(0.5424, abs=5e-5)         # the rule's (-0.217 + 1.220 + 0.625) / 3 = 0.542
    assert r['M']['M'] == pytest.approx(0.5424, abs=5e-5)
    assert T.conditions(r)['emotion']                                    # emotion passes: -0.0040 > -0.0046 although e < 0
    assert T.qualifies(r)


def test_worked_example_fails_when_style_or_genre_lower_bound_is_not_above_the_limit():
    e = dict(emotion=-0.0010, style=0.0100, genre=0.0080)
    lo = dict(emotion=-0.0040, style=-0.0083, genre=-0.0127)             # style below -0.0082
    assert not T.qualifies(rec(e, lo))
    lo = dict(emotion=-0.0040, style=-0.0081, genre=-0.0129)             # genre below -0.0128
    assert not T.qualifies(rec(e, lo))


# ------------------------------------------------------------------ qualifier edges
@pytest.mark.parametrize('aspect', ASP)
def test_lower_bound_must_be_strictly_above_minus_r(aspect):
    e = {a: 0.01 for a in ASP}
    lo = {a: 0.0 for a in ASP}
    lo[aspect] = -R[aspect]                                              # on the limit: does not qualify
    assert not T.qualifies(rec(e, lo))
    lo[aspect] = float(np.nextafter(-R[aspect], 0))                      # one step above: qualifies
    assert T.qualifies(rec(e, lo))
    lo[aspect] = float(np.nextafter(-R[aspect], -1))
    assert not T.qualifies(rec(e, lo))


def test_mean_gain_lower_bound_must_be_strictly_above_zero():
    e, lo = {a: 0.01 for a in ASP}, {a: 0.0 for a in ASP}
    assert not T.qualifies(rec(e, lo, M_lo=0.0))
    assert not T.qualifies(rec(e, lo, M_lo=-1e-12))
    assert T.qualifies(rec(e, lo, M_lo=float(np.nextafter(0.0, 1))))


def test_the_rule_tests_the_lower_bounds_not_the_point_gains():
    e, lo = {a: -0.001 for a in ASP}, {a: -0.002 for a in ASP}
    assert T.qualifies(rec(e, lo, M_lo=0.1))                             # a negative point gain can qualify
    lo['emotion'] = -0.0047
    assert not T.qualifies(rec({a: 0.05 for a in ASP}, lo, M_lo=0.1))    # a big point gain cannot save a low bound


def test_conditions_name_each_aspect_and_m():
    lo = dict(emotion=-0.01, style=0.0, genre=0.0)
    c = T.conditions(rec({a: 0.01 for a in ASP}, lo, M_lo=-0.5))
    assert c == dict(emotion=False, style=True, genre=True, M=False)


def test_failed_configuration_does_not_qualify():
    assert not T.qualifies(rec(None, None, status='failed'))
    assert T.conditions(rec(None, None, status='failed')) is None


def test_non_finite_numbers_in_an_ok_record_are_refused():
    r = good()
    r['effects']['genre']['lo'] = float('nan')
    with pytest.raises(AssertionError, match='finite'):
        T.qualifies(r)
    r = good()
    r['M']['M'] = float('inf')
    with pytest.raises(AssertionError, match='finite'):
        T.qualifies(r)


# ------------------------------------------------------------------ PICK, KEEP-G, ties, the lead
def test_pick_is_the_qualifier_with_the_largest_unrounded_mean_gain():
    d = T.decide({'a': good(M=0.50), 'b': good(M=0.5000001), 'c': good(M=0.4)}, ['a', 'b', 'c'])
    assert d['outcome'] == 'PICK' and d['pick'] == 'b' and d['qualifiers'] == ['b', 'a', 'c']


def test_a_larger_mean_gain_that_does_not_qualify_is_not_picked():
    bad = rec(dict(emotion=0.05, style=0.05, genre=0.05), dict(emotion=-0.0046, style=0.0, genre=0.0))
    d = T.decide({'a': good(M=0.2), 'b': bad}, ['a', 'b'])
    assert d['pick'] == 'a' and d['qualifiers'] == ['a'] and d['not_qualifying'] == ['b']


def test_ties_go_to_tops_order():
    recs = {'a': good(M=0.7), 'b': good(M=0.7), 'c': good(M=0.7)}
    assert T.decide(recs, ['b', 'c', 'a'])['pick'] == 'b'
    assert T.decide(recs, ['c', 'a', 'b'])['pick'] == 'c'
    assert T.decide(recs, ['c', 'a', 'b'])['qualifiers'] == ['c', 'a', 'b']


def test_keep_g_when_nothing_qualifies():
    bad = rec({a: 0.01 for a in ASP}, {a: 0.0 for a in ASP}, M_lo=-0.1)
    d = T.decide({'a': bad, 'b': rec(None, None, status='failed')}, ['a', 'b'])
    assert d['outcome'] == 'KEEP-G' and d['pick'] is None and d['qualifiers'] == [] and d['lead'] is None


def test_a_failed_seed_cannot_win():
    d = T.decide({'a': rec(None, None, status='failed'), 'b': good(M=0.1)}, ['a', 'b'])
    assert d['pick'] == 'b' and d['failed'] == ['a']


def test_decide_needs_a_record_per_top_configuration_and_no_more():
    with pytest.raises(AssertionError, match='TOP'):
        T.decide({'a': good()}, ['a', 'b'])
    with pytest.raises(AssertionError, match='TOP'):
        T.decide({'a': good(), 'b': good()}, ['a'])


def test_lead_over_the_next_qualifier_against_the_half_width():
    r1 = good(M=0.90, M_lo=0.4, M_hi=1.4)                                # half width 0.5
    r2 = good(M=0.60, M_lo=0.55, M_hi=0.65)                              # half width 0.05: the lead 0.30 is under c*'s 0.5 only
    d = T.decide({'a': r1, 'b': r2, 'c': good(M=0.2)}, ['a', 'b', 'c'])
    assert d['lead'] == dict(over='b', lead=pytest.approx(0.30), half_width=pytest.approx(0.5), less_than_half_width=True)
    d = T.decide({'a': r1, 'b': good(M=0.35, M_lo=0.01, M_hi=3.0)}, ['a', 'b'])     # a wide next interval does not change the half width
    assert d['lead']['lead'] == pytest.approx(0.55) and d['lead']['less_than_half_width'] is False


def test_lead_is_absent_with_one_qualifier():
    d = T.decide({'a': good(M=0.9), 'b': rec(None, None, status='failed')}, ['a', 'b'])
    assert d['outcome'] == 'PICK' and d['lead'] is None


# ------------------------------------------------------------------ the draws (rule §3.3: I_b, then S_b)
def test_draw_order_is_i_then_s_from_one_generator():
    I, S_ = T.joint_draws(6, 4, (1, 2, 3))
    rng = np.random.default_rng(0)
    for b in range(4):
        want_i = rng.integers(0, 6, size=6)
        want_s = rng.choice([1, 2, 3], size=3, replace=True)
        assert np.array_equal(I[b], want_i) and np.array_equal(S_[b], want_s)
    assert I.shape == (4, 6) and S_.shape == (4, 3)


def test_draws_on_a_tiny_case_are_pinned():
    """Literals from numpy's PCG64 under default_rng(0): I_1, S_1, I_2, S_2 in that order (S_b between the I_b)."""
    I, S_ = T.joint_draws(5, 2, (1, 2, 3))
    assert I.tolist() == [[4, 3, 2, 1, 1], [0, 4, 3, 4, 2]]
    assert S_.tolist() == [[1, 1, 1], [2, 3, 3]]
    swapped = np.random.default_rng(0)
    s_first = swapped.choice([1, 2, 3], size=3, replace=True).tolist()
    assert s_first != S_[0].tolist() or swapped.integers(0, 5, size=5).tolist() != I[0].tolist()   # the order is not free


def test_draws_use_all_rows_of_p_and_seeds_one_to_three():
    I, S_ = T.joint_draws(29232, 2, (1, 2, 3))
    assert I.shape == (2, 29232) and I.min() >= 0 and I.max() < 29232
    assert set(np.unique(S_)) <= {1, 2, 3}


# ------------------------------------------------------------------ the bootstrap on a tiny case, against a brute-force loop
N_ROWS, N_DRAWS, SEEDS = 36, 15, (1, 2, 3)


def tiny(seed=0):
    rng = np.random.default_rng(100 + seed)
    labels = dict(emotion=rng.integers(0, 4, N_ROWS), style=rng.integers(0, 5, N_ROWS), genre=rng.integers(0, 3, N_ROWS))
    labels['genre'][[1, 4, 5, 9, 20, 21, 30]] = -1                       # genre has unlabelled rows
    parts = {a: {n: {s: rng.integers(0, 4, (3, N_ROWS)) for s in SEEDS} for n in ('c', 'g')} for a in ASP}
    return labels, parts


def A_of(labels, parts, idx):
    return float(np.mean([ami(labels[idx], p[idx]) for p in parts]))


def brute_joint(labels, parts):
    """The rule's text as a loop: I_b over all rows, then S_b; unlabelled rows dropped per aspect, repeats kept."""
    rng = np.random.default_rng(0)
    L = {a: np.flatnonzero(labels[a] >= 0) for a in ASP}
    e_b = {a: [] for a in ASP}
    M_b = []
    for _ in range(N_DRAWS):
        I = rng.integers(0, N_ROWS, size=N_ROWS)
        Sb = rng.choice(list(SEEDS), size=3, replace=True)
        row = {}
        for a in ASP:
            idx = I[labels[a][I] >= 0]
            row[a] = np.mean([A_of(labels[a], parts[a]['c'][s], idx) - A_of(labels[a], parts[a]['g'][s], idx) for s in Sb])
            e_b[a].append(row[a])
        M_b.append(np.mean([row[a] / R[a] for a in ASP]))
    point = {a: {n: {s: A_of(labels[a], parts[a][n][s], L[a]) for s in SEEDS} for n in ('c', 'g')} for a in ASP}
    return e_b, M_b, point


def test_joint_bootstrap_equals_a_brute_force_loop():
    labels, parts = tiny()
    e_b, M_b, point = brute_joint(labels, parts)
    out = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=1)
    for a in ASP:
        Abar = {n: np.mean(list(point[a][n].values())) for n in ('c', 'g')}
        eff = out['effects'][a]
        assert eff['Abar']['c'] == pytest.approx(Abar['c'], abs=1e-12) and eff['Abar']['g'] == pytest.approx(Abar['g'], abs=1e-12)
        assert eff['e'] == pytest.approx(Abar['c'] - Abar['g'], abs=1e-12)
        lo, hi = np.percentile(e_b[a], [2.5, 97.5])
        assert eff['lo'] == pytest.approx(lo, abs=1e-12) and eff['hi'] == pytest.approx(hi, abs=1e-12)
        for n in ('c', 'g'):
            for s in SEEDS:
                assert eff['A'][n][s] == pytest.approx(point[a][n][s], abs=1e-12)
    lo, hi = np.percentile(M_b, [2.5, 97.5])
    e_point = {a: out['effects'][a]['e'] for a in ASP}
    assert out['M']['M'] == pytest.approx(sum(e_point[a] / R[a] for a in ASP) / 3, abs=1e-12)
    assert out['M']['lo'] == pytest.approx(lo, abs=1e-12) and out['M']['hi'] == pytest.approx(hi, abs=1e-12)


def test_workers_do_not_change_the_numbers():
    labels, parts = tiny(1)
    one = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=1)
    two = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=2)
    assert one == two


def test_the_reference_arm_is_a_parameter():
    labels, parts = tiny(2)
    ab = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=1)
    ba = T.joint_bootstrap(labels, parts, 'g', 'c', SEEDS, N_DRAWS, workers=1)
    for a in ASP:
        assert ba['effects'][a]['e'] == pytest.approx(-ab['effects'][a]['e'], abs=1e-12)
        assert ba['effects'][a]['lo'] == pytest.approx(-ab['effects'][a]['hi'], abs=1e-12)
    assert ba['M']['lo'] == pytest.approx(-ab['M']['hi'], abs=1e-12)


def test_per_aspect_mode_is_the_eg_rules_and_joint_differs_only_where_labels_are_missing():
    import eg_score3 as E
    labels, parts = tiny(3)
    joint = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=1)
    per = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=1, mode='per_aspect')
    assert per['M'] is None
    for a in ('emotion', 'style'):                                       # all rows labelled: the same draws
        assert per['effects'][a] == joint['effects'][a]
    assert per['effects']['genre']['e'] == joint['effects']['genre']['e']
    g_per, g_joint = per['effects']['genre'], joint['effects']['genre']
    assert (g_per['lo'], g_per['hi']) != (g_joint['lo'], g_joint['hi'])
    for a in ASP:                                                        # EG's own function: the original of this mode
        got = E.fresh_aspect(labels[a], parts[a], 'c', 'g', SEEDS, N_DRAWS, workers=1)
        assert per['effects'][a]['lo'] == pytest.approx(got['lo'], abs=1e-12)
        assert per['effects'][a]['hi'] == pytest.approx(got['hi'], abs=1e-12)
        assert per['effects'][a]['e'] == pytest.approx(got['e'], abs=1e-12)


def test_unlabelled_rows_are_dropped_per_aspect_and_repeats_are_kept():
    genre = np.array([0, 1, -1, -1, 2, 2])
    parts = np.array([[0, 1, 1, 1, 2, 0]] * 3)
    I = np.array([[2, 2, 0, 1, 4, 4, 3]])
    got = T.boot_scores(genre, {'x': parts}, I, workers=1)['x']
    idx = np.array([0, 1, 4, 4])                                         # rows 2 and 3 carry no genre label; the repeat stays
    assert got[0] == pytest.approx(A_of(genre, parts, idx))


def test_arms_that_are_equal_give_exactly_zero_everywhere():
    """ē_b = 0 in every draw whatever the draws are: every interval, and M's."""
    labels, parts = tiny(4)
    for a in ASP:
        parts[a]['g'] = {s: p.copy() for s, p in parts[a]['c'].items()}
    out = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=1)
    assert all(out['effects'][a][k] == 0 for a in ASP for k in ('e', 'lo', 'hi')) and out['M'] == dict(M=0, lo=0, hi=0)


def test_boot_cache_is_used_and_checked(tmp_path):
    labels, parts = tiny(5)
    img = {'c': 'x', 'g': 'y'}
    caches = {a: {s: T.Boots(tmp_path, a, s, 'p' * 64, 'l' * 64, img, N_DRAWS) for s in SEEDS} for a in ASP}
    first = T.joint_bootstrap(labels, parts, 'c', 'g', SEEDS, N_DRAWS, workers=1, caches=caches)
    assert list(tmp_path.glob('**/*__boot_*.npz'))
    zeros = {a: {n: {s: np.zeros((3, N_ROWS), np.int64) for s in SEEDS} for n in ('c', 'g')} for a in ASP}
    second = T.joint_bootstrap(labels, zeros, 'c', 'g', SEEDS, N_DRAWS, workers=1, caches=caches)   # the draws come from the cache
    assert second['effects']['emotion']['lo'] == first['effects']['emotion']['lo']
    other = T.Boots(tmp_path, 'emotion', 1, 'p' * 64, 'l' * 64, {'c': 'changed', 'g': 'y'}, N_DRAWS)
    with pytest.raises(AssertionError, match='other inputs'):
        other.get('c', 'P-n36')


# ------------------------------------------------------------------ the verdict file
def put_stage2(outputs, names, outcome='TOP', machine='local', smoke=False):
    top = [dict(rank=i + 1, name=n, M=1.0 - 0.1 * i) for i, n in enumerate(names)]
    path = outputs / 'sweep' / machine / 'stage2.json'
    S._json_dump(path, dict(stage='stage2', machine=machine, smoke=smoke, outcome=outcome, top=top,
                            p_sha256=rs_paths.P_SHA256))
    return path


def put_gate(outputs, smoke=False, ok=True, drop_items=(), **extra):
    """A passing (or not) V3a record with every item the gate must carry (`drop_items` leaves some out)."""
    names = T.v3a_item_names(smoke)
    items = {k: {'pass': True} for k in names if k not in drop_items}
    gate = dict(stage='v3a', smoke=smoke, **{'pass': ok}, code_sha256=T.code_sha256(), p_sha256=rs_paths.P_SHA256,
                bounds_checked=not smoke, n_draws=T.SMOKE_DRAWS if smoke else T.N_DRAWS, items=items,
                g_exports={str(s): {a: dict(file_sha256='0' * 64, array_sha256='0' * 64) for a in ASP} for s in T.SEEDS},
                g_run_sha256={str(s): '0' * 64 for s in T.SEEDS})
    gate.update(extra)
    S._json_dump(outputs / 'gates' / 'v3a.json', gate)
    return gate


def put_run(outputs, name, seed, **run):
    folder = rs_paths.run_dir(outputs, name, seed)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = H.parse(name)
    rec_ = dict(name=name, config=json.loads(json.dumps(cfg)), seed=seed, status='ok', cause=None, machine='local',
                p_sha256=rs_paths.P_SHA256, best_epoch=dict(emotion=3, style=2, genre=4), smoke=False)
    rec_.update(run)
    (folder / 'run.json').write_text(json.dumps(rec_))


def failed(outputs, name, seed):
    put_run(outputs, name, seed, status='failed', cause='nonfinite loss at epoch 2 (emotion)', failure_kind='nonfinite_loss')


def test_verdict_refuses_stage_2_without_a_top(tmp_path):
    path = put_stage2(tmp_path, [], outcome='TOP-NONE')
    put_gate(tmp_path)
    with pytest.raises(AssertionError, match='TOP-NONE'):
        T.run_verdict(tmp_path, path)


def test_verdict_refuses_a_top_configuration_that_lacks_a_seed(tmp_path):
    path = put_stage2(tmp_path, ['k6=0.0003', 'k2=8'])
    put_gate(tmp_path)
    for s in (1, 2, 3):
        failed(tmp_path, 'k6=0.0003', s)
    failed(tmp_path, 'k2=8', 1)
    failed(tmp_path, 'k2=8', 3)                                          # seed 2 of k2=8 is missing
    with pytest.raises(AssertionError, match=r'V3d.*k2=8.*s2'):
        T.run_verdict(tmp_path, path)
    assert not (tmp_path / 'stage3' / 'verdict.json').exists()


def test_verdict_refuses_without_a_passing_v3a_of_this_code(tmp_path):
    path = put_stage2(tmp_path, ['k6=0.0003'])
    for s in (1, 2, 3):
        failed(tmp_path, 'k6=0.0003', s)
    with pytest.raises(AssertionError, match='v3a.json'):
        T.run_verdict(tmp_path, path)
    put_gate(tmp_path, ok=False)
    with pytest.raises(AssertionError, match='did not pass'):
        T.run_verdict(tmp_path, path)
    put_gate(tmp_path, code_sha256='f' * 64)
    with pytest.raises(AssertionError, match='code'):
        T.run_verdict(tmp_path, path)
    put_gate(tmp_path, smoke=True)
    with pytest.raises(AssertionError, match='smoke'):
        T.run_verdict(tmp_path, path)


def test_a_configuration_with_a_failed_seed_does_not_qualify_and_all_failed_is_keep_g(tmp_path):
    names = ['k6=0.0003', 'k2=8']
    path = put_stage2(tmp_path, names)
    put_gate(tmp_path)
    for n in names:
        for s in (1, 2, 3):
            failed(tmp_path, n, s)
    v = T.run_verdict(tmp_path, path)
    assert v['outcome'] == 'KEEP-G' and v['pick'] is None and v['top'] == names
    assert v['results']['k2=8']['status'] == 'failed' and 'nonfinite' in v['results']['k2=8']['cause']
    assert v['inputs']['stage2_sha256'] == S._file_sha256(path)
    assert json.loads((tmp_path / 'stage3' / 'verdict.json').read_text()) == v


def test_verdict_is_kept_when_rerun_and_stops_on_other_inputs(tmp_path):
    path = put_stage2(tmp_path, ['k6=0.0003'])
    put_gate(tmp_path)
    for s in (1, 2, 3):
        failed(tmp_path, 'k6=0.0003', s)
    first = T.run_verdict(tmp_path, path)
    assert T.run_verdict(tmp_path, path) == first
    put_stage2(tmp_path, ['k6=0.0003', 'k2=8'])
    for s in (1, 2, 3):
        failed(tmp_path, 'k2=8', s)
    with pytest.raises(AssertionError, match='other inputs'):
        T.run_verdict(tmp_path, path)


def test_a_va3_failure_without_its_rerun_is_refused(tmp_path):
    path = put_stage2(tmp_path, ['k6=0.0003'])
    put_gate(tmp_path)
    for s in (1, 2):
        failed(tmp_path, 'k6=0.0003', s)
    put_run(tmp_path, 'k6=0.0003', 3, status='failed', cause='x', failure_kind='va3')
    with pytest.raises(AssertionError, match='rerun'):
        T.run_verdict(tmp_path, path)


def test_a_smoke_stage_2_is_refused_outside_a_smoke(tmp_path):
    path = put_stage2(tmp_path, ['k6=0.0003'], smoke=True)
    put_gate(tmp_path)
    with pytest.raises(AssertionError, match='smoke'):
        T.run_verdict(tmp_path, path)


def test_the_cli_takes_top_from_relative_to_outputs(tmp_path):
    put_stage2(tmp_path, ['k6=0.0003'])
    put_gate(tmp_path)
    for s in (1, 2, 3):
        failed(tmp_path, 'k6=0.0003', s)
    assert T.main(['verdict', '--top-from', 'sweep/local/stage2.json', '--outputs', str(tmp_path)]) == 0
    assert json.loads((tmp_path / 'stage3' / 'verdict.json').read_text())['outcome'] == 'KEEP-G'


# ------------------------------------------------------------------ the gate must carry V3a's whole item set
def test_v3a_items_are_named_by_aspect_arm_and_seed():
    names = T.v3a_item_names(False)
    assert len(names) == len(set(names))
    for a in ASP:
        for n in (T.G_ARM, T.R_ARM):
            assert f'{a}_Abar_{n}' in names and all(f'{a}_A_{n}_s{s}' in names for s in T.SEEDS)
        assert f'{a}_e' in names and f'{a}_lo_per_aspect' in names and f'{a}_hi_per_aspect' in names
    assert 'genre_lo_joint' not in names and 'genre_hi_joint' not in names        # genre's joint draw is not the EG rule's
    assert all(f'{a}_{k}_joint' in names for a in ('emotion', 'style') for k in ('lo', 'hi'))
    assert 'export_hashes_equal_stage_b' in names and 'partitions_equal_to_stage_b' in names
    smoke = T.v3a_item_names(True)                                                 # a smoke checks the point values only
    assert set(smoke) < set(names) and not [k for k in smoke if '_lo_' in k or '_hi_' in k]


@pytest.mark.parametrize('missing', ['emotion_lo_joint', 'genre_hi_per_aspect', 'export_hashes_equal_stage_b',
                                     'partitions_equal_to_stage_b', 'style_Abar_GS-0.02', 'genre_A_R6-5_s3', 'emotion_e'])
def test_a_gate_missing_one_item_is_refused(tmp_path, missing):
    put_gate(tmp_path, drop_items=(missing,))
    with pytest.raises(AssertionError, match=re.escape(missing)):
        T.require_v3a(tmp_path, smoke=False)


def test_a_gate_without_bounds_or_with_few_draws_is_refused_unless_smoke(tmp_path):
    put_gate(tmp_path, bounds_checked=False)
    with pytest.raises(AssertionError, match='bounds'):
        T.require_v3a(tmp_path, smoke=False)
    put_gate(tmp_path, n_draws=T.SMOKE_DRAWS)
    with pytest.raises(AssertionError, match='draws'):
        T.require_v3a(tmp_path, smoke=False)
    put_gate(tmp_path)
    assert T.require_v3a(tmp_path, smoke=False)['pass'] is True
    put_gate(tmp_path, smoke=True)
    assert T.require_v3a(tmp_path, smoke=True)['smoke'] is True


def test_partitions_that_differ_from_stage_b_fail_the_gate():
    equal = {n: {str(s): {a: True for a in ASP} for s in T.SEEDS} for n in (T.G_ARM, T.R_ARM)}
    assert T.partition_item(equal) == {'mismatches': [], 'pass': True}
    equal[T.R_ARM]['2']['style'] = False
    item = T.partition_item(equal)
    assert item['pass'] is False and item['mismatches'] == [f'{T.R_ARM} s2 style']


# ------------------------------------------------------------------ G's files: folder, hashes, and the only way to them
def test_g_folder_is_gs_002_seed_by_seed():
    for s in (1, 2, 3):
        assert T.g_folder(s) == rs_paths.RG_TRAIN / 'GS-0.02' / f's{s}'


def fake_run(seed, run='r', file='f', array='a'):
    return dict(run_sha256=run * 8 + str(seed), img=None,
                file_sha256={a: f'{file}-{a}-{seed}' for a in ASP}, array_sha256={a: f'{array}-{a}-{seed}' for a in ASP})


def g_gate():
    return dict(g_run_sha256={str(s): fake_run(s)['run_sha256'] for s in T.SEEDS},
                g_exports={str(s): {a: dict(file_sha256=fake_run(s)['file_sha256'][a], array_sha256=fake_run(s)['array_sha256'][a])
                                    for a in ASP} for s in T.SEEDS})


def patch_frozen(monkeypatch, runs):
    calls = []

    def read(folder, variant, seed):
        calls.append((folder, variant, seed))
        return runs[seed]
    monkeypatch.setattr(T, 'read_frozen_run', read)
    return calls


def test_g_runs_are_read_from_the_gs_folders_and_accepted_when_the_hashes_match(monkeypatch):
    calls = patch_frozen(monkeypatch, {s: fake_run(s) for s in T.SEEDS})
    out = T.read_g_runs(g_gate())
    assert sorted(out) == [1, 2, 3] and out[2] == fake_run(2)
    assert calls == [(rs_paths.RG_TRAIN / 'GS-0.02' / f's{s}', 'GS-0.02', s) for s in (1, 2, 3)]


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_g_run_json_that_differs_from_the_hashed_one_is_refused(monkeypatch, seed):
    runs = {s: fake_run(s) for s in T.SEEDS}
    runs[seed] = fake_run(seed, run='x')
    patch_frozen(monkeypatch, runs)
    with pytest.raises(AssertionError, match=rf's{seed}.*run.json'):
        T.read_g_runs(g_gate())


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('aspect', ASP)
@pytest.mark.parametrize('which', ['file', 'array'])
def test_g_export_that_differs_from_the_hashed_one_is_refused(monkeypatch, seed, aspect, which):
    runs = {s: fake_run(s) for s in T.SEEDS}
    runs[seed][f'{which}_sha256'][aspect] = 'changed'
    patch_frozen(monkeypatch, runs)
    with pytest.raises(AssertionError, match=rf'G s{seed} {aspect}.*hashed'):
        T.read_g_runs(g_gate())


class Reached(Exception):
    pass


def ok_runs():
    return {s: dict(status='ok', cause=None, img=None, run_sha256='r') for s in T.SEEDS}


def test_scoring_reaches_g_only_through_read_g_runs(monkeypatch):
    def sentinel(gate):
        raise Reached(gate)
    monkeypatch.setattr(T, 'read_g_runs', sentinel)
    monkeypatch.setattr(T, 'read_frozen_run', lambda *a: pytest.fail('G was read around read_g_runs'))
    with pytest.raises(Reached):
        T.score_configurations('out', ['a'], {'a': ok_runs()}, 'the-gate', dict(p_sha='p', labels={}), 3, 1)


def test_one_failed_seed_of_three_fails_the_configuration_and_it_is_never_bootstrapped(monkeypatch):
    """Seed 2 of 'a' failed, 1 and 3 ok: a failed record, no bootstrap for it; 'b' (all ok) is scored."""
    boot_calls = []

    def fake_bootstrap(labels, parts, arm, ref, seeds, n_draws, workers, caches, mode):
        boot_calls.append(arm)
        return dict(effects={a: dict(e=0.01, lo=0.0, hi=0.02) for a in ASP}, M=dict(M=0.5, lo=0.1, hi=0.9))
    monkeypatch.setattr(T, 'read_g_runs', lambda gate: ok_runs())
    monkeypatch.setattr(T, 'run_partitions', lambda *a: {asp: None for asp in ASP})
    monkeypatch.setattr(T, 'caches_for', lambda *a: None)
    monkeypatch.setattr(T, 'joint_bootstrap', fake_bootstrap)
    a_runs = ok_runs()
    a_runs[2] = dict(status='failed', cause='nonfinite loss at epoch 2', img=None, run_sha256='r')
    records, runs = T.score_configurations('out', ['a', 'b'], {'a': a_runs, 'b': ok_runs()}, 'gate', dict(p_sha='p', labels={}), 3, 1)
    assert records['a']['status'] == 'failed' and records['a']['qualifies'] is False and 'seed 2' in records['a']['cause']
    assert boot_calls == ['b'] and records['b']['qualifies'] is True and sorted(runs) == ['GS-0.02', 'b']
    assert T.decide(records, ['a', 'b'])['failed'] == ['a']


# ------------------------------------------------------------------ the bootstrap cache holds the draw code's hash
def test_boot_scores_cached_under_other_draw_code_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'code_sha256', lambda: 'a' * 64)
    img = {'c': 'x', 'g': 'y'}
    old = T.Boots(tmp_path, 'emotion', 1, 'p' * 64, 'l' * 64, img, 5)
    old.put('c', 'P-n36', np.arange(5.0))
    assert np.array_equal(T.Boots(tmp_path, 'emotion', 1, 'p' * 64, 'l' * 64, img, 5).get('c', 'P-n36'), np.arange(5.0))
    monkeypatch.setattr(T, 'code_sha256', lambda: 'b' * 64)                       # the draw code changed
    with pytest.raises(AssertionError, match='other inputs'):
        T.Boots(tmp_path, 'emotion', 1, 'p' * 64, 'l' * 64, img, 5).get('c', 'P-n36')


# ------------------------------------------------------------------ the verdict's written time
def test_the_time_is_amsterdam_local_to_the_minute():
    stamp = T.now_amsterdam()
    assert re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d', stamp)
    here = datetime.now(ZoneInfo('Europe/Amsterdam')).replace(tzinfo=None)
    assert abs((datetime.strptime(stamp, '%Y-%m-%d %H:%M') - here).total_seconds()) < 120


def test_verdict_carries_a_written_time_and_a_rewrite_keeps_the_file(tmp_path, monkeypatch):
    path = put_stage2(tmp_path, ['k6=0.0003'])
    put_gate(tmp_path)
    for s in (1, 2, 3):
        failed(tmp_path, 'k6=0.0003', s)
    monkeypatch.setattr(T, 'now_amsterdam', lambda: '2026-10-28 09:15')
    first = T.run_verdict(tmp_path, path)
    assert first['written'] == '2026-10-28 09:15'
    monkeypatch.setattr(T, 'now_amsterdam', lambda: '2026-10-28 11:40')
    again = T.run_verdict(tmp_path, path)                                         # equal content: the file, and its time, stay
    assert again == first
    assert json.loads((tmp_path / 'stage3' / 'verdict.json').read_text())['written'] == '2026-10-28 09:15'


# ------------------------------------------------------------------ the V3a check on a stub world (no frozen files, no k-means)
N_V3A = 12


def v3a_world(tmp_path, monkeypatch, joint_shift=None, partition_flip=False):
    """12-27's stage B 'verdict' and partition files in tmp_path; the frozen runs, k-means and the bootstrap are stubs that
    return the numbers the verdict holds (plus `joint_shift`: {(mode, aspect, key): delta})."""
    rng = np.random.default_rng(7)
    labels = {a: rng.integers(0, 3, N_V3A) for a in ASP}
    arms, shift = (T.G_ARM, T.R_ARM), joint_shift or {}
    part = {(n, s, a): rng.integers(0, 3, (3, N_V3A)) for n in arms for s in T.SEEDS for a in ASP}
    runs = {(n, s): dict(run_sha256=f'{n}{s}', img={a: None for a in ASP}, file_sha256={a: f'f{n}{s}{a}' for a in ASP},
                         array_sha256={a: f'a{n}{s}{a}' for a in ASP}) for n in arms for s in T.SEEDS}
    stage_b = tmp_path / 'stage_b'
    for (n, s, a), p in part.items():
        folder = stage_b / 'partitions' / n / f's{s}'
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f'{a}__partitions.npz', labels=p + (1 if partition_flip and (n, s, a) == (T.R_ARM, 2, 'style') else 0))
    want = {a: dict(Abar={n: 0.3 + 0.01 * i for i, n in enumerate(arms)}, e=-0.01, lo=-0.02 - 0.001 * j, hi=0.01 + 0.001 * j,
                    A={n: {str(s): 0.25 + 0.01 * s + 0.1 * i for s in T.SEEDS} for i, n in enumerate(arms)}) for j, a in enumerate(ASP)}
    ref = dict(smoke=False, n_draws=T.N_DRAWS, reference=T.R_ARM, seeds=list(T.SEEDS), identity=dict(
        p_sha256=rs_paths.P_SHA256, img_sha256={n: {str(s): dict(runs[(n, s)]['array_sha256']) for s in T.SEEDS} for n in arms}),
        results={T.G_ARM: dict(effects=want)})
    (stage_b / 'verdict.json').write_text(json.dumps(ref))

    def bootstrap(labels_, parts, arm, ref_arm, seeds, n_draws, workers, caches, mode):
        eff = {}
        for a in ASP:
            w = want[a]
            eff[a] = dict(A={n: {s: w['A'][n][str(s)] for s in seeds} for n in arms}, Abar=dict(w['Abar']), e=w['e'],
                          lo=w['lo'] + shift.get((mode, a, 'lo'), 0.0), hi=w['hi'] + shift.get((mode, a, 'hi'), 0.0))
        return dict(effects=eff, M=None if mode == 'per_aspect' else dict(M=0.1, lo=0.0, hi=0.2))
    monkeypatch.setattr(rs_paths, 'RG_STAGE_B', stage_b)
    monkeypatch.setattr(T.E, 'load_context', lambda: dict(p_sha=rs_paths.P_SHA256, labels=labels))
    monkeypatch.setattr(T, 'read_frozen_run', lambda folder, variant, seed: runs[(variant, seed)])
    monkeypatch.setattr(T.S, 'fixed_partitions', lambda root, n, a, img, p_sha, s: part[(n, s, a)])
    monkeypatch.setattr(T, 'joint_bootstrap', bootstrap)
    return ref


def test_v3a_check_writes_exactly_the_required_items_and_passes_on_equal_numbers(tmp_path, monkeypatch):
    v3a_world(tmp_path, monkeypatch)
    out = tmp_path / 'out'
    rec = T.run_check_v3a(out, workers=1)
    assert list(rec['items']) == T.v3a_item_names(False) and rec['pass'] is True and rec['bounds_checked'] is True
    assert all(i['pass'] for i in rec['items'].values()) and rec['items']['genre_lo_per_aspect']['diff'] == 0
    assert T.require_v3a(out, smoke=False)['g_run_sha256'] == {str(s): f'{T.G_ARM}{s}' for s in T.SEEDS}
    smoke = T.run_check_v3a(tmp_path / 'sm', workers=1, smoke=True)
    assert list(smoke['items']) == T.v3a_item_names(True) and smoke['bounds_checked'] is False and smoke['n_draws'] == T.SMOKE_DRAWS


@pytest.mark.parametrize('key', [('joint', 'emotion', 'lo'), ('joint', 'style', 'hi'), ('per_aspect', 'genre', 'lo'),
                                 ('per_aspect', 'genre', 'hi'), ('per_aspect', 'emotion', 'hi')])
def test_v3a_fails_when_a_bound_differs_from_stage_b_by_more_than_the_tolerance(tmp_path, monkeypatch, key):
    v3a_world(tmp_path, monkeypatch, joint_shift={key: 1e-4})
    rec = T.run_check_v3a(tmp_path / 'out', workers=1)
    mode, a, k = key
    assert rec['pass'] is False and rec['items'][f'{a}_{k}_{mode}']['pass'] is False
    assert sum(not i['pass'] for i in rec['items'].values()) == 1


def test_v3a_fails_when_a_partition_differs_from_stage_b(tmp_path, monkeypatch):
    v3a_world(tmp_path, monkeypatch, partition_flip=True)
    rec = T.run_check_v3a(tmp_path / 'out', workers=1)
    assert rec['pass'] is False and rec['items']['partitions_equal_to_stage_b']['mismatches'] == [f'{T.R_ARM} s2 style']
    assert sum(not i['pass'] for i in rec['items'].values()) == 1


def test_v3a_fails_when_an_export_hash_differs_from_the_stage_b_identity(tmp_path, monkeypatch):
    ref = v3a_world(tmp_path, monkeypatch)
    ref['identity']['img_sha256'][T.G_ARM]['1']['genre'] = 'other'
    (tmp_path / 'stage_b' / 'verdict.json').write_text(json.dumps(ref))
    rec = T.run_check_v3a(tmp_path / 'out', workers=1)
    assert rec['pass'] is False and rec['items']['export_hashes_equal_stage_b']['mismatches'] == [f'{T.G_ARM} s1 genre']
