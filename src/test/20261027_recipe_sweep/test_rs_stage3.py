"""Stage 3 (rule §3.3, §4.3): the qualifiers, PICK or KEEP-G, the joint bootstrap's draw order and statistics on a tiny
case against a brute-force loop, and the verdict file's refusals. No export of P is read here (the V3a gate does that on
the frozen files); the file layer runs on a tmp_path with failed runs, which carry no exports."""
import json

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
    d = T.decide({'a': r1, 'b': good(M=0.60), 'c': good(M=0.2)}, ['a', 'b', 'c'])
    assert d['lead'] == dict(over='b', lead=pytest.approx(0.30), half_width=pytest.approx(0.5), less_than_half_width=True)
    d = T.decide({'a': r1, 'b': good(M=0.35)}, ['a', 'b'])
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


def put_gate(outputs, smoke=False, ok=True, **extra):
    gate = dict(stage='v3a', smoke=smoke, **{'pass': ok}, code_sha256=T.code_sha256(), p_sha256=rs_paths.P_SHA256,
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
