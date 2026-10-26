"""Tests of rg_held (ticket 07) on small synthetic partitions; no real data is read.

Seams: the two bootstraps' draw order (rule §3.2), the outcome boundaries (rule §4.3), the G_half refusal, the
v_held / VC1 gate, the statistic against a hand-computed AMI, and the file readers' asserts."""
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score as ami

import rg_paths  # noqa: F401
import rg_held
import rg_variants
import rb_score

SEEDS5 = [0, 1, 2, 3, 4]
X = rg_variants.X


# --- the draw order (rule §3.2 "Intervals") ---------------------------------------------------------------------

def test_d_t_draws_are_i_b_then_s_b_for_g_and_t_draws_nothing():
    n, b = 7, 3
    rng = np.random.default_rng(0)                    # the rule's transcript, written out by hand
    expected = []
    for _ in range(b):
        rows = rng.integers(0, n, size=n)
        s_g = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        expected.append((rows, s_g))
    draws = rg_held.draws_d_t(n, b)
    assert len(draws) == b
    for (rows, seeds), (e_rows, e_s) in zip(draws, expected):
        assert np.array_equal(rows, e_rows)
        assert set(seeds) == {'G', 'T'}
        assert np.array_equal(seeds['G'], e_s)
        assert np.array_equal(seeds['T'], [0])         # T enters every draw with its one partition set


def test_d_r_draws_are_i_b_then_s_b_then_s_r_b():
    n, b = 7, 3
    rng = np.random.default_rng(0)
    expected = []
    for _ in range(b):
        rows = rng.integers(0, n, size=n)
        s_g = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        s_r = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        expected.append((rows, s_g, s_r))
    draws = rg_held.draws_d_r(n, b)
    for (rows, seeds), (e_rows, e_g, e_r) in zip(draws, expected):
        assert np.array_equal(rows, e_rows)
        assert np.array_equal(seeds['G'], e_g) and np.array_equal(seeds['R'], e_r)
    assert not all(np.array_equal(s['G'], s['R']) for _, s in draws)   # the two seed draws are different draws


def test_the_draws_equal_the_base_scorers_transcript():
    """12-24's `rb_score.draw_bootstrap` is the source of the margins "T" (T seedless) and "b_matched"."""
    n, b = 11, 4
    base_t = rb_score.draw_bootstrap(n, {'K4': SEEDS5}, ('K4', 'T'), b, seedless=('T',))
    base_r = rb_score.draw_bootstrap(n, {'K4': SEEDS5, 'B1': SEEDS5}, ('K4', 'B1'), b)
    for (rows, seeds), (b_rows, b_seeds) in zip(rg_held.draws_d_t(n, b), base_t):
        assert np.array_equal(rows, b_rows) and np.array_equal(seeds['G'], b_seeds['K4'])
    for (rows, seeds), (b_rows, b_seeds) in zip(rg_held.draws_d_r(n, b), base_r):
        assert np.array_equal(rows, b_rows)
        assert np.array_equal(seeds['G'], b_seeds['K4']) and np.array_equal(seeds['R'], b_seeds['B1'])


# --- the outcomes (rule §4.3) -----------------------------------------------------------------------------------

G_HALF = 0.0263670169701253


def out(d_t, lo_t, d_r, lo_r, g_half=G_HALF):
    return rg_held.outcome(d_t, lo_t, d_r, lo_r, g_half)


def test_go_needs_the_lower_bound_strictly_above_minus_x():
    assert out(-0.01, -X + 1e-9, 0, 0)['outcome_id'] == 'G-GO'
    assert out(-0.01, -X, 0, 0)['outcome_id'] != 'G-GO'          # equal to -X is not above it
    assert out(0.02, -X + 1e-9, 0, 0)['outcome_id'] == 'G-GO'


def test_matches_or_exceeds_flag_only_when_the_lower_bound_is_above_zero():
    assert out(0.02, 1e-9, 0, 0)['matches_or_exceeds'] is True
    assert out(0.02, 0.0, 0, 0)['matches_or_exceeds'] is False
    assert out(0.0, -0.005, 0, 0)['matches_or_exceeds'] is False
    assert out(-0.03, -0.04, 0, 0)['matches_or_exceeds'] is False


def test_mm_needs_d_at_or_above_minus_x_and_the_lower_bound_at_or_below_minus_x():
    assert out(-X, -X, 0, 0)['outcome_id'] == 'G-MM'             # both boundaries inclusive
    assert out(-X + 1e-9, -0.02, 0, 0)['outcome_id'] == 'G-MM'
    assert out(-X - 1e-9, -0.02, 0, 0)['outcome_id'] == 'G-NO'   # d just below -X is not MM


def test_mm_states_pc_either_way():
    held = out(-0.005, -0.02, G_HALF, 0.01)
    assert held['outcome_id'] == 'G-MM' and held['pc'] is True and held['partial_sentence'] is True
    no = out(-0.005, -0.02, 0.0, -0.01)
    assert no['outcome_id'] == 'G-MM' and no['pc'] is False and no['partial_sentence'] is False


def test_pc_needs_d_r_at_or_above_g_half_and_the_lower_bound_above_zero():
    assert out(-0.05, -0.06, G_HALF, 1e-9)['pc'] is True          # d_R == G_half passes
    assert out(-0.05, -0.06, G_HALF - 1e-9, 1e-9)['pc'] is False
    assert out(-0.05, -0.06, G_HALF, 0.0)['pc'] is False           # the lower bound must be > 0
    assert out(-0.05, -0.06, G_HALF, -1e-9)['pc'] is False


def test_part_is_pc_without_go_or_mm_and_no_is_the_rest():
    assert out(-0.05, -0.06, 0.03, 0.01)['outcome_id'] == 'G-PART'
    assert out(-0.05, -0.06, 0.03, 0.0)['outcome_id'] == 'G-NO'
    assert out(-0.05, -0.06, 0.01, 0.01)['outcome_id'] == 'G-NO'
    assert out(-0.05, -0.06, 0.03, 0.01, g_half=0.04)['outcome_id'] == 'G-NO'


def test_go_wins_over_pc_and_mm_wins_over_part():
    assert out(0.0, 0.01, 0.05, 0.02)['outcome_id'] == 'G-GO'
    assert out(-0.01, -0.02, 0.05, 0.02)['outcome_id'] == 'G-MM'


def test_outcome_refuses_non_finite_values():
    with pytest.raises(AssertionError):
        out(float('nan'), 0.0, 0.0, 0.0)


# --- G_half (rule §3.2) -----------------------------------------------------------------------------------------

def test_g_half_is_half_the_gap_and_must_match_the_frozen_value():
    assert rg_held.g_half(0.5061513009972589, 0.4534172670570083) == pytest.approx(G_HALF, abs=1e-15)
    assert rg_held.require_g_half(G_HALF + 0.9e-6) == pytest.approx(G_HALF + 0.9e-6)
    with pytest.raises(AssertionError, match='G_half'):
        rg_held.require_g_half(G_HALF + 1.1e-6)


# --- the gate: v_held and VC1 (ticket 07, `score --arm G`) ------------------------------------------------------

def write_gate(out_dir, *, v_held='GR-8', outcome='B-PICK', switches=None, train=True, scorer=True):
    sw = rg_variants.switches('GR-8') if switches is None else switches
    (out_dir / 'stage_b').mkdir(parents=True)
    (out_dir / 'stage_b/verdict.json').write_text(json.dumps(
        dict(outcome=outcome, v_held=v_held, v_held_switches=sw)))
    reg = out_dir / 'stage_c/regression'
    reg.mkdir(parents=True)
    for name, ok in (('vc1_train', train), ('vc1_scorer', scorer)):
        if ok is not None:
            (reg / f'{name}.json').write_text(json.dumps({'pass': ok, 'items': {}}))


def test_gate_passes_for_the_named_v_held_with_both_vc1_files_passing(tmp_path):
    write_gate(tmp_path)
    shas = rg_held.require_gate(tmp_path, 'GR-8')
    assert set(shas) == {'stage_b_verdict', 'vc1_train', 'vc1_scorer'}


@pytest.mark.parametrize('kw', [dict(v_held='GR-4'), dict(v_held=None, outcome='B-NONE'),
                                dict(outcome='B-NONE'), dict(train=False), dict(scorer=False),
                                dict(train=None), dict(scorer=None),
                                dict(switches=dict(m=1, lam=4, tau_t=0.05, anchors=0))])
def test_gate_refuses(tmp_path, kw):
    write_gate(tmp_path, **kw)
    with pytest.raises(AssertionError):
        rg_held.require_gate(tmp_path, 'GR-8')


def test_gate_refuses_a_missing_stage_b_verdict(tmp_path):
    (tmp_path / 'stage_c/regression').mkdir(parents=True)
    with pytest.raises(AssertionError):
        rg_held.require_gate(tmp_path, 'GR-8')


def test_gate_refuses_an_arm_the_round_does_not_define(tmp_path):
    write_gate(tmp_path, v_held='GR-9')
    with pytest.raises(AssertionError):
        rg_held.require_gate(tmp_path, 'GR-9')


# --- the statistic, against a hand-computed AMI ------------------------------------------------------------------

N = 60


def synthetic(seed=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, size=N)
    y[rng.choice(N, size=9, replace=False)] = -1                  # masked rows
    def parts(noise):
        out = np.empty((3, N), dtype=np.int64)
        for i in range(3):
            p = np.where(rng.random(N) < noise, rng.integers(0, 5, size=N), np.abs(y) % 4)
            out[i] = p
        return out
    fixed_g = {s: {'r20': parts(0.2 + 0.05 * s)} for s in SEEDS5}
    fixed_r = {s: {'r20': parts(0.5)} for s in SEEDS5}
    fixed_t = {0: {'r20': parts(0.1)}}
    return y, fixed_g, fixed_t, fixed_r


def hand_a(parts_by_seed, y, rows, seeds):
    """The rule's A: mean over the (drawn) seeds of the mean over the three partitions; rows indexed first, then
    the unlabelled rows masked."""
    yy = y[rows]
    keep = yy >= 0
    return float(np.mean([np.mean([ami(yy[keep], p[rows][keep]) for p in parts_by_seed[s]['r20']]) for s in seeds]))


def test_point_estimates_and_every_bootstrap_draw_match_a_hand_computation():
    y, fg, ft, fr = synthetic()
    b = 4
    res = rg_held.statistics(fg, ft, fr, y, n_resamples=b, workers=1)
    everything = np.arange(N)
    a_g, a_t, a_r = hand_a(fg, y, everything, SEEDS5), hand_a(ft, y, everything, [0]), hand_a(fr, y, everything, SEEDS5)
    assert res['d_T']['A_G'] == pytest.approx(a_g, abs=1e-12) and res['d_T']['A_T'] == pytest.approx(a_t, abs=1e-12)
    assert res['d_T']['d'] == pytest.approx(a_g - a_t, abs=1e-12)
    assert res['d_R']['A_R'] == pytest.approx(a_r, abs=1e-12) and res['d_R']['d'] == pytest.approx(a_g - a_r, abs=1e-12)
    assert res['G_half'] == pytest.approx((a_t - a_r) / 2, abs=1e-12)
    rng_t = np.random.default_rng(0)                              # d_T: I_b, S_b
    manual_t = []
    for _ in range(b):
        rows = rng_t.integers(0, N, size=N)
        s_g = rng_t.choice(SEEDS5, size=5, replace=True)
        manual_t.append(hand_a(fg, y, rows, s_g) - hand_a(ft, y, rows, [0]))
    rng_r = np.random.default_rng(0)                              # d_R: I_b, S_b, S_R,b
    manual_r = []
    for _ in range(b):
        rows = rng_r.integers(0, N, size=N)
        s_g = rng_r.choice(SEEDS5, size=5, replace=True)
        s_r = rng_r.choice(SEEDS5, size=5, replace=True)
        manual_r.append(hand_a(fg, y, rows, s_g) - hand_a(fr, y, rows, s_r))
    assert np.allclose(res['d_T']['bootstrap_d'], manual_t, atol=1e-12)
    assert np.allclose(res['d_R']['bootstrap_d'], manual_r, atol=1e-12)
    assert res['d_T']['interval'] == pytest.approx(np.percentile(manual_t, [2.5, 97.5]).tolist(), abs=1e-12)
    assert res['d_R']['interval'] == pytest.approx(np.percentile(manual_r, [2.5, 97.5]).tolist(), abs=1e-12)


def test_statistics_refuses_partitions_of_the_wrong_shape_or_seed_set():
    y, fg, ft, fr = synthetic()
    with pytest.raises(AssertionError):
        rg_held.statistics({s: fg[s] for s in (0, 1, 2)}, ft, fr, y, n_resamples=2, workers=1)
    bad = {s: {'r20': fg[s]['r20'][:2]} for s in SEEDS5}
    with pytest.raises(AssertionError):
        rg_held.statistics(bad, ft, fr, y, n_resamples=2, workers=1)
    with pytest.raises(AssertionError):
        rg_held.statistics(fg, ft, fr, y[:-1], n_resamples=2, workers=1)


# --- the file readers' asserts ------------------------------------------------------------------------------------

def test_output_reader_checks_rows_unit_norm_finiteness_and_the_sidecar(tmp_path):
    import rb_student
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 16)).astype(np.float32)
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
    meta = dict(kind='held', arm='GR-8', scale='full', aspect='genre', seed=2, rows=8, width=16, dtype='float16',
                list_sha256='h')
    path = tmp_path / 'genre__s2.npz'
    rb_student.write_outputs(path, dict(img=x), dict(meta, sha256=rb_student.sha_array(x)))
    got = rg_held.load_held_output(path, 'GR-8', 2, 'h', n_rows=8, width=16)
    assert got.dtype == np.float32 and got.shape == (8, 16)
    for bad in (dict(arm='GR-4'), dict(seed=3), dict(list_sha256='other')):
        with pytest.raises(AssertionError):
            rg_held.load_held_output(path, bad.get('arm', 'GR-8'), bad.get('seed', 2), bad.get('list_sha256', 'h'),
                                     n_rows=8, width=16)
    with pytest.raises(AssertionError):
        rg_held.load_held_output(path, 'GR-8', 2, 'h', n_rows=9, width=16)
    scaled = (x.astype(np.float32) * 1.5).astype(np.float16)       # not unit norm
    rb_student.write_outputs(tmp_path / 'genre__s0.npz', dict(img=scaled),
                             dict(meta, seed=0, sha256=rb_student.sha_array(scaled)))
    with pytest.raises(AssertionError):
        rg_held.load_held_output(tmp_path / 'genre__s0.npz', 'GR-8', 0, 'h', n_rows=8, width=16)
    nonfinite = x.copy()
    nonfinite[0, 0] = np.nan
    rb_student.write_outputs(tmp_path / 'genre__s1.npz', dict(img=nonfinite),
                             dict(meta, seed=1, sha256=rb_student.sha_array(nonfinite)))
    with pytest.raises(AssertionError):
        rg_held.load_held_output(tmp_path / 'genre__s1.npz', 'GR-8', 1, 'h', n_rows=8, width=16)


def test_a_vc1_scorer_record_counts_only_for_this_scorer_code():
    now = rg_held.code_provenance()
    assert rg_held.vc1_scorer_is_current(dict(identity=dict(rg_held_sha256=now['rg_held_sha256'],
                                                             rb_score_sha256=now['rb_score_sha256'])))
    assert not rg_held.vc1_scorer_is_current(dict(identity=dict(rg_held_sha256='x',
                                                                 rb_score_sha256=now['rb_score_sha256'])))
    assert not rg_held.vc1_scorer_is_current(dict(items={}))
