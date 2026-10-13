import numpy as np
import pytest
import rd9_stats as S


def test_largest_remainder_shares_use_language_order_for_ties():
    assert S.shares([1, 1, 1], 5) == [2, 2, 1]
    assert S.shares([2, 3, 5], 7) == [1, 2, 4]
    assert S.shares([0, 1, 1], 3) == [0, 2, 1]
    with pytest.raises(AssertionError):
        S.shares([0, 0], 4096)


@pytest.mark.parametrize('m, boundaries', [(7, [16,19,24,30,40,61,124]), (5, [24,30,40,61,124]), (2, [61,124])])
def test_holm_rule_integer_boundaries(m, boundaries):
    ids = [f'P{j}' for j in range(m)]
    result = S.holm(dict(zip(ids, boundaries)), ids)
    assert result['order'] == ids
    assert all(row['pass'] for row in result['checks'].values())
    for k in range(m):
        counts = [0] * k + [boundaries[k] + 1] * (m-k)
        result = S.holm(dict(zip(ids, counts)), ids)
        assert [result['checks'][j]['pass'] for j in ids] == [True]*k + [False]*(m-k)


def test_holm_tie_order_and_stop_even_when_later_count_passes():
    result = S.holm({'P4':62, 'P6':62}, ['P4','P6'])
    assert result['order'] == ['P4','P6']
    assert not result['checks']['P6']['pass']
    assert result['checks']['P6']['own_count_passes']
    assert result['checks']['P6']['not_reached']


@pytest.mark.parametrize('counts, claim', [([0]*7, 'full claim'), ([25,25,25,0,25,0,25], 'small claim'), ([0,0,0,125,0,125,0], 'NO-GO')])
def test_two_levels_gate_claims_and_do_not_test_level_two_after_failure(counts, claim):
    result = S.decide(dict(zip(S.CHECK_IDS, counts)), a28=True)
    assert result['outcome'] == claim
    if claim == 'NO-GO':
        assert result['families']['level2'] is None
        assert result['tested'] == ['P4','P6']
    else:
        assert result['families']['level1']['order'] == ['P4','P6']
        assert result['families']['level2']['order'] == ['P1','P2','P3','P5','P7']


def test_cluster_counts_include_zero_and_pool_repeated_paintings():
    draws = S.Draws(np.array(['same','same','other']))
    # Every mean is positive, and the observation-weighted point is 1/6.
    values = np.array([0.25, 0.25, 0.0])
    assert np.count_nonzero(draws.int_sums(np.zeros(3)) <= 0) == 5000
    assert np.count_nonzero(draws.int_sums(np.full(3, 0.25)) <= 0) == 0
    assert set(np.unique(draws.boots(values))) == {0.0, 1/6, 0.25}
    with pytest.raises(AssertionError):
        draws.int_sums(np.array([0.1, 0.25, 0.0]))


def test_strict_ties_miss_and_gain_is_quarter_integer():
    from rd9_scores import metrics
    scores = {c: {d: np.array([[3,3,0],[3,2,0]], np.float32) for d in ('i2t','t2i')} for c in ('a','b')}
    got = metrics(scores)
    np.testing.assert_array_equal(got['r1'], [0,0.5])
    np.testing.assert_array_equal(got['gain'], [0,0])


def test_cell_selection_uses_integer_minimum_and_first_tie():
    from rd9_scores import select_cells
    # cell 0 wins R@1 but loses gain; cell 1 wins min-margin; cell 2 ties it.
    rho = np.array([[20,20],[18,19],[18,19]])
    gamma = np.array([[0,0],[6,7],[6,7]])
    cf = np.array([[10,10],[11,11],[11,11]])
    assert select_cells(rho, gamma, cf, [12,12]) == ([1,1], [1,1])


def test_agreement_is_exact_for_discrete_values_and_arrays():
    from rd9_compare import compare_value,compare_array
    assert not compare_value(1,2,'checks.P4.n')
    assert not compare_value(True,False,'checks.P4.passes')
    assert compare_value(1.0,1.0+5e-10,'checks.P4.point')
    assert not compare_value(1.0,1.0+2e-9,'checks.P4.point')
    assert compare_value(0.5,0.5+1e-10,'taus.1')
    assert not compare_array(np.array([1.0]),np.array([1.0],np.float32))
    assert not compare_array(np.array([1.0]),np.array([1.0+1e-12]))


def test_unexpected_runtime_error_does_not_set_language_aside():
    from rd9_episodes import failed_draw_count
    assert failed_draw_count('emotion x style: could not fill 4 episodes (41 failed draws); the pools are too small for these constraints', 'emotion','style',4) == 41
    assert failed_draw_count('unexpected failed IO', 'emotion','style',4) is None
    assert failed_draw_count('emotion x style: could not fill 4 episodes (40 failed draws); the pools are too small for these constraints', 'emotion','style',4) is None


def test_agreement_requires_exact_frozen_weights_and_sigma():
    from rd9_compare import compare_value
    assert not compare_value(0.5,0.5+1e-12,'picks.rca_lambda.0')
    assert not compare_value(0.0,1e-12,'picks.sigma_star.0')


def test_pin_paths_keep_round9_outputs_in_the_running_worktree():
    from rd9_inputs import resolve_pin,HERE,MAIN
    assert resolve_pin('src/test/20261202_r9_scope/results/new.npz') == HERE/'results/new.npz'
    assert resolve_pin('src/test/20261031_pseudo_partitions/results/partitions.npz') == MAIN/'src/test/20261031_pseudo_partitions/results/partitions.npz'


def test_h5_archive_uses_concatenated_arrays_and_checks_stored_hashes():
    from rd9_phase2 import h5_parts
    from rd9_episodes import FIELDS
    # Minimal schema fixture; rows retain their original global positions.
    archive={'pair_index':np.repeat(np.arange(3),2), 'pair_names':np.array(['emotion__style','emotion__genre','style__genre'])}
    for f in FIELDS:
        archive[f]=np.arange(6) if f=='anchor' else np.zeros((6,13 if f=='candidates' else 4),np.int64)
    parts=h5_parts(archive,per_pair=2,check_hashes=False)
    np.testing.assert_array_equal(parts['style__genre'].anchor,[4,5])
    assert parts['emotion__genre'].candidates.shape==(2,13)
    with pytest.raises(AssertionError):
        h5_parts(archive,per_pair=3,check_hashes=False)


def test_arm_l_draw_identity_hashes_sorted_row_ids_without_changing_fit_order():
    from rd9_bundle import head_draw_hash
    rows=np.array([2,0,1],np.int64)
    assert head_draw_hash(rows)=='ab25350e3e65efebe24584461683ecda68725576e825e550038b90e7b1479946'
    np.testing.assert_array_equal(rows,[2,0,1])
