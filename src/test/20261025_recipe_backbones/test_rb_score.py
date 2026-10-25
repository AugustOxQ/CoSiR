"""Part 2 scorer seams (ticket 12): bootstrap draw order, indexed scores, eligibility, outcomes, verdict files."""
import numpy as np
import pytest

import rb_paths  # noqa: F401  (sys.path)
import ct_score
import rb_paths
import rb_score as score

CT_ARMS = ('K1', 'B1', 'B2', 'B3', 'B4', 'B5')
SEEDS5 = [0, 1, 2, 3, 4]
NEW_ARMS = ('K2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'M2')


def test_draws_equal_the_ct_transcript_for_the_ct_arm_list():
    eligible = {'K1': [0, 1, 2, 3, 4], 'B1': [0, 1, 2, 3, 4], 'B2': [], 'B3': [],
                'B4': [0, 1, 2, 3, 4], 'B5': [1, 3]}
    mine = score.draw_bootstrap(12281, eligible, CT_ARMS, n_resamples=20)
    ct = ct_score.draw_bootstrap(12281, eligible, n_resamples=20)
    assert len(mine) == len(ct) == 20
    for (rows, seeds), (ct_rows, ct_seeds) in zip(mine, ct):
        np.testing.assert_array_equal(rows, ct_rows)
        assert list(seeds) == list(ct_seeds)
        for arm in ct_seeds:
            np.testing.assert_array_equal(seeds[arm], ct_seeds[arm])


def test_draws_follow_rule_3_2_order_for_the_new_arm_list():
    # Independent transcript: I_b, then Kk, B1..B6, matched, skipping arms without seeds.
    eligible = {'M2': [0, 1, 2], 'B6': [0, 1, 2, 3, 4], 'B5': [2, 4], 'B4': [], 'B3': [], 'B2': [],
                'B1': [0, 1, 2, 3, 4], 'K2': [0, 1, 2, 3, 4]}
    draws = score.draw_bootstrap(746, eligible, NEW_ARMS, n_resamples=3)
    rng = np.random.default_rng(0)
    for rows, seeds in draws:
        np.testing.assert_array_equal(rows, rng.integers(0, 746, size=746))
        assert list(seeds) == ['K2', 'B1', 'B5', 'B6', 'M2']
        for arm in ('K2', 'B1', 'B5', 'B6', 'M2'):
            np.testing.assert_array_equal(
                seeds[arm], rng.choice(eligible[arm], size=len(eligible[arm]), replace=True))


def test_each_call_starts_its_own_default_rng_0_and_rejects_unlisted_or_unsorted_arms():
    eligible = {'K2': [0, 1], 'B1': [0, 1]}
    first = score.draw_bootstrap(50, eligible, NEW_ARMS, n_resamples=2)
    again = score.draw_bootstrap(50, eligible, NEW_ARMS, n_resamples=2)
    np.testing.assert_array_equal(first[1][0], again[1][0])
    with pytest.raises(AssertionError):
        score.draw_bootstrap(50, {'K9': [0]}, NEW_ARMS, n_resamples=1)
    with pytest.raises(AssertionError):
        score.draw_bootstrap(50, {'K2': [1, 0]}, NEW_ARMS, n_resamples=1)


def hand_case():
    # For labels [0,0,1,1]: perfect AMI=1, constant=0, crossed=-1/2; the fifth row is masked.
    perfect, constant, crossed = [0, 0, 1, 1, 9], [0, 0, 0, 0, 9], [0, 1, 0, 1, 9]
    parts = {
        'K2': {0: dict(r20=np.array([perfect, constant, crossed])), 1: dict(r20=np.array([perfect] * 3))},
        'B1': {0: dict(r20=np.array([crossed] * 3), own=np.array([constant])),
               1: dict(r20=np.array([constant] * 3), own=np.array([perfect]))},
        'B6': {2: dict(r20=np.array([perfect, constant, constant]), own=np.array([crossed]))},
        'M2': {0: dict(r20=np.array([perfect] * 3), own=np.array([perfect]))},
    }
    return parts, np.array([0, 0, 1, 1, -1]), {'K2': [0, 1], 'B1': [0, 1], 'B6': [2], 'M2': [0]}


def test_scores_mean_partitions_then_seeds_over_the_new_entry_list():
    parts, y, seeds = hand_case()
    for rows in (np.arange(5), np.array([4, 4, 0, 1, 2, 3])):
        result = score.score_indices(parts, y, rows, seeds, NEW_ARMS)
        assert result['A_cand'] == pytest.approx(7 / 12)
        assert result['A_e'] == pytest.approx({'B1-r20': -.25, 'B1-own': .5, 'B6-r20': 1 / 3, 'B6-own': -.5,
                                             'M2-r20': 1, 'M2-own': 1})
        assert result['d'] == pytest.approx(7 / 12 - 1)
        assert result['per_seed']['K2'][0]['r20'] == pytest.approx([1, 0, -.5])
    # Repeated seeds are weighted draws; both readings of an arm share the draw.
    result = score.score_indices(parts, y, np.arange(5), {'K2': [0, 0], 'B1': [1, 1], 'B6': [2]}, NEW_ARMS)
    assert result['A_cand'] == pytest.approx(1 / 6)
    assert result['A_e']['B1-own'] == pytest.approx(1)
    assert 'M2-r20' not in result['A_e']


def test_the_candidate_is_the_first_arm_and_needs_seeds():
    parts, y, seeds = hand_case()
    with pytest.raises(AssertionError):
        score.score_indices(parts, y, np.arange(5), dict(seeds, K2=[]), NEW_ARMS)
    with pytest.raises(AssertionError):
        score.score_indices(parts, y, np.arange(5), seeds, ('B1', 'K2'))


def test_scores_equal_the_ct_scorer_on_the_ct_arm_list():
    rng = np.random.default_rng(3)
    n = 300
    y = rng.integers(-1, 6, size=n)
    fixed = {arm: {s: dict(r20=rng.integers(0, 8, size=(3, n)), **({} if arm == 'K1' else
                                                                  dict(own=rng.integers(0, 5, size=(1, n)))))
                   for s in range(5)} for arm in CT_ARMS}
    seeds = {'K1': [0, 1, 2, 3, 4], 'B1': [0, 0, 3], 'B2': [], 'B3': [], 'B4': [1, 2], 'B5': [4]}
    for rows in (np.arange(n), rng.integers(0, n, size=n)):
        mine = score.score_indices(fixed, y, rows, seeds, CT_ARMS)
        ct = ct_score.score_indices(fixed, y, rows, seeds)
        assert mine['d'] == ct['d'] and mine['A_cand'] == ct['A_K1'] and mine['A_e'] == ct['A_e']
        assert mine['per_seed'] == ct['per_seed']


def synthetic(arms, n=200, seed=5):
    rng = np.random.default_rng(seed)
    y = rng.integers(-1, 5, size=n)
    fixed = {arm: {s: dict(r20=rng.integers(0, 6, size=(3, n)),
                           **({} if arm == arms[0] else dict(own=rng.integers(0, 4, size=(1, n)))))
                   for s in range(5)} for arm in arms}
    return fixed, y


def test_bootstrap_parallel_equals_serial_and_the_ct_bootstrap():
    fixed, y = synthetic(CT_ARMS)
    eligible = {'K1': SEEDS5, 'B1': SEEDS5, 'B2': [], 'B3': [], 'B4': [0, 2, 4], 'B5': SEEDS5}
    serial = score.paired_bootstrap(fixed, y, eligible, CT_ARMS, n_resamples=12, workers=1)
    parallel = score.paired_bootstrap(fixed, y, eligible, CT_ARMS, n_resamples=12, workers=3)
    ct = ct_score.paired_bootstrap(fixed, y, eligible, n_resamples=12, workers=1)
    assert serial['bootstrap_d'] == parallel['bootstrap_d'] == ct['bootstrap_d']
    assert serial['interval'] == parallel['interval'] == ct['interval']
    assert serial['d'] == ct['d'] and serial['A_e'] == ct['A_e'] and serial['A_cand'] == ct['A_K1']
    np.testing.assert_array_equal(serial['interval'], np.percentile(serial['bootstrap_d'], [2.5, 97.5]))


def test_bootstrap_takes_the_maximum_over_entries_again_in_each_draw_and_any_row_count():
    arms = NEW_ARMS
    fixed, y = synthetic(arms, n=60)
    eligible = {a: SEEDS5 for a in arms if a not in ('B2', 'B3')}
    result = score.paired_bootstrap(fixed, y, eligible, arms, n_resamples=5, workers=1)
    for (rows, seeds), got in zip(score.draw_bootstrap(60, eligible, arms, 5), result['bootstrap_d']):
        assert got == score.score_indices(fixed, y, rows, seeds, arms)['d']
    assert result['d'] == score.score_indices(fixed, y, np.arange(60), eligible, arms)['d']


def test_bootstrap_checks_partition_shapes_and_readings():
    fixed, y = synthetic(NEW_ARMS, n=40)
    eligible = {a: SEEDS5 for a in NEW_ARMS if a not in ('B2', 'B3')}
    fixed['B1'][0]['own'] = np.zeros((2, 40), dtype=int)
    with pytest.raises(AssertionError):
        score.paired_bootstrap(fixed, y, eligible, NEW_ARMS, n_resamples=2, workers=1)


@pytest.mark.parametrize('d, lower, expected', [
    (.01, 1e-12, 'GO'), (.5, 0.0, 'MM'), (.01, -.2, 'MM'), (np.nextafter(.01, 0), .005, 'IN'),
    (1e-12, -.1, 'IN'), (0.0, -.1, 'NO'), (-.3, -.4, 'NO'), (-.3, .1, 'NO')])
def test_outcomes_at_the_thresholds_for_both_scales(d, lower, expected):
    assert score.outcome(d, lower, 'sample') == f'S1-{expected}'
    assert score.outcome(d, lower, 'full') == f'S2-{expected}'


def test_outcome_rejects_nonfinite_values_and_unknown_scales():
    for args in ((np.nan, 0.0, 'sample'), (.1, np.inf, 'full'), (.1, 0.0, 'a28')):
        with pytest.raises(AssertionError):
            score.outcome(*args)


def test_arm_lists_follow_the_rule_candidate_b1_to_b6_matched():
    assert score.arms_for('K2') == NEW_ARMS
    assert score.arms_for('K3')[-1] == 'M3' and score.arms_for('K5')[-1] == 'M5'
    assert score.arms_for('K4') == ('K4', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6')   # its matched arm is B1
    assert score.arms_for('K1') == ('K1', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6')   # K1 on K4's entry set
    assert score.CT_ARMS == CT_ARMS


def test_eligibility_rcol_for_b2_to_b5_and_all_seeds_for_the_rest_minus_nonfinite_paper_runs():
    rcol = {'B2': [], 'B3': [], 'B4': [0, 1, 2, 3, 4], 'B5': [0, 1, 2, 3, 4]}
    got = score.eligible_seeds(NEW_ARMS, rcol, nonfinite={'B5': [3]})
    assert got == {'K2': SEEDS5, 'B1': SEEDS5, 'B2': [], 'B3': [], 'B4': SEEDS5, 'B5': [0, 1, 2, 4],
                   'B6': SEEDS5, 'M2': SEEDS5}
    assert score.eligible_seeds(CT_ARMS, rcol)['K1'] == SEEDS5
    with pytest.raises(AssertionError):
        score.eligible_seeds(NEW_ARMS, dict(rcol, B4=[4, 1]))
    with pytest.raises(AssertionError):
        score.eligible_seeds(NEW_ARMS, rcol, nonfinite={'B1': [0]})   # only paper runs may drop (V4)


@pytest.mark.parametrize('scale, path', [('sample', rb_paths.CT_OUT / 'percept/rcol_stage1.json'),
                                         ('full', rb_paths.CT_FULL / 'percept/rcol.json')])
def test_rcol_records_of_the_frozen_ct_give_the_rule_text_eligibility(scale, path):
    assert score.read_rcol_eligible(scale)[0] == {'B2': [], 'B3': [], 'B4': SEEDS5, 'B5': SEEDS5}
    assert score.read_rcol_eligible(scale)[1] == path


def write_k(path, x):
    np.savez(path, img=x.astype(np.float16))


def unit(n, w, seed):
    x = np.random.default_rng(seed).normal(size=(n, w)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def test_output_paths_ct_arms_from_the_ct_scale_root_new_arms_from_part2_outputs(tmp_path):
    p = score.output_path('sample', 'B6', 2, outputs=tmp_path)
    assert p == rb_paths.CT_OUT / 'held/percept/core__k40/P2__s2.npz'
    assert score.output_path('full', 'K1', 0, 'style', tmp_path) == rb_paths.CT_FULL / 'held/d2/style__s0.npz'
    assert score.output_path('full', 'K3', 4, 'emotion', tmp_path) == tmp_path / 'part2/full/held/K3/emotion__s4.npz'
    assert score.output_path('sample', 'M5', 1, outputs=tmp_path) == tmp_path / 'part2/sample/held/M5/P3__s1.npz'
    assert score.output_path('sample', 'B4', 0, outputs=tmp_path).parts[-3:] == ('percept', 'paper__k40', 'P1__s0.npz')
    assert score.output_path('sample', 'B2', 0, outputs=tmp_path).parts[-2:] == ('paper__strict', 'P1__s0.npz')
    assert score.output_path('sample', 'B5', 0, outputs=tmp_path).name == 'P4__s0.npz'


def test_load_output_checks_rows_width_dtype_range_and_norm(tmp_path):
    k = unit(30, 2048, 0)
    write_k(tmp_path / 'k.npz', k)
    np.testing.assert_array_equal(score.load_output(tmp_path / 'k.npz', 'K2', 30), k.astype(np.float32))
    for bad in (k[:29], np.concatenate([k[:, :-1], k[:, :1]], 1) * 2):
        write_k(tmp_path / 'bad.npz', bad)
        with pytest.raises(AssertionError):
            score.load_output(tmp_path / 'bad.npz', 'K2', 30)
    prob = np.random.default_rng(1).random((30, 40)).astype(np.float32)
    np.savez(tmp_path / 'p.npz', prob=prob)
    np.testing.assert_array_equal(score.load_output(tmp_path / 'p.npz', 'M2', 30), prob)
    np.savez(tmp_path / 'p.npz', prob=prob * 2)
    with pytest.raises(AssertionError):
        score.load_output(tmp_path / 'p.npz', 'B6', 30)
    np.savez(tmp_path / 'p.npz', prob=prob[:, :39])
    with pytest.raises(AssertionError):
        score.load_output(tmp_path / 'p.npz', 'B1', 30)


def test_nonfinite_paper_outputs_are_dropped_everything_else_fails(tmp_path):
    prob = np.random.default_rng(1).random((30, 40)).astype(np.float32)
    prob[3, 3] = np.nan
    np.savez(tmp_path / 'p.npz', prob=prob)
    assert score.load_output(tmp_path / 'p.npz', 'B4', 30) is None
    for arm in ('B1', 'B6', 'M3'):
        with pytest.raises(AssertionError):
            score.load_output(tmp_path / 'p.npz', arm, 30)


def test_fixed_partitions_are_computed_once_then_read_and_bound_to_their_source(tmp_path):
    x = unit(300, 64, 2)
    first = score.fixed_partitions(x, 'K2', tmp_path, 'emotion', 0, 'held', 'src')
    assert set(first) == {'r20'} and first['r20'].shape == (3, 300)
    assert (tmp_path / 'emotion__s0__r20.npz').exists() and not (tmp_path / 'emotion__s0__own.npz').exists()
    again = score.fixed_partitions(x, 'K2', tmp_path, 'emotion', 0, 'held', 'src')
    np.testing.assert_array_equal(first['r20'], again['r20'])
    with pytest.raises(AssertionError):
        score.fixed_partitions(x, 'K2', tmp_path, 'emotion', 0, 'held', 'other source')
    prob = np.random.default_rng(3).random((300, 40)).astype(np.float32)
    fixed = score.fixed_partitions(prob, 'B6', tmp_path, 'percept', 1, 'held', 'src')
    assert set(fixed) == {'r20', 'own'} and fixed['own'].shape == (1, 300)
    np.testing.assert_array_equal(fixed['own'][0], prob.argmax(axis=1))
    assert (tmp_path / 'percept__s1__own.npz').exists()


# ---- the CT's own saved outputs (real data; read only) -------------------------------------------------------

CT_VERDICT = {'sample': 'verdict_stage1.json', 'full': 'verdict_stage2.json'}


@pytest.fixture(scope='module')
def held():
    return score.held_setup()


def ct_setup(scale):
    import json
    ct = json.loads((rb_paths.MAIN_TEST / '20261020_claim_test' / CT_VERDICT[scale]).read_text())
    rcol, _ = score.read_rcol_eligible(scale)
    eligible = score.eligible_seeds(CT_ARMS, rcol)
    assert eligible == ct['eligibility']['eligible_seeds']
    return ct, eligible, score.ct_cached_partitions(scale, CT_ARMS, eligible)


def test_held_setup_is_the_frozen_list_with_emotion_labels(held):
    names, sha, labels = held
    assert len(names) == 12281 and sha == rb_paths.HELD_SHA256 and names == sorted(names)
    assert labels.shape == (12281,) and labels.dtype.kind == 'i'


@pytest.mark.parametrize('scale', ['sample', 'full'])
def test_scorer_on_the_ct_partitions_gives_the_ct_point_estimate_and_first_draws(scale, held):
    ct, eligible, fixed = ct_setup(scale)
    labels = held[2]
    result = score.paired_bootstrap(fixed, labels, eligible, CT_ARMS, n_resamples=20, workers=8)
    assert result['A_cand'] == pytest.approx(ct['A_K1'], abs=1e-12)
    assert result['A_e'].keys() == ct['A_e'].keys()
    for entry, value in ct['A_e'].items():
        assert result['A_e'][entry] == pytest.approx(value, abs=1e-12)
    assert result['d'] == pytest.approx(ct['d'], abs=1e-12)
    # Draw-order equality with the CT run: its first 20 resampled d values.
    assert result['bootstrap_d'] == pytest.approx(ct['bootstrap_d'][:20], abs=1e-12)


def test_prepare_arms_loads_fixes_and_drops_nonfinite_paper_seeds(tmp_path, monkeypatch):
    import json
    ct_root, out = tmp_path / 'ct', tmp_path / 'out'
    monkeypatch.setitem(score.CT_ROOT, 'sample', ct_root)
    n, rng = 60, np.random.default_rng(0)

    def put(path, **arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        if ct_root in path.parents:
            path.with_suffix('.json').write_text(json.dumps(dict(held_sha256='H', rows=n)))

    for s in range(5):
        put(score.output_path('sample', 'K2', s, outputs=out), img=unit(n, 2048, s))
        put(score.output_path('sample', 'B1', s, outputs=out), prob=rng.random((n, 40)).astype(np.float32))
        put(score.output_path('sample', 'M2', s, outputs=out), prob=rng.random((n, 40)).astype(np.float32))
    for s in (0, 1):
        prob = rng.random((n, 40)).astype(np.float32)
        prob[0, 0] = np.nan if s == 1 else prob[0, 0]
        put(score.output_path('sample', 'B4', s, outputs=out), prob=prob)
    rcol = {'B2': [], 'B3': [], 'B4': [0, 1], 'B5': []}
    arms = ('K2', 'B1', 'B4', 'M2')
    fixed, eligible, sources = score.prepare_arms('sample', arms, rcol, 'H', out, n_rows=n)
    assert eligible == {'K2': SEEDS5, 'B1': SEEDS5, 'B4': [0], 'M2': SEEDS5}
    assert sources['nonfinite_held'] == {'B4': [1]} and set(fixed['B4']) == {0}
    assert set(fixed['K2'][0]) == {'r20'} and set(fixed['M2'][4]) == {'r20', 'own'}
    assert (out / 'part2/partitions/sample/K2/emotion__s0__r20.npz').exists()
    assert (out / 'part2/partitions/sample/M2/percept__s4__own.npz').exists()
    (ct_root / 'held/percept/core__k40/P3__s2.npz').unlink()
    with pytest.raises(AssertionError, match='missing eligible output'):
        score.prepare_arms('sample', arms, rcol, 'H', out, n_rows=n)


# ---- verdict blocks, files, regression items -----------------------------------------------------------------

def synthetic_block(scale='sample', n_resamples=4):
    arms = ('K2', 'B1', 'B4', 'M2')
    fixed, y = synthetic(arms, n=80)
    eligible = {a: SEEDS5 for a in arms}
    return score.candidate_block(fixed, y, eligible, arms, scale, n_resamples=n_resamples, workers=1,
                                 eligibility=dict(eligible_seeds=eligible), sources={}), (fixed, y, eligible, arms)


def test_candidate_block_carries_statistics_and_the_outcome_of_the_scale():
    block, (fixed, y, eligible, arms) = synthetic_block('full')
    ref = score.paired_bootstrap(fixed, y, eligible, arms, n_resamples=4, workers=1)
    assert block['d'] == ref['d'] and block['interval'] == ref['interval'] and block['A_e'] == ref['A_e']
    assert block['candidate'] == 'K2' and block['arms'] == list(arms)
    assert block['outcome_id'] == score.outcome(ref['d'], ref['interval'][0], 'full')
    assert block['bootstrap'] == dict(resamples=4, rng='numpy.random.default_rng(0) per candidate',
                                      arm_order=list(arms), n_rows=80, percentiles=[2.5, 97.5], method='linear')
    assert block['label_counts']['missing'] == int((y < 0).sum())


def test_verdict_file_merges_candidates_and_refuses_changed_content(tmp_path):
    path = tmp_path / 'part2/verdict_stage3.json'
    block, _ = synthetic_block()
    header = dict(stage=3, scale='sample', held_sha256='H')
    first = score.write_verdict(path, header, {'K2': block})
    assert first['blocks'].keys() == {'K2'} and first['stage'] == 3
    both = score.write_verdict(path, header, {'K3': block})        # a later candidate joins
    assert both['blocks'].keys() == {'K2', 'K3'}
    assert score.write_verdict(path, header, {'K2': block}) == both    # identical content: left as is
    changed = dict(block, d=block['d'] + 1)
    with pytest.raises(AssertionError, match='replace'):
        score.write_verdict(path, header, {'K2': changed})
    with pytest.raises(AssertionError, match='header'):
        score.write_verdict(path, dict(header, stage=4), {'K4': block})
    out = score.write_verdict(path, header, {'K2': changed}, replace='V2 trace')
    assert out['blocks']['K2']['d'] == changed['d'] and out['replacement']['reason'] == 'V2 trace'


def test_regression_items_merge_by_name_and_pass_needs_every_item(tmp_path):
    path = tmp_path / 'part2/regression/stage2.json'
    score.write_regression_items(path, {'student/emotion_s0': {'pass': True}})
    rec = score.write_regression_items(path, {'scorer/d': {'pass': True}, 'scorer/A_e/B1-r20': {'pass': False}})
    assert rec['pass'] is False and set(rec['items']) == {'student/emotion_s0', 'scorer/d', 'scorer/A_e/B1-r20'}
    rec = score.write_regression_items(path, {'scorer/A_e/B1-r20': {'pass': True}})
    assert rec['pass'] is True


def test_comparison_with_ct_values_flags_every_difference_beyond_1e6():
    ct = dict(A_K1=.1, A_e={'B1-r20': .05, 'B1-own': .06}, d=.04, interval=[.01, .03], bootstrap_d=[.04, .03, .02])
    ours = dict(A_cand=.1 + 5e-7, A_e={'B1-r20': .05, 'B1-own': .06 + 2e-6}, d=.04, interval=[.01, .03],
                bootstrap_d=[.04, .03, .02])
    items = score.compare_with_ct(ours, ct)
    assert {k for k, v in items.items() if not v['pass']} == {'scorer/A_e/B1-own'}
    assert items['scorer/A_K1']['pass'] and items['scorer/d']['abs_diff'] == 0
    assert set(items) == {'scorer/A_K1', 'scorer/entries', 'scorer/A_e/B1-r20', 'scorer/A_e/B1-own', 'scorer/d',
                          'scorer/interval_lo', 'scorer/interval_hi', 'scorer/bootstrap_d'}
    short = score.compare_with_ct(dict(ours, bootstrap_d=[.04, .03]), ct)
    assert not short['scorer/bootstrap_d']['pass']                # a different draw count is a failure
    extra = score.compare_with_ct(dict(ours, A_e=dict(ours['A_e'], **{'B6-r20': .1})), ct)
    assert not extra['scorer/entries']['pass']                    # entry sets must be identical


def test_comparison_on_the_real_ct_partitions_with_20_draws_passes_points_not_bounds(held):
    ct, eligible, fixed = ct_setup('sample')
    result = score.paired_bootstrap(fixed, held[2], eligible, CT_ARMS, n_resamples=20, workers=8)
    items = score.compare_with_ct(result, ct)
    assert items['scorer/d']['pass'] and items['scorer/A_K1']['pass']
    assert all(items[f'scorer/A_e/{e}']['pass'] for e in ct['A_e'])
    assert not items['scorer/interval_lo']['pass']
    prefix = score.compare_with_ct(dict(result, interval=ct['interval']), dict(ct, bootstrap_d=ct['bootstrap_d'][:20]))
    assert prefix['scorer/bootstrap_d']['pass']


# ---- stage driver and CLI -----------------------------------------------------------------------------------

def tiny_world(tmp_path, monkeypatch, n=60):
    """A 60-painting stand-in for the held set: CT-layout files under tmp, K2 and M2 outputs under outputs."""
    import json
    ct_root, out = tmp_path / 'ct', tmp_path / 'out'
    monkeypatch.setitem(score.CT_ROOT, 'sample', ct_root)
    rng = np.random.default_rng(0)
    labels = np.arange(n) % 5
    monkeypatch.setattr(score, 'held_setup', lambda: (['x'] * n, 'H', labels))
    monkeypatch.setattr(score, 'read_rcol_eligible', lambda scale: (
        {'B2': [], 'B3': [], 'B4': SEEDS5, 'B5': SEEDS5}, tmp_path / 'rcol.json'))
    (tmp_path / 'rcol.json').write_text('{}')

    def put(path, **arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        if ct_root in path.parents:
            path.with_suffix('.json').write_text(json.dumps(dict(held_sha256='H', rows=n)))

    for s in SEEDS5:
        for aspect in ('emotion', 'style', 'genre'):
            put(score.output_path('sample', 'K2', s, aspect, out), img=unit(n, 2048, s))
        for arm in ('B1', 'B4', 'B5', 'B6', 'M2'):
            put(score.output_path('sample', arm, s, outputs=out), prob=rng.random((n, 40)).astype(np.float32))
    return out


def test_stage_driver_writes_a_block_per_candidate_after_the_gate(tmp_path, monkeypatch):
    out = tiny_world(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):                         # V1: no regression file yet
        score.run_stage(3, ['K2'], out, workers=1, n_resamples=3)
    score.write_regression_items(out / 'part2/regression/stage2.json', {'x': {'pass': False}})
    with pytest.raises(AssertionError, match='V1'):
        score.run_stage(3, ['K2'], out, workers=1, n_resamples=3)
    score.write_regression_items(out / 'part2/regression/stage2.json', {'x': {'pass': True}})
    record = score.run_stage(3, ['K2'], out, workers=1, n_resamples=3)
    block = record['blocks']['K2']
    assert record['stage'] == 3 and block['arms'] == list(NEW_ARMS)
    assert block['eligibility']['eligible_seeds']['B2'] == [] and block['outcome_id'].startswith('S1-')
    assert (out / 'part2/verdict_stage3.json').exists()
    assert len(block['bootstrap_d']) == 3


def test_stage_4_needs_the_regression_and_a_stage_3_block_that_is_not_s1_no(tmp_path):
    out = tmp_path
    score.write_regression_items(out / 'part2/regression/stage3a.json', {'x': {'pass': True}})
    with pytest.raises(FileNotFoundError):
        score.require_gate(out, 4, ['K2'])
    header = dict(stage=3, scale='sample', held_sha256='H')
    score.write_verdict(out / 'part2/verdict_stage3.json', header, {'K2': dict(outcome_id='S1-MM'),
                                                                    'K3': dict(outcome_id='S1-NO')})
    score.require_gate(out, 4, ['K2'])
    with pytest.raises(AssertionError, match='S1-NO'):
        score.require_gate(out, 4, ['K3'])
    with pytest.raises(AssertionError, match='no stage 3'):
        score.require_gate(out, 4, ['K5'])


def test_cli_refuses_smoke_outside_tmp_and_never_shrinks_a_check(tmp_path):
    with pytest.raises(AssertionError, match='/tmp/claude-0'):
        score.main(['--stage', '3', '--candidates', 'K2', '--smoke', '--outputs', str(tmp_path / 'not-tmp')])
    with pytest.raises(AssertionError):
        score.main(['--check', 'stage2', '--smoke', '--outputs', '/tmp/claude-0/x'])


# ---- arms without seeds (the frozen encoder image Fk of the reported margins) --------------------------------

def test_seedless_arms_draw_nothing_but_enter_every_resample_with_their_one_partition_set():
    arms = ('K2', 'B1', 'F2')
    eligible = {'K2': SEEDS5, 'B1': SEEDS5}
    plain = score.draw_bootstrap(40, eligible, arms, 3)
    draws = score.draw_bootstrap(40, eligible, arms, 3, seedless=('F2',))
    for (rows, seeds), (plain_rows, plain_seeds) in zip(draws, plain):
        np.testing.assert_array_equal(rows, plain_rows)                     # same rng stream: F2 consumed nothing
        assert list(seeds) == ['K2', 'B1', 'F2'] and list(seeds['F2']) == [0]
        np.testing.assert_array_equal(seeds['K2'], plain_seeds['K2'])
    fixed, y = synthetic(('K2', 'B1'), n=40)
    fixed['F2'] = {0: dict(r20=np.random.default_rng(9).integers(0, 6, size=(3, 40)))}
    result = score.paired_bootstrap(fixed, y, eligible, arms, n_resamples=3, workers=1, seedless=('F2',))
    assert 'F2-r20' in result['A_e'] and 'F2-own' not in result['A_e']
    ref = score.score_indices(fixed, y, np.arange(40), dict(eligible, F2=[0]), arms)
    assert result['d'] == ref['d']
    with pytest.raises(AssertionError):                                         # a seedless arm is not eligible-listed
        score.draw_bootstrap(40, dict(eligible, F2=[0]), arms, 1, seedless=('F2',))
