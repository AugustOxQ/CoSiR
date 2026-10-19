"""Frozen-protocol AMI, comparator verdicts and score-sheet boundaries."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_score as score

PASSING = '21 passed in 2.1s'


@pytest.mark.parametrize('line, expected', [
    ('21 passed in 1.1s', True), ('20 passed in 1s', False),
    ('21 passed, 1 skipped in 1s', False), ('22 passed in 1s', False),
    ('21 passed, 2 warnings in 1.0s', True),
    ('21 passed, 1 failed in 1s', False), ('21 passed, 1 error in 1s', False),
    ('21 passed, 1 deselected in 1s', False),
])
def test_unit_tests_require_exactly_all_four_fix_tests(line, expected):
    assert score.unit_tests_pass(line) is expected


@pytest.fixture(autouse=True)
def cpu_threads():
    with threadpool_limits(limits=8):
        yield


def test_ami_scores_accepts_other_dimensions_and_masks_missing_labels():
    groups = np.tile(np.arange(20), 250)
    x = np.eye(23, dtype=np.float32)[groups]
    y = groups.copy()
    y[:100] = -1
    result = score.ami_scores(x, dict(emotion=y))
    assert result['emotion'] == pytest.approx(dict(mean=1., sd=0.))
    with pytest.raises(AssertionError):
        score.ami_scores(x * 2, dict(emotion=y))
    with pytest.raises(AssertionError):
        score.ami_scores(x[:100], dict(emotion=y[:100]))


def test_row2_check_on_real_frozen_data(tmp_path):
    result = score.check_row2(outdir=tmp_path)
    assert result['status'] == 'PASS'
    assert result['both_mean'] == pytest.approx(dict(
        emotion=0.08855639551158387, style=0.17095358715401665,
        genre=0.3930761231906966), abs=1e-6)
    assert json.loads((tmp_path / 'row2_check.json').read_text()) == result


def test_native_ami_uses_argmax_and_ignores_unlabelled_paintings():
    prob = np.array([[.8, .2], [.6, .4], [.1, .9], [.3, .7], [.9, .1]], dtype=np.float32)
    labels = dict(emotion=np.array([3, 3, 8, 8, -1]))
    assert score.native_ami(prob, labels) == pytest.approx(dict(emotion=1.))


def seed_result(value, auc=.8):
    return dict(img={a: dict(mean=value, sd=.99) for a in score.LABELS},
                native_img={a: value for a in score.LABELS},
                native_topic={a: value / 2 for a in score.LABELS},
                K={'1.5': 2, '2': 3, '3': 4, '5': 100}, dev_auc=auc)


def test_seed_aggregation_uses_population_sd_and_all_seed_auc_gate():
    seeds = {str(s): seed_result(v) for s, v in enumerate((.1, .2, .3))}
    result = score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)
    assert result['established']
    assert result['img']['emotion'] == pytest.approx(dict(mean=.2, sd=.0816496580927726))
    assert result['native_topic']['style']['mean'] == pytest.approx(.1)
    assert result['K']['5'] == dict(mean=100., sd=0.)
    assert result['dev_auc'] == pytest.approx(dict(mean=.8, sd=0.))
    seeds['1']['dev_auc'] = .74
    assert not score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)['established']
    seeds['1']['dev_auc'] = .75
    assert score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)['established']
    seeds['1']['dev_auc'] = None
    assert not score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)['established']
    assert not score.aggregate_arm(seeds, stage0_pass=False, tests_pass=True)['established']
    del seeds['1']
    result = score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)
    assert not result['established'] and result['img'] is None


def test_comparator_ties_use_lowest_number_and_exclude_unestablished_best():
    arms = {}
    # P1 is outside the best's tie band, even though it is near P2.
    for arm, value in [('P5', .5), ('P4', .2), ('P3', .198), ('P2', .197), ('P1', .194)]:
        arms[arm] = dict(established=arm != 'P5',
                         img={a: dict(mean=value, sd=0.) for a in score.LABELS})
    result = score.comparators(arms)
    for aspect in score.LABELS:
        assert result[aspect]['established']['arm'] == 'P2'
        assert result[aspect]['best_all']['arm'] == 'P5'
        assert result[aspect]['best_all']['established'] is False
    arms['P1']['img']['style']['mean'] = .6
    assert score.comparators(arms)['style']['established']['arm'] == 'P1'
    for arm in arms.values():
        arm['established'] = False
    assert score.comparators(arms)['genre']['established'] is None
    assert score.comparators({})['emotion']['best_all'] is None


def test_row1_readers_normalize_captions_before_painting_mean(monkeypatch):
    from types import SimpleNamespace
    s = dict(first=np.array([0, 2, 3]), pid=np.array([0, 0, 1, 2]), subset_pid=np.array([2, 0]))
    texts = dict(clip_text=np.array([[30, 0], [0, 4], [2, 0], [0, 5]], dtype=np.float32),
                 roberta=np.array([[0, 8], [2, 0], [3, 4], [6, 8]], dtype=np.float32))
    images = np.array([[0, 5], [1, 0], [4, 3]], dtype=np.float32)
    readers = SimpleNamespace(caption_features=lambda sample, kind: texts[kind],
                              image_features=lambda sample, kind: images if kind == 'clip_image' else None)
    monkeypatch.setitem(sys.modules, 'pc_data', readers)
    views = []
    def measure(x, labels):
        views.append(x.copy())
        return {a: dict(mean=float(x[0, 0]), sd=0.) for a in labels}
    monkeypatch.setattr(score, 'ami_scores', measure)
    result = score.row1(s, dict(emotion=np.array([0, 1])))
    assert len(views) == 3
    np.testing.assert_allclose(views[0], [[.8, .6], [0, 1]], atol=1e-7)
    np.testing.assert_allclose(views[1], [[0, 1], [2**-.5, 2**-.5]], atol=1e-7)
    np.testing.assert_allclose(views[2], [[.6, .8], [2**-.5, 2**-.5]], atol=1e-7)
    assert result['clip_text']['both_mean']['emotion'] == pytest.approx(.4)
    assert result['roberta']['both_mean']['emotion'] == pytest.approx(.7)
    assert result['clip_text']['img'] == result['roberta']['img']


@pytest.fixture
def stage_files(tmp_path):
    n, k = 5000, 20
    groups = np.random.default_rng(42).permutation(np.arange(n)) % k
    sample = dict(sample_ids=np.arange(99999), pid=np.arange(99999) % 19889,
                  first=np.arange(19889), all_painting_ids=np.arange(19889).astype(str),
                  subset_pid=np.arange(n)[::-1], subset_image_indices=np.arange(n)[::-1])
    labels = {a: groups.copy() for a in score.LABELS}
    labels['genre'][:100] = -1
    for stage in ('stage1', 'stage2'):
        (tmp_path / stage).mkdir()
    for seed in range(3):
        path = tmp_path / 'stage1' / f'clip_roberta__s{seed}.npz'
        np.savez(path, center_norms=np.arange(1, 101, dtype=np.float32),
                 kept=np.arange(100) >= 80, topic=(np.arange(99999) % k).astype(np.int16),
                 q_paint_eval=np.eye(k, dtype=np.float32)[np.zeros(n, dtype=int)])  # one topic: AMI 0
        path.with_suffix('.json').write_text(json.dumps(dict(
            config=dict(version='clip_roberta', seed=seed), K={'1.5': 10, '2': k, '3': 30, '5': 100},
            sample_sha256=score.fingerprint(sample))))
        stage2 = tmp_path / 'stage2' / f'P1__s{seed}.npz'
        np.savez(stage2, eval_prob=.1 + .8 * np.eye(k, dtype=np.float32)[groups])
        stage2.with_suffix('.json').write_text(json.dumps(dict(
            config=dict(arm='P1', seed=seed), dev_auc=(.75, .8, .85)[seed],
            dev_valid_topics=k, K=k, stage1_sha256=score.sha_file(path))))
    (tmp_path / 'stage0.json').write_text(json.dumps(dict(status='PASS', fix_tests=PASSING)))
    return sample, labels, tmp_path


def test_score_stage_files_establishes_only_complete_arms_and_preserves_metrics(stage_files):
    sample, labels, outdir = stage_files
    result = score.score_outputs(sample, labels, outdir)
    arm = result['arms']['P1']
    assert arm['established']
    assert arm['img']['emotion'] == pytest.approx(dict(mean=1., sd=0.))
    assert arm['native_img']['genre'] == pytest.approx(dict(mean=1., sd=0.))
    assert arm['native_img']['emotion'] == pytest.approx(dict(mean=1., sd=0.))
    assert arm['native_topic']['style'] == pytest.approx(dict(mean=0., sd=0.))
    assert arm['native_topic']['genre'] == pytest.approx(dict(mean=0., sd=0.))
    assert arm['dev_auc'] == pytest.approx(dict(mean=.8, sd=.0408248290463863))
    assert arm['seeds']['0']['K'] == {'1.5': 10, '2': 20, '3': 30, '5': 100}
    assert arm['seeds']['0']['dev_valid_topics'] == 20
    assert not result['arms']['P2']['established']
    assert len(result['missing_outputs']) == 12  # Four incomplete arms, three seeds each.
    assert result['missing_outputs']['P2__s0'] == ['stage2/P2__s0.npz', 'stage2/P2__s0.json']
    assert result['comparator']['emotion']['established']['arm'] == 'P1'
    (outdir / 'stage0.json').write_text(json.dumps(dict(status='FAIL')))
    result = score.score_outputs(sample, labels, outdir)
    assert not result['arms']['P1']['established']
    assert result['comparator']['emotion']['established'] is None
    assert result['comparator']['emotion']['best_all']['established'] is False


def test_results_files_contain_row1_native_counts_auc_and_comparator(stage_files):
    sample, labels, outdir = stage_files
    result = score.score_outputs(sample, labels, outdir)
    result['row1'] = {kind: dict(img={a: dict(mean=.2, sd=.01) for a in score.LABELS},
                                cap={a: dict(mean=.4, sd=.02) for a in score.LABELS},
                                both_mean={a: .3 for a in score.LABELS})
                      for kind in ('clip_text', 'roberta')}
    score.write_results(result, outdir)
    assert json.loads((outdir / 'pc_results.json').read_text()) == result
    text = (outdir / 'pc_tables.md').read_text()
    for value in ('image-side AMI', 'K-native', 'K per gamma', 'Dev AUC', 'Row 1', 'Comparator',
                  'clip_text', 'roberta', '0.30000', '0.75000', '100', 'P5', 'not established'):
        assert value in text
    assert not list(outdir.glob('*.pending.*'))
    # A serialization failure must leave the published reports intact.
    old_json = (outdir / 'pc_results.json').read_bytes()
    old_text = (outdir / 'pc_tables.md').read_bytes()
    result['invalid'] = float('nan')
    with pytest.raises(ValueError):
        score.write_results(result, outdir)
    assert (outdir / 'pc_results.json').read_bytes() == old_json
    assert (outdir / 'pc_tables.md').read_bytes() == old_text


def test_default_cli_writes_sheet_with_standin_feature_readers(stage_files, monkeypatch):
    from types import SimpleNamespace
    sample, labels, outdir = stage_files
    groups = np.zeros(19889, dtype=int)
    groups[sample['subset_pid']] = labels['emotion']
    captions = np.eye(23, dtype=np.float32)[groups[sample['pid']]] * 3
    images = np.eye(23, dtype=np.float32)[groups] * 7
    readers = SimpleNamespace(caption_features=lambda s, kind: captions,
                              image_features=lambda s, kind: images)
    monkeypatch.setitem(sys.modules, 'pc_data', readers)
    monkeypatch.setattr(score, 'load_frozen', lambda: sample)
    def stand_in(s, nodes):
        assert np.array_equal(nodes, s['subset_pid'])
        return labels
    monkeypatch.setattr(score, 'label_vectors', stand_in)
    monkeypatch.setattr(sys, 'argv', ['pc_score.py', '--outdir', str(outdir)])
    assert score.main() == 0
    result = json.loads((outdir / 'pc_results.json').read_text())
    assert result['arms']['P1']['established']
    assert result['row1']['clip_text']['both_mean']['emotion'] == pytest.approx(1.)
    assert result['row1']['roberta']['both_mean']['style'] == pytest.approx(1.)


def write_stage2_without_training(outdir, seed, reason, k=1):
    """Ticket 05 for K below 2: sidecar only, dev_auc null, no npz."""
    stage1 = outdir / 'stage1' / f'clip_roberta__s{seed}.npz'
    stage2 = outdir / 'stage2' / f'P1__s{seed}'
    stage2.with_suffix('.npz').unlink()
    stage2.with_suffix('.json').write_text(json.dumps(dict(
        config=dict(arm='P1', seed=seed), dev_auc=None, dev_valid_topics=0, K=k,
        reason=reason, stage1_sha256=score.sha_file(stage1))))


def test_untrained_seed_reports_its_reason_not_incomplete_files(stage_files):
    sample, labels, outdir = stage_files
    meta = outdir / 'stage1' / 'clip_roberta__s1.json'
    stage1 = json.loads(meta.read_text())
    stage1['K'] = {g: 1 for g in ('1.5', '2', '3', '5')}
    meta.write_text(json.dumps(stage1))
    norms = np.random.default_rng(7).permutation(np.r_[0, np.arange(100, 199)]).astype(np.float32)
    np.savez(outdir / 'stage1' / 'clip_roberta__s1.npz', center_norms=norms,
             kept=norms == 0, topic=np.zeros(99999, dtype=np.int16),
             q_paint_eval=np.ones((5000, 1), dtype=np.float32))
    write_stage2_without_training(outdir, 1, 'K below 2: no topics to train')
    result = score.score_outputs(sample, labels, outdir)
    arm = result['arms']['P1']
    assert not arm['established'] and arm['img'] is None
    text = ' '.join(arm['reasons'])
    assert 'K below 2: no topics to train' in text and 'Incomplete' not in text
    assert 'P1__s1' not in result['missing_outputs']
    assert 'P1__s0' not in result['missing_outputs']
    md = score.markdown(dict(result, row1={}))
    assert 'K below 2: no topics to train' in md
    assert 'K = 1 at every γ' in md
    assert '50.00 x the mean gap' in md
    assert 'between sorted positions 1 and 2 of 100 (counted from the lowest norm)' in md
    assert '99 centers above it are dropped (Algorithm 1)' in md
    assert arm['K']['2'] == pytest.approx(dict(mean=13.6666666667, sd=8.9566858950))
    # Stage 1 counts remain available even when no Stage 2 sidecar exists (P2).
    assert result['arms']['P2']['K']['2'] == arm['K']['2']
    k_table = md.split('## K per gamma')[1].split('## Dev AUC')[0]
    assert '| P1 | 7.00000 ± 4.24264 | 13.66667 ± 8.95669 |' in k_table


def test_null_dev_auc_is_undefined_not_below_threshold_and_nan_still_asserts():
    seeds = {str(s): seed_result(.2) for s in range(3)}
    seeds['1']['dev_auc'] = None
    reasons = score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)['reasons']
    assert reasons == ['Seed 1 dev AUC undefined']
    seeds['1']['dev_auc'] = .5
    assert score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)['reasons'] == ['Seed 1 dev AUC below 0.75']
    seeds['1']['dev_auc'] = float('nan')
    with pytest.raises(AssertionError):
        score.aggregate_arm(seeds, stage0_pass=True, tests_pass=True)


def sheet(outdir, sample, labels):
    result = score.score_outputs(sample, labels, outdir)
    result['row1'] = {kind: dict(img={a: dict(mean=.2, sd=.01) for a in score.LABELS},
                                cap={a: dict(mean=.4, sd=.02) for a in score.LABELS},
                                both_mean={a: .3 for a in score.LABELS})
                      for kind in ('clip_text', 'roberta')}
    return result, score.markdown(result)


@pytest.mark.parametrize('arrays, sidecar, equal, detail', [
    (dict(topic=True), dict(config=True), True, ''),
    (dict(topic=False), dict(config=True), False, 'arrays: topic'),
    (dict(topic=True), dict(config=False), False, 'sidecar: config'),
])
def test_sheet_reports_fresh_repeat_without_gating(stage_files, arrays, sidecar, equal, detail):
    sample, labels, outdir = stage_files
    report = dict(first_npz_sha256=score.sha_file(outdir / 'stage1/clip_roberta__s0.npz'),
                  bitwise_equal=equal, arrays=arrays, sidecar=sidecar)
    path = outdir / 'stage1_repeat.json'
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    result, text = sheet(outdir, sample, labels)
    assert result['stage1_repeat'] == dict(status='fresh', bitwise_equal=equal,
                                           arrays=arrays, sidecar=sidecar)
    line = next(line for line in text.splitlines() if line.startswith('Stage 1 repeat'))
    assert line.startswith('Stage 1 repeat (A2, reported, not a gate): fresh; bitwise equal: '
                           + ('yes' if equal else 'no'))
    assert detail in line
    assert text.index('Stage 0:') < text.index(line) < text.index('## Arms:')
    assert result['arms']['P1']['established']
    score.write_results(result, outdir)
    assert json.loads((outdir / 'pc_results.json').read_text())['stage1_repeat'] == result['stage1_repeat']
    assert path.read_bytes() == original


@pytest.mark.parametrize('kind, status', [
    ('missing', 'missing'), ('changed_npz', 'stale'), ('missing_npz', 'stale'),
])
def test_sheet_hides_missing_or_stale_repeat_details(stage_files, kind, status):
    sample, labels, outdir = stage_files
    report = dict(first_npz_sha256=score.sha_file(outdir / 'stage1/clip_roberta__s0.npz'),
                  bitwise_equal=False, arrays=dict(topic=False), sidecar=dict(config=False))
    if kind == 'missing_npz':
        outdir = outdir / 'empty'
        outdir.mkdir()
    elif kind == 'changed_npz':
        with (outdir / 'stage1/clip_roberta__s0.npz').open('ab') as stream:
            stream.write(b'changed')
        stage2 = outdir / 'stage2/P1__s0.json'
        meta = json.loads(stage2.read_text())
        meta['stage1_sha256'] = score.sha_file(outdir / 'stage1/clip_roberta__s0.npz')
        stage2.write_text(json.dumps(meta))
    path = outdir / 'stage1_repeat.json'
    if kind != 'missing':
        path.write_text(json.dumps(report))
    result, text = sheet(outdir, sample, labels)
    assert result['stage1_repeat'] == dict(status=status, bitwise_equal=None, arrays=None, sidecar=None)
    line = next(line for line in text.splitlines() if line.startswith('Stage 1 repeat'))
    assert line == f'Stage 1 repeat (A2, reported, not a gate): {status}; bitwise equal: —'
    if kind != 'missing':
        assert json.loads(path.read_text()) == report


def test_sheet_states_the_stage0_fix_tests_line(stage_files):
    sample, labels, outdir = stage_files
    (outdir / 'stage0.json').write_text(json.dumps(dict(status='PASS', fix_tests='23 passed in 4.56s')))
    assert '23 passed in 4.56s' in sheet(outdir, sample, labels)[1]
    (outdir / 'stage0.json').write_text(json.dumps(dict(status='PASS')))
    assert 'not recorded' in sheet(outdir, sample, labels)[1]


def test_comparator_names_the_highest_arm_before_the_tie_rule():
    arms = {}
    for arm, value in [('P1', .1930), ('P2', .1975), ('P3', .1985), ('P4', .2)]:
        arms[arm] = dict(established=True, img={a: dict(mean=value, sd=0.) for a in score.LABELS})
    pick = score.comparators(arms)['emotion']['established']
    assert pick['arm'] == 'P2' and pick['highest_arm'] == 'P4' and pick['highest_mean'] == .2
    text = score.markdown(dict(
        stage0=dict(status='PASS'), arms={}, row1={}, missing_outputs={},
        comparator=score.comparators(arms)))
    line = next(row for row in text.splitlines() if row.startswith('| emotion |') and 'P4' in row)
    assert 'P4 (0.20000)' in line and 'P2 (0.19750)' in line


def test_per_seed_image_ami_table_shows_seeds_of_an_incomplete_arm(stage_files):
    sample, labels, outdir = stage_files
    for suffix in ('.npz', '.json'):
        (outdir / 'stage2' / f'P1__s2{suffix}').unlink()
    result, text = sheet(outdir, sample, labels)
    assert result['arms']['P1']['img'] is None
    head = text.index('## Per-seed image-side AMI')
    table = text[head:].split('\n\n')[1:3]
    rows = [r for r in '\n'.join(table).splitlines() if r.startswith('| P1 |')]
    assert [r.split('|')[2].strip() for r in rows] == ['0', '1']
    assert '1.00000 ± 0.00000' in rows[0]


def test_failed_or_missing_unit_tests_establish_no_arm(stage_files):
    sample, labels, outdir = stage_files
    path = outdir / 'stage0.json'
    for stage0, reason in [
            (dict(status='PASS', fix_tests='1 failed, 18 passed in 2.1s'),
             'Unit tests (spec §6) failed: 1 failed, 18 passed in 2.1s'),
            (dict(status='PASS', fix_tests='2 errors in 0.3s'),
             'Unit tests (spec §6) failed: 2 errors in 0.3s'),
            (dict(status='PASS', fix_tests='no tests ran in 0.1s'),
             'Unit tests (spec §6) failed: no tests ran in 0.1s'),
            (dict(status='PASS'), 'Unit tests (spec §6) not recorded')]:
        path.write_text(json.dumps(stage0))
        result, text = sheet(outdir, sample, labels)
        arm = result['arms']['P1']
        assert not arm['established'] and reason in arm['reasons']
        assert result['stage0']['status'] == 'PASS'
        assert result['comparator']['emotion']['established'] is None
        assert reason in text
    path.write_text(json.dumps(dict(status='PASS', fix_tests='21 passed in 2.13s')))
    assert sheet(outdir, sample, labels)[0]['arms']['P1']['established']


def test_passing_unit_test_line_still_establishes_and_sheet_lists_condition(stage_files):
    sample, labels, outdir = stage_files
    result, text = sheet(outdir, sample, labels)
    assert result['arms']['P1']['established'] and result['arms']['P1']['reasons'] == []
    sentence = next(line for line in text.splitlines() if line.startswith('Stage 0:'))
    assert 'unit tests' in sentence.lower()


def test_stage2_reason_with_k_of_two_or_more_names_the_file(stage_files):
    sample, labels, outdir = stage_files
    sidecar = outdir / 'stage2' / 'P1__s0.json'
    meta = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps(dict(meta, reason='bogus')))
    with pytest.raises(AssertionError, match='P1__s0.*reason with K=20'):
        score.score_outputs(sample, labels, outdir)
