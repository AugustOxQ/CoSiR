"""Frozen-protocol AMI, comparator verdicts and score-sheet boundaries."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_score as score


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
    result = score.aggregate_arm(seeds, stage0_pass=True)
    assert result['established']
    assert result['img']['emotion'] == pytest.approx(dict(mean=.2, sd=.0816496580927726))
    assert result['native_topic']['style']['mean'] == pytest.approx(.1)
    assert result['K']['5'] == dict(mean=100., sd=0.)
    assert result['dev_auc'] == pytest.approx(dict(mean=.8, sd=0.))
    seeds['1']['dev_auc'] = .74
    assert not score.aggregate_arm(seeds, stage0_pass=True)['established']
    seeds['1']['dev_auc'] = .75
    assert score.aggregate_arm(seeds, stage0_pass=True)['established']
    seeds['1']['dev_auc'] = None
    assert not score.aggregate_arm(seeds, stage0_pass=True)['established']
    assert not score.aggregate_arm(seeds, stage0_pass=False)['established']
    del seeds['1']
    result = score.aggregate_arm(seeds, stage0_pass=True)
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
    groups = np.tile(np.arange(k), n // k)
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
                 q_paint_eval=np.eye(k, dtype=np.float32)[(groups + 1) % k])
        path.with_suffix('.json').write_text(json.dumps(dict(
            config=dict(version='clip_roberta', seed=seed), K={'1.5': 10, '2': k, '3': 30, '5': 100},
            sample_sha256=score.fingerprint(sample))))
        stage2 = tmp_path / 'stage2' / f'P1__s{seed}.npz'
        np.savez(stage2, eval_prob=.1 + .8 * np.eye(k, dtype=np.float32)[groups])
        stage2.with_suffix('.json').write_text(json.dumps(dict(
            config=dict(arm='P1', seed=seed), dev_auc=(.75, .8, .85)[seed],
            dev_valid_topics=k, K=k, stage1_sha256=score.sha_file(path))))
    (tmp_path / 'stage0.json').write_text(json.dumps(dict(status='PASS')))
    return sample, labels, tmp_path


def test_score_stage_files_establishes_only_complete_arms_and_preserves_metrics(stage_files):
    sample, labels, outdir = stage_files
    result = score.score_outputs(sample, labels, outdir)
    arm = result['arms']['P1']
    assert arm['established']
    assert arm['img']['emotion'] == pytest.approx(dict(mean=1., sd=0.))
    assert arm['native_img']['genre'] == pytest.approx(dict(mean=1., sd=0.))
    assert arm['native_topic']['style'] == pytest.approx(dict(mean=1., sd=0.))
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
    captions = np.eye(23, dtype=np.float32)[sample['pid'] % 20] * 3
    images = np.eye(23, dtype=np.float32)[np.arange(19889) % 20] * 7
    readers = SimpleNamespace(caption_features=lambda s, kind: captions,
                              image_features=lambda s, kind: images)
    monkeypatch.setitem(sys.modules, 'pc_data', readers)
    monkeypatch.setattr(score, 'load_frozen', lambda: sample)
    monkeypatch.setattr(score, 'label_vectors', lambda s, nodes: labels)
    monkeypatch.setattr(sys, 'argv', ['pc_score.py', '--outdir', str(outdir)])
    assert score.main() == 0
    result = json.loads((outdir / 'pc_results.json').read_text())
    assert result['arms']['P1']['established']
    assert result['row1']['clip_text']['both_mean']['emotion'] == pytest.approx(1.)
    assert result['row1']['roberta']['both_mean']['style'] == pytest.approx(1.)
