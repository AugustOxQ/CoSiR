"""Stage 2 R-COL, outcome/comparator rules and persisted verdict seams."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import ct_rcol
import ct_score
import ct2_verdict as verdict
import ct2_data


def test_sample_rcol_reproduces_real_stage1_report(tmp_path):
    reader = ct_rcol.read_stage1
    expected = json.loads(ct_rcol.OUTPUT.read_text())
    actual = verdict.write_rcol(tmp_path, mode='sample', stage1_dir=ct_rcol.STAGE1)
    assert {k: v for k, v in actual.items() if k != 'written'} == {
        k: v for k, v in expected.items() if k != 'written'}
    path = tmp_path / 'percept/rcol.json'
    assert json.loads(path.read_text()) == actual
    # An existing report retains the original timestamp and exact content.
    path.write_text(json.dumps(expected))
    assert verdict.write_rcol(tmp_path, mode='sample', stage1_dir=ct_rcol.STAGE1) == expected
    expected['runs'][0]['share'] = .123
    path.write_text(json.dumps(expected))
    before = path.read_bytes()
    with pytest.raises(AssertionError, match='refusing to overwrite'):
        verdict.write_rcol(tmp_path, mode='sample', stage1_dir=ct_rcol.STAGE1)
    assert path.read_bytes() == before
    assert ct_rcol.read_stage1 is reader


@pytest.fixture(scope='module')
def full_names():
    return ct2_data.split2('full')['names_sha256']


@pytest.fixture(scope='module')
def full_paper_root(tmp_path_factory, full_names):
    root = tmp_path_factory.mktemp('full_paper')
    stage1 = root / 'percept/paper/stage1'
    stage1.mkdir(parents=True)
    # Every pair coincides with center 0, which holds all fit pairs.
    count = 216107
    centers = np.eye(100, 128, dtype=np.float32)
    centers[0] = 0
    first = stage1 / 'clip_roberta__s0.npz'
    np.savez_compressed(first, centers=centers,
                        fit_latent=np.zeros((count, 128), np.float32),
                        center_norms=np.linalg.norm(centers, axis=1),
                        fit_counts=np.r_[count, np.zeros(99, dtype=np.int64)],
                        kept_strict=np.arange(100) == 0, kept_k40=np.arange(100) < 40,
                        fit_topics_strict=np.zeros(count, np.int64),
                        fit_topics_k40=np.zeros(count, np.int64))
    for version in ('clip_roberta', 'qwen_qwen'):
        for seed in range(5):
            path = stage1 / f'{version}__s{seed}.npz'
            if path != first:
                path.hardlink_to(first)
            path.with_suffix('.json').write_text(json.dumps(dict(config=dict(
                mode='full', names_sha256=full_names))))
    return root


def test_full_rcol_uses_all_216107_fit_pairs_and_freezes_content(full_paper_root):
    report = verdict.write_rcol(full_paper_root)
    assert len(report['runs']) == 20
    assert all(r['share'] == 1 and r['collapsed'] for r in report['runs'])
    assert report['entries'] == {arm: dict(eligible=False, seeds=[])
                                 for arm in ('B2', 'B3', 'B4', 'B5')}
    path = full_paper_root / 'percept/rcol.json'
    before = path.read_bytes()
    with pytest.raises(AssertionError, match='fit pairs'):
        verdict.read_stage1(full_paper_root / 'percept/paper/stage1/clip_roberta__s0.npz', 67327)
    assert path.read_bytes() == before


@pytest.mark.parametrize('d,lower,expected', [
    (.01, .000001, 'S2-GO'), (.01, 0., 'S2-MM'), (.02, -.01, 'S2-MM'),
    (np.nextafter(.01, 0), .001, 'S2-IN'), (.000001, -.01, 'S2-IN'),
    (0., .001, 'S2-NO'), (-.001, -.02, 'S2-NO'),
])
def test_outcome_boundaries_use_unrounded_values(d, lower, expected):
    assert verdict.stage2_outcome(d, lower) == expected


@pytest.mark.parametrize('entries,seeds,expected', [
    ({'B1-own': .9, 'B2-r20': .2}, {'B2': [0]}, None),
    ({'B1-own': .9, 'B2-r20': .2},
     {'K1': [0, 1, 2, 3, 4], 'B1': [0, 1, 2, 3, 4], 'B2': [0]}, None),
    ({'B1-own': .9, 'B2-r20': .4},
     {'K1': [0, 1, 2, 3, 4], 'B1': [0, 1, 2, 3, 4], 'B2': [0]}, 'B2-r20'),
    ({'B2-r20': .3}, {'B2': [2]}, 'B2-r20'),
    ({'B2-own': .4, 'B3-r20': .6, 'B4-own': .5},
     {'B2': [0], 'B3': [4], 'B4': [1]}, 'B3-r20'),
    ({'B2-own': .9, 'B3-own': .4}, {'B2': [], 'B3': [0]}, 'B3-own'),
    ({'B2-own': np.nextafter(.3, 0)}, {'B2': [0]}, None),
    ({'B1-r20': .9}, {}, None),
    ({'B3-own': .4, 'B2-r20': .4}, {'B2': [0], 'B3': [0]}, 'B2-r20'),
])
def test_comparator_is_largest_eligible_paper_entry_at_least_k1(entries, seeds, expected):
    result = dict(A_K1=.3, A_e=entries)
    assert verdict.improve_against(result, {'eligible_seeds': seeds}) == expected


@pytest.fixture
def verdict_inputs(tmp_path):
    perfect = np.array([[0, 0, 1, 1, 9]] * 3)
    constant = np.array([[0, 0, 0, 0, 9]] * 3)
    labels = np.array([0, 0, 1, 1, -1])
    fixed = {'K1': {0: dict(r20=perfect)},
             'B1': {0: dict(r20=constant, own=constant[:1])},
             'B2': {2: dict(r20=perfect, own=perfect[:1])}}
    seeds = {'K1': [0], 'B1': [0], 'B2': [2], 'B3': [], 'B4': [], 'B5': []}
    result = ct_score.paired_bootstrap(fixed, labels, seeds, n_resamples=4, workers=1)
    return dict(result=result, eligibility={'eligible_seeds': seeds},
                held_sha256='e' * 64, labels={'emotion': labels, 'genre': np.array([0, 0, -1, -1, -1])},
                code_commit='a' * 40,
                sources=dict(stage2a={'sha256': 'b' * 64}, rcol={'sha256': 'c' * 64},
                             held_outputs='fixture'))


def test_verdict_saves_stage2_statistics_labels_and_provenance(tmp_path, verdict_inputs):
    path = tmp_path / 'verdict_stage2.json'
    actual = verdict.write_verdict(path, **verdict_inputs)
    assert actual == json.loads(path.read_text())
    assert actual['stage'] == 2 and actual['outcome_id'] == 'S2-NO'
    assert actual['improve_against'] == 'B2-own'
    assert actual['sources'] == verdict_inputs['sources']
    assert actual['held_sha256'] == 'e' * 64 and actual['code_commit'] == 'a' * 40
    assert actual['label_counts'] == {
        'emotion': {'classes': {'0': 2, '1': 2}, 'missing': 1},
        'genre': {'classes': {'0': 2}, 'missing': 3}}
    assert actual['d'] == 0 and actual['A_K1'] == 1
    assert actual['bootstrap']['resamples'] == 4
    assert 'V2' in actual['reporting_requires']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['verdict_stage2.json']


def test_verdict_replacement_requires_reason_and_records_old_hash(tmp_path, verdict_inputs):
    path = tmp_path / 'verdict_stage2.json'
    original = verdict.write_verdict(path, **verdict_inputs)
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    assert verdict.write_verdict(path, **verdict_inputs) == original
    assert path.stat().st_mtime_ns == mtime
    verdict_inputs['result'] = dict(verdict_inputs['result'], d=.02, interval=[.001, .03])
    with pytest.raises(AssertionError, match='--replace'):
        verdict.write_verdict(path, **verdict_inputs)
    assert path.read_bytes() == before
    with pytest.raises(AssertionError, match='non-empty'):
        verdict.write_verdict(path, **verdict_inputs, replace='  ')
    reason = 'V3 rerun: saved order corrected, same seeds and code'
    actual = verdict.write_verdict(path, **verdict_inputs, replace=reason)
    assert actual['outcome_id'] == 'S2-GO'
    assert actual['replacement'] == dict(reason=reason, previous_sha256=hashlib.sha256(before).hexdigest())
    assert verdict.write_verdict(path, **verdict_inputs) == actual


def test_stage1_verdict_path_is_never_written(tmp_path, verdict_inputs):
    path = tmp_path / 'verdict_stage1.json'
    with pytest.raises(AssertionError, match='stage 2 verdict path'):
        verdict.write_verdict(path, **verdict_inputs)
    assert not path.exists()
    path.write_text('stage 1 sentinel\n')
    with pytest.raises(AssertionError, match='stage 2 verdict path'):
        verdict.write_verdict(path, **verdict_inputs, replace='V3 rerun')
    assert path.read_text() == 'stage 1 sentinel\n'


def test_scoring_and_bootstrap_are_stage1_functions_unchanged():
    assert verdict.prepare_held is ct_score.prepare_held
    assert verdict.paired_bootstrap is ct_score.paired_bootstrap


@pytest.mark.parametrize('frozen_matches', [True, False], ids=['frozen-match', 'frozen-mismatch'])
def test_run_saves_rcol_before_held_scoring_and_hashes_both_reports(
        tmp_path, full_paper_root, verdict_inputs, monkeypatch, frozen_matches):
    import sys
    from types import SimpleNamespace
    import ct_held

    root = tmp_path / 'full'
    stage1_dir = root / 'percept/paper/stage1'
    stage1_dir.mkdir(parents=True)
    for source in (full_paper_root / 'percept/paper/stage1').glob('*'):
        (stage1_dir / source.name).hardlink_to(source)
    rcol_path = root / 'percept/rcol.json'
    assert not rcol_path.exists()
    report = tmp_path / 'stage2a.json'
    report.write_text('{"status": "PASS"}\n')
    gate_calls = []

    def require_pass(path, device):
        assert Path(path) == report and device == 'cpu'
        gate_calls.append('pass')

    monkeypatch.setitem(sys.modules, 'ct2_stage2a', SimpleNamespace(require_pass=require_pass))
    frozen_list = ct_rcol.STAGE1.parents[2] / 'held/held_paintings.json'
    names, frozen_sha = ct_held.read_held_list(frozen_list)
    if not frozen_matches:
        names = [f'painting_{i:05}' for i in range(12281)]
    held_sha = ct_held.names_sha256(names)
    held_list = root / 'held/held_paintings.json'
    held_list.parent.mkdir()
    held_list.write_text(json.dumps(dict(names=names, sha256=held_sha)))
    monkeypatch.setattr(ct_held, 'held_names', lambda: names)
    labels = {aspect: np.arange(12281, dtype=np.int64) % 2
              for aspect in ('emotion', 'style', 'genre')}

    def painting_labels(ns):
        assert frozen_matches, 'labels read before checking the frozen held list'
        assert ns == names and rcol_path.exists()
        return labels

    monkeypatch.setattr(ct_held, 'painting_labels', painting_labels)
    fixed = {'fixture': 'all held partitions'}
    eligibility = {'eligible_seeds': dict(K1=list(range(5)), B1=list(range(5)),
                                         **{arm: [] for arm in ('B2', 'B3', 'B4', 'B5')})}
    calls = []

    # Orchestration seam: stand in for the costly full-held clustering/bootstrap;
    # the identity test above guards that production uses the unmodified functions.
    def prepare(ns, sha, eligible, widths, output_root):
        assert gate_calls == ['pass']
        assert rcol_path.exists()
        rcol = json.loads(rcol_path.read_text())
        assert len(rcol['runs']) == 20
        assert eligible == {arm: [] for arm in ('B2', 'B3', 'B4', 'B5')}
        assert set(widths) == {'B2', 'B3'}
        assert ns == names and sha == held_sha and Path(output_root) == root
        calls.append('prepare')
        return fixed, eligibility, {'held_source': 'fixture'}

    def bootstrap(parts, y, seeds, **kwargs):
        assert parts is fixed and y is labels['emotion']
        assert seeds == eligibility['eligible_seeds'] and kwargs == {'workers': 2}
        calls.append('bootstrap')
        return verdict_inputs['result']

    monkeypatch.setattr(verdict, 'prepare_held', prepare)
    monkeypatch.setattr(verdict, 'paired_bootstrap', bootstrap)
    stage1 = tmp_path / 'verdict_stage1.json'
    stage1.write_text('untouched stage 1\n')
    path = tmp_path / 'verdict_stage2.json'
    if not frozen_matches:
        assert held_sha != frozen_sha
        with pytest.raises(AssertionError, match='stage 1 frozen held list'):
            verdict.run(root, stage2a=report, device='cpu', workers=2, verdict_path=path)
        assert not path.exists() and calls == []
        return
    actual = verdict.run(root, stage2a=report, device='cpu', workers=2, verdict_path=path)
    assert calls == ['prepare', 'bootstrap']
    assert json.loads(path.read_text()) == actual
    assert actual['held_sha256'] == held_sha
    assert actual['sources']['held_list'] == dict(
        path=str(held_list.resolve()), held_sha256=held_sha,
        sha256=hashlib.sha256(held_list.read_bytes()).hexdigest())
    assert actual['sources']['stage1_held_list'] == dict(
        path=str(frozen_list.resolve()), held_sha256=frozen_sha,
        sha256=hashlib.sha256(frozen_list.read_bytes()).hexdigest())
    assert actual['sources']['stage2a']['sha256'] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert actual['sources']['rcol']['sha256'] == hashlib.sha256(
        rcol_path.read_bytes()).hexdigest()
    assert actual['sources']['held_outputs'] == {'held_source': 'fixture'}
    assert len(actual['code_commit']) == 40
    assert actual['label_counts']['emotion'] == {'classes': {'0': 6141, '1': 6140}, 'missing': 0}
    assert stage1.read_text() == 'untouched stage 1\n'


def test_run_requires_stage2a_before_any_input_or_rcol(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    def require_pass(path, device):
        raise AssertionError('stage 2a did not pass')

    monkeypatch.setitem(sys.modules, 'ct2_stage2a', SimpleNamespace(require_pass=require_pass))
    with pytest.raises(AssertionError, match='stage 2a did not pass'):
        verdict.run(tmp_path, stage2a=tmp_path / 'missing.json', device='cpu',
                    verdict_path=tmp_path / 'verdict_stage2.json')
    assert list(tmp_path.iterdir()) == []


def test_cli_replace_reason_reaches_verdict_policy(tmp_path, verdict_inputs, monkeypatch):
    path = tmp_path / 'verdict_stage2.json'

    def run(root, *, stage2a, device, workers, verdict_path, replace):
        assert Path(root) == tmp_path and device == 'cpu' and workers == 1
        return verdict.write_verdict(verdict_path, **verdict_inputs, replace=replace)

    monkeypatch.setattr(verdict, 'run', run)
    args = ['--root', str(tmp_path), '--verdict', str(path), '--device', 'cpu', '--workers', '1']
    verdict.main(args)
    original = path.read_bytes()
    verdict_inputs['result'] = dict(verdict_inputs['result'], d=.015, interval=[-.001, .03])
    with pytest.raises(AssertionError, match='--replace'):
        verdict.main(args)
    assert path.read_bytes() == original
    verdict.main(args + ['--replace', 'V3 rerun: corrected order'])
    actual = json.loads(path.read_text())
    assert actual['replacement']['reason'] == 'V3 rerun: corrected order'
    assert actual['outcome_id'] == 'S2-MM'


def test_cli_sample_mode_only_writes_rcol(tmp_path):
    stage1 = tmp_path / 'percept/paper/stage1'
    stage1.mkdir(parents=True)
    for path in ct_rcol.STAGE1.glob('*.npz'):
        (stage1 / path.name).symlink_to(path)
    with pytest.raises(AssertionError, match='full mode'):
        verdict.main(['--root', str(tmp_path), '--mode', 'sample'])
    verdict.main(['--root', str(tmp_path), '--mode', 'sample', '--rcol-only'])
    report = json.loads((tmp_path / 'percept/rcol.json').read_text())
    assert report['entries'] == json.loads(ct_rcol.OUTPUT.read_text())['entries']
    assert not (tmp_path / 'verdict_stage1.json').exists()
    assert not (tmp_path / 'verdict_stage2.json').exists()


@pytest.mark.parametrize('fault', ['mode', 'names_sha256'])
def test_full_rcol_refuses_sample_or_wrong_split_sidecars(tmp_path, full_paper_root, full_names, fault):
    stage1 = tmp_path / 'percept/paper/stage1'
    stage1.mkdir(parents=True)
    names = full_names
    config = dict(mode='full', names_sha256=names)
    config[fault] = 'sample' if fault == 'mode' else dict(names, fit='other hash')
    path = stage1 / 'clip_roberta__s0.npz'
    path.hardlink_to(full_paper_root / 'percept/paper/stage1/clip_roberta__s0.npz')
    path.with_suffix('.json').write_text(json.dumps(dict(config=config)))
    with pytest.raises(AssertionError, match=fault):
        verdict.read_stage1(path, 216107)


@pytest.mark.parametrize('arm', ct_score.ARM_ORDER)
@pytest.mark.parametrize('fault', ['mode', 'names_sha256', 'stage1-copy', None])
def test_verdict_refuses_wrong_training_provenance_before_scoring(tmp_path, monkeypatch, full_names, arm, fault):
    import ct2_stage2a
    import ct_held
    root = tmp_path / 'full'
    names = [f'held-{i:05}' for i in range(12281)]
    held_sha = ct_held.names_sha256(names)
    held_list = root / 'held/held_paintings.json'
    held_list.parent.mkdir(parents=True)
    held_list.write_text(json.dumps(dict(names=names, sha256=held_sha)))
    report = tmp_path / 'stage2a.json'
    report.write_text('{}')
    monkeypatch.setattr(ct2_stage2a, 'require_pass', lambda *a: None)
    monkeypatch.setattr(verdict, 'write_rcol', lambda *a, **k: None)
    rcol = root / 'percept/rcol.json'
    rcol.parent.mkdir(parents=True)
    rcol.write_text('{}')
    monkeypatch.setattr(ct_score, 'read_rcol', lambda p: ({}, {}, {}))
    monkeypatch.setattr(verdict, 'FROZEN_HELD_LIST', held_list)
    monkeypatch.setattr(ct_held, 'held_names', lambda: names)
    monkeypatch.setattr(ct_held, 'painting_labels', lambda ns: {
        a: np.arange(12281) % 2 for a in ('emotion', 'style', 'genre')})
    def prepare(*a, **k):
        raise RuntimeError('provenance accepted for scoring')

    monkeypatch.setattr(verdict, 'prepare_held', prepare)
    expected = full_names
    config = dict(mode='full', names_sha256=expected)
    if fault == 'stage1-copy':
        config = dict(seed=0, smoke=False)  # Stage 1 has neither ct2 field.
    elif fault is not None:
        config[fault] = 'sample' if fault == 'mode' else dict(expected, dev='other hash')
    path = root / 'held' / f'{ct_score.HELD_STEMS[arm]}__s0.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'held output')
    meta = dict(config=config)
    if arm == 'K1':
        # D2 held sidecars bind the training config through the weights hash.
        meta = dict(weights_sha256='weights')
        weight_meta = root / 'd2/weights/emotion__s0.json'
        weight_meta.parent.mkdir(parents=True)
        if fault == 'stage1-copy':
            meta['weights_sha256'] = 'stage1 weights'
            config = dict(mode='full', names_sha256=expected)
        weight_meta.write_text(json.dumps(dict(config=config, weights_sha256='weights')))
    path.with_suffix('.json').write_text(json.dumps(meta))
    error = RuntimeError if fault is None else AssertionError
    message = 'provenance accepted' if fault is None else 'mode|names_sha256|weights'
    with pytest.raises(error, match=message):
        verdict.run(root, stage2a=report, device='cpu', verdict_path=tmp_path / 'verdict_stage2.json')
