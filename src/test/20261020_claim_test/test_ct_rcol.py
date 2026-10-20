"""Rule §4 cuts, collapse boundaries and the saved Stage 1 judgment."""
import numpy as np
import pytest

import ct_rcol
import pc_fixes


@pytest.mark.parametrize('norms,expected', [
    (np.r_[np.full(20, 1.), np.full(80, 10.)][::-1],
     np.r_[np.zeros(80, dtype=bool), np.ones(20, dtype=bool)]),
    (np.arange(100, dtype=np.float64), np.ones(100, dtype=bool)),
    (np.ones(100), np.ones(100, dtype=bool)),
    (np.r_[0., np.cumsum(np.r_[np.ones(97), 2., 0.])], np.ones(100, dtype=bool)),
    (np.r_[np.zeros(10), np.full(40, 10.), np.full(50, 20.)][::-1],
     np.r_[np.zeros(90, dtype=bool), np.ones(10, dtype=bool)]),
])
def test_strict_mask_follows_algorithm_one_and_core(norms, expected):
    actual = ct_rcol.strict_mask(norms)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, pc_fixes.kept_centers(norms, gamma=2))
    assert actual.dtype == np.bool_


@pytest.mark.parametrize('counts,indices', [
    (np.arange(100, dtype=np.int64), np.arange(60, 100)),
    (np.ones(100, dtype=np.int64), np.arange(40)),
    (np.r_[np.full(20, 2), np.zeros(60, dtype=int), np.full(20, 3)],
     np.r_[np.arange(20), np.arange(80, 100)]),
])
def test_k40_mask_keeps_most_used_with_lower_index_ties(counts, indices):
    actual = ct_rcol.k40_mask(counts)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(np.flatnonzero(actual), indices)
    np.testing.assert_array_equal(actual, pc_fixes.top_used(counts, 40))


@pytest.mark.parametrize('cut,count,topics,share,collapsed', [
    ('strict', 9, [0, 1, 2, 3], .25, True),
    ('strict', 10, [0, 0, 1, 2], .5, False),
    ('strict', 90, [0, 0, 1, 2], .5, False),
    ('strict', 91, [0, 1, 2, 3], .25, True),
    ('strict', 40, [0, 0, 0, 1], .75, True),
    ('k40', 40, [0, 0, 1, 2], .5, False),
    ('k40', 40, [0, 0, 0, 1], .75, True),
    ('k40', 40, [0, 1, 2, 3], .25, False),
])
def test_judge_uses_kept_count_and_fit_pair_share(cut, count, topics, share, collapsed):
    # Non-contiguous original centers; hard-topic IDs index the kept subset.
    kept = np.zeros(100, dtype=bool)
    kept[np.arange(100)[::-1][:count]] = True
    actual = ct_rcol.judge(cut, kept, np.array(topics, dtype=np.int64))
    assert actual == dict(kept_count=count, share=share, collapsed=collapsed)


def test_eligible_entries_use_each_versions_cut_and_sort_surviving_seeds():
    runs = []
    for version in ('qwen_qwen', 'clip_roberta'):
        for seed in (4, 2, 0, 3, 1):
            for cut in ('k40', 'strict'):
                survivors = {('clip_roberta', 'strict'): {3},
                             ('clip_roberta', 'k40'): {0, 2, 4},
                             ('qwen_qwen', 'strict'): set(),
                             ('qwen_qwen', 'k40'): set(range(5))}
                runs.append(dict(version=version, seed=seed, cut=cut,
                                 collapsed=seed not in survivors[version, cut]))
    assert ct_rcol.eligible_entries(runs) == {
        'B2': dict(eligible=True, seeds=[3]),
        'B3': dict(eligible=False, seeds=[]),
        'B4': dict(eligible=True, seeds=[0, 2, 4]),
        'B5': dict(eligible=True, seeds=[0, 1, 2, 3, 4]),
    }


@pytest.fixture(scope='session')
def paper_artifacts(tmp_path_factory):
    folder = tmp_path_factory.mktemp('paper_artifacts') / 'stage1'
    folder.mkdir(parents=True)
    for version in ('clip_roberta', 'qwen_qwen'):
        for seed in range(5):
            norms = np.r_[np.ones(20), np.full(80, 10.)].astype(np.float32)
            # Qwen has no norm gap; strict retains all 100 and collapses.
            if version == 'qwen_qwen':
                norms = np.arange(1, 101, dtype=np.float32)
            centers = np.eye(100, 128, dtype=np.float32) * norms[:, None]
            full_topics = np.arange(67327, dtype=np.int64) % (20 if seed == 2 else 40)
            if seed == 4:
                full_topics[:] = 3
            counts = np.bincount(full_topics, minlength=100)
            kept_strict = np.arange(100) < (20 if version == 'clip_roberta' else 100)
            # Orthogonal excluded centers are equidistant from the 20 low-norm
            # centers: their pairs go to the first kept center on a tie.
            strict_topics = (np.where(full_topics < 20, full_topics, 0)
                             if version == 'clip_roberta' else full_topics)
            np.savez_compressed(folder / f'{version}__s{seed}.npz',
                                centers=centers, fit_latent=centers[full_topics],
                                center_norms=norms, fit_counts=counts,
                                kept_strict=kept_strict, kept_k40=np.arange(100) < 40,
                                fit_topics_strict=strict_topics, fit_topics_k40=full_topics)
    return folder


@pytest.fixture
def paper_stage1(tmp_path, paper_artifacts):
    import shutil

    return shutil.copytree(paper_artifacts, tmp_path / 'percept' / 'paper' / 'stage1')


def test_cli_writes_all_runs_eligibility_and_input_array_hashes(paper_stage1, tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import sys

    output = tmp_path / 'outputs' / 'percept' / 'rcol_stage1.json'
    completed = subprocess.run(
        [sys.executable, str(Path(ct_rcol.__file__)), '--stage1-dir', str(paper_stage1),
         '--output', str(output)], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    assert len(report['runs']) == 20
    assert {(r['version'], r['seed'], r['cut']) for r in report['runs']} == {
        (v, s, c) for v in ('clip_roberta', 'qwen_qwen')
        for s in range(5) for c in ('strict', 'k40')}
    assert report['entries'] == {
        'B2': dict(eligible=True, seeds=[2]), 'B3': dict(eligible=False, seeds=[]),
        'B4': dict(eligible=True, seeds=[0, 1, 2, 3]),
        'B5': dict(eligible=True, seeds=[0, 1, 2, 3]),
    }
    for run in report['runs']:
        with np.load(paper_stage1 / f"{run['version']}__s{run['seed']}.npz") as z:
            assert run['array_sha256'] == {
                key: hashlib.sha256(z[key].tobytes(order='C')).hexdigest()
                for key in ('centers', 'fit_latent', 'center_norms', 'fit_counts', f"kept_{run['cut']}",
                            f"fit_topics_{run['cut']}")}
        if run['version'] == 'clip_roberta' and run['seed'] == 2 and run['cut'] == 'strict':
            assert run['kept_count'] == 20
            assert run['share'] == 3367 / 67327
            assert run['collapsed'] is False
    assert sorted(p.name for p in output.parent.iterdir()) == ['rcol_stage1.json']


@pytest.mark.parametrize('key,value', [
    ('center_norms', None),
    ('fit_topics_k40', None),
    ('center_norms', np.ones(99, dtype=np.float32)),
    ('center_norms', np.r_[np.nan, np.ones(99)]),
    ('center_norms', np.r_[-1., np.ones(99)]),
    ('fit_counts', np.ones(99, dtype=np.int64)),
    ('fit_counts', np.ones(100, dtype=np.int64)),
    ('fit_counts', np.full(100, 673.27)),
    ('kept_strict', np.ones(100, dtype=bool)),
    ('kept_k40', np.arange(100) >= 60),
    ('kept_k40', np.ones(99, dtype=bool)),
    ('kept_strict', np.ones(100, dtype=np.int64)),
    ('fit_topics_strict', np.zeros(67326, dtype=np.int64)),
    ('fit_topics_k40', np.zeros((67327, 1), dtype=np.int64)),
    ('fit_topics_k40', np.zeros(67327, dtype=np.float32)),
    ('fit_topics_strict', np.full(67327, 20, dtype=np.int64)),
    ('fit_topics_k40', np.full(67327, -1, dtype=np.int64)),
])
def test_reader_asserts_artifact_keys_shapes_and_consistency(paper_stage1, key, value):
    path = paper_stage1 / 'clip_roberta__s0.npz'
    with np.load(path) as z:
        arrays = dict(z)
    if value is None:
        arrays.pop(key)
    else:
        arrays[key] = value
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError):
        ct_rcol.read_stage1(path)


def test_writer_creates_complete_json_atomically(paper_stage1, tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path

    output = tmp_path / 'rcol_stage1.json'
    replace = os.replace
    observed = []

    def observe_replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert destination == output
        assert source != output and source.parent == output.parent
        assert not output.exists()
        report = json.loads(source.read_text())
        assert len(report['runs']) == 20 and set(report['entries']) == {'B2', 'B3', 'B4', 'B5'}
        observed.append(report)
        replace(source, destination)

    monkeypatch.setattr(os, 'replace', observe_replace)
    report = ct_rcol.write_stage1(paper_stage1, output)
    assert observed == [report]
    assert json.loads(output.read_text()) == report


def test_writer_records_amsterdam_local_minute(paper_stage1, tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import json

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz):
            assert str(tz) == 'Europe/Amsterdam'
            return datetime(2026, 10, 20, 0, 35, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(ct_rcol, 'datetime', FixedDatetime, raising=False)
    output = tmp_path / 'rcol_stage1.json'
    report = ct_rcol.write_stage1(paper_stage1, output)
    assert report['written'] == '2026-10-20 02:35'
    assert json.loads(output.read_text())['written'] == report['written']


def test_writer_leaves_identical_content_and_timestamp_untouched(paper_stage1, tmp_path):
    import json

    output = tmp_path / 'rcol_stage1.json'
    report = ct_rcol.write_stage1(paper_stage1, output)
    report['written'] = '2000-01-01 00:00'
    original = json.dumps(report, sort_keys=True) + '\n'
    output.write_text(original)
    modified = output.stat().st_mtime_ns
    result = ct_rcol.write_stage1(paper_stage1, output)
    assert result == report
    assert output.read_text() == original
    assert output.stat().st_mtime_ns == modified


def test_writer_refuses_changed_content_and_preserves_existing_file(paper_stage1, tmp_path):
    import json

    output = tmp_path / 'rcol_stage1.json'
    report = ct_rcol.write_stage1(paper_stage1, output)
    report['runs'][0]['share'] = .123
    original = json.dumps(report) + '\n'
    output.write_text(original)
    modified = output.stat().st_mtime_ns
    with pytest.raises(AssertionError, match='refusing to overwrite'):
        ct_rcol.write_stage1(paper_stage1, output)
    assert output.read_text() == original
    assert output.stat().st_mtime_ns == modified


def test_default_output_is_in_main_checkout_percept_folder():
    from pathlib import Path

    assert ct_rcol.OUTPUT == Path('/project/CoSiR/src/test/20261020_claim_test/outputs/percept/rcol_stage1.json')


def test_invalid_artifact_cannot_replace_previous_report(paper_stage1, tmp_path):
    output = tmp_path / 'rcol_stage1.json'
    output.write_text('previous complete report\n')
    np.savez_compressed(paper_stage1 / 'qwen_qwen__s4.npz', center_norms=np.ones(100))
    with pytest.raises(AssertionError):
        ct_rcol.write_stage1(paper_stage1, output)
    assert output.read_text() == 'previous complete report\n'


def test_reader_recomputes_consistent_student_t_topics_and_counts(paper_stage1):
    runs = ct_rcol.read_stage1(paper_stage1 / 'clip_roberta__s0.npz')
    for run in runs:
        assert run['fit_counts_l1_difference'] == 0
        assert run['fit_topics_disagreements'] == 0
    # Half the fit pairs were on removed centers and move to local topic zero.
    assert runs[0]['share'] == 35344 / 67327
    assert runs[0]['collapsed'] is True


@pytest.mark.parametrize('key,problem', [
    ('centers', 'missing'), ('fit_latent', 'missing'),
    ('centers', 'shape'), ('fit_latent', 'shape'),
    ('centers', 'dtype'), ('fit_latent', 'dtype'),
    ('centers', 'nonfinite'), ('fit_latent', 'nonfinite'),
    ('center_norms', 'mismatch'),
])
def test_reader_validates_centers_and_fit_latent(paper_stage1, key, problem):
    path = paper_stage1 / 'clip_roberta__s0.npz'
    with np.load(path) as z:
        arrays = dict(z)
    if problem == 'missing':
        arrays.pop(key)
    elif problem == 'shape':
        arrays[key] = arrays[key][:, :-1]
    elif problem == 'dtype':
        arrays[key] = arrays[key].astype(np.float64)
    elif problem == 'nonfinite':
        arrays[key][0, 0] = np.nan
    else:
        arrays[key][0] += .01
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError):
        ct_rcol.read_stage1(path)


@pytest.mark.parametrize('changed,l1,disagreements', [(3, 6, [2, 3]), (4, 8, None)])
def test_reader_limits_and_records_count_l1_disagreement(paper_stage1, changed, l1, disagreements):
    path = paper_stage1 / 'clip_roberta__s0.npz'
    with np.load(path) as z:
        arrays = dict(z)
    arrays['fit_latent'][:changed] = arrays['centers'][39]
    np.savez_compressed(path, **arrays)
    if disagreements is None:
        with pytest.raises(AssertionError, match='count L1'):
            ct_rcol.read_stage1(path)
    else:
        runs = ct_rcol.read_stage1(path)
        assert [r['fit_counts_l1_difference'] for r in runs] == [l1, l1]
        assert [r['fit_topics_disagreements'] for r in runs] == disagreements


@pytest.mark.parametrize('changed', [6, 7])
def test_reader_limits_and_records_post_cut_topic_disagreement(paper_stage1, changed):
    path = paper_stage1 / 'clip_roberta__s0.npz'
    with np.load(path) as z:
        arrays = dict(z)
    rows = np.flatnonzero(arrays['fit_topics_strict'] == 0)[:changed]
    arrays['fit_topics_strict'][rows] = 1
    np.savez_compressed(path, **arrays)
    if changed == 7:
        with pytest.raises(AssertionError, match='topic disagreements'):
            ct_rcol.read_stage1(path)
    else:
        runs = ct_rcol.read_stage1(path)
        assert [r['fit_topics_disagreements'] for r in runs] == [6, 0]


@pytest.mark.parametrize('saved_topic,valid', [(5, True), (10, False)])
def test_reader_requires_local_topics_for_noncontiguous_kept_centers(paper_stage1, saved_topic, valid):
    path = paper_stage1 / 'clip_roberta__s0.npz'
    with np.load(path) as z:
        arrays = dict(z)
    # Original center 10 is local center 5 in the strict cut. Both are in
    # range, so a mistakenly saved full argmax must fail the count check.
    arrays['kept_strict'] = (np.arange(100) < 40) & (np.arange(100) % 2 == 0)
    arrays['center_norms'] = np.where(arrays['kept_strict'], 1., 10.).astype(np.float32)
    arrays['centers'] = np.eye(100, 128, dtype=np.float32) * arrays['center_norms'][:, None]
    arrays['fit_latent'][:] = arrays['centers'][10]
    arrays['fit_counts'][:] = 0
    arrays['fit_counts'][10] = 67327
    arrays['fit_topics_strict'][:] = saved_topic
    arrays['fit_topics_k40'][:] = 10
    np.savez_compressed(path, **arrays)
    if valid:
        runs = ct_rcol.read_stage1(path)
        assert [r['fit_topics_disagreements'] for r in runs] == [0, 0]
    else:
        with pytest.raises(AssertionError, match='post-cut counts'):
            ct_rcol.read_stage1(path)


@pytest.mark.parametrize('version,cut', [('clip_roberta', 'strict'), ('qwen_qwen', 'strict'),
                                       ('clip_roberta', 'k40')])
def test_reader_rejects_loss_of_a_kept_centers_pairs_beyond_tolerance(paper_stage1, version, cut):
    path = paper_stage1 / f'{version}__s2.npz'
    with np.load(path) as z:
        arrays = dict(z)
    # Counts and topics come from separate GPU passes, so a near-tie may move a
    # pair (0.01% allowed); 200 moved pairs are far beyond that.
    topics = arrays[f'fit_topics_{cut}']
    topics[np.flatnonzero(topics != 1)[:200]] = 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError, match='post-cut counts'):
        ct_rcol.read_stage1(path)
