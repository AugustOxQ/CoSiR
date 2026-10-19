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


@pytest.fixture
def paper_stage1(tmp_path):
    folder = tmp_path / 'percept' / 'paper' / 'stage1'
    folder.mkdir(parents=True)
    for version in ('clip_roberta', 'qwen_qwen'):
        for seed in range(5):
            norms = np.r_[np.ones(20), np.full(80, 10.)].astype(np.float32)
            # Qwen has no norm gap; strict retains all 100 and collapses.
            if version == 'qwen_qwen':
                norms = np.arange(100, dtype=np.float32)
            counts = np.r_[np.full(40, 1683), np.zeros(60, dtype=int)]
            counts[0] += 7  # 67,327 fit pairs, before either cut.
            kept_strict = np.arange(100) < (20 if version == 'clip_roberta' else 100)
            strict_topics = np.arange(67327, dtype=np.int64) % 20
            k40_topics = np.arange(67327, dtype=np.int64) % 40
            if seed != 2:
                strict_topics[:] = 3
            if seed == 4:
                k40_topics[:] = 7
            np.savez_compressed(folder / f'{version}__s{seed}.npz',
                                center_norms=norms, fit_counts=counts,
                                kept_strict=kept_strict, kept_k40=np.arange(100) < 40,
                                fit_topics_strict=strict_topics, fit_topics_k40=k40_topics)
    return folder


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
                for key in ('center_norms', 'fit_counts', f"kept_{run['cut']}",
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


def test_writer_replaces_complete_json_atomically(paper_stage1, tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path

    output = tmp_path / 'rcol_stage1.json'
    old = '{"previous": true}\n'
    output.write_text(old)
    replace = os.replace
    observed = []

    def observe_replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert destination == output
        assert source != output and source.parent == output.parent
        assert output.read_text() == old
        report = json.loads(source.read_text())
        assert len(report['runs']) == 20 and set(report['entries']) == {'B2', 'B3', 'B4', 'B5'}
        observed.append(report)
        replace(source, destination)

    monkeypatch.setattr(os, 'replace', observe_replace)
    report = ct_rcol.write_stage1(paper_stage1, output)
    assert observed == [report]
    assert json.loads(output.read_text()) == report


def test_invalid_artifact_cannot_replace_previous_report(paper_stage1, tmp_path):
    output = tmp_path / 'rcol_stage1.json'
    output.write_text('previous complete report\n')
    np.savez_compressed(paper_stage1 / 'qwen_qwen__s4.npz', center_norms=np.ones(100))
    with pytest.raises(AssertionError):
        ct_rcol.write_stage1(paper_stage1, output)
    assert output.read_text() == 'previous complete report\n'
