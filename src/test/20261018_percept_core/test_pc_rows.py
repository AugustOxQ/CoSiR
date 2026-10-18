"""Archived graph identity, head gate and seed aggregation checks."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_rows as rows


def test_packaged_graphs_load_unchanged_and_match_both_sources(tmp_path):
    s = rows.head.load_frozen()
    rows.build_graphs(s, tmp_path)
    for aspect in rows.head.ASPECTS:
        graph, manifest = rows.head.load_graph(s, tmp_path, aspect)
        source, _ = rows.head.load_graph(s, rows.ARCHIVE, aspect, ['R0o'])
        assert set(graph['rows']) == {'Ddef', 'R0o'}
        for side in ('fit', 'dev'):
            with np.load(rows.PILOT / f'{side}.npz', allow_pickle=False) as z:
                assert np.array_equal(graph[f'{side}_node_ids'], z['node_ids'])
                pairs, weights = graph['rows']['Ddef'][side]
                assert np.array_equal(pairs, z['I' if aspect == 'style' else 'D'])
                assert np.array_equal(weights, np.ones(len(pairs), dtype=np.float32))
            for actual, expected in zip(graph['rows']['R0o'][side], source['rows']['R0o'][side]):
                assert np.array_equal(actual, expected)
        assert manifest['identity']['sample_sha256'] == rows.head.fingerprint(s)
    before = (tmp_path / 'graphs/manifest.json').read_bytes()
    rows.build_graphs(s, tmp_path)
    assert (tmp_path / 'graphs/manifest.json').read_bytes() == before


@pytest.mark.parametrize('target,status,within', [
    (0.40609876614182405, 'PASS', True),
    (0.4060995, 'PASS', True),
    (0.40609876614182405 + 2e-6, 'FAIL', True),
    (0.40609876614182405 - 2e-6, 'FAIL', True),
    (0.40609876614182405 + 0.0019, 'FAIL', True),
    (0.40609876614182405 - 0.0025, 'FAIL', False),
    (0.4075, 'FAIL', True),
    (0.41, 'FAIL', False),
])
def test_gate_uses_both_sides_and_reports_strict_and_fallback_tolerances(target, status, within):
    scores = dict(img={'genre': {'mean': 0.3}}, cap={'genre': {'mean': 2 * target - 0.3}})
    verdict = rows.gate_result(scores)
    assert verdict['status'] == status
    assert verdict['pass_gate'] == (status == 'PASS')
    assert verdict['within_002'] == within
    assert verdict['target_ami'] == pytest.approx(target)
    assert verdict['difference'] == pytest.approx(target - 0.40609876614182405)
    assert rows.gate_result(scores, smoke=True)['pass_gate'] is False


def fake_score(row, seed, img, cap, input_key='default', complete=True):
    config = dict(aspect='emotion', row=row, seed=seed, input_img=input_key, input_cap=input_key)
    ami = {side: {label: {'mean': value, 'sd': 0.99} for label in rows.head.LABELS}
           for side, value in (('img', img), ('cap', cap))}
    return dict(config=config, ami=ami, target_ami=(img + cap) / 2, complete=complete)


def test_aggregation_keeps_rows_separate_and_uses_population_sd_of_seed_means():
    outputs = {
        'r6s0': fake_score('Ddef', 0, .1, .5),
        'r6s1': fake_score('Ddef', 1, .2, .3),
        'r6s2': fake_score('Ddef', 2, .3, .1),
        'r10s0': fake_score('R0o', 0, .9, .7),
        'r13s0': fake_score('R0o', 0, .6, .4, 'emotion_focus'),
        'unfinished': fake_score('R0o', 1, 1., 1., complete=False),
    }
    result = rows.aggregate(outputs)
    r6 = result['row6']['emotion']
    assert r6['seeds'] == [0, 1, 2] and r6['complete']
    assert r6['ami']['img']['emotion'] == pytest.approx(dict(mean=.2, sd=.0816496580927726))
    assert r6['ami']['cap']['emotion'] == pytest.approx(dict(mean=.3, sd=.1632993161855452))
    assert r6['ami']['mean']['emotion'] == pytest.approx(dict(mean=.25, sd=.0408248290463863))
    assert r6['target_ami'] == pytest.approx(dict(mean=.25, sd=.0408248290463863))
    assert result['row10']['emotion']['target_ami'] == dict(mean=pytest.approx(.8), sd=None)
    assert result['row13']['emotion']['target_ami'] == dict(mean=pytest.approx(.5), sd=None)
    assert result['row10']['emotion']['seeds'] == [0]
    assert not result['row10']['emotion']['complete']


def test_run_sets_cover_the_24_ablation_heads_and_one_archived_gate():
    r6, r10, r13 = (rows.jobs(name) for name in ('row6', 'row10', 'row13'))
    assert len(r6) == 9 and len(r10) == 6 and len(r13) == 9
    assert set(r6) == {(a, 'Ddef', 'default', seed)
                       for a in ('emotion', 'style', 'genre') for seed in (0, 1, 2)}
    assert set(r10) == {(a, 'R0o', 'default', seed)
                        for a in ('emotion', 'style') for seed in (0, 1, 2)}
    assert set(r13) == {(a, 'R0o', key, seed) for a, key in
                        (('emotion', 'emotion_focus'), ('style', 'style_word'), ('genre', 'genre_word'))
                        for seed in (0, 1, 2)}
    assert rows.jobs('gate') == [('genre', 'R0', 'default', 0)]
    assert rows.jobs('row13', ['style'], [2]) == [('style', 'R0o', 'style_word', 2)]


def test_full_head_rows_require_a_passing_gate_before_loading_inputs(tmp_path):
    with pytest.raises(AssertionError, match='gate'):
        rows.train(None, tmp_path, 'row6', ['emotion'], [0])
    assert not (tmp_path / 'heads').exists()


def test_gate_check_requires_a_local_rerun_instead_of_the_archived_output(tmp_path):
    with pytest.raises(AssertionError, match='rerun'):
        rows.gate_check(None, tmp_path)
    assert not (tmp_path / 'gate.json').exists()


def test_train_rechecks_head_without_writing_shared_gate(tmp_path, monkeypatch):
    marker = tmp_path / 'gate.json'
    marker.write_text(json.dumps(dict(pass_gate=True)))
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    folder = tmp_path / 'heads'
    folder.mkdir()
    path = folder / 'genre__R0__s0.npz'
    np.savez(path, img=np.ones((1, 1)), cap=np.ones((1, 1)))
    verified, trained = [], []

    def verify(s, base, output, smoke):
        verified.append(output)
        return dict(config={}, output_sha256='verified')

    def no_write(*args):
        pytest.fail('training must not rewrite gate.json')

    monkeypatch.setattr(rows, 'verify_head', verify)
    monkeypatch.setattr(rows.head, 'label_vectors', lambda *args: {})
    monkeypatch.setattr(rows.head, 'ami_scores', lambda *args: {'genre': {'mean': rows.GATE_TARGET}})
    monkeypatch.setattr(rows.head, 'atomic_json', no_write)
    monkeypatch.setattr(rows.head, 'load_graph', lambda *args: ({}, {}))
    monkeypatch.setattr(rows.head, 'load_inputs', lambda *args: {})
    monkeypatch.setattr(rows.head, 'run', lambda *args, **kwargs: trained.append(args))
    rows.train(dict(subset_pid=np.array([0])), tmp_path, 'row6', ['emotion'], [0])
    assert verified == [path] and len(trained) == 1
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before


def test_score_uses_the_frozen_scorer_on_sealed_heads_and_reports_missing_runs(tmp_path):
    s = rows.head.load_frozen()
    rows.build_graphs(s, tmp_path)
    _, manifest = rows.head.load_graph(s, tmp_path, 'emotion', ['Ddef'])
    config = rows.head.make_config(s, manifest, 'emotion', 'Ddef', 0, False)
    folder = tmp_path / 'heads'
    folder.mkdir()
    path = folder / 'emotion__Ddef__s0.npz'
    x = np.zeros((5000, 2048), dtype=np.float16)
    x[np.arange(5000), np.arange(5000) % 20] = 1
    np.savez(path, img=x, cap=x)
    rows.head.atomic_json(path.with_suffix('.json'), dict(config=config, best_epoch=1,
        epochs_run=1, best_dev_loss=.5, history=[{}], output_sha256=rows.head.sha_file(path)))
    (folder / 'style__Ddef__s0.pending.npz').write_bytes(b'unsealed')
    rows.head.atomic_json(folder / 'genre__Ddef__s0.json', dict(config={}))
    expected = rows.head.ami_scores(x, rows.head.label_vectors(s, s['subset_pid']))
    result = rows.score(s, tmp_path)
    output = result['outputs']['emotion__Ddef__s0']
    assert output['complete'] and output['ami'] == dict(img=expected, cap=expected)
    assert output['target_ami'] == expected['emotion']['mean']
    assert len(result['outputs']) == 1 and len(result['missing_outputs']) == 24
    assert result['aggregates']['row6']['emotion']['seeds'] == [0]
    assert 'pending' in (tmp_path / 'pc_rows_tables.md').read_text()
    assert json.loads((tmp_path / 'pc_rows_results.json').read_text()) == result
    assert rows.score(s, tmp_path) == result
    with path.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(AssertionError, match='checksum'):
        rows.score(s, tmp_path)
