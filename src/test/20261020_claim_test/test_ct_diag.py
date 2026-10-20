"""Ticket 08 graph recipes, fixed partitions, CPU inference and post-verdict gate."""
import importlib.util
import json
from pathlib import Path

import igraph as ig
import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

import ct_diag as diag
import ct_score as sc


def test_verdict_gate_precedes_every_input_read(tmp_path, monkeypatch):
    monkeypatch.setattr(diag.ct_held, 'read_held_list', lambda *args: pytest.fail('read before verdict'))
    with pytest.raises(AssertionError, match='verdict_stage1.json first'):
        diag.run(tmp_path / 'verdict_stage1.json')


def test_union_graph_matches_probe_and_is_simple():
    rng = np.random.default_rng(31)
    x = diag.normalize(rng.normal(size=(42, 8)))
    graph = diag.union_knn(x, k=20, block=7)
    expected = set()
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    for i, row in enumerate(np.argpartition(-similarity, 19, axis=1)[:, :20]):
        expected.update(tuple(sorted((i, int(j)))) for j in row)
    assert set(graph.get_edgelist()) == expected
    assert graph.vcount() == 42 and graph.is_simple() and not graph.is_directed()
    assert 'weight' not in graph.es.attributes()
    # There must be one-sided choices: union, rather than mutual, is binding.
    assert any(i not in np.argpartition(-similarity[j], 19)[:20] for i, j in expected)


def test_leiden_uses_modularity_seeds_0_to_2(monkeypatch):
    seen = []
    def find(graph, partition, **kwargs):
        assert partition is diag.leidenalg.ModularityVertexPartition
        assert kwargs == dict(seed=len(seen))
        seen.append(kwargs['seed'])
        return type('Partition', (), dict(membership=[0, 1, 1]))()
    monkeypatch.setattr(diag.leidenalg, 'find_partition', find)
    np.testing.assert_array_equal(diag.leiden_partitions(ig.Graph.Ring(3)), [[0, 1, 1]] * 3)
    assert seen == [0, 1, 2]


def test_mutual_graph_repairs_degree_and_connectivity_on_cpu(monkeypatch):
    # Two dense clusters plus an isolated point, requiring both repair steps.
    import torch
    torch.set_num_threads(8)
    rng = np.random.default_rng(2)
    x = np.r_[np.tile([1., 0., 0.], (12, 1)), np.tile([-1., 0., 0.], (12, 1)), [[0., 1., 0.]]]
    x += rng.normal(scale=.001, size=x.shape)
    helpers = diag.september_graph_helpers()
    seen = []
    class Observe:
        def __getattr__(self, name):
            function = getattr(helpers, name)
            def call(*args, **kwargs):
                assert kwargs['device'] == 'cpu' and kwargs['use_half'] is False
                seen.append(name)
                return function(*args, **kwargs)
            return call
    monkeypatch.setattr(diag, 'september_graph_helpers', lambda: Observe())
    graph, stats = diag.fit_graph(x, k=3)
    assert graph.is_connected() and graph.is_simple() and min(graph.degree()) >= 1
    assert stats['degree']['num_isolated'] >= 1
    assert stats['connectivity']['bridges_added'] >= 1
    assert seen == ['mutual_knn', 'ensure_min_degree', 'ensure_connected']


def test_vote_transfer_matches_september_including_nearest_ties():
    rng = np.random.default_rng(45)
    train = diag.normalize(rng.normal(size=(27, 5)))
    held = diag.normalize(rng.normal(size=(8, 5)))
    communities = np.asarray([np.arange(27), np.arange(27) % 3, np.arange(27) % 2])
    path = diag.SEPT / 'src/test/20260923_artelingo_buddy_analysis/run_heldout_label_transfer_pilot.py'
    spec = importlib.util.spec_from_file_location('ct_diag_vote_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    actual, fallback = diag.vote_transfer(train, communities, held, k=20)
    for i, community in enumerate(communities):
        expected = reference.assign_to_train_communities(train, community, held, k=20)
        np.testing.assert_array_equal(actual[i], expected)
        assert fallback[i] == reference.assign_to_train_communities.last_fallback_count
    assert fallback[0] == len(held)


def test_all_metrics_mask_missing_labels_and_random_preserves_sizes():
    p = np.array([[0, 0, 1, 1, 1, 2], [1, 1, 0, 0, 0, 3]])
    y = np.array([0, 0, 1, 1, 1, -1])
    records = diag.partition_metrics(p, y)
    rng = np.random.default_rng(0)
    for partition, record in zip(p, records):
        random = rng.permutation(partition)
        assert record['count'] == 3
        assert set(record['scores']) == {'ami', 'v', 'ari', 'fmi'}
        for key, metric in diag.METRICS.items():
            assert record['scores'][key] == pytest.approx(1)
            assert record['random'][key] == pytest.approx(metric(y[:-1], random[:-1]))
        np.testing.assert_array_equal(np.sort(random), np.sort(partition))


def test_diagnostic_uses_fixed_k20_and_rounded_mean_count(monkeypatch):
    rng = np.random.default_rng(5)
    x = diag.normalize(rng.normal(size=(40, 5))) * np.float32(.9995)
    y = np.arange(40) % 2
    fixed = dict(r20=np.array([y] * 3))
    ps = np.array([np.arange(40) % count for count in (2, 3, 3)])
    monkeypatch.setattr(sc, 'output_partitions', lambda *args: pytest.fail('recomputed fixed partitions'))
    monkeypatch.setattr(diag, 'leiden_partitions', lambda graph: ps)
    seen = []
    def partitions(value, k):
        np.testing.assert_array_equal(value, x)  # no K1 re-normalization
        seen.append(k)
        return np.array([y] * 3)
    monkeypatch.setattr(sc, 'partitions', partitions)
    record, actual, returned = diag.diagnose_vectors(x, y, 'k1', fixed=fixed)
    assert seen == [3] and record['rounded_mean_count'] == 3
    assert record['leiden_counts'] == [2, 3, 3]
    assert returned is fixed
    np.testing.assert_array_equal(actual, ps)
    assert len(record['held']['r20']) == len(record['held']['leiden3a']) == 3


def test_leiden_margin_uses_03_bootstrap_eligibility_and_three_readings(monkeypatch):
    perfect = np.array([[0, 0, 1, 1]] * 3)
    constant = np.zeros((3, 4), dtype=np.int64)
    partitions = {'K1': {0: perfect}, 'B1': {0: constant}, 'B2': {0: perfect}}
    eligible = {'K1': [0], 'B1': [0], 'B2': []}
    called = []
    original = sc.paired_bootstrap
    def observe(*args, **kwargs):
        called.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(sc, 'paired_bootstrap', observe)
    result = diag.leiden_margin(partitions, perfect[0], eligible, n_resamples=4, workers=1)
    assert called == [dict(n_resamples=4, workers=1, comparator_readings=('r20',))]
    assert result['d'] == pytest.approx(1) and result['A_e'] == {'B1-r20': 0.}
    assert len(result['bootstrap_d']) == 4
    eligible['B2'] = [0]
    assert diag.leiden_margin(partitions, perfect[0], eligible, n_resamples=4, workers=1)['d'] == 0
    with pytest.raises(AssertionError):
        original({arm: {s: dict(r20=p) for s, p in runs.items()} for arm, runs in partitions.items()},
                 perfect[0], eligible, n_resamples=4, workers=1)


def test_saved_partitions_bind_verdict_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, 'HELD_ROWS', 4)
    path = tmp_path / 'K1__s0.npz'
    p = np.zeros((3, 4), dtype=np.int64)
    np.savez_compressed(path, r20=p)
    sha = sc._sha_file(path)
    source = dict(partitions_sha256=sha, output_sha256='output')
    meta = dict(partitions_sha256=sha, identity=dict(held_sha256='names', source_sha256='output'))
    path.with_suffix('.json').write_text(json.dumps(meta))
    np.testing.assert_array_equal(diag.saved_partitions(path, source, 'names')['r20'], p)
    with pytest.raises(AssertionError):
        diag.saved_partitions(path, source, 'other-names')
    source['output_sha256'] = 'other-output'
    with pytest.raises(AssertionError):
        diag.saved_partitions(path, source, 'names')


def test_paper_rows_include_collapsed_and_skipped_runs(tmp_path):
    runs = [dict(version='clip_roberta', seed=0, cut='strict', kept_count=1, share=1., collapsed=True),
            dict(version='qwen_qwen', seed=2, cut='k40', kept_count=40, share=.6, collapsed=True)]
    verdict = dict(sources=dict(rcol=dict(record=dict(runs=runs)),
                               held_outputs={'B2': {'0': dict(status='rcol_collapsed_no_output')},
                                             'B5': {'2': dict(status='finite')}}),
                   eligibility=dict(eligible_seeds={'B2': [], 'B5': []}))
    path = tmp_path / 'percept/paper/stage2/k40/P4__s2.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(K=40, dev_auc=.7, config=dict(settings='paper', cut='k40', arm='P4', seed=2))))
    rows = diag.paper_runs(verdict, tmp_path)
    assert rows[0]['stage2_dev_auc'] is None and rows[0]['collapsed']
    assert rows[1]['stage2_dev_auc'] == .7 and rows[1]['collapsed']
    assert rows[0]['stage2_status'] == 'not_trained_kept_count_under_2'


def test_target_loader_checks_names_and_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, 'HELD_ROWS', 3)
    names = ['a', 'b', 'c']
    path = tmp_path / 'emotion_focus_held'
    path.mkdir()
    (path / 'meta.json').write_text(json.dumps(dict(held_sha256='hash',
                                                   instruction=diag.QWEN_INSTRUCTIONS['emotion_focus'],
                                                   shape=[3, 2048], dtype='float16')))
    np.savez(path / 'ids.npz', painting_names=names)
    np.save(path / 'img.npy', np.ones((3, 2048), dtype=np.float16))
    x = diag.target_images('emotion', names, 'hash', tmp_path)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1)
    with pytest.raises(AssertionError):
        diag.target_images('emotion', names[::-1], 'hash', tmp_path)
    with pytest.raises(AssertionError):
        diag.target_images('emotion', names, 'other', tmp_path)


def test_cpu_fit_inference_uses_saved_d2_and_percept_builders(tmp_path, monkeypatch):
    import torch
    import ct_d2 as d2
    import ct_percept as pc
    torch.set_num_threads(8)
    engine = diag.FitInference.__new__(diag.FitInference)
    engine.d2, engine.pc, engine.torch = d2, pc, torch
    engine.root, engine.names, engine.sources = tmp_path, ['fit-a', 'fit-b'], {}
    engine.sample_sha256 = 'sample'
    engine.inputs = dict(qwen_default=np.ones((2, 3), dtype=np.float32))
    calls = []
    class Unit(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.normalize(x, dim=1)
    def builder(row, aspects, device):
        assert device.type == 'cpu'
        calls.append('d2')
        view = torch.nn.ModuleDict(dict(img=torch.nn.Sequential(torch.nn.Linear(3, 2048), Unit())))
        return None, dict(emotion=view)
    monkeypatch.setattr(d2.ms_train, 'build_model', builder)
    _, views = builder(None, None, torch.device('cpu'))
    path = d2._weight_path(tmp_path, 'emotion', 0)
    path.parent.mkdir(parents=True)
    torch.save(views['emotion'].state_dict(), path)
    monkeypatch.setattr(d2, '_weights_info', lambda *args: dict(weights_sha256=sc._sha_file(path), config=dict(device='cuda', sample_sha256='sample')))
    x = engine.predict('K1', 'emotion', 0)
    assert x.shape == (2, 2048) and x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-3)
    assert engine.sources['K1/emotion/0']['inference_device'] == 'cpu'
    assert engine.sources['K1/emotion/0']['training_device'] == 'cuda'

    config = dict(settings='core', cut='k40', arm='P3', seed=0, smoke=False,
                  mapper='global', input_dim=3, hidden_dim=4, device='cuda')
    mapper = pc.build_mapper(config, 40)
    path = tmp_path / 'percept/core/stage2/k40/P3__s0.npz'
    path.parent.mkdir(parents=True)
    torch.save(dict(mapper=mapper.state_dict()), path.with_suffix('.pt'))
    meta = dict(config=config, K=40, sample_sha256='sample', weights_sha256=sc._sha_file(path.with_suffix('.pt')))
    path.with_suffix('.json').write_text(json.dumps(meta))
    prob = engine.predict('B1', 'emotion', 0)
    with torch.no_grad():
        expected = mapper(torch.from_numpy(diag.normalize(engine.inputs['qwen_default']))).sigmoid().numpy()
    np.testing.assert_array_equal(prob, expected)
    assert engine.sources['B1/emotion/0']['inference_device'] == 'cpu'
    meta['weights_sha256'] = 'wrong'
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        engine.predict('B1', 'emotion', 0)


def test_run_writes_json_and_markdown_after_verdict_and_keeps_eligibility(tmp_path, monkeypatch):
    from types import SimpleNamespace
    n = 24
    monkeypatch.setattr(sc, 'HELD_ROWS', n)
    monkeypatch.setattr(sc, 'SEEDS', [0])
    names = [f'held-{i:02}' for i in range(n)]
    y = np.arange(n) % 3
    x = diag.normalize(np.random.default_rng(13).normal(size=(n, 8)))
    monkeypatch.setattr(diag.ct_held, 'read_held_list', lambda path: (names, 'names'))
    monkeypatch.setattr(diag.ct_held, 'held_names', lambda: names)
    monkeypatch.setattr(diag.ct_held, 'painting_labels', lambda names: {a: y for a in diag.PROMPTS})
    monkeypatch.setattr(diag.ct_held, 'qwen_default_images', lambda names: x)
    monkeypatch.setattr(diag, 'target_images', lambda *args: x)
    eligible = {arm: [0] if arm in ('K1', 'B1', 'B4') else [] for arm in sc.ARM_ORDER}
    sources = {arm: {'0': dict(status='finite' if eligible[arm] else 'rcol_collapsed_no_output')}
               for arm in sc.ARM_ORDER}
    original_verdict = dict(stage=1, outcome_id='S1-NO', held_sha256='names', eligibility=dict(eligible_seeds=eligible),
                            sources=dict(held_outputs=sources, rcol=dict(record=dict(runs=[]))))
    root = tmp_path / 'outputs'
    for aspect in diag.PROMPTS:
        for arm in sc.ARM_ORDER if aspect == 'emotion' else ('K1', 'B1'):
            if aspect == 'emotion' and not eligible[arm]:
                continue
            stem = sc.HELD_STEMS[arm]
            if aspect != 'emotion':
                stem = f'd2/{aspect}' if arm == 'K1' else f'percept/core__k40/{"P2" if aspect == "style" else "P3"}'
            path = root / 'held' / f'{stem}__s0.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
            path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256='weights')))
            if aspect == 'emotion':
                sources[arm]['0'].update(kind='k1' if arm == 'K1' else 'percept',
                                         output_sha256=sc._sha_file(path),
                                         sidecar_sha256=sc._sha_file(path.with_suffix('.json')),
                                         partitions='fixture')
    verdict_path = tmp_path / 'verdict_stage1.json'
    verdict_path.write_text(json.dumps(original_verdict))
    before = verdict_path.read_bytes()
    def predict(arm, aspect, seed):
        inference.sources[f'{arm}/{aspect}/{seed}'] = dict(weights_sha256='weights')
        return x
    inference = SimpleNamespace(names=['fit-a'], labels={a: y for a in diag.PROMPTS},
                                predict=predict, sources={})
    monkeypatch.setattr(diag, 'FitInference', lambda *args: inference)
    monkeypatch.setattr(sc, 'load_held_output', lambda *args, **kwargs: x)
    def fixed(kind):
        p = dict(r20=np.array([y] * 3))
        if kind != 'k1':
            p['own'] = y[None, :]
        return p
    monkeypatch.setattr(diag, 'saved_partitions', lambda path, source, hash: fixed(source['kind']))
    def diagnose(x, labels, kind, prior=None, fit=None, fit_labels=None):
        partitions = fixed(kind)
        record = dict(held={key: diag.partition_metrics(ps, labels) for key, ps in partitions.items()},
                      leiden_counts=[3, 3, 3])
        record['held']['leiden3a'] = record['held']['r20']
        record['held']['kmeans_at_leiden_count'] = record['held']['r20']
        if fit is not None:
            record['fit'] = dict(r20=record['held']['r20'])
            record['held']['leiden3b'] = record['held']['r20']
        return record, partitions['r20'], partitions
    monkeypatch.setattr(diag, 'diagnose_vectors', diagnose)
    monkeypatch.setattr(diag, 'HERE', tmp_path)
    report = diag.run(verdict_path, root, tmp_path / 'diag_stage1.json', workers=1, n_resamples=2)
    assert verdict_path.read_bytes() == before
    assert report['eligibility'] == eligible
    assert set(report['margins']) == {'leiden_emotion', 'style', 'genre'}
    assert set(report['margins']['leiden_emotion']['A_e']) == {'B1-r20', 'B4-r20'}
    assert len(report['records']) == 13  # 3 emotion + 4 style/genre + 6 references
    assert json.loads((tmp_path / 'diag_stage1.json').read_text())['diagnostic_only']
    table = (tmp_path / 'diag_stage1.md').read_text()
    assert '| target | style |' in table and '| leiden_emotion |' in table


def test_in_sample_reading_and_held_vote_score_their_own_populations(monkeypatch):
    fit_y = np.repeat([0, 1], 25)
    held_y = np.repeat([0, 1], 12)
    fit = np.eye(2, dtype=np.float32)[fit_y]
    held = np.eye(2, dtype=np.float32)[held_y]
    fit_labels = fit_y.copy()
    fit_labels[-1] = -1
    monkeypatch.setattr(diag, 'fit_graph', lambda x: (ig.Graph.Ring(len(x)), dict(fixture=True)))
    monkeypatch.setattr(diag, 'leiden_partitions',
                        lambda graph: np.array([fit_y if graph.vcount() == 50 else held_y] * 3))
    # Isolate the fit/transfer population seam from k-means' empty-center warnings.
    monkeypatch.setattr(sc, 'output_partitions', lambda x, kind: dict(r20=np.array([fit_y] * 3)))
    monkeypatch.setattr(sc, 'partitions', lambda x, k: np.array([held_y] * 3))
    record, _, _ = diag.diagnose_vectors(held, held_y, 'k1',
                                        dict(r20=np.array([held_y] * 3)), fit, fit_labels)
    assert set(record['fit']) == {'r20', 'leiden3b'}
    assert record['leiden3b']['counts'] == [2, 2, 2]
    assert record['leiden3b']['graph_vertices'] == 50
    for population in ('fit', 'held'):
        assert [p['scores']['ami'] for p in record[population]['leiden3b']] == [1., 1., 1.]


def test_main_refuses_without_a_stage0_pass(monkeypatch, gate_refuses):
    error, text = gate_refuses
    monkeypatch.setattr(diag, 'run', lambda *args, **kwargs: pytest.fail('ran before the V1 gate'))
    with pytest.raises(error, match=text):
        diag.main([])
