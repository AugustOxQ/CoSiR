"""Ticket 08: public stage-2 array seams; frozen replay is controller-only."""
import json

import numpy as np
import pytest

import pf_paths


def synthetic():
    rng = np.random.default_rng(8)
    labels = np.tile(np.arange(4, dtype=np.int64), 30)
    centers = rng.standard_normal((4, 8))
    raw, unit = {}, {}
    for seed in pf_paths.SEEDS:
        x = centers[labels] + (0.5 + seed * 0.4) * rng.standard_normal((len(labels), 8))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        raw[seed] = x.astype(np.float16)
        y = raw[seed].astype(np.float32)
        unit[seed] = y / np.linalg.norm(y, axis=1, keepdims=True)
    dist = np.eye(9, dtype=np.float64)[labels]
    return raw, unit, labels, dist


def test_ranges_measure_five_runs_and_keep_each_unrounded_spread():
    import pf_stage2

    raw, unit, labels, dist = synthetic()
    result = pf_stage2.ranges(raw.__getitem__, unit.__getitem__, labels=labels, dist=dist, workers=1)
    assert set(result) == {'clu', 'map', 'ndcg'}
    for metric in result.values():
        assert len(metric['per_seed']) == 5
        assert metric['r'] == max(metric['per_seed']) - min(metric['per_seed'])
        assert metric['r'] > 0
    # Independent, loop-based all-gallery AP; excludes self and preserves row-order ties.
    expected = []
    for x in unit.values():
        ap = []
        for q in range(len(labels)):
            order = sorted((g for g in range(len(labels)) if g != q),
                           key=lambda g: -float(x[q].astype(np.float64) @ x[g].astype(np.float64)))
            hits, total = 0, 0.0
            for rank, g in enumerate(order, 1):
                if labels[g] == labels[q]:
                    hits += 1
                    total += hits / rank
            ap.append(total / hits)
        expected.append(float(np.mean(ap)))
    np.testing.assert_allclose(result['map']['per_seed'], expected, atol=1e-14, rtol=0)


def test_ranges_cluster_stored_rows_and_retrieve_embedded_rows():
    import ac_cluster
    import pf_stage2
    import rt_metrics

    raw, unit, labels, dist = synthetic()
    # Preserve valid dtypes/norms, but break the agreement of the two representations.
    rng = np.random.default_rng(42)
    unit = {s: x[rng.permutation(len(x))] for s, x in unit.items()}
    expected_clu = []
    expected_map = []
    for s in pf_paths.SEEDS:
        stored_parts = ac_cluster.partitions(raw[s])
        embedded_parts = ac_cluster.partitions(unit[s])
        stored_ami = ac_cluster.ami(stored_parts, labels)
        assert not np.array_equal(stored_parts, embedded_parts)
        assert abs(stored_ami - ac_cluster.ami(embedded_parts, labels)) > 0.01
        expected_clu.append(stored_ami)
        expected_map.append(float(rt_metrics.retrieval(unit[s], labels, workers=1)['ap'].mean()))
    result = pf_stage2.ranges(raw.__getitem__, unit.__getitem__, labels=labels, dist=dist, workers=1)
    np.testing.assert_array_equal(result['clu']['per_seed'], expected_clu)
    np.testing.assert_array_equal(result['map']['per_seed'], expected_map)


@pytest.mark.parametrize('missing', ['stored', 'embed'])
def test_ranges_require_all_five_runs(missing):
    import pf_stage2

    raw, unit, labels, dist = synthetic()
    (raw if missing == 'stored' else unit).pop(4)
    with pytest.raises(AssertionError, match='five runs'):
        pf_stage2.ranges(raw.__getitem__, unit.__getitem__, labels=labels, dist=dist, workers=1)


@pytest.mark.parametrize('bad', ['stored_dtype', 'embed_norm', 'rows', 'dist_shape'])
def test_ranges_check_array_boundaries_without_repair(bad):
    import pf_stage2

    raw, unit, labels, dist = synthetic()
    if bad == 'stored_dtype':
        raw[4] = raw[4].astype(np.float32)
    elif bad == 'embed_norm':
        unit[4] *= 2
    elif bad == 'rows':
        raw[4] = raw[4][:-1]
    else:
        dist = dist[:, :4]
    with pytest.raises(AssertionError):
        pf_stage2.ranges(raw.__getitem__, unit.__getitem__, labels=labels, dist=dist, workers=1)


def test_readout_uses_the_passed_range_for_scores_guard_and_pick():
    import pf_stage2

    _, unit, labels, _ = synthetic()
    small = pf_stage2.readout(unit, 0.01, labels=labels, workers=1)
    large = pf_stage2.readout(unit, 1000.0, labels=labels, workers=1)
    assert small['pick'] != 'none+k0'
    assert large['pick'] == 'none+k0'  # all gains are within 0.001 r of R0
    assert set(small['scores']) == set(large['scores'])
    base = small['scores']['none+k0']['map']['emotion']
    for name, row in small['scores'].items():
        assert row['map'] == large['scores'][name]['map']
        assert row['score'] == (row['map']['emotion'] - base) / 0.01
        assert row['score'] == pytest.approx(large['scores'][name]['score'] * 100000)
        assert large['scores'][name]['kept']
    assert any(not row['kept'] for row in small['scores'].values())


@pytest.mark.parametrize('seeds', [(0, 1, 2, 3), (None,), (0, 1, 2, 3, 4, 5)])
def test_readout_requires_exactly_five_runs(seeds):
    import pf_stage2

    _, unit, labels, _ = synthetic()
    with pytest.raises(AssertionError, match='five runs'):
        pf_stage2.readout({s: unit[0] for s in seeds}, 0.01, labels=labels, workers=1)


@pytest.mark.parametrize('r_map', [0.0, -0.1, float('nan'), float('inf')])
def test_readout_rejects_nonpositive_or_nonfinite_ranges(r_map):
    import pf_stage2

    _, unit, labels, _ = synthetic()
    with pytest.raises(AssertionError, match='finite positive'):
        pf_stage2.readout(unit, r_map, labels=labels, workers=1)


def test_readout_refuses_nonunit_rows_and_duplicate_keys():
    import pf_stage2

    _, unit, labels, _ = synthetic()
    with pytest.raises(AssertionError):
        pf_stage2.readout(unit, 0.01, labels=labels, keys=['same/key'] * len(labels), workers=1)
    unit[4] *= 2
    with pytest.raises(AssertionError):
        pf_stage2.readout(unit, 0.01, labels=labels, workers=1)


def test_record_has_the_shared_schema_and_keeps_unrounded_numbers():
    import pf_stage2

    measured = {metric: {'per_seed': [0.1, 0.12, 0.11, 0.1234567890123, 0.115],
                         'r': 0.1234567890123 - 0.1} for metric in ('clu', 'map', 'ndcg')}
    frozen = {metric: {'per_seed': [0.2, 0.21, 0.22, 0.23, 0.24], 'r': 0.24 - 0.2}
              for metric in measured}
    _, unit, labels, _ = synthetic()
    selected = pf_stage2.readout(unit, measured['map']['r'], labels=labels, workers=1)
    result = pf_stage2.record(measured, selected, r12_ranges=frozen, width=8)
    assert result == {'ranges': measured, 'r12_ranges': frozen, 'readout': selected}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    measured['clu']['per_seed'][0] = 999
    assert result['ranges']['clu']['per_seed'][0] == 0.1


def test_record_refuses_a_missing_run_or_an_edited_range():
    import pf_stage2

    measured = {metric: {'per_seed': [0.1, 0.2, 0.3, 0.4, 0.5], 'r': 0.4}
                for metric in ('clu', 'map', 'ndcg')}
    selected = {'pick': 'none+k0', 'scores': {'none+k0': {'map': {'emotion': 0.25}, 'score': 0.0, 'kept': True}}}
    for bad in ({'per_seed': [0.1] * 4, 'r': 0.1}, {'per_seed': [0.1, 0.2, 0.3, 0.4, 0.5], 'r': 0.3}):
        with pytest.raises(AssertionError):
            pf_stage2.record(dict(measured, clu=bad), selected, r12_ranges=measured)


def test_record_refuses_readout_scored_with_another_range():
    import pf_stage2

    measured = {metric: {'per_seed': [0.1, 0.2, 0.3, 0.4, 0.5], 'r': 0.4}
                for metric in ('clu', 'map', 'ndcg')}
    _, unit, labels, _ = synthetic()
    wrong = pf_stage2.readout(unit, pf_paths.R12_R_MAP, labels=labels, workers=1)
    with pytest.raises(AssertionError):
        pf_stage2.record(measured, wrong, r12_ranges=measured, width=8)
    right = pf_stage2.readout(unit, measured['map']['r'], labels=labels, workers=1)
    assert pf_stage2.record(measured, right, r12_ranges=measured, width=8)['readout'] == right


def test_record_requires_the_complete_menu_in_tie_order():
    import pf_stage2
    import rt_readout

    measured = {metric: {'per_seed': [0.1, 0.2, 0.3, 0.4, 0.5], 'r': 0.4}
                for metric in ('clu', 'map', 'ndcg')}
    scores = {rt_readout.name(ro): {'map': {'emotion': 0.25}, 'score': 0.0, 'kept': True}
              for ro in rt_readout.menu(2048)}
    selected = {'pick': 'none+k0', 'scores': scores}
    assert pf_stage2.record(measured, selected, r12_ranges=measured)['readout'] == selected
    for bad in (dict(list(scores.items())[:-1]), dict(reversed(list(scores.items())))):
        with pytest.raises(AssertionError):
            pf_stage2.record(measured, dict(selected, scores=bad), r12_ranges=measured)


def test_gate_writes_failures_for_each_range_and_readout_field(tmp_path, monkeypatch):
    """Replace only frozen-file readers; exercise real scoring against bad references."""
    import ac_data
    import pf_stage2
    import rt_readout

    raw, unit, labels, dist = synthetic()
    monkeypatch.setenv('PF_OUTPUTS', str(tmp_path / 'outputs'))
    monkeypatch.setattr(pf_paths, 'OUTPUTS', tmp_path / 'outputs')
    monkeypatch.setattr(pf_paths, 'R12_OUT', tmp_path / 'r12')
    monkeypatch.setattr(ac_data, 'stored', lambda arm, split, seed, **kw: raw[seed])
    monkeypatch.setattr(ac_data, 'embed', lambda arm, split, seed, **kw: unit[seed])
    monkeypatch.setattr(ac_data, 'labels', lambda split: labels)
    monkeypatch.setattr(ac_data, 'emotion_dist', lambda split: dist)
    monkeypatch.setattr(ac_data, 'keys', lambda split: [f'source/{i:04d}' for i in range(len(labels))])
    # Valid positive reference ranges deliberately unrelated to the synthetic scores.
    reference = {m: {'per_seed': [0.10, 0.11, 0.12, 0.13, 0.14], 'r': 0.14 - 0.10}
                 for m in ('clu', 'map', 'ndcg')}
    scores = {rt_readout.name(ro): {'map': {'emotion': -1.0}, 'score': -1.0, 'kept': False}
              for ro in rt_readout.menu(8)}
    for path, doc in (
        ('stage0b/ranges.json', reference),
        ('stage1/scores.json', {'readout': {'affection': {'cstar': scores}}}),
        ('stage1/picks.json', {'affection': {'readout': {'cstar': 'center+k10'}}}),
    ):
        target = pf_paths.R12_OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc))
    result = pf_stage2.gate_ii_stage2(workers=1)
    assert result['ok'] is False
    assert json.loads((pf_paths.OUTPUTS / 'stage0a/stage2_replay.json').read_text()) == result
    for metric in ('clu', 'map', 'ndcg'):
        for field in ('r', *(f'per_seed.{s}' for s in pf_paths.SEEDS)):
            item = result['items'][f'ranges.{metric}.{field}']
            assert item['ok'] is False
            assert set(item) == {'value', 'reference', 'ok'}
    for name in scores:
        for field in ('map.emotion', 'score'):
            assert result['items'][f'readout.{name}.{field}']['ok'] is False
    assert result['items']['readout.none+k0.kept']['ok'] is False


@pytest.mark.slow
@pytest.mark.real_data
def test_gate_ii_stage2(tmp_path, monkeypatch):
    """Controller: full c* val replay, CPU; estimate 1–3 minutes from r12 timings."""
    import pf_stage2

    monkeypatch.setenv('PF_OUTPUTS', str(tmp_path))
    monkeypatch.setattr(pf_paths, 'OUTPUTS', tmp_path)
    result = pf_stage2.gate_ii_stage2()
    assert result['ok'], {k: v for k, v in result['items'].items() if not v['ok']}
    assert result['items']['readout.pick']['value'] == 'center+k10'
    assert sum(k.endswith('.map.emotion') for k in result['items']) == 25
