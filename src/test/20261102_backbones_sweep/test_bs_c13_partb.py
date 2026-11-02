"""Fast CPU checks of the independent arithmetic and bidirectional sheet join."""
import copy

import numpy as np
import pytest

import bs_c13_partb as C


def test_effect_sign_seed_mean_and_range_interval():
    rec = C.effect_record([0.3, 0.2, 0.4], [0.2, 0.3, 0.1], (1, 2, 3), 0.02, (-0.01, 0.2))
    assert rec['effect'] == pytest.approx(0.1)
    assert rec['effect_r'] == pytest.approx(5)
    assert (rec['lo_r'], rec['hi_r']) == pytest.approx((-0.5, 10))
    assert rec['arm_seed'] == {'1': 0.3, '2': 0.2, '3': 0.4}
    assert 'lo' not in C.effect_record([0.1], [0.2], (4,), 0.01)


def test_glossary_ranges_and_m_score():
    e = {a: r * unit for (a, r), unit in zip(C.CLUSTER_R.items(), (1, -2, 4))}
    assert C.m_score(e, C.CLUSTER_R) == pytest.approx(1)
    assert C.m_score(e, C.CLUSTER_R) == C.rs_sweep.mean_gain(e)
    assert C.m_score({'emotion': 0.0028}, {'emotion': C.MAP_R['emotion']}) == 2
    with pytest.raises(AssertionError):
        C.m_score(e, {'emotion': 0.1})


def test_three_seed_paired_transcript_and_query_seed_pairing():
    rng = np.random.default_rng(91)
    x, y = rng.random((3, 13)), rng.random((3, 13))
    got = C.paired_n(x, y, 17)
    rng, want = np.random.default_rng(0), []
    for _ in range(17):
        rows = rng.integers(0, 13, size=13)
        seeds = rng.choice([0, 1, 2], size=3, replace=True)
        want.append(x[seeds][:, rows].mean() - y[seeds][:, rows].mean())
    np.testing.assert_allclose(got['boot'], want, rtol=0, atol=5e-16)
    np.testing.assert_allclose(got['interval'], np.percentile(want, [2.5, 97.5]), atol=5e-16)
    np.testing.assert_allclose(C.paired_n(x, x, 7)['boot'], 0, atol=1e-15)


def test_five_seed_bootstrap_is_frozen_function():
    x = np.arange(55, dtype=np.float64).reshape(5, 11) / 100
    ref = np.flip(x, axis=0).copy()
    got, want = C.paired_n(x, ref, 7), C.rt_stats.paired(x, ref, draws=7)
    np.testing.assert_array_equal(got['boot'], want['boot'])


def test_query_mask_drops_unlabelled_and_singletons():
    np.testing.assert_array_equal(C.expected_queries(np.array([0, -1, 1, 0, 2, 2])), [0, 3, 4, 5])


def test_content_correction_uses_frozen_graph_rows():
    assert C.export_path('k11=40-no_content_correction', 'val', 'emotion', 0).name == 'emotion__R0__s0.npz'
    assert C.export_path('k11=40-no_content_correction', 'val', 'style', 4).name == 'style__R0__s4.npz'
    assert C.export_path('k11=40-no_relational', 'val', 'emotion', 0).name == 'emotion__Gres__s0.npz'


def effect():
    return C.effect_record([0.2, 0.3, 0.4], [0.1, 0.2, 0.3], (0, 1, 2), 0.01, (-0.05, 0.2))


def sheet_effect(rec):
    return {target: rec[source] for source, target in C.EFFECT_FIELDS.items() if source in rec}


def photo_document():
    e = effect()
    return {'rows': [dict(kind='photo', row='lr=0.003', validation=dict(
        ami={'emotion': e}, map={'center+k10': {'emotion': e}},
        M_ami={'M': 10}, M_map={'M': 10}, M=10))]}


def test_photo_adapter_exact_and_unknown_and_reverse():
    doc = photo_document()
    e = sheet_effect(effect())
    row = dict(row='lr=0.003', ami=e, map=e, M={'M': 10, 'note': 'two tasks'}, status='ok')
    sheet = {'rows': {'lr=0.003': row}}
    ours, theirs = C.derived_numbers(doc), C.adapt_sheet('photo', sheet)
    got = C.compare_values(ours, theirs)
    assert got['items'] == 19 and not got['mismatches']
    assert not got['missing_derived'] and not got['missing_sheet']
    row['extra_result'] = 5.0
    got = C.compare_values(ours, C.adapt_sheet('photo', sheet))
    assert got['missing_derived'] == [['photo', 'lr=0.003', 'unmapped', 'extra_result']]
    del row['extra_result']
    del row['map']
    assert len(C.compare_values(ours, C.adapt_sheet('photo', sheet))['missing_sheet']) == 9


def test_ablation_adapter_two_splits_readouts_and_m():
    e = effect()
    split = dict(ami={'genre': e}, map={'R0': {'genre': e}, 'center+k10': {'genre': e}},
                 M_ami={'M': 0.5}, M_map={'R0': {'M': 0.3}, 'center+k10': {'M': -0.1}})
    doc = {'rows': [dict(kind='ablation', row='relational', validation=split, held=split)]}
    se = sheet_effect(e)
    row = dict(removal='relational', val={'genre': dict(ami=se, map_r0=se, map_center_k10=se)},
               held={'genre': dict(ami=se, map_r0=se, map_center_k10=se)},
               M_val=dict(ami=0.5, map_r0=0.3, map_center_k10=-0.1),
               M_held=dict(ami=0.5, map_r0=0.3, map_center_k10=-0.1))
    got = C.compare_values(C.derived_numbers(doc), C.adapt_sheet('ablation', {'rows': {'relational': row}}))
    assert got['items'] == 60 and not got['mismatches']
    assert not got['missing_derived'] and not got['missing_sheet']


def test_knob_adapter_and_tolerance_boundary():
    e = effect()
    split = dict(ami={'emotion': e}, map={'R0': {'emotion': e}},
                 M_ami={'M': 2, 'lo': -1, 'hi': 3}, M_map={'M': 1})
    doc = {'rows': [dict(kind='knob', row='k2=16', validation=split)]}
    row = dict(row='k2=16', aspects={'emotion': {'ami': sheet_effect(e), 'map_r0': sheet_effect(e)}},
               M={'M': 2, 'lo': -1, 'hi': 3}, M_map={'M': 1})
    got = C.compare_values(C.derived_numbers(doc), C.adapt_sheet('knob', {'rows': {'k2=16': row}}))
    assert got['items'] == 22 and not got['missing_derived'] and not got['missing_sheet']
    key = ('same',)
    assert not C.compare_values({key: 0}, {key: 1e-6})['mismatches']
    assert C.compare_values({key: 0}, {key: 1.001e-6})['mismatches']
    assert C.compare_values({key: 0}, {key: float('nan')})['mismatches']


def test_duplicate_sheet_and_derived_rows_rejected():
    row = dict(row='lr=0.003', ami={}, map={})
    with pytest.raises(AssertionError, match='duplicate'):
        C.adapt_sheet('photo', {'rows': [row, row]})
    doc = photo_document()
    doc['rows'].append(copy.deepcopy(doc['rows'][0]))
    with pytest.raises(AssertionError, match='duplicate'):
        C.derived_numbers(doc)


def test_compare_cannot_certify_smoke_or_completely_absent_rows(monkeypatch, capsys):
    reports = []
    monkeypatch.setattr(C, 'read_json', lambda p: dict(smoke=True, selected=['photo:lr=0.003']))
    monkeypatch.setattr(C, 'sheet_numbers', lambda d: ({('photo', 'lr=0.003', 'M'): 1.0},
                                                       {('photo', 'lr=0.003', 'M'): 1.0}, []))
    monkeypatch.setattr(C, 'sha_file', lambda p: 'hash')
    monkeypatch.setattr(C, 'write_json', lambda p, doc: reports.append(doc))
    assert C.compare() == 1
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert reports[-1]['rows_missing_derived'] == ['knob:k2=16', 'knob:k11=80',
                                                  'ablation:content_correction', 'ablation:relational']


def test_ablation_center_fits_own_entire_validation_for_each_split(monkeypatch):
    calls = []

    class NoDisk:
        def cached(self, operation, arrays, params, compute):
            return compute()

    def stage5(ro, val, target):
        calls.append((ro, val.copy(), target.copy()))
        return target

    def retrieval(x, y, qe_k, workers):
        return dict(ap=np.full(len(y), 0.5), rows=np.arange(len(y)))

    monkeypatch.setattr(C.rt_readout, 'stage5_apply', stage5)
    monkeypatch.setattr(C.rt_metrics, 'retrieval', retrieval)
    val = np.array([[1, 2], [2, 1], [-1, -2], [-2, -1]], np.float16)
    held = val * 2
    y = np.array([0, 0, 1, 1])
    for split, raw in [('val', val), ('held', held)]:
        got = C.ablation_ap(NoDisk(), val, raw, y, split, 'center+k10')
        assert float(got['point']) == 0.5
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][1], C.unit(val))
    np.testing.assert_array_equal(calls[1][1], C.unit(val))
    np.testing.assert_array_equal(calls[1][2], C.unit(held))
