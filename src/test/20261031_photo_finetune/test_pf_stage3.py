"""Tests for pf_stage3 (ticket 09): SC1 to SC5 of spec 2026-10-31 §6, outcomes (§3.4), the verdict, SC5's claim, the
comparators from r12's outputs and the stage 3 verdict document. Verdict code: written test first, through the public
functions. The slow gate (ii) replay is `test_gate_ii_sc` (controller, CPU).

Seams: outcome / sc_pass / verdict / claims (numbers and outcomes -> decisions), load_comparators (r12's files -> the
comparator rows, with their checks), score (arrays + comparators -> the verdict document), gate_ii_sc (the replay).
"""
import itertools
import json

import numpy as np
import pytest

import pf_paths  # noqa: F401  (puts r12's round on sys.path)
import pf_stage3

OUT = ('WIN', 'LEVEL', 'BELOW')


# ---- outcomes, bars, the verdict -----------------------------------------------------------------------------------

def test_outcome_uses_the_r_passed_in():
    # the photo model's own range decides LEVEL, not r12's
    assert pf_stage3.outcome(-0.004, 0.005) == 'LEVEL'
    assert pf_stage3.outcome(-0.004, 0.003) == 'BELOW'
    assert pf_stage3.outcome(0.0, 0.003) == 'LEVEL'            # lower bound 0 is not a WIN
    assert pf_stage3.outcome(-0.003, 0.003) == 'BELOW'         # lower bound -r is not a LEVEL
    assert pf_stage3.outcome(1e-9, 0.003) == 'WIN'


def test_bars_by_sc():
    for sid in ('SC1', 'SC2', 'SC5'):                           # WIN bars
        assert [pf_stage3.sc_pass(sid, o) for o in OUT] == [True, False, False]
    for sid in ('SC3', 'SC4'):                                  # LEVEL or WIN
        assert [pf_stage3.sc_pass(sid, o) for o in OUT] == [True, True, False]


def _verdict_spec(passes):
    """Spec §6 stated independently: holds = all four pass; qualified = SC1 and SC2 pass and exactly one of SC3, SC4."""
    p1, p2, p3, p4 = passes
    if p1 and p2 and p3 and p4:
        return 'holds'
    if p1 and p2 and (p3 + p4) == 1:
        return 'qualified'
    return 'fails'


def test_every_outcome_combination_maps_to_the_spec_verdict():
    seen = set()
    for combo in itertools.product(OUT, repeat=4):
        outcomes = dict(zip(('SC1', 'SC2', 'SC3', 'SC4'), combo))
        passes = [pf_stage3.sc_pass(s, o) for s, o in outcomes.items()]
        got = pf_stage3.verdict(outcomes)
        assert got == _verdict_spec(passes), combo
        seen.add(got)
    assert seen == {'holds', 'qualified', 'fails'}


def test_verdict_literal_cases():
    v = pf_stage3.verdict
    assert v(dict(SC1='WIN', SC2='WIN', SC3='LEVEL', SC4='WIN')) == 'holds'
    assert v(dict(SC1='WIN', SC2='WIN', SC3='LEVEL', SC4='BELOW')) == 'qualified'
    assert v(dict(SC1='WIN', SC2='WIN', SC3='BELOW', SC4='WIN')) == 'qualified'
    assert v(dict(SC1='WIN', SC2='WIN', SC3='BELOW', SC4='BELOW')) == 'fails'
    assert v(dict(SC1='WIN', SC2='LEVEL', SC3='WIN', SC4='WIN')) == 'fails'      # a plain bar is WIN
    assert v(dict(SC1='LEVEL', SC2='WIN', SC3='WIN', SC4='BELOW')) == 'fails'    # qualified needs both plain WINs


def test_verdict_needs_exactly_sc1_to_sc4():
    with pytest.raises(AssertionError):
        pf_stage3.verdict(dict(SC1='WIN', SC2='WIN', SC3='WIN'))
    with pytest.raises(AssertionError):
        pf_stage3.verdict(dict(SC1='WIN', SC2='WIN', SC3='WIN', SC4='WIN', SC5='WIN'))


def test_sc5_claim_is_its_own_and_follows_its_bar():
    assert pf_stage3.claims({'SC5': {'pass': True}}) == {'SC5': True}
    assert pf_stage3.claims({'SC5': {'pass': False}}) == {'SC5': False}


# ---- comparators from r12's saved files -----------------------------------------------------------------------------

N_Q = 40            # tiny test split: 40 photos, four emotions, ten each
PICKS = {'readout': {'plain': 'center+k10', 'teacher_artwork': 'center+k10', 'crl_clip_painting': 'center+k10',
                     'cstar': 'center+k10'},
         'teacher_wording': {'clustering': 'artwork', 'retrieval': 'artwork'}, 'crl_version': 'clip_painting'}


def _write_partition(root, arm, labels):
    import rs_vh
    d = root / 'partitions'
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f'affection__test__{arm}__s-.npz', labels=labels)
    side = dict(array_sha256='x' * 64, shape=[labels.shape[1], 2048], settings={'k': 20},
                partitions_sha256=rs_vh.file_sha(d / f'affection__test__{arm}__s-.npz'))
    (d / f'affection__test__{arm}__s-.json').write_text(json.dumps(side))


def _write_ap(root, stage, arm, ap, rows):
    d = root / f'stage{stage}/perquery'
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f'affection__{arm}__emotion__s-.npz', ap=ap, rows=rows)


@pytest.fixture
def fake_r12(tmp_path, monkeypatch):
    import ac_data
    monkeypatch.setattr(ac_data, 'labels', lambda split: LABELS)
    rng = np.random.default_rng(1)
    rows = np.arange(N_Q, dtype=np.int64)
    aps = {a: rng.uniform(0.1, 0.5, N_Q) for a in ('plain', 'teacher_artwork', 'crl_clip_painting')}
    for arm in ('plain', 'teacher_artwork'):
        _write_partition(tmp_path, arm, rng.integers(0, 20, (3, N_Q)).astype(np.int32))
        _write_ap(tmp_path, 2, arm, aps[arm], rows)
    _write_ap(tmp_path, 2, 'crl_clip_painting', aps['crl_clip_painting'], rows)
    _write_ap(tmp_path, 3, 'crl_clip_painting', aps['crl_clip_painting'], rows)
    _write_verdicts(tmp_path)
    return tmp_path, aps


def test_load_comparators_follow_r12_picks(fake_r12):
    root, aps = fake_r12
    comps = pf_stage3.load_comparators(PICKS, N_Q, r12=root)
    assert set(comps) == {'plain', 'teacher', 'crl'}
    assert [comps[k]['arm'] for k in ('plain', 'teacher', 'crl')] == ['plain', 'teacher_artwork', 'crl_clip_painting']
    assert comps['teacher']['readout'] == 'center+k10'
    np.testing.assert_array_equal(comps['teacher']['ap'], aps['teacher_artwork'])
    assert comps['plain']['point'] == aps['plain'].mean()
    assert comps['plain']['labels'].shape == (3, N_Q) and comps['crl']['labels'] is None


def test_a_changed_partition_file_is_refused(fake_r12):
    root, _ = fake_r12
    np.savez(root / 'partitions/affection__test__plain__s-.npz', labels=np.zeros((3, N_Q), dtype=np.int32))
    with pytest.raises(AssertionError, match='partition file changed'):
        pf_stage3.load_comparators(PICKS, N_Q, r12=root)


def test_a_wrong_sized_comparator_is_refused(fake_r12):
    root, _ = fake_r12
    with pytest.raises(AssertionError):
        pf_stage3.load_comparators(PICKS, N_Q + 1, r12=root)


DIM, NV = 16, 60
LABELS = np.repeat(np.arange(4), 10).astype(np.int64)          # 40 test photos, four emotions of ten


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _candidate(good, seed=0, name='photo'):
    """Five runs. good: each photo sits at its emotion's corner of a simplex (+ small noise); else pure noise."""
    rng = np.random.default_rng(seed)
    corners = np.eye(DIM, dtype=np.float32)[:4] * 1.0
    stored, test, val = [], [], []
    for s in range(5):
        x = (corners[LABELS] if good else 0.0) + (0.2 if good else 1.0) * rng.normal(size=(N_Q, DIM))
        st = _unit(x).astype(np.float16)
        stored.append(st)
        test.append(_unit(st.astype(np.float32)))
        val.append(_unit(rng.normal(size=(NV, DIM))))
    return pf_stage3.Candidate(name=name, readout='center+k10', test_stored=np.stack(stored),
                               test_embed=np.stack(test), val_embed=np.stack(val))


def _perfect_comparators(fake, good):
    """Comparators whose r12 files say: perfect (AP 1, partitions = classes) or no better than chance."""
    root, aps = fake
    rows = np.arange(N_Q, dtype=np.int64)
    perfect = (np.repeat(np.arange(4), 10) * 5 + np.arange(N_Q) % 5).astype(np.int32)
    for arm in ('plain', 'teacher_artwork'):
        lab = np.tile(perfect, (3, 1)) if good else np.random.default_rng(3).integers(0, 20, (3, N_Q)).astype(np.int32)
        _write_partition(root, arm, lab)
        _write_ap(root, 2, arm, np.ones(N_Q) if good else np.full(N_Q, 0.3), rows)
    for stage in (2, 3):
        _write_ap(root, stage, 'crl_clip_painting', np.ones(N_Q) if good else np.full(N_Q, 0.3), rows)
    _write_verdicts(root)
    return pf_stage3.load_comparators(PICKS, N_Q, r12=root)


R = {'clu': 0.02, 'map': 0.01}
DIST = np.eye(9)[LABELS % 9]


def _score(cand, comps, tmp_path, **kw):
    return pf_stage3.score(cand, R, LABELS, DIST, comps, PICKS, partition_root=tmp_path / 'parts',
                           draws_clu=60, draws_ret=200, workers=2, **kw)


def test_a_clearly_better_candidate_wins_every_bar(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    doc = _score(_candidate(good=True), comps, tmp_path)['doc']
    assert list(doc['sc']) == list(pf_stage3.SC_IDS)
    assert [e['outcome'] for e in doc['sc'].values()] == ['WIN'] * 5
    assert doc['verdict'] == 'holds' and doc['claims'] == {'SC5': True}
    e = doc['sc']['SC3']
    assert (e['candidate'], e['comparator'], e['metric'], e['readout_c'], e['readout_t']) == \
        ('photo', 'teacher_artwork', 'ami', None, None)
    e = doc['sc']['SC5']
    assert (e['comparator'], e['metric'], e['readout_c'], e['readout_t'], e['r']) == \
        ('crl_clip_painting', 'map', 'center+k10', 'center+k10', R['map'])
    assert e['point_t'] == pytest.approx(0.3) and e['d'] == pytest.approx(e['point_c'] - e['point_t'])
    # the spec's section 6 table: (comparator, metric, r) of every SC
    table = {'SC1': ('plain', 'ami', R['clu']), 'SC2': ('plain', 'map', R['map']),
             'SC3': ('teacher_artwork', 'ami', R['clu']), 'SC4': ('teacher_artwork', 'map', R['map']),
             'SC5': ('crl_clip_painting', 'map', R['map'])}
    assert {k: (v['comparator'], v['metric'], v['r']) for k, v in doc['sc'].items()} == table


def test_a_clearly_worse_candidate_is_below_and_fails(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=True)
    doc = _score(_candidate(good=False), comps, tmp_path)['doc']
    assert [e['outcome'] for e in doc['sc'].values()] == ['BELOW'] * 5
    assert doc['verdict'] == 'fails' and doc['claims'] == {'SC5': False}
    assert all(not e['pass'] for e in doc['sc'].values())


def test_level_is_decided_by_the_r_passed_in(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=True)
    cand = _candidate(good=True)
    wide = pf_stage3.score(cand, {'clu': 5.0, 'map': 5.0}, LABELS, DIST, comps, PICKS, partition_root=tmp_path / 'p',
                           draws_clu=60, draws_ret=200, workers=2)['doc']
    narrow = pf_stage3.score(cand, {'clu': 1e-6, 'map': 1e-6}, LABELS, DIST, comps, PICKS,
                             partition_root=tmp_path / 'p', draws_clu=60, draws_ret=200, workers=2)['doc']
    for sid in ('SC1', 'SC2', 'SC3', 'SC4'):
        assert wide['sc'][sid]['interval'] == narrow['sc'][sid]['interval']          # same data, same bootstrap
        lower = wide['sc'][sid]['interval'][0]
        assert wide['sc'][sid]['outcome'] == ('WIN' if lower > 0 else 'LEVEL')
        assert narrow['sc'][sid]['outcome'] == ('WIN' if lower > 0 else 'BELOW' if lower <= -1e-6 else 'LEVEL')


def test_a_comparator_row_mismatch_raises(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    comps['teacher'] = dict(comps['teacher'], rows=comps['teacher']['rows'][::-1].copy())
    with pytest.raises(AssertionError, match='different queries'):
        _score(_candidate(good=True), comps, tmp_path)


def test_a_comparator_missing_a_query_raises(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    short = {k: comps['crl'][k] for k in ('arm', 'readout', 'labels')}
    short.update(ap=comps['crl']['ap'][:-1], rows=comps['crl']['rows'][:-1], point=float(comps['crl']['ap'][:-1].mean()))
    comps['crl'] = short
    with pytest.raises(AssertionError, match='different queries'):
        _score(_candidate(good=True), comps, tmp_path)


def test_outputs_stay_out_of_r12s_folder(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    with pytest.raises(AssertionError, match='r12'):
        pf_stage3.score(_candidate(good=True), R, LABELS, DIST, comps, PICKS,
                        partition_root=pf_paths.R12_OUT / 'partitions', draws_clu=60, draws_ret=200, workers=2)


def test_the_stored_and_embedded_arrays_must_be_the_same_rows(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    cand = _candidate(good=True)
    bad = cand._replace(test_embed=cand.test_embed[:, ::-1].copy())
    with pytest.raises(AssertionError, match='stored'):
        _score(bad, comps, tmp_path)


def test_photo_rows_and_perquery_are_returned(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    res = _score(_candidate(good=True), comps, tmp_path, frozen_rows={'cstar': {'ami': 0.1}})
    row = res['doc']['rows']['photo']
    assert row['readout'] == 'center+k10' and len(row['map_picked']['per_seed']) == 5
    assert row['map_picked']['map'] == pytest.approx(np.mean(row['map_picked']['per_seed']))
    assert len(row['map_r0']['per_seed']) == 5
    assert 0 <= row['ndcg_r0'] <= 1 and row['ami'] == res['doc']['sc']['SC1']['point_c']
    assert res['doc']['rows']['cstar'] == {'ami': 0.1}
    assert set(res['perquery']) == {f'photo{suf}__s{s}' for suf in ('', '_R0') for s in range(5)}
    ap, rows = res['perquery']['photo__s3']
    assert ap.shape == rows.shape == (N_Q,)
    assert res['partitions'].shape == (5, 3, N_Q) and res['partitions'].dtype == np.int32
    assert (tmp_path / 'parts/affection__test__photo__s4.npz').is_file()


def test_write_stage3_writes_r12s_names_and_refuses_r12s_folder(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    res = _score(_candidate(good=True), comps, tmp_path)
    path = pf_stage3.write_stage3(res, tmp_path / 'stage3')
    assert json.loads(path.read_text()) == res['doc']
    with np.load(tmp_path / 'stage3/perquery/affection__photo_R0__emotion__s2.npz') as z:
        assert z['ap'].dtype == np.float64 and z['rows'].dtype == np.int64 and z['ap'].shape == (N_Q,)
    with pytest.raises(AssertionError, match='r12'):
        pf_stage3.write_stage3(res, pf_paths.R12_OUT / 'stage3')


def test_a_frozen_row_cannot_replace_the_candidates_row(fake_r12, tmp_path):
    comps = _perfect_comparators(fake_r12, good=False)
    with pytest.raises(AssertionError, match='overwrite'):
        _score(_candidate(good=True), comps, tmp_path, frozen_rows={'photo': {}})


def test_compare_entry_flags_a_changed_number_or_outcome():
    ref = dict(candidate='cstar', comparator='plain', dataset='affection', aspect='emotion', metric='ami', readout_c=None,
               readout_t=None, point_c=0.1, point_t=0.05, d=0.05, interval=[0.04, 0.06], r=0.003, outcome='WIN', **{'pass': True})
    assert pf_stage3.compare_entry(dict(ref), ref, 1e-9)['ok']
    assert not pf_stage3.compare_entry(dict(ref, d=0.05 + 1e-6), ref, 1e-9)['ok']
    assert not pf_stage3.compare_entry(dict(ref, interval=[0.04, 0.06 + 1e-6]), ref, 1e-9)['ok']
    assert not pf_stage3.compare_entry(dict(ref, outcome='LEVEL'), ref, 1e-9)['ok']


@pytest.mark.slow
@pytest.mark.real_data
def test_gate_ii_sc(tmp_path):
    """Controller, CPU: r12's c* through pf_stage3 reproduces r12's SC1 to SC4 and SC8 (2,000 and 10,000 draws)."""
    doc = pf_stage3.gate_ii_sc(out=tmp_path)
    assert doc['full_draws'] and doc['ok'], {k: (i['ok'], i.get('max_abs_diff', i['value'])) for k, i in doc['items'].items()}


def test_crl_ap_must_agree_between_r12_stages(fake_r12):
    root, aps = fake_r12
    _write_ap(root, 3, 'crl_clip_painting', aps['crl_clip_painting'] + 1e-3, np.arange(N_Q, dtype=np.int64))
    with pytest.raises(AssertionError, match='stages 2 and 3'):
        pf_stage3.load_comparators(PICKS, N_Q, r12=root)


def _write_verdicts(root):
    """Save independent SC comparator points and row AMIs alongside synthetic exports."""
    sc, rows = {}, {}
    for sid, arm in (('SC2', 'plain'), ('SC4', 'teacher_artwork'), ('SC8', 'crl_clip_painting')):
        stage = 3 if sid == 'SC8' else 2
        with np.load(root / f'stage{stage}/perquery/affection__{arm}__emotion__s-.npz') as z:
            sc[sid] = {'point_t': float(z['ap'].mean())}
        if stage == 2:
            with np.load(root / f'partitions/affection__test__{arm}__s-.npz') as z:
                rows[arm] = {'ami': pf_stage3.ac_cluster.ami(z['labels'], LABELS)}
    (root / 'stage2/verdict.json').write_text(json.dumps({'sc': {k: sc[k] for k in ('SC2', 'SC4')}, 'rows': rows}))
    (root / 'stage3/verdict.json').write_text(json.dumps({'sc': {'SC8': sc['SC8']}}))


def test_teacher_wordings_must_agree():
    bad = dict(PICKS, teacher_wording={'clustering': 'artwork', 'retrieval': 'photo'})
    with pytest.raises(AssertionError, match='wording differs'): pf_stage3.comparator_arms(bad)


def test_ap_count_cannot_exceed_test_rows(fake_r12):
    root, _ = fake_r12
    path = root / 'stage2/perquery/affection__plain__emotion__s-.npz'
    with pytest.raises(AssertionError): pf_stage3._load_ap(path, N_Q - 1)


def test_partition_sidecar_row_count_must_match(fake_r12):
    root, _ = fake_r12
    path = root / 'partitions/affection__test__plain__s-.json'
    side = json.loads(path.read_text())
    side['shape'][0] += 1
    path.write_text(json.dumps(side))
    with pytest.raises(AssertionError): pf_stage3._load_partition(root, 'plain', N_Q)


@pytest.mark.parametrize('bad', ['unit', 'val_shape'])
def test_candidate_array_guards(bad):
    cand = _candidate(good=True)
    cand = cand._replace(val_embed=cand.val_embed * 2 if bad == 'unit' else cand.val_embed[:4])
    with pytest.raises(AssertionError): pf_stage3._check_candidate(cand)


def test_candidate_retrieval_refuses_dropped_queries(monkeypatch):
    monkeypatch.setattr(pf_stage3.ac_stats, 'arm_retrieval', lambda *a, **kw: {'dropped': 1})
    with pytest.raises(AssertionError, match='queries without'): pf_stage3.candidate_retrieval(_candidate(True), 'none+k0', LABELS)


@pytest.mark.parametrize('metric', ['clu', 'map'])
@pytest.mark.parametrize('value', [0.0, -0.1, float('nan'), float('inf')])
def test_seed_ranges_must_be_finite_positive(metric, value):
    with pytest.raises(AssertionError, match='finite and positive'): pf_stage3._check_r(dict(R, **{metric: value}))


def test_score_refuses_wrong_candidate_row_count(tmp_path, monkeypatch):
    # Stop at the next boundary when the guard is missing; no fitting is needed.
    monkeypatch.setattr(pf_stage3, 'candidate_partitions', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('past row guard')))
    with pytest.raises(AssertionError, match='candidate rows'):
        pf_stage3.score(_candidate(True), R, LABELS[:-1], DIST[:-1], {}, PICKS, tmp_path)


@pytest.mark.parametrize('which', ['plain', 'teacher', 'crl'])
def test_comparator_ap_point_must_match_r12(fake_r12, which):
    root, aps = fake_r12
    arm = pf_stage3.comparator_arms(PICKS)[which]
    # CRL agrees across both stages, so only the new provenance check can refuse it.
    for stage in ((2, 3) if which == 'crl' else (2,)):
        _write_ap(root, stage, arm, aps[arm] + 1e-6, np.arange(N_Q, dtype=np.int64))
    with pytest.raises(AssertionError, match='AP point differs'): pf_stage3.load_comparators(PICKS, N_Q, r12=root)


@pytest.mark.parametrize('arm', ['plain', 'teacher_artwork'])
def test_comparator_partition_ami_must_match_r12(fake_r12, arm):
    root, _ = fake_r12
    # Re-seal a different, valid partition: its hash and shape still pass.
    _write_partition(root, arm, np.tile(LABELS, (3, 1)).astype(np.int32))
    with pytest.raises(AssertionError, match='partition AMI differs'): pf_stage3.load_comparators(PICKS, N_Q, r12=root)
