"""Small, data-free C13 join and decision-rule tests."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bs_c13 as c13


def cell(sc, encoder='siglip2', aspect='emotion', task='clu', data='artelingo_held', outcome='WIN'):
    return dict(sc=sc, encoder=encoder, aspect=aspect, task=task, data=data,
                candidate='candidate', comparator='bar', outcome=outcome)


def complete_cells():
    rows = []
    for enc in c13.P.NEW_ENCODERS:
        for a in c13.P.ASPECTS:
            for sc in ('SC1', 'SC2', 'SC3'):
                rows.append(cell(sc, enc, a, 'ret' if sc == 'SC2' else 'clu'))
            for task in ('clu', 'ret'):
                rows.append(cell('SC6', enc, a, task))
        for sc, task in (('SC4', 'clu'), ('SC5', 'ret'), ('SC6', 'clu'), ('SC6', 'ret')):
            rows.append(cell(sc, enc, task=task, data='affection_test'))
    for sc, task in (('SC7', 'clu'), ('SC8', 'ret'), ('SC9', 'clu'), ('SC10', 'ret')):
        rows.append(cell(sc, 'qwen', task=task, data='emoset_test'))
    return rows


def status():
    return {'encoders': {e: {'cstar_runs': 'ok', 'photo_runs': 'ok',
                            'stage0': {'paintings': 'ok', 'photos': 'ok'}} for e in c13.P.NEW_ENCODERS},
            'part_c': {'stage0': 'ok', 'stage2': 'ok'}}


def test_join_reorders_and_preserves_subset_identity():
    x = np.asarray([10, 20, 30])
    assert x[c13.join_rows(['z', 'x', 'y'], ['y', 'z'])].tolist() == [30, 10]


@pytest.mark.parametrize('stored,requested', [(['x', 'x'], ['x']), (['x'], ['x', 'x']), (['x'], ['y'])])
def test_join_rejects_ambiguous_or_missing_keys(stored, requested):
    with pytest.raises(AssertionError):
        c13.join_rows(stored, requested)


def test_hash_order_is_of_full_key_and_not_label_grouped():
    keys = [f'image/{a}/{i}.jpg' for a in ('a', 'b', 'c') for i in range(30)]
    ordered = c13.hashed_order(reversed(keys))
    assert ordered == sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    assert len({k.split('/')[1] for k in ordered[:20]}) == 3
    assert len({k.split('/')[1] for k in ordered[::2]}) == 3


def test_cell_join_uses_every_identifier_and_rejects_duplicates():
    a = cell('SC6')
    b = {**a, 'task': 'ret'}
    assert len(c13.index_cells([a, b])) == 2
    for field in c13.IDENTITY:
        assert c13.cell_id({**a, field: a[field] + '_other'}) != c13.cell_id(a)
    with pytest.raises(AssertionError):
        c13.index_cells([a, deepcopy(a)])


@pytest.mark.parametrize('lower,r,expected', [(1e-15, .1, 'WIN'), (0., .1, 'LEVEL'),
                                           (-.099, .1, 'LEVEL'), (-.1, .1, 'BELOW'),
                                           (-.101, .1, 'BELOW'), (0., 0., 'BELOW')])
def test_strict_unrounded_outcome_boundaries(lower, r, expected):
    assert c13.cell_outcome(lower, r) == expected


def test_all_wins_and_claims():
    result = c13.verdicts(complete_cells())
    assert result['part_c'] == 'holds'
    assert all(v == {'paintings': 'holds', 'photos': 'holds'} for v in result['encoders'].values())
    assert len(result['claims']) == 24 and all(c['claim'] is True for c in result['claims'])


@pytest.mark.parametrize('sc,expected', [('SC1', 'fails'), ('SC2', 'fails'), ('SC3', 'qualified')])
def test_paintings_bars_are_conjunctive(sc, expected):
    rows = complete_cells()
    next(c for c in rows if c['sc'] == sc and c['encoder'] == 'siglip2')['outcome'] = 'LEVEL'
    assert c13.verdicts(rows)['encoders']['siglip2']['paintings'] == expected


@pytest.mark.parametrize('nine,ten,expected', [('WIN', 'WIN', 'holds'), ('WIN', 'LEVEL', 'qualified'),
                                             ('LEVEL', 'WIN', 'qualified'), ('LEVEL', 'LEVEL', 'fails'),
                                             ('WIN', 'BELOW', 'fails'), ('BELOW', 'WIN', 'fails')])
def test_part_c_requires_teacher_win_for_qualified(nine, ten, expected):
    rows = complete_cells()
    for row in rows:
        if row['sc'] in ('SC9', 'SC10'):
            row['outcome'] = nine if row['sc'] == 'SC9' else ten
    assert c13.verdicts(rows)['part_c'] == expected


def test_photo_failure_leaves_paintings_and_other_encoders():
    st = status()
    st['encoders']['siglip2']['photo_runs'] = 'failed'
    rows = [c for c in complete_cells() if not (c['encoder'] == 'siglip2' and c['data'] == 'affection_test')]
    result = c13.verdicts(rows, st)
    assert result['encoders']['siglip2'] == {'paintings': 'holds', 'photos': 'inconclusive'}
    assert result['encoders']['siglip1']['photos'] == 'holds'


def test_gate_failure_and_sc6_inconclusive():
    st = status()
    st['encoders']['siglip2']['stage0']['paintings'] = 'failed'
    st['part_c']['stage2'] = 'failed'
    result = c13.verdicts(complete_cells(), st)
    assert result['part_c'] == 'inconclusive'
    assert all(c['claim'] == 'inconclusive' for c in result['claims'] if c['encoder'] == 'siglip2')


def test_missing_live_cells_fail_and_subset_stays_pending():
    rows = complete_cells()[1:]
    with pytest.raises(AssertionError, match='need 9'):
        c13.verdicts(rows)
    assert c13.verdicts(rows, partial=True)['encoders']['siglip2']['paintings'] == 'pending'


def test_comparison_tolerance_sets_and_exact_decisions():
    result = c13.Comparison()
    result.add('d', .1, .1 + 5e-7, 'd')
    result.add('keys', ['a', 'b'], ['b', 'a'], sets=True)
    result.add('pick', 'center+k10', 'none+k0')
    result.add('order', ['a', 'b'], ['b', 'a'])
    result.add('missing', .2, None, 'range')
    assert len(result.mismatches) == 3
    assert 0 < result.max_diff['d'] < 1e-6


def test_seedwise_crossfit_adapter_matches_frozen_ensemble(monkeypatch):
    rng = np.random.default_rng(42)
    xs = {('emotion', s): c13.ac_data._unit(rng.normal(size=(48, 8)).astype(np.float32))
          for s in c13.SEEDS}
    labels = {'emotion': np.arange(48, dtype=np.int64) % 3}
    original = c13.rt_readout.crossfit_all
    monkeypatch.setattr(c13.rt_readout, 'crossfit_all',
                        lambda x, y, ros, workers=8: original(x, y, ros, workers=1))
    work = c13.Computation(c13.SCRATCH, 50, 100)
    monkeypatch.setattr(work, 'array', lambda name, inputs, fn: np.asarray(fn()))
    actual = work.menu(lambda a, s: xs[a, s], labels, {'emotion': .01}, True)
    expected, pick = c13.ac_select.arm_table(xs, labels, {'emotion': .01}, workers=1)
    assert actual['pick'] == c13.rt_readout.name(pick)
    for ro in expected:
        assert actual['scores'][ro]['map']['emotion'] == expected[ro]['map']['emotion']


def test_stage2_compare_projects_numbers_and_exact_picks():
    derived = {'ranges': {m: {'r': .1, 'per_seed': [.1, .2, .15, .12, .13]} for m in ('clu', 'map')},
               'readouts': {'photo_qwen': {'pick': 'none+k0', 'scores': {
                   'none+k0': {'map': {'emotion': .4}, 'score': 0., 'kept': True}}}},
               'teacher_wording': {'clustering': 'artwork', 'retrieval': 'photo'},
               'wording_scores': {t: {'emotion_focus': .3, 'emotion_focus_photo': .4}
                                  for t in ('clustering', 'retrieval')}}
    record = deepcopy(derived)
    result = c13.Comparison()
    c13.compare_stage2(result, derived, record)
    assert not result.mismatches
    record['teacher_wording']['retrieval'] = 'artwork'
    result = c13.Comparison()
    c13.compare_stage2(result, derived, record)
    assert len(result.mismatches) == 1


def test_written_verdicts_and_individual_claims_compare():
    rows = [{**row, 'r': .1, 'lo': .2} for row in complete_cells()]
    expected = c13.verdicts(rows)
    record = {'verdicts': {k: v for k, v in expected.items() if k != 'claims'},
              'cells': deepcopy(rows), 'claims': deepcopy(expected['claims'])}
    result = c13.Comparison()
    c13.compare_outcomes(result, {'cells': rows, 'partial': False}, record, status())
    assert not result.mismatches
    record['claims'][0]['claim'] = False
    result = c13.Comparison()
    c13.compare_outcomes(result, {'cells': rows, 'partial': False}, record, status())
    assert len(result.mismatches) == 1


def test_missing_input_requires_explicit_failure_on_its_own_path():
    st = status()
    assert not c13.missing_path_excused('artclu:siglip2', None)
    assert not c13.missing_path_excused('artclu:siglip2', st)
    st['encoders']['siglip2']['photo_runs'] = 'failed'
    assert c13.missing_path_excused('affection:siglip2', st)
    assert not c13.missing_path_excused('artclu:siglip2', st)
    assert not c13.missing_path_excused('affection:siglip1', st)
    st['part_c']['stage2'] = 'failed'
    assert c13.missing_path_excused('stage2', st)
