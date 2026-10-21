"""The ablation table is a set of explicit edits to the frozen D2 recipe."""
import copy

import pytest

import ab_rows
from ab_paths import ms_rows


def test_complete_table_and_reference_rows():
    assert list(ab_rows.ROWS) == [f'R{i}' for i in range(13)]
    assert ab_rows.ROWS['R0']['base'] == ms_rows.ROWS['D2']
    assert ab_rows.ROWS['R2']['base'] == ms_rows.ROWS['T1']
    for row in ab_rows.ROWS.values():
        assert ab_rows.validate(row) == row
        assert ms_rows.validate(row['base']) == row['base']


@pytest.mark.parametrize('name,changes', [
    ('R0', {}),
    ('R1', {'base.share': 'separate'}),
    ('R2', {'base.distill': 'none', 'base.distill_weight': 0.0}),
    ('R3', {'switches.teacher': 'default'}),
    ('R4', {'base.graphs': None, 'base.graph_terms': False}),
    ('R5', {'switches.buddy': 0.0}),
    ('R6', {'switches.own': 0.0}),
    ('R7', {'switches.endpoints': 'image'}),
    ('R8', {'base.graphs': dict(emotion='R0', style='R0', genre='R0o')}),
    ('R9', {'base.graphs': dict.fromkeys(ms_rows.ASPECTS, 'Dview'), 'store': 'ab'}),
    ('R10', {'base.graphs': dict.fromkeys(ms_rows.ASPECTS, 'rand_s{seed}'), 'store': 'ab'}),
    ('R11', {'base.graphs': dict.fromkeys(ms_rows.ASPECTS, 'Dview'), 'store': 'ab',
             'base.distill': 'none', 'base.distill_weight': 0.0}),
    ('R12', {'base.inputs': 'instructed'}),
])
def test_exact_row_edit(name, changes):
    expected = copy.deepcopy(ab_rows.ROWS['R0'])
    for key, value in changes.items():
        target = expected
        parts = key.split('.')
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    assert ab_rows.ROWS[name] == expected


def test_seed_resolution_does_not_mutate_table():
    before = copy.deepcopy(ab_rows.ROWS)
    for seed in range(3):
        resolved = ab_rows.resolve('R10', seed)
        assert set(resolved['base']['graphs'].values()) == {f'rand_s{seed}'}
    assert ab_rows.ROWS == before


@pytest.mark.parametrize('key,value', [('buddy', 0.5), ('own', -1),
                                      ('teacher', 'caption'), ('endpoints', 'caption')])
def test_invalid_switch_rejected(key, value):
    row = copy.deepcopy(ab_rows.ROWS['R0'])
    row['switches'][key] = value
    with pytest.raises(AssertionError):
        ab_rows.validate(row)
