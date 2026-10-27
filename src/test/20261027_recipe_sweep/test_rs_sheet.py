"""The number sheet equals the spec's table (spec §3.2 at the rule's spec commit); names round-trip."""
import subprocess

import pytest

import rs_paths
import rs_sheet as S


def spec_rows():
    text = subprocess.run(['git', 'show', f'{rs_paths.RULE_COMMIT}:docs/superpowers/specs/2026-10-27-recipe-sweep-design.md'],
                          cwd=rs_paths.ROUND, capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 4 and cells[0] in S.KNOBS:
            rows[cells[0]] = (cells[2], tuple(v.strip() for v in cells[3].split(',')))
    return rows


def test_sheet_equals_spec():
    rows = spec_rows()
    assert set(rows) == set(S.KNOBS)
    for k, (g, values) in rows.items():
        expected_g = 0 if g == 'off' else float(g)
        assert S.G[k] == expected_g, k
        assert S.VALUES[k] == tuple(float(v) for v in values), k


def test_stage1_count_and_order():
    runs = S.stage1()
    assert len(runs) == 39
    assert [S.changed(c) for c in runs[:3]] == [('k1',)] * 3
    assert S.name(runs[0]) == 'k1=2' and S.name(runs[-1]) == 'k13=30'


@pytest.mark.parametrize('cfg', [S.config(), *S.stage1(), S.config({'k2': 8, 'k6': 5e-4}),
                                 *[S.config(remove=[r]) for r in S.REMOVALS],
                                 S.config({'k6': 5e-4}, ['buddy'])])
def test_names_round_trip(cfg):
    assert S.parse(S.name(cfg)) == cfg


def test_tie_keys():
    c = S.config({'k6': 2e-3, 'k2': 2})
    assert S.name(c) == 'k2=2+k6=0.002'
    assert S.knob_numbers(c) == (2, 6) and S.positions(c) == (0, 2)
    assert S.name(S.config()) == 'G' and S.changed(S.config()) == ()


def test_types():
    c = S.config({'k8': 1024.0, 'k7': 0})
    assert isinstance(c['k8'], int) and isinstance(c['k7'], float)
    with pytest.raises(AssertionError):
        S.config({'k8': 1000.5})
