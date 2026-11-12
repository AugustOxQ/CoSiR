"""Tests for r28_settings: the six-cell comparison, the tie rule, the self-comparison, the paired bootstrap on a tiny
fixture, the export paths. Synthetic data only.
Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_r28_settings.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r28_settings as S  # noqa: E402


def cells(vals):
    assert len(vals) == 6
    return dict(zip(S.CELLS, vals))


REF = cells([0.10, 0.45, 0.52, 0.27, 0.35, 0.60])


def test_cstar_against_itself_is_zero_and_wins_the_tie():
    c = S.compare(REF, dict(REF))
    assert all(v == 0.0 for v in c['diff'].values()) and c['mean_diff'] == 0.0
    assert not c['row_wins'] and S.outcome('k11=80', c) == 'c* wins or ties'
    assert S.overall({'k11=80': c}) == dict(outcome='c* wins or ties', winners=[], best=None)


def test_row_wins_on_the_mean_not_on_cells():
    row = cells([0.20, 0.45, 0.52, 0.27, 0.35, 0.50])       # +0.10 on one cell, -0.10 on another: mean ties
    assert not S.compare(REF, row)['row_wins']
    row = cells([0.20, 0.45, 0.52, 0.27, 0.35, 0.55])       # mean +0.05/6
    c = S.compare(REF, row)
    assert c['row_wins'] and c['mean_diff'] == pytest.approx(0.05 / 6)
    assert S.outcome('k11=10', c) == 'k11=10 wins'


def test_tie_within_tolerance_goes_to_cstar():
    row = {k: v + 1e-13 for k, v in REF.items()}
    assert not S.compare(REF, row)['row_wins']
    row = {k: v + 1e-9 for k, v in REF.items()}
    assert S.compare(REF, row)['row_wins']


def test_overall_names_the_best_winner():
    r = {'k11=10': S.compare(REF, {k: v + 0.01 for k, v in REF.items()}),
         'k11=20': S.compare(REF, {k: v + 0.03 for k, v in REF.items()}),
         'k11=80': S.compare(REF, {k: v - 0.01 for k, v in REF.items()})}
    o = S.overall(r)
    assert o['outcome'] == 'k11=20 wins' and o['winners'] == ['k11=10', 'k11=20']


def test_six_mean_needs_exactly_the_six_cells():
    with pytest.raises(AssertionError):
        S.six_mean({'ami/emotion': 0.1})
    assert S.six_mean(REF) == pytest.approx(np.mean(list(REF.values())))


def _fixture(n=120, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    y = {a: rng.integers(0, 3, size=n).astype(np.int64) for a in S.ASPECTS}
    y['genre'][::5] = -1
    parts, ap = {}, {}
    for a in S.ASPECTS:
        noise = rng.random((3, n)) < (0.3 + shift)
        parts[a] = np.where(noise, rng.integers(0, 3, size=(3, n)), y[a][None]).clip(0)
        ap[a] = rng.random(n)
    return dict(parts=parts, ap=ap, labels=y)


def test_bootstrap_tiny_fixture_is_paired_and_deterministic():
    a, b = _fixture(), _fixture()
    same = S.boot_diff(a, a, 20)
    assert same['mean_diff'] == 0.0 and same['lo'] == 0.0 and same['hi'] == 0.0     # identical arms: every draw is 0
    r1 = S.boot_diff(a, b, 20)
    assert r1 == S.boot_diff(a, b, 20)                                               # default_rng(0)
    assert r1['lo'] <= r1['mean_diff'] <= r1['hi'] and r1['n_draws'] == 20
    # an arm with every AP raised by 0.1 and the same partitions: the six-cell mean rises by 0.1 / 2 in every draw
    c = dict(b, ap={k: v + 0.1 for k, v in b['ap'].items()})
    d = S.boot_diff(b, c, 20)
    assert d['lo'] == pytest.approx(0.05) and d['hi'] == pytest.approx(0.05)


def test_bootstrap_ignores_non_query_rows_in_ap():
    a = _fixture()
    a2 = dict(a, ap={k: v.copy() for k, v in a['ap'].items()})
    for k in a2['ap']:
        a2['ap'][k][::7] = np.nan
    r = S.boot_diff(a2, a2, 5)
    assert np.isfinite([r['lo'], r['hi'], r['mean_diff']]).all()


def test_export_path_is_found_by_glob_with_one_match(tmp_path):
    d = tmp_path / 'train/k6=0.0005/R6-5'
    d.mkdir(parents=True)
    (d / 'genre__R0o__s0.npz').write_bytes(b'')
    (d / 'genre__R0o__s1.npz').write_bytes(b'')
    p, g = S.export_path('k6=0.0005', 'genre', 0, r28_root=tmp_path)
    assert p == d / 'genre__R0o__s0.npz' and g == 'R0o'
    (d / 'genre__Dview__s0.npz').write_bytes(b'')       # a second graph row for the same aspect and seed: ambiguous
    with pytest.raises(AssertionError):
        S.export_path('k6=0.0005', 'genre', 0, r28_root=tmp_path)
    with pytest.raises(AssertionError):
        S.export_path('k6=0.0005', 'style', 0, r28_root=tmp_path)    # no file


def test_export_dir_c_star_from_r14_rows_from_r28():
    assert S.export_dir('k11=40') == S.R14_ROOT / 'train/k11=40/R6-5'
    assert S.export_dir('k11=10') == S.R28_ROOT / 'train/k11=10/R6-5'
