"""Tests for ns_clu (r17 ticket 06): metric table, paired cell (joint bootstrap), ranges.

Verdict code: each test names the logic it guards and fails when that logic is broken. Synthetic data only.
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m pytest -q test_ns_clu.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ns_clu as C  # noqa: E402
import ns_metrics as M  # noqa: E402
import rb_score  # noqa: E402  (r14's scorer: the function that produced the stored numbers)
import rt_stats  # noqa: E402

N, K, DRAWS = 240, 6, 30
METRICS = ns_paths.CLASS_METRICS


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    monkeypatch.setattr(C, 'WORKERS', 1)


def make_labels(rng, n=N, k=K, unlabelled=0.2):
    y = rng.integers(0, k, n)
    y[rng.random(n) < unlabelled] = -1
    return y.astype(np.int64)


def noisy(rng, y, noise, k=K, states=3):
    """[states, n] partitions: the class (random for unlabelled rows) with a share `noise` of rows reassigned."""
    base = np.where(y >= 0, y, rng.integers(0, k, len(y)))
    out = []
    for _ in range(states):
        p = base.copy()
        flip = rng.random(len(y)) < noise
        p[flip] = rng.integers(0, k, int(flip.sum()))
        out.append(p)
    return np.stack(out).astype(np.int32)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    y = make_labels(rng)
    cand = np.stack([noisy(rng, y, 0.15) for _ in range(5)])
    comp5 = np.stack([noisy(rng, y, 0.35) for _ in range(5)])
    comp1 = noisy(rng, y, 0.35)
    own = np.stack([noisy(rng, y, 0.30, states=1) for _ in range(5)])
    return y, cand, comp5, comp1, own


def r14_fixed(cand, comps):
    """rb_score's `fixed[arm][seed] = {reading: [P, n]}` from arrays."""
    fixed = {'C': {s: {'r20': cand[s]} for s in range(5)}}
    for arm, readings in comps.items():
        fixed[arm] = {s: {name: (p[s] if p.ndim == 3 else p) for name, p in readings.items()} for s in range(5)}
    return fixed


def r14_distribution(y, fixed, arms, eligible, seedless, draws):
    """r14's `paired_bootstrap` loop (its draws, its `score_indices`) without its reading-set asserts."""
    point = rb_score.score_indices(fixed, y, np.arange(len(y)), dict(eligible, **{a: [0] for a in seedless}), arms)
    ds = [rb_score.score_indices(fixed, y, rows, seeds, arms)['d']
          for rows, seeds in rb_score.draw_bootstrap(len(y), eligible, arms, draws, seedless)]
    return point, ds


# ------------------------------------------------------------------------------------------------ draw order
def test_draw_transcript_equals_r14_seeded_comparator():
    """Guards: rows, then c*'s seeds, then the comparator's seeds, from one default_rng(0) (r14's order). A swap of the
    seed and row draws, or of the two arms' seeds, changes the transcript."""
    ours = list(C.draws_iter(N, 5, True, 12))
    theirs = rb_score.draw_bootstrap(N, {'C': list(range(5)), 'M': list(range(5))}, ('C', 'M'), 12)
    for (rows, cs, ms), (rows14, seeds14) in zip(ours, theirs):
        assert np.array_equal(rows, rows14) and np.array_equal(cs, seeds14['C']) and np.array_equal(ms, seeds14['M'])


def test_unseeded_comparator_draws_rows_and_cand_seeds_only():
    """Guards: an unseeded comparator consumes nothing from the stream (r14's `seedless` arm)."""
    ours = list(C.draws_iter(N, 5, False, 12))
    theirs = rb_score.draw_bootstrap(N, {'C': list(range(5))}, ('C', 'F'), 12, seedless=('F',))
    for (rows, cs, ms), (rows14, seeds14) in zip(ours, theirs):
        assert ms is None and np.array_equal(rows, rows14) and np.array_equal(cs, seeds14['C'])
    rng = np.random.default_rng(0)                      # by hand: rows then seeds, nothing else
    rows = rng.integers(0, N, size=N)
    assert np.array_equal(ours[0][0], rows)
    assert np.array_equal(ours[0][1], rng.choice(list(range(5)), size=5, replace=True))
    assert np.array_equal(ours[1][0], rng.integers(0, N, size=N))


def test_paired_cell_uses_the_transcript_once_for_all_metrics(data, monkeypatch):
    """Guards: all metrics of a cell read the same draws (one transcript per cell, not one per metric)."""
    y, cand, comp5, _, _ = data
    calls = []
    real = C.draws_iter
    monkeypatch.setattr(C, 'draws_iter', lambda *a, **kw: (calls.append(a), real(*a, **kw))[1])
    C.paired_cell(cand, comp5, y, K, draws=DRAWS)
    assert len(calls) == 1


# ------------------------------------------------------------------------------------------------ equals r14
def test_ami_equals_r14_unseeded_comparator(data):
    """Guards: AMI value, d and interval equal r14's `paired_bootstrap` with a seedless arm (SC1/SC4/SC6 form), with
    unlabelled rows drawn and masked after indexing."""
    y, cand, _, comp1, _ = data
    assert (y < 0).any()
    cell = C.paired_cell(cand, comp1, y, K, draws=DRAWS)['ami']
    fixed = r14_fixed(cand, {})
    fixed['F'] = {0: {'r20': comp1}}
    point, ds = r14_distribution(y, fixed, ('C', 'F'), {'C': list(range(5))}, ('F',), DRAWS)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    assert abs(cell['cand'] - point['A_cand']) < 1e-12 and abs(cell['comp'] - point['A_e']['F-r20']) < 1e-12
    assert abs(cell['d'] - point['d']) < 1e-12
    assert abs(cell['lo'] - lo) < 1e-12 and abs(cell['hi'] - hi) < 1e-12


def test_ami_equals_r14_seeded_comparator_one_reading(data):
    """Guards: the one-reading seeded path equals r14's `b_matched` with the own reading removed, on the same draws (the
    matched arm's five seeds are drawn in the same slot)."""
    y, cand, comp5, _, _ = data
    cell = C.paired_cell(cand, comp5, y, K, draws=DRAWS)['ami']
    fixed = r14_fixed(cand, {'M': {'r20': comp5}})
    point, ds = r14_distribution(y, fixed, ('C', 'M'), {'C': list(range(5)), 'M': list(range(5))}, (), DRAWS)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    assert abs(cell['d'] - point['d']) < 1e-12 and abs(cell['comp'] - point['A_e']['M-r20']) < 1e-12
    assert abs(cell['lo'] - lo) < 1e-12 and abs(cell['hi'] - hi) < 1e-12


def test_two_reading_path_equals_b_matched_and_own_matters(data):
    """Guards: the two-reading comparator (larger reading re-taken per draw, one seed draw) equals r14's `b_matched`; and
    the one-reading result differs from it when the own reading is the larger one (the test would otherwise be blind)."""
    y, cand, comp5, _, own = data
    fixed = r14_fixed(cand, {'M': {'r20': comp5, 'own': own}})
    arms, elig = ('C', 'M'), {'C': list(range(5)), 'M': list(range(5))}
    point, ds = r14_distribution(y, fixed, arms, elig, (), DRAWS)
    k_own = int(max(comp5.max(), own.max())) + 1
    res = C._boot(cand.astype(np.int64), {'r20': comp5, 'own': own}, y, k_own, ('ami',), True, DRAWS, 0, 1)
    assert abs(res['cand'][0] - point['A_cand']) < 1e-12
    assert abs(res['comp'][0] - max(point['A_e'].values())) < 1e-12
    assert res['reading'] == [max(point['A_e'], key=point['A_e'].get).replace('M-', '')]
    assert np.abs(res['d_draws'][:, 0] - np.array(ds)).max() < 1e-12
    one = C._boot(cand.astype(np.int64), {'r20': comp5}, y, k_own, ('ami',), True, DRAWS, 0, 1)
    assert np.abs(one['d_draws'] - res['d_draws']).max() > 1e-6      # own reading changes d: the comparison is not vacuous
    # the one-reading draws are b_matched's draws: same d as r14 with the own reading removed
    f1 = r14_fixed(cand, {'M': {'r20': comp5}})
    _, ds1 = r14_distribution(y, f1, arms, elig, (), DRAWS)
    assert np.abs(one['d_draws'][:, 0] - np.array(ds1)).max() < 1e-12


# ------------------------------------------------------------------------------------------------ metrics per draw
def test_every_metric_equals_array_forms_on_a_hand_built_draw(data):
    """Guards: contingency forms per draw equal the array forms on rows indexed with repeats then masked, for all four
    metrics (state mean, then drawn-seed mean, d = cand - comp), on draw 0 and on the point."""
    y, cand, comp5, _, _ = data
    rows, cs, ms = next(C.draws_iter(N, 5, True, 1))
    yy = y[rows]
    keep = yy >= 0

    res = C._boot(cand.astype(np.int64), {'comp': comp5}, y, K, METRICS, True, 1, 0, 1)
    for j, m in enumerate(METRICS):
        want = np.mean([np.mean([getattr(M, m)(yy[keep], p[rows][keep]) for p in cand[s]]) for s in cs]) \
            - np.mean([np.mean([getattr(M, m)(yy[keep], p[rows][keep]) for p in comp5[s]]) for s in ms])
        assert abs(res['d_draws'][0, j] - want) < 1e-12, m
        ym = y[y >= 0]
        pt = np.mean([np.mean([getattr(M, m)(ym, p[y >= 0]) for p in cand[s]]) for s in range(5)])
        assert abs(res['cand'][j] - pt) < 1e-12, m


def test_masking_happens_after_indexing_not_before(data):
    """Guards: rows are drawn over all n rows (unlabelled included): with the same labels, the draws of a data set with
    unlabelled rows differ from those over its labelled rows alone (r14 trap: drawing among labelled rows only)."""
    y, cand, comp5, _, _ = data
    full = C._boot(cand.astype(np.int64), {'c': comp5}, y, K, ('ami',), True, 10, 0, 1)
    lab = y >= 0
    sub = C._boot(cand[:, :, lab].astype(np.int64), {'c': comp5[:, :, lab]}, y[lab], K, ('ami',), True, 10, 0, 1)
    assert abs(full['cand'][0] - sub['cand'][0]) < 1e-12              # the point does not care
    assert np.abs(full['d_draws'] - sub['d_draws']).max() > 1e-4      # the draws do


def test_parallel_equals_serial(data, monkeypatch):
    y, cand, comp5, comp1, _ = data
    one = C.paired_cell(cand, comp5, y, K, draws=60)
    monkeypatch.setattr(C, 'WORKERS', 3)
    many = C.paired_cell(cand, comp5, y, K, draws=60)
    for m in METRICS:
        for key in ('cand', 'comp', 'd', 'lo', 'hi'):
            assert one[m][key] == many[m][key], (m, key)


# ------------------------------------------------------------------------------------------------ cell contract
def test_cell_fields_and_outcome(data):
    y, cand, comp5, _, _ = data
    cell = C.paired_cell(cand, comp5, y, K, draws=DRAWS)
    assert set(cell) == set(METRICS)
    assert set(cell['ami']) == {'cand', 'comp', 'd', 'lo', 'hi', 'r', 'outcome', 'n_draws'}
    assert cell['ami']['outcome'] is None and cell['ami']['r'] is None and cell['ami']['n_draws'] == DRAWS
    lo = cell['ari']['lo']
    for r, want in (({m: 1e-9 for m in METRICS}, rt_stats.outcome(lo, 1e-9)), ({m: abs(lo) + 1 for m in METRICS}, 'LEVEL')):
        got = C.paired_cell(cand, comp5, y, K, r=r, draws=DRAWS)
        for m in METRICS:
            assert got[m]['outcome'] == rt_stats.outcome(got[m]['lo'], r[m]) and got[m]['r'] == r[m]
        if lo <= 0:
            assert got['ari']['outcome'] == want
    swapped = C.paired_cell(comp5, cand, y, K, r={m: 0.01 for m in METRICS}, draws=DRAWS)
    assert swapped['ami']['d'] < 0 and swapped['ami']['outcome'] == 'BELOW'      # a weaker candidate is BELOW


def test_metric_subset_gives_same_numbers(data):
    y, cand, comp5, _, _ = data
    full = C.paired_cell(cand, comp5, y, K, draws=DRAWS)
    part = C.paired_cell(cand, comp5, y, K, metrics=('acc', 'ari'), draws=DRAWS)
    assert set(part) == {'acc', 'ari'}
    for m in part:
        assert part[m] == full[m]


def test_rng_seed_changes_draws(data):
    y, cand, comp5, _, _ = data
    a = C.paired_cell(cand, comp5, y, K, metrics=('ami',), draws=DRAWS)['ami']
    b = C.paired_cell(cand, comp5, y, K, metrics=('ami',), draws=DRAWS, rng_seed=1)['ami']
    assert a['d'] == b['d'] and (a['lo'], a['hi']) != (b['lo'], b['hi'])


def test_input_checks(data):
    y, cand, comp5, comp1, _ = data
    with pytest.raises(AssertionError):
        C.paired_cell(cand[:4], comp1, y, K, draws=5)                  # four seeds
    with pytest.raises(AssertionError):
        C.paired_cell(comp1, comp1, y, K, draws=5)                     # unseeded candidate
    with pytest.raises(AssertionError):
        C.paired_cell(cand, comp5[:, :, :-1], y, K, draws=5)           # row count
    with pytest.raises(AssertionError):
        C.paired_cell(cand, comp5, y, K - 1, draws=5)                  # cluster id outside [0, k)
    with pytest.raises(AssertionError):
        C.paired_cell(cand.astype(float), comp5, y, K, draws=5)        # float partitions
    with pytest.raises(AssertionError):
        C.paired_cell(cand, comp5, y, K, metrics=('map',), draws=5)
    with pytest.raises(AssertionError):
        C.paired_cell(cand, comp5, y, K, r={'ami': 0.1}, draws=5)      # r lacks metrics


# ------------------------------------------------------------------------------------------------ table and ranges
def test_metric_table_unseeded_and_seeded(data):
    y, cand, _, comp1, _ = data
    t = C.metric_table('plain', 'emotion', comp1, y, K)
    assert [r['metric'] for r in t] == list(METRICS) and all(r['k'] == K and r['arm'] == 'plain' for r in t)
    keep = y >= 0
    for r in t:
        want = np.mean([getattr(M, r['metric'])(y[keep], p[keep]) for p in comp1])
        assert r['per_seed'] == [pytest.approx(want, abs=1e-15)] and abs(r['mean'] - want) < 1e-15
    t5 = C.metric_table('cstar', 'emotion', cand, y, K, metrics=('ami',))
    per = [np.mean([M.ami(y[keep], p[keep]) for p in cand[s]]) for s in range(5)]
    assert len(t5) == 1 and np.allclose(t5[0]['per_seed'], per, atol=1e-15) and abs(t5[0]['mean'] - np.mean(per)) < 1e-15
    with pytest.raises(AssertionError):
        C.metric_table('x', 'emotion', cand, y, K - 2)


def test_ranges(data):
    y, cand, _, _, _ = data
    out = C.ranges(cand, y, K)
    keep = y >= 0
    for m in METRICS:
        vals = [np.mean([getattr(M, m)(y[keep], p[keep]) for p in cand[s]]) for s in range(5)]
        assert abs(out[m]['r'] - (max(vals) - min(vals))) < 1e-15 and out[m]['r'] > 0
        assert [out[m]['per_seed'][str(s)] for s in range(5)] == pytest.approx(vals, abs=1e-15)
    as_dict = C.ranges({s: cand[s] for s in range(5)}, y, K, metrics=('ami',))
    assert as_dict['ami'] == out['ami']
    with pytest.raises(AssertionError):
        C.ranges({s: cand[s] for s in range(4)}, y, K)
