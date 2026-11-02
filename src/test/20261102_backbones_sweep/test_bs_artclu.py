"""Unit tests of bs_artclu's statistic paths on small synthetic partitions (ticket 05). Each expected value is a hand-written
bootstrap that follows the spec's draw order with `default_rng(0)`; a wrapper that changes the draws, the seed handling or the
reading choice gives other numbers."""
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score as ami

import bs_artclu as A

N, K, B = 240, 6, 60
SEEDS = list(range(5))


def _labels(rng, masked=0):
    y = rng.integers(0, K, N)
    y[:masked] = -1
    return y


def _noisy(rng, y, noise, sets=3):
    """`sets` partitions: the labels with a share `noise` of the rows moved to a random cluster."""
    base = np.where(y < 0, rng.integers(0, K, N), y)
    out = []
    for _ in range(sets):
        p = base.copy()
        move = rng.random(N) < noise
        p[move] = rng.integers(0, K, int(move.sum()))
        out.append(p)
    return np.array(out, dtype=np.int64)


def _fixed(rng, y, noise, readings=('r20',), seeds=SEEDS):
    out = {}
    for s in seeds:
        out[s] = {r: _noisy(rng, y, noise, 1 if r == 'own' else 3) for r in readings}
    return out


def _score(fixed_seed_list, y, rows, reading):
    """Mean over drawn seeds of the mean over a set's partitions of the AMI on rows `rows` (labels < 0 masked)."""
    yy = y[rows]
    m = yy >= 0
    return float(np.mean([np.mean([ami(yy[m], p[rows][m]) for p in fs[reading]]) for fs in fixed_seed_list]))


def _interval(diffs):
    return np.percentile(diffs, [2.5, 97.5]).tolist()


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(123)
    y = _labels(rng, masked=17)
    return dict(y=y, cand=_fixed(rng, y, 0.45), frozen=_fixed(rng, y, 0.6, seeds=[0]),
                matched=_fixed(rng, y, 0.5, ('r20', 'own')), teacher=_fixed(rng, y, 0.55, seeds=[0]))


# ------------------------------------------------------------------------------------------------ seed ranges
def test_seed_range_is_unrounded_max_minus_min():
    per_seed = {0: 0.1000001, 1: 0.1049, 2: 0.1031, 3: 0.1, 4: 0.1020}
    assert A.seed_range(per_seed) == pytest.approx(0.1049 - 0.1, abs=0, rel=0)
    assert A.seed_range(per_seed) != round(0.1049 - 0.1, 4)


def test_seed_range_needs_seeds_0_to_4():
    with pytest.raises(AssertionError):
        A.seed_range({0: 0.1, 1: 0.2, 2: 0.3})


# ------------------------------------------------------------------------------------------------ SC1
def test_frozen_margin_resamples_paintings_only(data):
    """Hand bootstrap: per draw I_b, then S_b for X-c* only. A frozen image that also drew a seed would shift the stream."""
    y = data['y']
    res = A.frozen_margin(data['cand'], data['frozen'], y, B, 1, 'C', 'F')
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(B):
        rows = rng.integers(0, N, size=N)
        s = rng.choice(SEEDS, size=5, replace=True)
        diffs.append(_score([data['cand'][int(i)] for i in s], y, rows, 'r20') -
                     _score([data['frozen'][0]], y, rows, 'r20'))
    assert res['interval'] == pytest.approx(_interval(diffs), abs=1e-12, rel=0)
    rows0 = np.arange(N)
    point = _score([data['cand'][s] for s in SEEDS], y, rows0, 'r20') - _score([data['frozen'][0]], y, rows0, 'r20')
    assert res['d'] == pytest.approx(point, abs=1e-12, rel=0)
    assert res['seedless'] == ['F']


def test_frozen_margin_uses_the_frozen_arms_single_reading(data):
    with pytest.raises(AssertionError):
        A.frozen_margin(data['cand'], data['matched'], data['y'], 4, 1)      # 'own' reading present: not a frozen image


# ------------------------------------------------------------------------------------------------ SC3
def _hand_matched(data, shared_seeds=False, fixed_max=False):
    y = data['y']
    rng = np.random.default_rng(0)
    r20_all = _score([data['matched'][s] for s in SEEDS], y, np.arange(N), 'r20')
    own_all = _score([data['matched'][s] for s in SEEDS], y, np.arange(N), 'own')
    point_reading = 'r20' if r20_all >= own_all else 'own'
    diffs, flips = [], 0
    for _ in range(B):
        rows = rng.integers(0, N, size=N)
        s_c = rng.choice(SEEDS, size=5, replace=True)
        s_m = s_c if shared_seeds else rng.choice(SEEDS, size=5, replace=True)
        a = _score([data['cand'][int(i)] for i in s_c], y, rows, 'r20')
        m = {r: _score([data['matched'][int(i)] for i in s_m], y, rows, r) for r in ('r20', 'own')}
        best = point_reading if fixed_max else max(m, key=m.get)
        flips += best != point_reading
        diffs.append(a - m[best])
    return diffs, flips, point_reading, r20_all, own_all


def test_matched_margin_is_10_26s_b_matched(data):
    res = A.matched_margin(data['cand'], data['matched'], data['y'], B, 1, 'C', 'M')
    diffs, _, reading, r20_all, own_all = _hand_matched(data)
    assert res['interval'] == pytest.approx(_interval(diffs), abs=1e-12, rel=0)
    assert res['max_entry'] == f'M-{reading}'
    a_all = _score([data['cand'][s] for s in SEEDS], data['y'], np.arange(N), 'r20')
    assert res['d'] == pytest.approx(a_all - max(r20_all, own_all), abs=1e-12, rel=0)
    assert res['A_e'] == {'M-r20': pytest.approx(r20_all, abs=1e-12), 'M-own': pytest.approx(own_all, abs=1e-12)}


def test_matched_margin_seeds_are_drawn_independently_per_arm(data):
    res = A.matched_margin(data['cand'], data['matched'], data['y'], B, 1)
    shared, _, _, _, _ = _hand_matched(data, shared_seeds=True)
    independent, _, _, _, _ = _hand_matched(data)
    assert _interval(shared) != pytest.approx(_interval(independent), abs=1e-9)    # the test can tell them apart
    assert res['interval'] == pytest.approx(_interval(independent), abs=1e-12, rel=0)
    assert res['interval'] != pytest.approx(_interval(shared), abs=1e-9)


def test_matched_margin_takes_the_larger_reading_again_in_each_draw():
    """Data where the two readings are close, so the larger one changes between draws; a reading fixed from the point
    values gives a different interval."""
    rng = np.random.default_rng(7)
    y = _labels(rng)
    d = dict(y=y, cand=_fixed(rng, y, 0.40), matched=None)
    d['matched'] = {s: {'r20': _noisy(rng, y, 0.50, 3), 'own': _noisy(rng, y, 0.50, 1)} for s in SEEDS}
    res = A.matched_margin(d['cand'], d['matched'], y, B, 1)
    retaken, flips, _, _, _ = _hand_matched(d)
    fixed, _, _, _, _ = _hand_matched(d, fixed_max=True)
    assert flips > 0, 'the synthetic readings must cross over in some draws'
    assert _interval(retaken) != pytest.approx(_interval(fixed), abs=1e-9)
    assert res['interval'] == pytest.approx(_interval(retaken), abs=1e-12, rel=0)


def test_matched_margin_needs_both_readings_and_five_seeds(data):
    one_reading = {s: {'r20': data['matched'][s]['r20']} for s in SEEDS}
    with pytest.raises(AssertionError):
        A.matched_margin(data['cand'], one_reading, data['y'], 4, 1)
    with pytest.raises(AssertionError):
        A.matched_margin({s: data['cand'][s] for s in (0, 1, 2)}, data['matched'], data['y'], 4, 1)


# ------------------------------------------------------------------------------------------------ SC6
def test_teacher_margin_draws_seeds_for_x_c_only(data):
    y = data['y']
    res = A.teacher_margin(data['cand'], data['teacher'], y, B, 1)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(B):
        rows = rng.integers(0, N, size=N)
        s = rng.choice(SEEDS, size=5, replace=True)
        diffs.append(_score([data['cand'][int(i)] for i in s], y, rows, 'r20') -
                     _score([data['teacher'][0]], y, rows, 'r20'))
    assert res['interval'] == pytest.approx(_interval(diffs), abs=1e-12, rel=0)
    assert res['A_cand'] - res['A_T'] == pytest.approx(res['d'], abs=1e-12)


def test_teacher_and_frozen_margins_share_one_draw_transcript(data):
    """SC1 and SC6 both draw I_b then S_b for X-c*: against the same comparator their intervals agree."""
    y = data['y']
    t = A.teacher_margin(data['cand'], data['teacher'], y, B, 1)
    f = A.frozen_margin(data['cand'], data['teacher'], y, B, 1)
    assert f['interval'] == pytest.approx(t['interval'], abs=1e-12, rel=0)
    assert f['d'] == pytest.approx(t['d'], abs=1e-12, rel=0)


# ------------------------------------------------------------------------------------------------ cells
def _margins(data, matched='M2', frozen='F2'):
    y = data['y']
    return dict(frozen=A.frozen_margin(data['cand'], data['frozen'], y, 8, 1, 'C', frozen),
                matched=A.matched_margin(data['cand'], data['matched'], y, 8, 1, 'C', matched),
                teacher=A.teacher_margin(data['cand'], data['teacher'], y, 8, 1))


def test_cells_follow_the_index_format_and_units(data):
    ranges = {'emotion': 0.00531, 'style': 0.00712, 'genre': 0.01499}
    cells = A.aspect_cells('siglip2', 'genre', 'C', _margins(data), ranges, 8, 'M2', 'F2')
    assert [c['sc'] for c in cells] == ['SC1', 'SC3', 'SC6']
    for c in cells:
        assert set(A.CELL_KEYS) <= set(c) and c['task'] == 'clu' and c['seeds'] == SEEDS
        assert c['encoder'] == 'siglip2' and c['aspect'] == 'genre' and c['n_draws'] == 8
    sc1, sc3, sc6 = cells
    assert sc1['r'] is None and sc3['r'] is None                      # WIN-only bars carry no unit
    assert sc6['r'] == 0.01499                                        # X-c*'s own range decides, unrounded
    assert sc6['r_glossary'] == 0.0128                                # the glossary's unit beside it
    assert 'validation' in sc6['r_source']
    assert sc1['comparator'] == 'frozen_siglip2' and sc3['comparator'] == 'matched_M2'
    assert sc3['comparator_reading'] in ('M2-r20', 'M2-own')
    assert sc3['comparator_value'] == sc3['comparator_readings'][sc3['comparator_reading']]
    assert sc3['comparator_value'] == max(sc3['comparator_readings'].values())


def test_sc6_cell_needs_a_unit_and_win_cells_must_not_have_one():
    base = dict(encoder='siglip2', aspect='emotion', candidate='c', comparator='t', candidate_value=0.1,
                comparator_value=0.09, d=0.01, lo=0.0, hi=0.02, n_draws=8)
    with pytest.raises(AssertionError):
        A.make_cell('SC6', r=None, r_source=None, **base)
    with pytest.raises(AssertionError):
        A.make_cell('SC1', r=0.005, r_source='x', **base)


def test_write_cells_replaces_only_the_named_encoders(tmp_path, data):
    ranges = {a: 0.01 for a in A.ASPECTS}
    path = tmp_path / 'stage3' / A.CELLS_NAME
    c2 = A.aspect_cells('siglip2', 'genre', 'C', _margins(data), ranges, 8, 'M2', 'F2')
    c8 = A.aspect_cells('siglip1', 'genre', 'C', _margins(data, 'M8', 'F8'), ranges, 8, 'M8', 'F8')
    A.write_cells(path, c2, ['siglip2'])
    A.write_cells(path, c8, ['siglip1'])
    again = [dict(c, d=c['d'] + 1) for c in c2]
    out = A.write_cells(path, again, ['siglip2'])
    saved = json.loads(path.read_text())
    assert saved == out and len(saved) == 6
    assert {c['encoder'] for c in saved} == {'siglip2', 'siglip1'}
    assert [c['d'] for c in saved if c['encoder'] == 'siglip1'] == [c['d'] for c in c8]


# ------------------------------------------------------------------------------------------------ validation ranges
def test_validation_ami_masks_unlabelled_rows_and_averages_three_states():
    import ct_score
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, 300)
    y[:40] = -1
    x = rng.normal(size=(300, 16)).astype(np.float32)
    x += 3 * np.eye(4, 16, dtype=np.float32)[np.where(y < 0, 0, y)]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ps = ct_score.output_partitions(x, 'k1')['r20']
    assert ps.shape == (3, 300)
    want = np.mean([ami(y[y >= 0], p[y >= 0]) for p in ps])
    assert A.validation_ami(x, y) == pytest.approx(want, abs=1e-12, rel=0)


def test_clustering_ranges_are_max_minus_min_per_aspect(monkeypatch, tmp_path):
    per = {a: {s: 0.1 + 0.001 * ((s * 7 + len(a)) % 5) for s in A.SEEDS} for a in A.ASPECTS}
    monkeypatch.setattr(A.rt_data, 'names', lambda split: ['p%d' % i for i in range(3)])
    monkeypatch.setattr(A.rt_data, 'labels', lambda split: {a: np.array([0, 1, -1]) for a in A.ASPECTS})
    seen = {}

    def fake_cstar(split, aspect, seed, names):
        seen[(aspect, seed)] = str(A.rt_data.CSTAR_ROOT)
        return np.full((3, 4), seed, dtype=np.float16)
    monkeypatch.setattr(A.rt_data, '_cstar', fake_cstar)
    calls = iter([per[a][s] for a in A.ASPECTS for s in A.SEEDS])
    monkeypatch.setattr(A, 'validation_ami', lambda x, y: next(calls))
    monkeypatch.setattr(A, 'sha_file', lambda p: 'sha')
    old = A.rt_data.CSTAR_ROOT
    res = A.clustering_ranges(tmp_path)
    assert A.rt_data.CSTAR_ROOT == old                                  # the reader's root is restored
    assert set(seen.values()) == {str(tmp_path)}                        # and was X-c*'s root while reading
    for a in A.ASPECTS:
        vals = [per[a][s] for s in A.SEEDS]
        assert res['ranges'][a] == max(vals) - min(vals)
        assert res['per_seed'][a] == per[a]


def test_read_ranges_checks_encoder_and_aspects(tmp_path):
    p = tmp_path / 'r.json'
    A.write_ranges(p, 'siglip2', dict(ranges={a: 0.01 for a in A.ASPECTS}))
    assert A.read_ranges(p, 'siglip2')['genre'] == 0.01
    with pytest.raises(AssertionError):
        A.read_ranges(p, 'siglip1')
