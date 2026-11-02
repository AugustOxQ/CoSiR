"""Tests of bs_artret's verdict code on small synthetic data (ticket 06): the pick with own ranges, the frozen image's
pick with r11's r_a, the SC6 `r` unit, the cell format, the cell values and the ablation row function."""
import json

import numpy as np
import pytest

import bs_artret as A
import rt_metrics
import rt_readout
from rt_readout import R0, ReadOut

ASP = A.ASPECTS
CENTER_K10 = ReadOut('center', None, 10)


def unit_rows(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rows_with(gains):
    """Read-out table of a width-8 arm: R0 at 0.5 everywhere, center+k10 at 0.5 + gains[aspect]."""
    base = {a: 0.5 for a in ASP}
    return {R0: base, CENTER_K10: {a: 0.5 + gains[a] for a in ASP}}


# ---- ranges -------------------------------------------------------------------------------------------------------

def test_seed_ranges_are_max_minus_min_over_seeds_unrounded():
    per_seed = {a: {s: 0.3 + 0.001234567 * ((s * 7) % 5) + i for s in range(5)} for i, a in enumerate(ASP)}
    got = A.own_ranges(per_seed)
    for a in ASP:
        v = list(per_seed[a].values())
        assert got[a] == max(v) - min(v)
        assert got[a] != round(got[a], 4)


def test_seed_ranges_need_seeds_zero_to_four():
    per_seed = {a: {s: 0.3 + 0.01 * s for s in range(4)} for a in ASP}
    with pytest.raises(AssertionError):
        A.own_ranges(per_seed)


def test_r11_ranges_are_the_committed_stage_0a_values():
    r = A.r11_ranges()
    assert r == {'emotion': 0.0014116107779215126, 'style': 0.002229334648849224, 'genre': 0.0038237977312404192}


# ---- the pick with own ranges -------------------------------------------------------------------------------------

def test_cstar_pick_uses_its_own_ranges():
    # center+k10 gains on emotion and style and loses 0.01 on genre
    rows = rows_with(dict(emotion=0.02, style=0.02, genre=-0.01))
    tight = {a: 0.005 for a in ASP}      # genre loss 0.01 > r: the guard drops center+k10
    loose = {a: 0.02 for a in ASP}       # genre loss 0.01 < r: kept, score (1 + 1 - 0.5) / 3 > 0
    assert A.cstar_pick(rows, tight)['pick'] == 'none+k0'
    assert A.cstar_pick(rows, loose)['pick'] == 'center+k10'


def test_cstar_pick_reports_the_table():
    rows = rows_with(dict(emotion=0.02, style=0.02, genre=0.02))
    out = A.cstar_pick(rows, {a: 0.01 for a in ASP})
    assert out['table']['center+k10']['score'] == pytest.approx(2.0)
    assert out['table']['none+k0']['score'] == 0.0
    assert out['table']['center+k10']['kept'] is True
    assert out['table']['center+k10']['map'] == dict(emotion=0.52, style=0.52, genre=0.52)
    assert out['ranges'] == {a: 0.01 for a in ASP}


def test_frozen_pick_uses_r11s_r_a_not_the_encoders_own():
    r = A.r11_ranges()
    # a genre loss of 0.8 r_genre: kept under r11's r_a, dropped under a tenfold tighter range
    rows = rows_with(dict(emotion=2 * r['emotion'], style=2 * r['style'], genre=-0.8 * r['genre']))
    assert A.frozen_pick(rows)['pick'] == 'center+k10'
    own = {a: 0.1 * r[a] for a in ASP}
    assert A.cstar_pick(rows, own)['pick'] == 'none+k0'
    assert A.frozen_pick(rows)['ranges'] == r


def test_frozen_pick_takes_no_ranges_argument():
    with pytest.raises(TypeError):
        A.frozen_pick(rows_with(dict(emotion=0, style=0, genre=0)), {a: 1.0 for a in ASP})


# ---- the cells ----------------------------------------------------------------------------------------------------

OWN = {'emotion': 0.0021, 'style': 0.0033, 'genre': 0.0047}


def synthetic_ap(seed=0, n=300, gain=0.02):
    rng = np.random.default_rng(seed)
    comp = rng.uniform(0.2, 0.6, n)
    cand = np.stack([comp + gain + rng.normal(0, 0.01, n) for _ in range(5)])
    return cand, comp, rng.integers(0, 1000, n)


def test_sc6_unit_is_the_encoders_own_range_and_sc2_has_none():
    cand, comp, rows = synthetic_ap()
    teacher = comp + 0.001
    cells = A.aspect_cells('siglip2', 'style', dict(ap=cand, rows=rows), dict(ap=comp, rows=rows),
                           dict(ap=teacher, rows=rows), OWN, 200, readouts=dict(cstar='center+k10', frozen='center+k10',
                                                                                  teacher='center+k10'))
    sc2, sc6 = cells
    assert (sc2['sc'], sc6['sc']) == ('SC2', 'SC6')
    assert sc2['r'] is None
    assert sc6['r'] == OWN['style']                      # X-c*'s own range, not r11's r_a
    assert sc6['r'] != A.r11_ranges()['style']
    assert 'own validation seed range' in sc6['r_source']


def test_cell_values_and_format():
    cand, comp, rows = synthetic_ap(gain=0.05)
    teacher = comp - 0.01
    cells = A.aspect_cells('clip_l14', 'genre', dict(ap=cand, rows=rows), dict(ap=comp, rows=rows),
                           dict(ap=teacher, rows=rows), OWN, 200,
                           readouts=dict(cstar='center+k10', frozen='none+k0', teacher='center+k10'))
    sc2, sc6 = cells
    assert set(A.CELL_KEYS) <= set(sc2)
    assert (sc2['encoder'], sc2['aspect'], sc2['task'], sc2['data']) == ('clip_l14', 'genre', 'ret', 'artelingo_held')
    assert (sc2['candidate'], sc2['comparator']) == ('cstar_clip_l14', 'frozen_clip_l14')
    assert sc6['comparator'] == 'teacher_genre'
    assert sc2['candidate_value'] == pytest.approx(cand.mean())
    assert sc2['comparator_value'] == pytest.approx(comp.mean())
    assert sc2['d'] == pytest.approx(cand.mean() - comp.mean())
    assert sc6['d'] == pytest.approx(cand.mean() - teacher.mean())
    assert sc2['lo'] < sc2['d'] < sc2['hi'] and sc2['lo'] > 0           # a clear gain
    assert sc2['n_draws'] == 200 and sc2['seeds'] == [0, 1, 2, 3, 4]
    assert sc2['comparator_readout'] == 'none+k0' and sc2['candidate_readout'] == 'center+k10'
    json.dumps(cells, allow_nan=False)


def test_cell_interval_is_r11s_joint_bootstrap():
    cand, comp, rows = synthetic_ap()
    cells = A.aspect_cells('siglip2', 'emotion', dict(ap=cand, rows=rows), dict(ap=comp, rows=rows),
                           dict(ap=comp, rows=rows), OWN, 300,
                           readouts=dict(cstar='center+k10', frozen='center+k10', teacher='center+k10'))
    p = A.rt_stats.paired(cand, comp, 300)
    assert (cells[0]['lo'], cells[0]['hi']) == (p['interval'][0], p['interval'][1])


def test_cells_need_the_same_queries():
    cand, comp, rows = synthetic_ap()
    with pytest.raises(AssertionError):
        A.aspect_cells('siglip2', 'emotion', dict(ap=cand, rows=rows), dict(ap=comp, rows=rows[::-1].copy()),
                       dict(ap=comp, rows=rows), OWN, 100,
                       readouts=dict(cstar='center+k10', frozen='center+k10', teacher='center+k10'))


def test_make_cell_rejects_a_level_unit_on_sc2_and_nan():
    base = dict(encoder='siglip2', aspect='style', candidate='c', comparator='f', candidate_value=.3,
                comparator_value=.2, d=.1, lo=.05, hi=.15, n_draws=10)
    with pytest.raises(AssertionError):
        A.make_cell('SC2', r=0.002, r_source='x', **base)
    with pytest.raises(AssertionError):
        A.make_cell('SC6', r=None, r_source=None, **base)
    with pytest.raises(AssertionError):
        A.make_cell('SC6', r=0.002, r_source='x', **{**base, 'lo': float('nan')})
    c = A.make_cell('SC6', r=0.002, r_source='x', **base)
    assert c['task'] == 'ret' and c['data'] == 'artelingo_held'


def test_write_cells_replaces_only_that_encoders_cells(tmp_path):
    def cell(enc, aspect):
        return A.make_cell('SC2', encoder=enc, aspect=aspect, candidate='c', comparator='f', candidate_value=.3,
                           comparator_value=.2, d=.1, lo=.05, hi=.15, r=None, r_source=None, n_draws=10)
    path = tmp_path / A.CELLS_NAME
    A.write_cells(path, [cell('siglip2', 'style'), cell('clip_l14', 'style')], ['siglip2', 'clip_l14'])
    A.write_cells(path, [cell('siglip2', 'genre')], ['siglip2'])
    got = json.loads(path.read_text())
    assert sorted((c['encoder'], c['aspect']) for c in got) == [('clip_l14', 'style'), ('siglip2', 'genre')]
    with pytest.raises(AssertionError):
        A.write_cells(path, [cell('siglip1', 'style')], ['siglip2'])


# ---- part b's ablation rows ---------------------------------------------------------------------------------------

def test_ablation_row_fits_on_the_validation_export_not_the_held_one():
    rng = np.random.default_rng(3)
    n = 60
    shift = rng.standard_normal(16).astype(np.float32) * 3
    x_val = unit_rows(rng, n, 16) + shift
    x_val = (x_val / np.linalg.norm(x_val, axis=1, keepdims=True)).astype(np.float32)
    x_held = unit_rows(rng, n, 16)                       # another mean: centering on held would differ
    y_val = {a: rng.integers(0, 4, n) for a in ASP}
    y_held = {a: rng.integers(0, 4, n) for a in ASP}
    out = A.ablation_retrieval(x_val, x_held, y_val, y_held, 'emotion', workers=1)
    r0 = rt_metrics.retrieval(x_held, y_held['emotion'], workers=1)
    assert np.array_equal(out['held']['r0']['ap'], r0['ap'])
    xa = rt_readout.apply(rt_readout.fit(CENTER_K10, x_val), x_held)
    want = rt_metrics.retrieval(xa, y_held['emotion'], qe_k=10, workers=1)
    assert np.array_equal(out['held']['center_k10']['ap'], want['ap'])
    xb = rt_readout.apply(rt_readout.fit(CENTER_K10, x_held), x_held)
    other = rt_metrics.retrieval(xb, y_held['emotion'], qe_k=10, workers=1)
    assert not np.array_equal(want['ap'], other['ap'])
    assert out['readout'] == 'center+k10'
    assert np.array_equal(out['held']['r0']['rows'], out['held']['center_k10']['rows'])


def test_ablation_row_validation_scores_are_r0_and_crossfit():
    rng = np.random.default_rng(4)
    n = 80
    x_val, x_held = unit_rows(rng, n, 12), unit_rows(rng, n, 12)
    y = {a: rng.integers(0, 3, n) for a in ASP}
    out = A.ablation_retrieval(x_val, x_held, y, y, 'style', workers=1)
    assert out['val']['r0_map'] == float(rt_metrics.retrieval(x_val, y['style'], workers=1)['ap'].mean())
    cf = rt_readout.crossfit_all({(a, None): x_val for a in ASP}, y, [CENTER_K10], workers=1)[CENTER_K10]['style']
    assert out['val']['center_k10_crossfit_map'] == cf
