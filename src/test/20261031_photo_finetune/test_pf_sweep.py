"""Tests for pf_sweep (ticket 07): the sweep score, row 9, the pick (D9), the sheet and gate (ii)'s sweep part.

Seams: changed (settings -> count), row9 (rows -> settings, skipped, equals), pick (rows -> pick), gaps and score
(embeddings -> {ami, map, g_clu, g_map, score}), sheet (rows -> the committed document), gate_ii_sweep (slow).
Expected values are worked by hand from spec 2026-10-31-photo-finetune-design.md §3.4, §3.5, §6 and D9.
"""
import copy
import json

import numpy as np
import pytest

import pf_paths as P
import pf_sweep
import ac_cluster
import rt_readout
from pf_paths import BASE, GRID


def ok(settings, score):
    return {'settings': settings, 'status': 'ok', 'score': score}


def failed(settings):
    return {'settings': settings, 'status': 'failed', 'score': None}


def rows_with(scores, failed_rows=()):
    """Rows 1 to 8 of the grid with the given scores (a list for rows 1 to 8)."""
    out = {}
    for n, s in zip(range(1, 9), scores):
        out[n] = failed(GRID[n]) if n in failed_rows else ok(GRID[n], s)
    return out


# ---- changed settings (D9) -----------------------------------------------------------------------------------------

def test_changed_counts_columns_that_differ_from_row_1():
    assert pf_sweep.changed(GRID[1]) == 0
    assert [pf_sweep.changed(GRID[n]) for n in range(2, 9)] == [1] * 7
    assert pf_sweep.changed(dict(BASE, lr=0.0003, k=20)) == 2
    assert pf_sweep.changed(dict(BASE, lr=0.0003, k=20, correction=False, teacher_weight=1.0, start='random')) == 5


def test_changed_refuses_settings_with_missing_or_extra_columns():
    with pytest.raises(AssertionError):
        pf_sweep.changed({k: v for k, v in BASE.items() if k != 'k'})
    with pytest.raises(AssertionError):
        pf_sweep.changed(dict(BASE, extra=1))


# ---- row 9 ---------------------------------------------------------------------------------------------------------

def test_row9_takes_each_columns_value_from_its_best_row_above_row_1():
    # row 1 scores 0; lr: row 3 (2.0) beats row 2 (1.0); k: row 5 (0.5); correction: row 6 below row 1; weight: row 7 up
    rows = rows_with([0.0, 1.0, 2.0, -1.0, 0.5, -0.2, 0.3, -9.0])
    settings, skipped, equals = pf_sweep.row9(rows)
    assert settings == dict(BASE, lr=0.0001, k=80, teacher_weight=1.0)
    assert (skipped, equals) == (False, None)
    assert pf_sweep.changed(settings) == 3


@pytest.mark.parametrize('row, column, value', [(2, 'lr', 0.0003), (3, 'lr', 0.0001), (4, 'k', 20), (5, 'k', 80),
                                               (6, 'correction', False), (7, 'teacher_weight', 1.0)])
def test_row9_each_of_rows_2_to_7_alone_above_row_1_sets_its_own_column(row, column, value):
    scores = [0.0] + [1.0 if n == row else -1.0 for n in range(2, 8)] + [5.0]
    settings, skipped, equals = pf_sweep.row9(rows_with(scores))
    assert settings == dict(BASE, **{column: value})
    assert skipped and equals == row                                    # it equals that row's own settings


def test_row9_row_8_never_sets_a_value():
    rows = rows_with([0.0, -1, -1, -1, -1, -1, -1, 50.0])
    settings, skipped, equals = pf_sweep.row9(rows)
    assert settings == dict(BASE) and settings['start'] == 'cstar'
    assert (skipped, equals) == (True, 1)


def test_row9_needs_a_strictly_higher_score_than_row_1():
    rows = rows_with([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0])        # all equal to row 1: nothing changes
    settings, skipped, equals = pf_sweep.row9(rows)
    assert settings == dict(BASE) and skipped and equals == 1
    rows[6] = ok(GRID[6], 0.7 + 1e-9)
    settings, skipped, _ = pf_sweep.row9(rows)
    assert settings == dict(BASE, correction=False) and skipped


def test_row9_equal_scores_in_a_column_go_to_the_lower_row_number():
    rows = rows_with([0.0, 1.0, 1.0, -1, -1, -1, -1, 0.0])
    assert pf_sweep.row9(rows)[0]['lr'] == 0.0003                       # row 2, not row 3
    rows = rows_with([0.0, -1, -1, 2.0, 2.0, -1, -1, 0.0])
    assert pf_sweep.row9(rows)[0]['k'] == 20                            # row 4, not row 5


def test_row9_is_skipped_when_it_equals_an_earlier_row_and_names_the_lowest_one():
    rows = rows_with([0.0, 1.0, 0.5, -1, -1, -1, -1, 0.0])              # row 9 = row 2's settings
    settings, skipped, equals = pf_sweep.row9(rows)
    assert settings == GRID[2] and skipped and equals == 2


def test_row9_ignores_failed_rows_and_a_failed_row_cannot_win():
    rows = rows_with([0.0, 3.0, 2.0, -1, -1, -1, -1, 0.0], failed_rows=(2,))
    settings, skipped, equals = pf_sweep.row9(rows)
    assert settings['lr'] == 0.0001 and skipped and equals == 3          # row 3 wins instead
    rows = rows_with([0.0, 3.0, 2.0, 1.0, -1, -1, 4.0, 0.0], failed_rows=(2, 7))
    settings, skipped, equals = pf_sweep.row9(rows)
    assert settings == dict(BASE, lr=0.0001, k=20) and not skipped and equals is None


def test_row9_with_row_1_failed_stops():
    rows = rows_with([0.0, 1, 1, 1, 1, 1, 1, 1], failed_rows=(1,))
    with pytest.raises(RuntimeError, match='row 1'):
        pf_sweep.row9(rows)


def test_row9_does_not_modify_its_input_and_returns_fresh_settings():
    rows = rows_with([0.0, 1.0, 2.0, -1.0, 0.5, -0.2, 0.3, 0.0])
    before = copy.deepcopy(rows)
    settings = pf_sweep.row9(rows)[0]
    settings['lr'] = 5
    assert rows == before and GRID[1]['lr'] == 0.001


# ---- the pick (D9) -------------------------------------------------------------------------------------------------

def with_row9(rows, settings, score):
    return {**rows, 9: ok(settings, score)}


def test_pick_takes_the_best_score_when_nothing_is_within_the_margin():
    rows = rows_with([1.0, 0.2, 0.3, 0.1, 0.0, -2.0, 0.4, 1.6])         # row 1 is 0.6 below row 8
    p = pf_sweep.pick(rows)
    assert p['row'] == 8 and p['settings'] == GRID[8] and p['best_score'] == 1.6 and p['tied_rows'] == [8]


def test_pick_ties_within_half_go_to_fewer_changed_settings_even_at_a_lower_score():
    rows = rows_with([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5])         # row 1 is 0.3 below row 8
    p = pf_sweep.pick(rows)
    assert p['row'] == 1 and p['best_score'] == 1.5 and p['tied_rows'] == [1, 8]


def test_pick_margin_is_inclusive():
    rows = rows_with([1.0, -5, -5, -5, -5, -5, -5, 1.5])                # exactly 0.5 below
    assert pf_sweep.pick(rows)['row'] == 1
    rows = rows_with([0.9999, -5, -5, -5, -5, -5, -5, 1.5])
    assert pf_sweep.pick(rows)['row'] == 8


def test_pick_then_lowest_row_among_equal_change_counts():
    rows = rows_with([-5, 1.0, 1.2, 1.1, 0.9, -5, -5, -5])             # rows 2 to 5 within 0.5 of 1.2, one change each
    p = pf_sweep.pick(rows)
    assert p['row'] == 2 and p['tied_rows'] == [2, 3, 4, 5]


def test_pick_row_9_has_two_or_more_changes_so_loses_ties_to_single_change_rows():
    two = dict(BASE, lr=0.0003, teacher_weight=1.0)
    rows = with_row9(rows_with([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), two, 1.4)
    assert pf_sweep.pick(rows)['row'] == 2                              # rows 2, 3, 9 tied; row 2 has one change
    rows = with_row9(rows_with([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), two, 1.6)
    p = pf_sweep.pick(rows)
    assert p['row'] == 9 and p['settings'] == two and p['tied_rows'] == [9]


def test_pick_row_8_beats_a_tied_row_9():
    rows = with_row9(rows_with([-5, -5, -5, -5, -5, -5, -5, 2.0]), dict(BASE, lr=0.0003, k=20), 2.2)
    assert pf_sweep.pick(rows)['row'] == 8


def test_pick_ignores_failed_rows_and_skipped_row_9():
    rows = rows_with([1.0, -5, -5, -5, -5, -5, -5, 9.0], failed_rows=(8,))
    rows[9] = {'settings': GRID[2], 'status': 'skipped', 'score': None}
    p = pf_sweep.pick(rows)
    assert p['row'] == 1 and p['best_score'] == 1.0 and p['tied_rows'] == [1]


def test_pick_never_takes_row_0():
    rows = rows_with([-1.0, -5, -5, -5, -5, -5, -5, -5])
    rows[0] = {'settings': GRID[1], 'status': 'ok', 'score': 100.0}
    assert pf_sweep.pick(rows)['row'] == 1


def test_pick_with_row_1_failed_stops():
    rows = rows_with([0.0, 1, 1, 1, 1, 1, 1, 1], failed_rows=(1,))
    with pytest.raises(RuntimeError, match='row 1'):
        pf_sweep.pick(rows)


def test_pick_returns_a_copy_of_the_settings_of_the_row():
    rows = rows_with([1.0, 0, 0, 0, 0, 0, 0, 0])
    p = pf_sweep.pick(rows)
    p['settings']['lr'] = 5
    assert rows[1]['settings']['lr'] == 0.001 and GRID[1]['lr'] == 0.001


# ---- gaps and score ------------------------------------------------------------------------------------------------

def test_gaps_are_in_units_of_the_r12_ranges_and_the_score_is_the_smaller():
    g = pf_sweep.gaps(P.TEACHER_AMI + 2 * P.R12_R_CLU, P.TEACHER_MAP - 3 * P.R12_R_MAP)
    assert g['g_clu'] == pytest.approx(2.0, abs=1e-9) and g['g_map'] == pytest.approx(-3.0, abs=1e-9)
    assert g['score'] == g['g_map']
    g = pf_sweep.gaps(P.TEACHER_AMI + 5 * P.R12_R_CLU, P.TEACHER_MAP + 1 * P.R12_R_MAP)
    assert g['score'] == g['g_map'] == pytest.approx(1.0, abs=1e-9)
    g = pf_sweep.gaps(P.TEACHER_AMI - 1 * P.R12_R_CLU, P.TEACHER_MAP + 4 * P.R12_R_MAP)
    assert g['score'] == g['g_clu'] == pytest.approx(-1.0, abs=1e-9)
    assert g['ami'] == P.TEACHER_AMI - P.R12_R_CLU


# ---- score on embeddings (synthetic stand-in for one run's validation export) --------------------------------------

def synthetic_run(n_classes=20, per_class=40, dim=64, noise=0.05, seed=0):
    """Float16 unit rows of `n_classes` tight, well-separated clusters, and their int64 labels."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_classes, dim))
    labels = np.repeat(np.arange(n_classes), per_class).astype(np.int64)
    x = centres[labels] + noise * rng.normal(size=(len(labels), dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float16), labels


def test_score_of_perfectly_clustered_rows_is_near_one_for_both_tasks_and_has_the_gap_fields():
    x, labels = synthetic_run()
    s = pf_sweep.score(x, labels=labels, workers=2)
    assert set(s) == {'ami', 'map', 'g_clu', 'g_map', 'score'}
    assert s['ami'] > 0.95 and s['map'] > 0.99
    assert s['g_clu'] == pytest.approx((s['ami'] - P.TEACHER_AMI) / P.R12_R_CLU, abs=1e-9)
    assert s['g_map'] == pytest.approx((s['map'] - P.TEACHER_MAP) / P.R12_R_MAP, abs=1e-9)
    assert s['score'] == min(s['g_clu'], s['g_map'])


def test_score_clusters_the_stored_float16_rows_and_retrieves_on_float32_unit_rows():
    x, labels = synthetic_run(per_class=30, seed=3)
    s = pf_sweep.score(x, labels=labels, workers=2)
    assert s['ami'] == ac_cluster.ami(ac_cluster.partitions(x), labels)             # float16 rows, no repair
    x32 = x.astype(np.float32)
    x32 = x32 / np.linalg.norm(x32, axis=1, keepdims=True)
    menu = rt_readout.menu(x.shape[1])
    ro = menu[[rt_readout.name(r) for r in menu].index('center+k10')]
    want = rt_readout.crossfit({('emotion', None): x32}, {'emotion': labels}, ro, workers=2)['emotion']
    assert s['map'] == want


def test_score_refuses_rows_that_are_not_float16_or_not_unit():
    x, labels = synthetic_run(per_class=30)
    with pytest.raises(AssertionError):
        pf_sweep.score(x.astype(np.float32), labels=labels)
    with pytest.raises(AssertionError):
        pf_sweep.score((x * 2).astype(np.float16), labels=labels)                   # clustering checks, never repairs
    with pytest.raises(AssertionError):
        pf_sweep.score(x[:-1], labels=labels)                                       # one label per row


# ---- the sheet -----------------------------------------------------------------------------------------------------

def scores_for(score):
    return {'ami': 0.1 + score / 100, 'map': 0.2 + score / 100, 'g_clu': score, 'g_map': score + 1, 'score': score}


def full_rows(scores, row0_score=-3.0, failed_rows=()):
    rows = {0: pf_sweep.entry(None, 'reference', scores_for(row0_score))}
    for n, s in zip(range(1, 9), scores):
        rows[n] = pf_sweep.entry(GRID[n], 'failed') if n in failed_rows else pf_sweep.entry(GRID[n], 'ok', scores_for(s))
    return rows


def test_entry_failed_rows_have_null_scores_and_ok_rows_need_all_five_numbers():
    e = pf_sweep.entry(GRID[3], 'failed')
    assert e == {'settings': GRID[3], 'status': 'failed', 'ami': None, 'map': None, 'g_clu': None, 'g_map': None,
                 'score': None}
    with pytest.raises(AssertionError):
        pf_sweep.entry(GRID[3], 'ok')
    with pytest.raises(AssertionError):
        pf_sweep.entry(GRID[3], 'failed', scores_for(1.0))
    with pytest.raises(AssertionError):
        pf_sweep.entry(GRID[3], 'ok', {'score': 1.0})


def test_sheet_has_the_index_schema_with_a_skipped_row_9_and_survives_json():
    rows = full_rows([0.0, 1.0, 0.5, -1, -1, -1, -1, 0.2])                           # row 9 = row 2: skipped
    doc = pf_sweep.sheet(rows)
    assert set(doc) == {'rows', 'row9', 'pick'}
    assert sorted(doc['rows'], key=int) == [str(n) for n in range(10)]
    assert set(doc['rows']['2']) == {'settings', 'changed', 'status', 'ami', 'map', 'g_clu', 'g_map', 'score'}
    assert doc['rows']['0']['status'] == 'reference' and doc['rows']['0']['score'] == -3.0
    assert doc['rows']['0']['changed'] is None and doc['rows']['0']['settings'] is None
    assert [doc['rows'][str(n)]['changed'] for n in range(1, 9)] == [0] + [1] * 7
    assert doc['row9'] == {'settings': GRID[2], 'skipped': True, 'equals': 2}
    assert doc['rows']['9'] == {'settings': GRID[2], 'changed': 1, 'status': 'skipped', 'ami': None, 'map': None,
                                'g_clu': None, 'g_map': None, 'score': None}
    assert doc['pick'] == {'row': 2, 'settings': GRID[2], 'best_score': 1.0, 'tied_rows': [2, 3]}
    assert json.loads(json.dumps(doc)) == doc


def test_sheet_pick_leaves_out_row_0_and_failed_rows():
    rows = full_rows([0.0, 1.0, 0.5, -1, -1, -1, -1, 9.0], row0_score=99.0, failed_rows=(8,))
    doc = pf_sweep.sheet(rows)
    assert doc['pick']['row'] == 2 and doc['pick']['best_score'] == 1.0 and doc['pick']['tied_rows'] == [2, 3]
    assert doc['rows']['8']['status'] == 'failed' and doc['rows']['8']['score'] is None


def test_sheet_with_a_scored_row_9_includes_it_and_it_can_be_picked():
    base = full_rows([0.0, 1.0, 2.0, -1, 0.5, -1, 0.7, -9])
    s9, skipped, equals = pf_sweep.row9(base)
    assert not skipped and s9 == dict(BASE, lr=0.0001, k=80, teacher_weight=1.0)
    rows = {**base, 9: pf_sweep.entry(s9, 'ok', scores_for(3.0))}
    doc = pf_sweep.sheet(rows)
    assert doc['row9'] == {'settings': s9, 'skipped': False, 'equals': None}
    assert doc['rows']['9']['changed'] == 3 and doc['rows']['9']['score'] == 3.0
    assert doc['pick']['row'] == 9 and doc['pick']['settings'] == s9 and doc['pick']['tied_rows'] == [9]


def test_sheet_refuses_a_row_9_that_does_not_follow_the_rule_or_is_missing():
    base = full_rows([0.0, 1.0, 2.0, -1, 0.5, -1, 0.7, -9])
    with pytest.raises(AssertionError):                                              # row 9 should have run
        pf_sweep.sheet(base)
    with pytest.raises(AssertionError):                                              # row 9 with other settings
        pf_sweep.sheet({**base, 9: pf_sweep.entry(GRID[3], 'ok', scores_for(3.0))})


def test_sheet_with_row_1_failed_stops():
    with pytest.raises(RuntimeError, match='row 1'):
        pf_sweep.sheet(full_rows([0.0, 1, 1, 1, 1, 1, 1, 1], failed_rows=(1,)))


# ---- gate (ii), sweep part -----------------------------------------------------------------------------------------

def test_replay_check_passes_within_1e_9_and_fails_outside_it():
    per_seed = {s: {'ami': 0.1, 'map': P.CSTAR_MAP_5SEED + d} for s, d in zip(range(5), (-0.01, 0.01, 0.02, -0.02, 0.0))}
    per_seed[0]['ami'] = P.FROZEN_AMI_S0 + 5e-10
    res = pf_sweep.replay_check(per_seed)
    assert res['ok'] and res['complete'] and res['map_seed0'] == per_seed[0]['map']
    assert res['ami_diff'] == pytest.approx(5e-10, abs=1e-12) and abs(res['map_diff']) < 1e-9
    per_seed[0]['ami'] = P.FROZEN_AMI_S0 + 2e-9
    assert not pf_sweep.replay_check(per_seed)['ok']
    per_seed[0]['ami'] = P.FROZEN_AMI_S0
    per_seed[4]['map'] += 1e-8                                                         # the mean moves by 2e-9
    assert not pf_sweep.replay_check(per_seed)['ok']


def test_replay_check_with_fewer_than_five_seeds_is_incomplete_and_never_ok():
    res = pf_sweep.replay_check({0: {'ami': P.FROZEN_AMI_S0, 'map': 0.3}})
    assert not res['complete'] and not res['ok'] and res['map_diff'] is None and res['ami_diff'] == 0.0


@pytest.mark.slow
@pytest.mark.real_data
def test_gate_ii_sweep_replays_frozen_cstar_through_the_sweep_score():
    res = pf_sweep.gate_ii_sweep()
    assert res['complete'] and res['ok'], res
    assert abs(res['ami_diff']) <= P.TOL['ii'] and abs(res['map_diff']) <= P.TOL['ii']
    assert (P.OUTPUTS / 'stage0a' / 'sweep_replay.json').exists()


@pytest.mark.parametrize('bad', ['reference', 'missing', 'skipped', 'status', 'settings'])
def test_sheet_refuses_invalid_rows(bad):
    rows = full_rows([0.0, 1.0, 0.5, -1, -1, -1, -1, 0.2])
    if bad == 'reference':
        rows[0]['status'] = 'ok'
    elif bad == 'missing':
        rows.pop(3)
    elif bad == 'skipped':
        rows[9] = pf_sweep.entry(GRID[2], 'ok', scores_for(2.0))
    elif bad == 'status':
        rows = full_rows([0.0, 1.0, 2.0, -1, 0.5, -1, 0.7, -9])
        rows[9] = pf_sweep.entry(pf_sweep.row9(rows)[0], 'skipped')
    else:
        rows[3]['settings'] = dict(GRID[3], lr=0.0002)
    with pytest.raises(AssertionError): pf_sweep.sheet(rows)


def test_gate_sweep_refuses_seeds_without_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(pf_sweep.ac_data, 'stored', lambda *a, **kw: None)
    monkeypatch.setattr(pf_sweep, 'score', lambda *a, **kw: {'ami': 0.1, 'map': 0.2})
    with pytest.raises(AssertionError, match='requires seed 0'):
        pf_sweep.gate_ii_sweep(seeds=(1,), out=tmp_path / 'replay.json')
