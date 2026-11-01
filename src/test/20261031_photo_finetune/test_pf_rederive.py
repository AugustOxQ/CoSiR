"""Independent decision-rule examples and frozen-data replays for C13."""
import json
import time

import numpy as np
import pytest

import pf_paths as paths
import pf_rederive as rd


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv('PF_OUTPUTS', str(tmp_path))
    monkeypatch.setattr(paths, 'OUTPUTS', tmp_path)
    for attr, name in [('NEARCOPY_FILE', 'stage0a_nearcopy.json'), ('SHEET_FILE', 'stage1_sheet.json'),
                       ('STAGE2_FILE', 'stage2_records.json')]:
        monkeypatch.setattr(paths, attr, tmp_path / 'records' / name)


def row(n, score, status='ok', settings=None):
    return {'settings': dict(paths.GRID.get(n, paths.BASE) if settings is None else settings),
            'score': score, 'status': status}


def test_sweep_pick_includes_boundary_prefers_fewest_changes_then_lowest_row():
    rows = {'0': row(0, 100, 'reference'), '1': row(1, 1.5),
            '2': row(2, 2.0), '3': row(3, 2.0), '7': row(7, 99, 'failed')}
    assert rd.pick_sweep(rows) == {'row': 1, 'settings': dict(paths.BASE),
                                   'best_score': 2.0, 'tied_rows': [1, 2, 3]}
    rows['1']['score'] = 1.499999999
    rows['0']['status'] = 'ok'  # exclusion is by row number, regardless of status
    assert rd.pick_sweep(rows)['row'] == 2
    rows['9'] = row(9, 2.4, settings=dict(paths.BASE, lr=0.0003, k=20))
    assert rd.pick_sweep(rows)['row'] == 2


def test_reference_and_failed_rows_cannot_supply_a_pick():
    with pytest.raises(AssertionError, match='candidate'):
        rd.pick_sweep({'0': row(0, 100, 'reference'), '1': row(1, None, 'failed')})


def test_row9_combines_only_strict_improvements_ties_go_to_lower_row():
    rows = {str(n): row(n, score) for n, score in
            [(1, 1), (2, 2), (3, 2), (4, 3), (5, 2.5), (6, 1), (7, 99), (8, 100)]}
    rows['7'] = row(7, None, 'failed')
    assert rd.row9(rows) == {'settings': dict(paths.BASE, lr=0.0003, k=20),
                             'skipped': False, 'equals': None}
    rows['4']['score'] = rows['5']['score'] = 1
    assert rd.row9(rows) == {'settings': dict(paths.GRID[2]), 'skipped': True, 'equals': 2}
    rows['2']['score'] = rows['3']['score'] = 1
    assert rd.row9(rows) == {'settings': dict(paths.BASE), 'skipped': True, 'equals': 1}


def test_row9_smoke_uses_available_rows_and_row1_must_succeed():
    assert rd.row9({'1': row(1, 0), '8': row(8, 100)})['equals'] == 1
    with pytest.raises(AssertionError, match='row 1'):
        rd.row9({'1': row(1, None, 'failed')})


@pytest.mark.parametrize('lower,r,want', [(1e-12, .1, 'WIN'), (0, .1, 'LEVEL'),
    (-.1, .1, 'BELOW'), (-.099, .1, 'LEVEL'), (-.099, .01, 'BELOW')])
def test_outcome_strict_thresholds_use_the_supplied_range(lower, r, want):
    assert rd.outcome(lower, r) == want


@pytest.mark.parametrize('outs,want', [
    (['WIN', 'WIN', 'LEVEL', 'WIN'], 'holds'),
    (['WIN', 'WIN', 'BELOW', 'LEVEL'], 'qualified'),
    (['WIN', 'WIN', 'LEVEL', 'BELOW'], 'qualified'),
    (['WIN', 'WIN', 'BELOW', 'BELOW'], 'fails'),
    (['LEVEL', 'WIN', 'WIN', 'WIN'], 'fails'),
    (['WIN', 'LEVEL', 'WIN', 'WIN'], 'fails')])
def test_verdict_mapping(outs, want):
    assert rd.verdict(dict(zip(('SC1', 'SC2', 'SC3', 'SC4'), outs))) == want


def test_nearcopy_keys_counts_histogram_and_train_copies():
    # Two train photos copy one another but are unlike held photos: they stay.
    train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    want = {'cut': .85, 'n_train': 4, 'n_removed': 2, 'removed_keys': ['a/a', 'b/a'],
            'by_source': {'a': {'val': 1, 'test': 0}, 'b': {'val': 0, 'test': 1}},
            'within_train_copies': 2}
    for block in (1, 3):
        got = rd.nearcopy_arrays(['a/a', 'b/a', 'b/b', 'b/c'], train,
                                ['a/v'], train[:1], ['b/t'], train[1:2], block=block)
        assert {k: got[k] for k in want} == want
        assert sum(got['hist']['counts']) == 2
        assert got['hist']['counts'][-1] == 2


def test_nearcopy_inclusive_cut_and_val_first_tie():
    x = np.array([[.85, np.sqrt(1 - .85 ** 2)]], dtype=np.float32)
    held = np.array([[1, 0]], dtype=np.float32)
    got = rd.nearcopy_arrays(['a/t'], x, ['a/v'], held, ['a/e'], held)
    assert got['removed_keys'] == ['a/t']
    assert got['by_source']['a'] == {'val': 1, 'test': 0}


def test_export_rejects_wrong_key_order_and_array_corruption(tmp_path, monkeypatch):
    import ac_data
    import rs_vh
    monkeypatch.setattr(paths, 'OUTPUTS', tmp_path)
    monkeypatch.setattr(ac_data, 'keys', lambda split: ['a/1', 'a/2'])
    out = tmp_path / 'exports/sweep/row1/val'
    out.mkdir(parents=True)
    x = np.zeros((2, 2048), dtype=np.float16)
    x[:, 0] = 1
    np.savez(out / 'emotion.npz', img=x)
    side = dict(keys_sha256=rs_vh.names_sha(['a/1', 'a/2']), array_sha256=rs_vh.array_sha(x),
                weights_sha256='a' * 64, run_id='sweep/row1', split='val', smoke=bool(paths.SMOKE))
    p = out / 'emotion.json'
    p.write_text(json.dumps(side))
    assert np.array_equal(rd.saved_export('sweep/row1', 'val'), x)
    side['keys_sha256'] = rs_vh.names_sha(['a/2', 'a/1'])
    p.write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='keys'):
        rd.saved_export('sweep/row1', 'val')
    side['keys_sha256'] = rs_vh.names_sha(['a/1', 'a/2'])
    side['array_sha256'] = 'b' * 64
    p.write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='array'):
        rd.saved_export('sweep/row1', 'val')


def test_compare_reports_every_leaf_with_exact_choices_sets_and_numeric_tolerance(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, 'OUTPUTS', tmp_path)
    monkeypatch.setattr(paths, 'SMOKE', 0)
    for attr, name in [('NEARCOPY_FILE', 'near.json'), ('SHEET_FILE', 'sheet.json'), ('STAGE2_FILE', 'stage2.json')]:
        monkeypatch.setattr(paths, attr, tmp_path / name)
    theirs = {'smoke': 0, 'nearcopy': {'removed_keys': ['a/1', 'b/1'], 'n_removed': 2},
              'sheet': {'rows': {'1': row(1, .25)}, 'row9': {'skipped': True}, 'pick': {'row': 1}},
              'stage2': {'ranges': {'map': {'per_seed': [.1, .2], 'r': .1}},
                         'readout': {'pick': 'center+k10'}},
              'stage3': {'sc': {'SC1': {'interval': [.01, .02], 'outcome': 'WIN'}},
                         'verdict': 'holds', 'claims': {'SC5': True}}}
    rd.write_json(tmp_path / 'rederive/theirs.json', theirs)
    ours = json.loads(json.dumps(theirs))
    ours['nearcopy']['removed_keys'].reverse()
    ours['sheet']['rows']['1']['score'] += .0000005
    for attr, key in [('NEARCOPY_FILE', 'nearcopy'), ('SHEET_FILE', 'sheet'), ('STAGE2_FILE', 'stage2')]:
        rd.write_json(getattr(paths, attr), ours[key])
    rd.write_json(tmp_path / 'stage3/verdict.json', ours['stage3'])
    assert rd.main(['compare']) == 0
    report = rd.read_json(tmp_path / 'rederive/rederive.json')
    assert report['ok']
    assert 'stage3.sc.SC1.interval.0' in report['items']
    assert report['items']['nearcopy.removed_keys']['ok']
    # A grid setting is a choice: it must match exactly, even within float tolerance.
    ours['sheet']['rows']['1']['settings']['lr'] += 1e-8
    rd.write_json(paths.SHEET_FILE, ours['sheet'])
    assert rd.main(['compare']) == 1
    report = rd.read_json(tmp_path / 'rederive/rederive.json')
    assert not report['items']['sheet.rows.1.settings.lr']['ok']
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_smoke_compare_intersects_rows_but_missing_sc_is_failure():
    ours = {'sheet': {'rows': {'1': row(1, 1), '8': row(8, 3)}}, 'stage3': {'sc': {}}}
    theirs = {'sheet': {'rows': {'1': row(1, 1), '2': row(2, 2)}},
              'stage3': {'sc': {'SC5': {'outcome': 'WIN'}}}}
    result = rd.compare_documents(ours, theirs, smoke=True)
    assert not result['ok']
    assert 'stage3.sc.SC5.outcome' in result['items']
    assert not any(k.startswith(('sheet.rows.2.', 'sheet.rows.8.')) for k in result['items'])
    assert any(not v['ok'] for k, v in rd.compare_documents(ours, theirs)['items'].items()
               if k.startswith('sheet.rows.2.'))


def test_compare_does_not_apply_float_tolerance_to_pick_row_or_counts():
    result = rd.compare_documents({'sheet': {'pick': {'row': 1.00000001}}, 'nearcopy': {'n_removed': 2.00000001}},
                                  {'sheet': {'pick': {'row': 1}}, 'nearcopy': {'n_removed': 2}})
    assert not result['items']['sheet.pick.row']['ok']
    assert not result['items']['nearcopy.n_removed']['ok']


@pytest.mark.slow
@pytest.mark.real_data
def test_real_nearcopy(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'OUTPUTS', tmp_path)
    monkeypatch.setattr(paths, 'SMOKE', 0)
    start = time.monotonic()
    got = rd.derive_nearcopy()
    seconds = time.monotonic() - start
    assert got['n_train'] == 72255
    assert got['n_removed'] == len(set(got['removed_keys']))
    assert got['n_removed'] == sum(sum(v.values()) for v in got['by_source'].values())
    assert got['n_removed'] > 0
    print(f"near-copy count={got['n_removed']}; within-train={got['within_train_copies']}; seconds={seconds:.3f}")
    # The audit result is kept only in pytest's scratch directory.
    (tmp_path / 'nearcopy.json').write_text(json.dumps(got))


@pytest.mark.slow
@pytest.mark.real_data
def test_r12_ranges():
    got = rd.seed_ranges(lambda seed, split: rd.frozen_export(seed, split))
    expected = rd.read_json(paths.R12_OUT / 'stage0b/ranges.json')
    for metric in ('clu', 'map', 'ndcg'):
        assert got[metric]['per_seed'] == pytest.approx(expected[metric]['per_seed'], abs=1e-9, rel=0)
        assert got[metric]['r'] == pytest.approx(expected[metric]['r'], abs=1e-9, rel=0)


@pytest.mark.slow
@pytest.mark.real_data
def test_r12_readout_menu():
    ranges = rd.read_json(paths.R12_OUT / 'stage0b/ranges.json')
    got = rd.readout_menu(lambda seed, split: rd.frozen_export(seed, split), ranges)
    expected = rd.read_json(paths.R12_OUT / 'stage1/scores.json')['readout']['affection']['cstar']
    pick = rd.read_json(paths.R12_OUT / 'stage1/picks.json')['affection']['readout']['cstar']
    assert got['pick'] == pick
    assert set(got['scores']) == set(expected)
    for name, e in expected.items():
        assert got['scores'][name]['map']['emotion'] == pytest.approx(e['map']['emotion'], abs=1e-9, rel=0)
        assert got['scores'][name]['score'] == pytest.approx(e['score'], abs=1e-9, rel=0)
        assert got['scores'][name]['kept'] is e['kept']


@pytest.mark.slow
@pytest.mark.real_data
@pytest.mark.parametrize('sid', ['SC1', 'SC2', 'SC3', 'SC4', 'SC5'])
def test_r12_criteria(sid):
    ranges = rd.read_json(paths.R12_OUT / 'stage0b/ranges.json')
    readout = rd.read_json(paths.R12_OUT / 'stage1/picks.json')['affection']['readout']['cstar']
    got = rd.score_criteria(rd.frozen_export, ranges, readout, candidate='cstar', only=(sid,))[sid]
    old_id, stage = ('SC8', 'stage3') if sid == 'SC5' else (sid, 'stage2')
    expected = rd.read_json(paths.R12_OUT / stage / 'verdict.json')['sc'][old_id]
    for field in ('point_c', 'point_t', 'd', 'interval', 'r'):
        assert got[field] == pytest.approx(expected[field], abs=1e-9, rel=0)
    for field in ('outcome', 'pass', 'readout_c', 'readout_t', 'comparator'):
        assert got[field] == expected[field]
