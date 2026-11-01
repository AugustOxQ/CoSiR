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
    rd.write_json(tmp_path / 'runs/sweep/row1/run.json', {'status': 'ok', 'weights_sha256': 'a' * 64})
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
    side['array_sha256'] = rs_vh.array_sha(x)
    side['weights_sha256'] = 'b' * 64
    p.write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='weights'):
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
                         'verdict': 'holds', 'claims': {'SC5': True}, 'draws_clu': 2000, 'draws_ret': 10000}}
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


def test_smoke_compare_requires_derived_rows_but_ignores_extra_ours_rows():
    ours = {'sheet': {'rows': {'1': row(1, 1), '8': row(8, 3)}}, 'stage3': {'sc': {}}}
    theirs = {'sheet': {'rows': {'1': row(1, 1), '2': row(2, 2)}},
              'stage3': {'sc': {'SC5': {'outcome': 'WIN'}}}}
    result = rd.compare_documents(ours, theirs, smoke=True)
    assert not result['ok']
    assert 'stage3.sc.SC5.outcome' in result['items']
    assert not result['items']['sheet.rows.2.score']['ok']
    assert not any(k.startswith('sheet.rows.8.') for k in result['items'])
    assert any(not v['ok'] for k, v in rd.compare_documents(ours, theirs)['items'].items()
               if k.startswith('sheet.rows.2.'))


@pytest.fixture
def synthetic_sweep(tmp_path, monkeypatch):
    stored = np.array([[1, .02], [.02, 1]], dtype=np.float16)
    monkeypatch.setattr(rd.ac_data, 'keys', lambda split: ['a/1', 'a/2'])
    monkeypatch.setattr(rd.ac_data, 'labels', lambda split: np.array([0, 1]))
    def partitions(x):
        # A norm repair or float32 cast before clustering must fail this test.
        assert x is stored and x.dtype == np.float16
        return np.zeros((3, 2), dtype=np.int32)
    monkeypatch.setattr(rd.ac_cluster, 'partitions', partitions)
    monkeypatch.setattr(rd.ac_cluster, 'ami', lambda p, lab: .5)
    def table(xs, labels, ranges, fixed, **kwargs):
        assert set(xs) == {('emotion', None)}
        assert xs['emotion', None].dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(xs['emotion', None], axis=1), 1)
        assert rd.rt_readout.name(fixed) == 'center+k10'
        return {'center+k10': {'map': {'emotion': .4375}}}, fixed
    monkeypatch.setattr(rd.ac_select, 'arm_table', table)
    monkeypatch.setattr(paths, 'TEACHER_AMI', .25)
    monkeypatch.setattr(paths, 'TEACHER_MAP', .5)
    monkeypatch.setattr(paths, 'R12_R_CLU', .125)
    monkeypatch.setattr(paths, 'R12_R_MAP', .0625)
    monkeypatch.setattr(rd, 'frozen_export', lambda seed, split: stored)
    monkeypatch.setattr(rd, 'saved_export', lambda run, split: stored)
    for n in (1, 8):
        rd.write_json(tmp_path / f'runs/sweep/row{n}/run.json', {'status': 'ok'})
    return stored


def test_sweep_formula_and_stored_clustering(synthetic_sweep):
    got = rd.sweep_score(synthetic_sweep, paths.BASE)
    assert (got['ami'], got['map'], got['g_clu'], got['g_map'], got['score']) == (.5, .4375, 2., -1., -1.)


def test_sheet_matches_handwritten_reference_and_skipped_row9(synthetic_sweep):
    numbers = dict(ami=.5, map=.4375, g_clu=2., g_map=-1., score=-1.)
    expected = {'rows': {
        '0': dict(settings=None, changed=None, status='reference', **numbers),
        '1': dict(settings=dict(paths.BASE), changed=0, status='ok', **numbers),
        '8': dict(settings=dict(paths.GRID[8]), changed=1, status='ok', **numbers),
        '9': dict(settings=dict(paths.BASE), changed=0, status='skipped',
                  ami=None, map=None, g_clu=None, g_map=None, score=None)},
        'row9': dict(settings=dict(paths.BASE), skipped=True, equals=1),
        'pick': dict(row=1, settings=dict(paths.BASE), best_score=-1., tied_rows=[1, 8])}
    got = rd.derive_sweep()
    assert got == expected
    assert rd.compare_documents({'sheet': expected}, {'sheet': got})['ok']


def test_row9_run_is_refused_when_combination_is_skipped(synthetic_sweep, tmp_path):
    rd.write_json(tmp_path / 'runs/sweep/row9/run.json', {'status': 'ok'})
    with pytest.raises(AssertionError, match='row 9 ran'):
        rd.derive_sweep()


def test_nearcopy_normalizes_each_input():
    got = rd.nearcopy_arrays(['a/t'], np.array([[.1, 0]], np.float32),
                             ['a/v'], np.array([[.2, 0]], np.float32),
                             ['a/e'], np.array([[0, 5]], np.float32))
    assert got['removed_keys'] == ['a/t']
    assert got['by_source'] == {'a': {'val': 1, 'test': 0}}


@pytest.mark.parametrize('trail,replacement', [
    (('sheet', 'rows', '1', 'score'), .250002),
    (('nearcopy', 'removed_keys'), ['b/other']),
    (('sheet', 'pick', 'row'), 8),
    (('stage3', 'sc', 'SC1', 'outcome'), 'BELOW'),
    (('stage3', 'draws_clu'), 19),
    (('stage3', 'draws_ret'), 29),
])
def test_compare_rejects_wrong_evidence_and_exits_one(tmp_path, trail, replacement):
    theirs = {'smoke': paths.SMOKE, 'nearcopy': {'removed_keys': ['a/copy']},
              'sheet': {'rows': {'1': row(1, .25)}, 'pick': {'row': 1}},
              'stage2': {}, 'stage3': {'sc': {'SC1': {'outcome': 'WIN'}},
                                      'draws_clu': 20, 'draws_ret': 30}}
    ours = json.loads(json.dumps(theirs))
    target = ours
    for key in trail[:-1]:
        target = target[key]
    target[trail[-1]] = replacement
    rd.write_json(tmp_path / 'rederive/theirs.json', theirs)
    for attr, key in [('NEARCOPY_FILE', 'nearcopy'), ('SHEET_FILE', 'sheet'), ('STAGE2_FILE', 'stage2')]:
        rd.write_json(getattr(paths, attr), ours[key])
    rd.write_json(tmp_path / 'stage3/verdict.json', ours['stage3'])
    assert rd.main(['compare']) == 1
    report = rd.read_json(tmp_path / 'rederive/rederive.json')
    assert not report['ok'] and not report['items']['.'.join(trail)]['ok']


def test_smoke_compare_rejects_zero_overlap():
    got = rd.compare_documents({'sheet': {'rows': {'row1': row(1, 0.)}}},
                               {'sheet': {'rows': {'1': row(1, 0.)}}}, smoke=True)
    assert not got['ok'] and not got['items']['sheet.rows.1.score']['ok']


def test_photo_ranges_and_readout_flow_through_menu_and_all_criteria(monkeypatch):
    monkeypatch.setattr(paths, 'SMOKE', 1)
    rng = np.random.default_rng(7)
    arrays = {s: rd.ac_data._unit(rng.normal(size=(30, 6)) + [2, 0, 0, 0, 0, 0]).astype(np.float16)
              for s in paths.SEEDS}
    get = lambda s, split: arrays[s]
    labels = np.repeat(np.arange(3), 10)
    monkeypatch.setattr(rd.ac_data, 'labels', lambda split: labels)
    monkeypatch.setattr(rd.ac_data, 'keys', lambda split: [f'a/{i:02}' for i in range(30)])
    picks = {'affection': {'readout': {a: 'none+k0' for a in
             ('cstar', 'plain', 'teacher_artwork', 'crl_clip_painting')},
             'teacher_wording': dict(clustering='artwork', retrieval='artwork'), 'crl_version': 'clip_painting'}}
    def frozen_json(path):
        if str(path).endswith('stage1/picks.json'):
            return json.loads(json.dumps(picks))
        assert str(path).endswith('stage0b/ranges.json')
        return {m: {'r': .001} for m in ('clu', 'map', 'ndcg')}
    monkeypatch.setattr(rd, 'read_json', frozen_json)
    monkeypatch.setattr(rd.ac_cluster, 'partitions', lambda x: np.zeros((3, 30), np.int32))
    monkeypatch.setattr(rd, 'comparator_partitions', lambda arm: np.zeros((3, 30), np.int32))
    def cluster_pair(*args, draws, **kwargs):
        assert draws == 7
        return dict(A_cand=.25, A_comp=.5, d=-.25, interval=[-.25, -.1])
    def retrieval_pair(cand, comp, draws):
        assert draws == 11
        return dict(d=float(cand.mean() - comp.mean()), interval=[-.25, -.1])
    monkeypatch.setattr(rd.ac_cluster, 'paired', cluster_pair)
    monkeypatch.setattr(rd.rt_stats, 'paired', retrieval_pair)
    monkeypatch.setattr(rd, 'comparator_ap', lambda *args: dict(
        ap=np.full(30, .6), rows=np.arange(30), point=.6, per_seed=None, dropped=0))
    ranges = {m: {'r': .5} for m in ('clu', 'map', 'ndcg')}
    menu = rd.readout_menu(get, ranges, workers=1)
    baseline = menu['scores']['none+k0']['map']['emotion']
    gain = menu['scores']['center+k10']['map']['emotion'] - baseline
    assert abs(gain) > 1e-4
    assert menu['scores']['center+k10']['score'] == pytest.approx(gain / .5)
    smaller = {m: {'r': .25} for m in ranges}
    assert rd.readout_menu(get, smaller, workers=1)['scores']['center+k10']['score'] == pytest.approx(gain / .25)
    level = rd.score_criteria(get, ranges, 'center+k10', workers=1, draws_clu=7, draws_ret=11)
    below = rd.score_criteria(get, smaller, 'none+k0', workers=1, draws_clu=7, draws_ret=11)
    assert set(level) == {'SC1', 'SC2', 'SC3', 'SC4', 'SC5'}
    for sid in level:
        assert level[sid]['r'] == .5 and level[sid]['outcome'] == 'LEVEL'
        assert below[sid]['r'] == .25 and below[sid]['outcome'] == 'BELOW'
    assert level['SC2']['readout_c'] == 'center+k10'
    assert below['SC2']['readout_c'] == 'none+k0'
    assert abs(level['SC2']['point_c'] - below['SC2']['point_c']) > 1e-4


@pytest.mark.parametrize('option', ['--draws-clu', '--draws-ret'])
def test_draw_overrides_are_smoke_only(option, monkeypatch):
    monkeypatch.setattr(paths, 'SMOKE', 0)
    with pytest.raises(SystemExit) as exc:
        rd.main(['derive', option, '7'])
    assert exc.value.code == 2


def test_smoke_cli_forwards_draws(monkeypatch):
    monkeypatch.setattr(paths, 'SMOKE', 1)
    def derive(workers, draws_clu, draws_ret):
        assert (workers, draws_clu, draws_ret) == (1, 7, 11)
        return dict(stage3={'verdict': 'holds'}, seconds=0.)
    monkeypatch.setattr(rd, 'derive', derive)
    assert rd.main(['derive', '--workers', '1', '--draws-clu', '7', '--draws-ret', '11']) == 0


@pytest.fixture
def run_records(tmp_path, monkeypatch):
    train = ['a/copy', 'a/fit', 'b/fit']
    monkeypatch.setattr(rd, 'store_split_keys', lambda: (train, {'train': train}))
    fit_sha = rd.rs_vh.names_sha(['a/fit', 'b/fit'])
    settings = dict(paths.GRID[8])
    for s in paths.SEEDS:
        rd.write_json(tmp_path / f'runs/final/s{s}/run.json', dict(
            status='ok', settings=settings, cstar_seed=s, train_seed=100 + s,
            fit_keys_sha256=fit_sha, weights_sha256='a' * 64))
    for n, status in [(1, 'ok'), (2, 'failed')]:
        rd.write_json(tmp_path / f'runs/sweep/row{n}/run.json', dict(status=status, fit_keys_sha256=fit_sha))
    return {'pick': {'row': 8, 'settings': settings}}, {'removed_keys': ['a/copy']}


@pytest.mark.parametrize('run_id,field,wrong', [
    ('final/s2', 'settings', dict(paths.BASE)),
    ('final/s2', 'train_seed', 2),
    ('final/s2', 'cstar_seed', 0),
    ('final/s2', 'fit_keys_sha256', 'a' * 64),
    ('sweep/row1', 'fit_keys_sha256', 'a' * 64),
    ('sweep/row2', 'fit_keys_sha256', 'a' * 64),
])
def test_run_records_match_derived_pick_seeds_and_fit_set(run_records, tmp_path, run_id, field, wrong):
    sheet, nearcopy = run_records
    assert rd.validate_runs(sheet, nearcopy) == ['a/fit', 'b/fit']
    path = tmp_path / 'runs' / run_id / 'run.json'
    record = rd.read_json(path)
    record[field] = wrong
    rd.write_json(path, record)
    with pytest.raises(AssertionError, match=field):
        rd.validate_runs(sheet, nearcopy)


def test_derive_verdict_uses_sc1_to_sc4_and_keeps_sc5_separate(run_records, monkeypatch):
    sheet, nearcopy = run_records
    monkeypatch.setattr(rd, 'derive_sweep', lambda workers: sheet)
    monkeypatch.setattr(rd, 'derive_nearcopy', lambda: nearcopy)
    ranges = {m: {'r': .5} for m in ('clu', 'map', 'ndcg')}
    monkeypatch.setattr(rd, 'seed_ranges', lambda get, workers: ranges)
    monkeypatch.setattr(rd, 'readout_menu', lambda get, ranges, workers: {'pick': 'center+k10'})
    def criteria(get, ranges, pick, **kwargs):
        assert kwargs['draws_clu'] == 7 and kwargs['draws_ret'] == 11
        return {sid: dict(outcome=o, **{'pass': p}) for sid, o, p in
                [('SC1', 'WIN', True), ('SC2', 'WIN', True), ('SC3', 'LEVEL', True),
                 ('SC4', 'LEVEL', True), ('SC5', 'BELOW', False)]}
    monkeypatch.setattr(paths, 'SMOKE', 1)
    monkeypatch.setattr(rd, 'score_criteria', criteria)
    got = rd.derive(draws_clu=7, draws_ret=11)
    assert got['stage3']['verdict'] == 'holds'
    assert got['stage3']['claims'] == {'SC5': False}
    assert (got['stage3']['draws_clu'], got['stage3']['draws_ret']) == (7, 11)
    path = paths.OUTPUTS / 'runs/final/s0/run.json'
    run = rd.read_json(path)
    run['settings'] = dict(paths.BASE)
    rd.write_json(path, run)
    with pytest.raises(AssertionError, match='settings'):
        rd.derive(draws_clu=7, draws_ret=11)


def test_run_validation_checks_unused_export_sidecars(run_records, tmp_path):
    sheet, nearcopy = run_records
    path = tmp_path / 'runs/sweep/row1/run.json'
    run = rd.read_json(path)
    run['weights_sha256'] = 'a' * 64
    rd.write_json(path, run)
    side_path = tmp_path / 'exports/sweep/row1/test/emotion.json'
    rd.write_json(side_path, {'weights_sha256': 'a' * 64})
    rd.validate_runs(sheet, nearcopy)
    rd.write_json(side_path, {'weights_sha256': 'b' * 64})
    with pytest.raises(AssertionError, match='weights_sha256'):
        rd.validate_runs(sheet, nearcopy)


@pytest.mark.slow
@pytest.mark.real_data
def test_r12_sweep_row0():
    stored = rd.frozen_export(0, 'val')
    got = rd.sweep_score(stored, None, 'reference')
    # Frozen r12's one-run cross-fit path, independent of our sweep adapter.
    ro = rd.ac_select.readout_from_name('center+k10')
    expected = rd.rt_readout.crossfit({('emotion', None): rd.ac_data._unit(stored)},
                                      {'emotion': rd.ac_data.labels('val')}, ro, workers=8)['emotion']
    assert got['ami'] == pytest.approx(paths.FROZEN_AMI_S0, abs=1e-9, rel=0)
    assert got['map'] == pytest.approx(expected, abs=1e-9, rel=0)


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
