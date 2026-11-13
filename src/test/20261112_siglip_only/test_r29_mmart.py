"""Tests of r29_mmart (ticket 11): MMArt exports, the wiring of r17's scorers to them, the six MMArt cells, the order rule, the
gate comparisons. Small synthetic MMArt sets (240 test, 180 validation paintings; planted classes); no real data.

Seams: (1) the order rule's refusals of `run_cells` / `run_picks` / `run_export`; (2) export -> wiring round trip (key join, stale
checkpoint, split guard); (3) `score_cells` end to end on planted data: ids, comparators, cell shape (accepted by ticket 08's reader),
direction of the paired differences; (4) the picks record; (5) the gate comparisons (tolerance, per-comparator failure).
"""
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_mmart as mm  # noqa: E402
import r29_ladder as LD  # noqa: E402

import mt_data  # noqa: E402
import mt_export  # noqa: E402
import ns_data  # noqa: E402
import ns_paths  # noqa: E402
import bs_cstar  # noqa: E402

N_TEST, N_VAL, N_EXTRA = 240, 180, 30
W_IN = 24                                   # a small image width for the synthetic rows
CLASSES = {'emotion': 9, 'style': 20, 'genre': 10}
ARM = 'siglip1'


# ------------------------------------------------------------------------------------------------ synthetic world
def _keys(prefix, n):
    return mt_data.order([f'{prefix}{i:04d}' for i in range(n)])


KEYS = dict(test=_keys('t', N_TEST), val=_keys('v', N_VAL))
KEYS['new'] = mt_data.order(KEYS['test'] + _keys('x', N_EXTRA))       # the test rows are a subset of `new`
KEYSET = {'val': 'val', 'test': 'new'}


def _labels(split, aspect, rng):
    n = len(KEYS[split])
    y = np.arange(n) % CLASSES[aspect]
    y = y[rng.permutation(n)].astype(np.int64)
    if aspect != 'style':
        y[::7] = -1                          # MMArt has no emotion or genre label for some paintings
    return y


def _rows(y, rng, signal, width):
    """Unit rows whose class centre is `signal` strong (labels -1: noise only)."""
    centres = rng.normal(size=(max(CLASSES.values()), width))
    x = rng.normal(size=(len(y), width)) + signal * np.where((y >= 0)[:, None], centres[np.maximum(y, 0)], 0.0)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope='module')
def world():
    rng = np.random.default_rng(0)
    y = {(s, a): _labels(s, a, rng) for s in ('val', 'test') for a in mm.ASPECTS}
    return dict(
        y=y,
        # the candidate heads: strong class signal; the plain image: weak; the teacher: very strong (its aspect view)
        head={(s, a, seed): _rows(y[(s, a)], np.random.default_rng(100 + seed), 0.22, 2048)
              for s in ('val', 'test') for a in mm.ASPECTS for seed in range(5)},
        plain={s: _rows(y[(s, 'style')], np.random.default_rng(7), 0.0, W_IN) for s in ('val', 'test')},
        teacher={(s, a): _rows(y[(s, a)], np.random.default_rng(8), 1.2, 32) for s in ('val', 'test') for a in mm.ASPECTS})


def _checkpoints(root, arm, content=b'weights'):
    for a in mm.ASPECTS:
        for s in range(5):
            p = bs_cstar.rb_student.weights_path(root, arm, a, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content + f'{a}{s}'.encode())


def _fake_export(world, tmp_path, arm=ARM, content=b'weights'):
    """Write the exports of the synthetic world through `export_arm` with an injected inference that returns the planted heads
    (rows of the keyset in key order)."""
    root, export_root = tmp_path / 'cstar', tmp_path / 'mmart'
    _checkpoints(root, arm, content)
    position = {k: {key: i for i, key in enumerate(KEYS[k])} for k in ('test', 'new')}

    def rows(keyset):
        return KEYS[keyset], np.zeros((len(KEYS[keyset]), W_IN), dtype=np.float32)

    def infer(weights, aspect, seed, x):
        if len(x) == N_VAL:
            return world['head'][('val', aspect, seed)].astype(np.float16)
        full = np.zeros((len(KEYS['new']), 2048), dtype=np.float32)
        full[[position['new'][k] for k in KEYS['test']]] = world['head'][('test', aspect, seed)]
        rest = [i for i, k in enumerate(KEYS['new']) if k not in set(KEYS['test'])]
        extra = np.random.default_rng(5).normal(size=(len(rest), 2048))
        full[rest] = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        return full.astype(np.float16)

    mm.export_arm(root, arm, export_root, rows=rows, infer=infer, log=lambda *a: None)
    return root, export_root


@pytest.fixture
def exported(world, tmp_path):
    root, export_root = _fake_export(world, tmp_path)
    return root, export_root


def _wiring(root, export_root, **kw):
    return mm.Wiring(root, ARM, export_root, names=lambda split: KEYS[split], **kw)


# ------------------------------------------------------------------------------------------------ exports and the wiring
def test_export_round_trip_joins_by_key_and_gives_the_stored_float16_rows(world, exported):
    root, export_root = exported
    w = _wiring(root, export_root)
    got = w.head('test', 'style', 3)
    assert got.dtype == np.float16 and got.shape == (N_TEST, 2048)
    assert np.array_equal(got, world['head'][('test', 'style', 3)].astype(np.float16))


def test_export_files_are_r15s_layout_and_name_the_checkpoint(exported):
    root, export_root = exported
    f = mt_export.export_path(export_root, 'new', 'siglip1', 'genre', 2)
    side = json.loads(f.with_suffix('.json').read_text())
    assert side['keyset'] == 'new' and side['n'] == N_TEST + N_EXTRA and side['width'] == 2048
    assert side['checkpoint'] == str(bs_cstar.rb_student.weights_path(root, ARM, 'genre', 2))
    assert sorted(np.load(f).files) == ['emb', 'keys']


def test_wiring_refuses_a_stale_export_whose_checkpoint_changed(exported):
    root, export_root = exported
    bs_cstar.rb_student.weights_path(root, ARM, 'emotion', 1).write_bytes(b'retrained')
    with pytest.raises(AssertionError, match='stale export'):
        _wiring(root, export_root).head('val', 'emotion', 1)


def test_validation_wiring_never_reads_test_rows(exported):
    root, export_root = exported
    w = _wiring(root, export_root, splits=('val',))
    w.head('val', 'emotion', 0)
    with pytest.raises(AssertionError, match='validation'):
        w.head('test', 'emotion', 0)


def test_wiring_refuses_rows_for_another_key_list(exported):
    root, export_root = exported
    w = mm.Wiring(root, ARM, export_root, names=lambda split: KEYS[split] + ['not-exported'])
    with pytest.raises(AssertionError, match='not exported'):
        w.head('val', 'emotion', 0)


def test_wiring_patches_r17s_loaders_for_mmart_only_and_unwinds(exported):
    root, export_root = exported
    original = ns_data.cstar
    with mock.patch.object(ns_data, 'keys', lambda s, split: KEYS[split]), _wiring(root, export_root) as w:
        x = ns_data.cstar('mmart', 'test', 'siglip1', 'emotion', 0, unit=True)
        assert x.dtype == np.float32 and x.shape == (N_TEST, 2048)
        raw = ns_data.cstar('mmart', 'test', 'siglip1', 'emotion', 0, unit=False)
        assert np.array_equal(raw, w.head('test', 'emotion', 0).astype(np.float32))      # the clustering form: not renormalised
        with pytest.raises(AssertionError):
            ns_data.cstar('artelingo', 'held', 'siglip1', 'emotion', 0)
        assert ns_data._head_id('mmart', 'val', 'siglip1', 'emotion', 0) == KEYS['val']
    assert ns_data.cstar is original


# ------------------------------------------------------------------------------------------------ the six cells, end to end
PICKS = dict(
    ami_ranges={a: dict(r=0.01, per_seed={str(s): 0.1 for s in range(5)}, k=CLASSES[a], n_val=N_VAL) for a in mm.ASPECTS},
    map_ranges={a: dict(r=0.01, per_seed={str(s): 0.3 for s in range(5)}) for a in mm.ASPECTS},
    readouts=dict(untold=dict(pick='none+k0'), cstar=dict(pick='none+k0'), plain=dict(pick='none+k0'),
                  teacher=dict(pick='none+k0')))


@pytest.fixture
def small_mmart(world):
    """r17's and r15's data loaders on the synthetic MMArt (labels, plain image, teacher views); the row counts r17 asserts."""
    teacher = lambda set_, split, aspect: world['teacher'][(split, aspect)]          # noqa: E731
    with mock.patch.object(ns_data, 'keys', lambda s, split: KEYS[split]), \
            mock.patch.object(ns_data, 'labels', lambda s, split, a: world['y'][(split, a)].copy()), \
            mock.patch.object(ns_data, 'plain', lambda s, split, enc: world['plain'][split]), \
            mock.patch.object(ns_data, 'teacher', teacher), \
            mock.patch.dict(ns_paths.N_ROWS, {('mmart', 'test'): N_TEST, ('mmart', 'val'): N_VAL}), \
            mock.patch.dict(ns_paths.WIDTH, {'siglip1': W_IN}):
        yield


@pytest.fixture
def cells(world, exported, small_mmart):
    root, export_root = exported
    w = _wiring(root, export_root)
    return mm.score_cells(w, PICKS, draws=dict(ami=40, map=100), workers=2, check_files=False)


def test_twelve_cells_with_the_ladders_ids(cells):
    ids = sorted(c['id'] for c in cells)
    want = sorted(f'mmart/siglip1/{ARM}/{a}/{m}/{c}' for a, k in (('emotion', 9), ('style', 20), ('genre', 10))
                  for m, c in ((f'ami{k}', 'plain'), (f'ami{k}', 'teacher'), ('map', 'plain'), ('map', 'teacher')))
    assert ids == want


def test_every_cell_is_read_by_ticket_08s_reader_as_an_sc5_cell(cells, tmp_path):
    rec = dict(arm=ARM, encoder='siglip1', cells=cells)
    (tmp_path / f'mmart_cells_{ARM}.json').write_text(json.dumps(rec))
    got, info = LD.read_source('mmart', 'SC5', ARM, 'siglip1', tmp_path)
    assert len(got) == 12 and info['n_cells'] == 12
    assert {k[0][2] for k in got} == {'ami', 'map'} and {k[1] for k in got} == {'P1', 'Q1'}


def test_map_cells_follow_r17_untold_against_plain_and_r15_told_against_teacher(cells):
    by = {c['id'].split('/', 3)[3]: c for c in cells}
    assert by['emotion/map/plain']['type'] == 'map_plain' and by['emotion/map/plain']['protocol'] == 'r17_SC7'
    assert by['emotion/map/teacher']['type'] == 'map_teacher' and by['emotion/map/teacher']['protocol'] == 'r15_SC4'
    assert by['emotion/ami9/plain']['protocol'] == 'r17_SC1' and by['emotion/ami9/teacher']['protocol'] == 'r17_SC3'
    for a in mm.ASPECTS:
        assert by[f'{a}/map/plain']['candidate_readout'] == 'none+k0' and by[f'{a}/map/teacher']['comparator_readout'] == 'none+k0'


def test_directions_follow_the_planted_data(cells):
    """Heads strong, plain image weak, teacher stronger than the heads: against plain the candidate wins, against the teacher it
    does not (a swapped comparator would flip the signs)."""
    for c in cells:
        if c['comparator'] == 'plain':
            assert c['d'] > 0 and c['outcome'] == 'WIN', (c['id'], c['value'], c['comparator_value'], c['lo'], c['hi'], c['r'])
        else:
            assert c['d'] < 0 and c['outcome'] in ('LEVEL', 'BELOW'), (c['id'], c['value'], c['comparator_value'], c['lo'], c['hi'], c['r'])


def test_cells_score_the_test_rows_not_the_validation_rows(cells):
    for c in cells:
        assert c['set'] == 'mmart' and c['n_rows'] == N_TEST, c['id']


def test_ami_cells_count_the_labelled_test_rows(cells, world):
    by = {c['id'].split('/', 3)[3]: c for c in cells}
    assert by['emotion/ami9/plain']['n_labelled'] == int((world['y'][('test', 'emotion')] >= 0).sum())


def test_scoring_unwinds_the_patches_of_ticket_06(cells):
    assert mm.IC.SET == 'artelingo' and mm.IC.SPLIT_HELD == 'held'


# ------------------------------------------------------------------------------------------------ the order rule
def _git(path, *args):
    subprocess.run(['git', '-C', str(path), '-c', 'user.name=t', '-c', 'user.email=t@t', *args], check=True, capture_output=True)


@pytest.fixture
def records(tmp_path):
    """A records folder inside a throwaway git repo: pass_rule (pass), prefix_pick (K* = siglip1) and nothing else, committed."""
    repo = tmp_path / 'repo'
    rec = repo / 'records'
    rec.mkdir(parents=True)
    _git(repo, 'init', '-q')
    (rec / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))
    (rec / 'prefix_pick.json').write_text(json.dumps(dict(kstar='siglip1', kstar_form='plain', p1_form='plain', flag=None)))
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-qm', 'rules')
    return rec


def _commit(rec, name, doc):
    (rec / name).write_text(json.dumps(doc))
    _git(rec.parent, 'add', '-A')
    _git(rec.parent, 'commit', '-qm', name)


def _picks_record(arm=ARM):
    return dict(PICKS, arm=arm, encoder='siglip1', root='r', export_root='e', split='val', validation_files={})


def _run(rec, scorer=None, arm=ARM, **kw):
    scorer = scorer or mock.Mock(return_value=[])
    with mock.patch.object(mm, 'score_cells', scorer), mock.patch.object(mm, 'held_files', lambda w: {}), \
            mock.patch.object(mm, 'Wiring', mock.MagicMock()):
        out = mm.run_cells(arm, records=rec, root=Path('r'), export_root=Path('e'), workers=1, **kw)
    return scorer, out


def test_cells_refuse_without_the_picks_record_and_score_nothing(records):
    scorer = mock.Mock()
    with pytest.raises(FileNotFoundError, match='picks'):
        _run(records, scorer)
    scorer.assert_not_called()


def test_cells_refuse_a_picks_record_not_committed(records):
    (records / f'mmart_picks_{ARM}.json').write_text(json.dumps(_picks_record()))     # present, not committed
    scorer = mock.Mock()
    with pytest.raises(RuntimeError, match='committed'):
        _run(records, scorer)
    scorer.assert_not_called()


def test_cells_refuse_a_picks_record_edited_after_the_commit(records):
    _commit(records, f'mmart_picks_{ARM}.json', _picks_record())
    (records / f'mmart_picks_{ARM}.json').write_text(json.dumps(dict(_picks_record(), written='later')))
    with pytest.raises(RuntimeError, match='committed'):
        _run(records)


def test_cells_refuse_without_a_committed_pass_rule(records, tmp_path):
    _commit(records, f'mmart_picks_{ARM}.json', _picks_record())
    (records / 'pass_rule.json').write_text(json.dumps(dict(status='pass', edited=True)))
    scorer = mock.Mock()
    with pytest.raises(RuntimeError, match='pass_rule'):
        _run(records, scorer)
    scorer.assert_not_called()


@pytest.mark.parametrize('status', ['fail', 'parked', 'fallback'])
def test_stage2_cells_need_status_pass(records, status):
    _commit(records, 'pass_rule.json', dict(status=status))
    _commit(records, f'mmart_picks_{ARM}.json', _picks_record())
    with pytest.raises(RuntimeError, match='pass'):
        _run(records)


def test_cells_refuse_an_arm_that_is_not_kstar(records):
    _commit(records, 'prefix_pick.json', dict(kstar='siglip1_prefix', kstar_form='prefixed', p1_form='plain', flag=None))
    _commit(records, f'mmart_picks_{ARM}.json', _picks_record())
    with pytest.raises(AssertionError, match='K\\*'):
        _run(records)


def test_prefix_pick_keys_are_asserted(records):
    _commit(records, 'prefix_pick.json', dict(kstar='siglip1', kstar_form='plain'))
    _commit(records, f'mmart_picks_{ARM}.json', _picks_record())
    with pytest.raises(AssertionError, match='p1_form'):
        _run(records)


def test_cells_run_and_write_the_record_after_the_picks_are_committed(records):
    _commit(records, f'mmart_picks_{ARM}.json', _picks_record())
    scorer, path = _run(records)
    scorer.assert_called_once()
    rec = json.loads(Path(path).read_text())
    assert Path(path).name == f'mmart_cells_{ARM}.json' and rec['arm'] == ARM and rec['encoder'] == 'siglip1'
    assert rec['kstar'] == 'siglip1' and rec['kstar_form'] == 'plain' and rec['flag'] is None and rec['cells'] == []


def test_picks_record_is_written_once(records, tmp_path):
    body = dict(ami_ranges={}, map_ranges={}, readouts={}, validation_files={}, n_val=N_VAL)
    mm.write_picks(ARM, tmp_path / 'r', tmp_path / 'e', body, records=records)
    with pytest.raises(FileExistsError):
        mm.write_picks(ARM, tmp_path / 'r', tmp_path / 'e', body, records=records)


def test_picks_and_export_refuse_without_a_pass(records):
    _commit(records, 'pass_rule.json', dict(status='fail'))
    with pytest.raises(RuntimeError, match='pass'):
        mm.run_picks(ARM, records=records, root=Path('r'), export_root=Path('e'))
    with pytest.raises(RuntimeError, match='pass'):
        mm.run_export(ARM, records=records, root=Path('r'), export_root=Path('e'))


def test_validation_files_must_be_the_ones_the_picks_read(exported, world, small_mmart):
    root, export_root = exported
    w = _wiring(root, export_root)
    files = mm.validation_files(w)
    mm.check_validation_files(w, dict(validation_files=files))
    p = mt_export.export_path(export_root, 'val', 'siglip1', 'genre', 4)
    p.write_bytes(p.read_bytes() + b'0')
    with pytest.raises(AssertionError, match='changed since the picks'):
        mm.check_validation_files(w, dict(validation_files=files))


# ------------------------------------------------------------------------------------------------ picks on validation
def test_picks_read_validation_rows_only_and_record_ranges_and_readouts(world, exported, small_mmart):
    root, export_root = exported
    w = _wiring(root, export_root, splits=('val',))
    rt_readout = mm.IC.ns_picks.ac_select.rt_readout
    menu = rt_readout.menu
    with mock.patch.object(mm, 'teacher_pick', lambda: 'center+k10'), \
            mock.patch.object(mm.IC.ns_ret, 'reused_pick', lambda arm, s, e: 'none+k1'), \
            mock.patch.object(rt_readout, 'menu', lambda width: [r for r in menu(width) if r.w != 'whiten'][:4]):   # R0 first; whitening's eigendecompositions take minutes
        body = mm.compute_picks(w, workers=2)
    assert body['n_val'] == N_VAL
    for a in mm.ASPECTS:
        assert body['ami_ranges'][a]['k'] == CLASSES[a] and body['ami_ranges'][a]['r'] >= 0
        assert body['map_ranges'][a]['r'] > 0 and len(body['map_ranges'][a]['per_seed']) == 5
    assert set(body['readouts']) == {'untold', 'cstar', 'plain', 'teacher'}
    assert body['readouts']['plain']['pick'] == 'none+k1' and body['readouts']['teacher']['pick'] == 'center+k10'
    json.dumps(body)                                                       # a record: plain JSON types only


# ------------------------------------------------------------------------------------------------ gate comparisons
def test_rebuild_comparison_is_per_comparator_and_inclusive_at_the_tolerance():
    rebuilt = dict(teacher=dict(emotion=0.5, style=0.25), plain=dict(emotion=0.125, style=0.0625))
    recorded = dict(teacher=dict(emotion=[('r15 SC4', 0.5)], style=[('r15 SC4', 0.25)]),
                    plain=dict(emotion=[('r17 SC7', 0.125 + 1e-9)], style=[('r17 SC7', 0.0625)]))
    items, ok = mm.compare_rebuilds(rebuilt, recorded, tol=1e-9)
    assert ok == dict(teacher=True, plain=True) and len(items) == 4


def test_rebuild_comparison_fails_only_the_comparator_that_differs():
    rebuilt = dict(teacher=dict(emotion=0.5), plain=dict(emotion=0.125))
    recorded = dict(teacher=dict(emotion=[('a', 0.5)]), plain=dict(emotion=[('b', 0.125 + 2e-9)]))
    items, ok = mm.compare_rebuilds(rebuilt, recorded, tol=1e-9)
    assert ok == dict(teacher=True, plain=False)
    assert [i['ok'] for i in items if i['comparator'] == 'plain'] == [False]


def test_rebuild_comparison_checks_every_recorded_value_of_a_cell():
    rebuilt = dict(teacher=dict(emotion=0.5))
    recorded = dict(teacher=dict(emotion=[('r15 SC4 qwen', 0.5), ('r15 SC4 clip_l14', 0.5 + 1e-3)]))
    _, ok = mm.compare_rebuilds(rebuilt, recorded, tol=1e-9)
    assert ok == dict(teacher=False)


def test_rebuild_comparison_refuses_a_missing_value():
    with pytest.raises(AssertionError):
        mm.compare_rebuilds(dict(teacher=dict(emotion=float('nan'))), dict(teacher=dict(emotion=[('a', 0.5)])), tol=1e-9)


def test_export_comparison_is_bit_for_bit(tmp_path):
    a = tmp_path / 'a.npz'
    b = tmp_path / 'b.npz'
    emb = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float16)
    np.savez(a, keys=np.asarray(['k1', 'k2', 'k3', 'k4']), emb=emb)
    np.savez(b, keys=np.asarray(['k1', 'k2', 'k3', 'k4']), emb=emb)
    assert mm.compare_export_files(a, b)['equal']
    emb2 = emb.copy()
    emb2[2, 3] = np.nextafter(emb2[2, 3], np.float16(10))                     # one float16 step
    np.savez(b, keys=np.asarray(['k1', 'k2', 'k3', 'k4']), emb=emb2)
    r = mm.compare_export_files(a, b)
    assert not r['equal'] and r['differing_values'] == 1
    np.savez(b, keys=np.asarray(['k1', 'k2', 'k4', 'k3']), emb=emb)
    assert not mm.compare_export_files(a, b)['equal']                          # same rows, other key order: not equal


def test_cell_comparison_names_the_record_values_and_the_tolerance():
    cell = dict(id='mmart/siglip1/k11=40/emotion/ami9/plain', type='ami', comparator='plain', aspect='emotion', k=9,
                value=0.1, comparator_value=0.05, d=0.05, lo=0.04, hi=0.06, r=0.003, outcome='WIN', encoder='siglip1')
    rec = dict(value=0.1, comparator_value=0.05, d=0.05, lo=0.04, hi=0.06, r=0.003, outcome='WIN')
    items = mm.compare_cell(cell, 'r17 SC1', rec, check_interval=True)
    assert items['ok'] and items['max_abs_diff'] == 0
    assert not mm.compare_cell(dict(cell, hi=0.06 + 2e-9), 'r17 SC1', rec, check_interval=True)['ok']
    assert mm.compare_cell(dict(cell, hi=0.07), 'r17 SC1', rec, check_interval=False)['ok']           # reduced draws: no interval
    assert not mm.compare_cell(dict(cell, outcome='LEVEL'), 'r17 SC1', rec, check_interval=True)['ok']


def test_gate_status_rule_pass_then_fail_fail_parked_and_a_fourth_try_raises(tmp_path):
    """`pass`; a failure is `fail` before try 3 and `parked` at try 3; one try per write (the count is the record's, not ours)."""
    got = [mm.write_gate(tmp_path, 'vi', False, dict(x=1)) for _ in range(3)]
    assert [(r['status'], r['tries']) for r in got] == [('fail', 1), ('fail', 2), ('parked', 3)]
    with pytest.raises(AssertionError, match='fourth'):
        mm.write_gate(tmp_path, 'vi', False, dict(x=1))
    rec = json.loads((tmp_path / 'vi.json').read_text())
    assert rec['gate'] == 'vi' and rec['tries'] == 3 and rec['status'] == 'parked'


def test_gate_status_a_pass_at_the_second_try_is_pass(tmp_path):
    assert mm.write_gate(tmp_path, 'v_mmart', False, dict(x=1))['status'] == 'fail'
    rec = mm.write_gate(tmp_path, 'v_mmart', True, dict(x=1))
    assert (rec['status'], rec['tries']) == ('pass', 2)


def test_gate_v_mmart_command_writes_the_status_rule_and_the_ladder_reads_it(tmp_path):
    """Absent record: the ladder's reader raises "gate v_mmart not run"; fail, fail, parked on three failures; pass on a pass."""
    with pytest.raises(RuntimeError, match='gate v_mmart not run'):
        LD.gate_status(tmp_path, 'v_mmart')
    details = dict(failed_comparators=['plain'], max_abs_diff=0.5)
    with mock.patch.object(mm, 'records_dir', lambda: tmp_path), mock.patch.object(mm, 'gate_v_mmart', lambda **kw: (False, details)):
        assert mm.main(['gate-v-mmart']) == 1
        assert LD.gate_status(tmp_path, 'v_mmart', open_ok=True) == 'open'                 # fail below try 3 is still open
        mm.main(['gate-v-mmart'])
        mm.main(['gate-v-mmart'])
    assert LD.gate_status(tmp_path, 'v_mmart') == 'parked'
    other = tmp_path / 'other'
    with mock.patch.object(mm, 'records_dir', lambda: other), \
            mock.patch.object(mm, 'gate_v_mmart', lambda **kw: (True, dict(failed_comparators=[], max_abs_diff=0.0))):
        assert mm.main(['gate-v-mmart']) == 0
    assert LD.gate_status(other, 'v_mmart') == 'pass'


def test_gate_vi_score_command_writes_the_status_rule_and_a_reduced_check_writes_no_record(tmp_path):
    n_files = len(mm.KEYSETS) * len(mm.ASPECTS) * len(mm.SEEDS)
    out = tmp_path / 'out' / 'gate_vi' / 'try1'
    out.mkdir(parents=True)
    (out / 'export_result.json').write_text(json.dumps(dict(ok=True, n_files=n_files)))
    score = lambda ok: (lambda *a, **kw: (ok, dict(intervals_checked=True, picks=[], max_abs_diff=0.0)))    # noqa: E731
    rec = tmp_path / 'rec'
    with mock.patch.object(mm, 'records_dir', lambda: rec), mock.patch.object(mm.P, 'OUT', tmp_path / 'out'):
        with pytest.raises(RuntimeError, match='gate vi not run'):
            LD.gate_status(rec, 'vi')
        with mock.patch.object(mm, 'gate_vi_score', score(False)):
            assert mm.main(['gate-vi', 'score', '--try', '1']) == 1
            assert LD.gate_status(rec, 'vi', open_ok=True) == 'open'
            mm.main(['gate-vi', 'score', '--try', '1'])
            mm.main(['gate-vi', 'score', '--try', '1'])
        assert LD.gate_status(rec, 'vi') == 'parked'
        assert json.loads((rec / 'gates' / 'vi.json').read_text())['tries'] == 3
        with mock.patch.object(mm, 'gate_vi_score', score(True)):                 # a reduced check: one aspect, no record
            assert mm.main(['gate-vi', 'score', '--try', '1', '--aspects', 'emotion']) == 0
        assert json.loads((rec / 'gates' / 'vi.json').read_text())['tries'] == 3
    assert (out / 'score_partial.json').exists()


def test_gate_vi_score_passes_into_a_pass_record(tmp_path):
    n_files = len(mm.KEYSETS) * len(mm.ASPECTS) * len(mm.SEEDS)
    out = tmp_path / 'out' / 'gate_vi' / 'try1'
    out.mkdir(parents=True)
    (out / 'export_result.json').write_text(json.dumps(dict(ok=True, n_files=n_files)))
    rec = tmp_path / 'rec'
    with mock.patch.object(mm, 'records_dir', lambda: rec), mock.patch.object(mm.P, 'OUT', tmp_path / 'out'), \
            mock.patch.object(mm, 'gate_vi_score', lambda *a, **kw: (True, dict(intervals_checked=True, picks=[], max_abs_diff=0.0))):
        assert mm.main(['gate-vi', 'score', '--try', '1']) == 0
    assert LD.gate_status(rec, 'vi') == 'pass'
