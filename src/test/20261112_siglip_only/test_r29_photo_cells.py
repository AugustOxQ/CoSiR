"""Tests of r29_photo_cells (ticket 13): the order rule and the stage 2 guard, F1's picks and ranges (validation only, r_E for
EmoSet's mAP), the eight photo cells (ticket 06's shape, readable by ticket 08's ladder), the parking of a cell whose rebuilt
comparator differs, the gate status rule and the "gate not run" refusal, the export reader (join by key, sidecar, key set).
Small synthetic fixtures (90 and 80 photos, 2,048 wide as the real exports); the real records are read only for r_E and the
comparators' recorded values (cheap JSON reads)."""
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_photo_cells as PC  # noqa: E402
import r29_ladder as L  # noqa: E402

import ns_clu  # noqa: E402
import rs_vh  # noqa: E402

W = 2048
KS = {'affection': 9, 'emoset': 8}
N_PER = 10
SEEDS = (0, 1, 2, 3, 4)


@pytest.fixture(autouse=True)
def small_bootstrap(monkeypatch):
    monkeypatch.setattr(ns_clu, 'WORKERS', 1)       # r17: "set to 1 in small tests"


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class FakeData:
    """Class-structured rows: F1's noise per quality, plain very noisy, teacher in between. `forbid`: splits it must not serve."""

    def __init__(self, f1_noise=0.4, plain_noise=2.0, teacher_noise=1.0, forbid=()):
        self.f1_noise, self.plain_noise, self.teacher_noise, self.forbid = f1_noise, plain_noise, teacher_noise, forbid
        rng = np.random.default_rng(7)
        self.centers = {s: _unit(rng.normal(size=(k, W))) for s, k in KS.items()}
        self.asked = []

    def _check(self, set_name, split):
        self.asked.append((set_name, split))
        assert split not in self.forbid, f'{split} rows were read'

    def keys(self, set_name, split):
        return [f'{set_name}/{split}/{i:03d}' for i in range(KS[set_name] * N_PER)]

    def labels(self, set_name, split):
        self._check(set_name, split)
        return np.repeat(np.arange(KS[set_name]), N_PER).astype(np.int64)

    def _rows(self, set_name, split, noise, seed):
        rng = np.random.default_rng(zlib.crc32(f'{set_name}/{split}/{seed}'.encode()))
        y = np.repeat(np.arange(KS[set_name]), N_PER)
        return _unit(self.centers[set_name][y] + noise * _unit(rng.normal(size=(len(y), W))))

    def f1(self, set_name, split, seed):
        self._check(set_name, split)
        return self._rows(set_name, split, self.f1_noise, 100 + seed).astype(np.float16)

    def f1_sha(self, set_name, split, seed):
        return f'sha-{set_name}-{split}-{seed}'

    def plain(self, set_name, split):
        self._check(set_name, split)
        return self._rows(set_name, split, self.plain_noise, 50)

    def teacher(self, set_name, split):
        self._check(set_name, split)
        return self._rows(set_name, split, self.teacher_noise, 60)


COMPS = {(s, c): dict(map_value=0.5, map_pick='none+k0', ami_value=0.1, source='fake') for s in KS for c in ('plain', 'teacher')}


def fake_pick(calls=None):
    def pick(set_name, xs_val, y_val, keys_val, r_map, workers):
        if calls is not None:
            calls.append(dict(set=set_name, r_map=r_map, seeds=sorted(xs_val), n=len(y_val)))
        return dict(pick='none+k0', scores={'none+k0': {'map': {'emotion': 0.5}, 'score': 0.5, 'kept': True}})
    return pick


def picks_body(monkeypatch, data=None, calls=None):
    monkeypatch.setattr(PC, 'pick_readout', fake_pick(calls))
    return PC.compute_picks(data or FakeData(), 'siglip1', workers=1)


# ------------------------------------------------------------------------------------------------ the rule and the guards
@pytest.mark.parametrize('lo, r, want', [(1e-12, 0.01, 'WIN'), (0.0, 0.01, 'LEVEL'), (-0.01, 0.01, 'BELOW')])
def test_outcome_is_the_shared_rule(lo, r, want):
    assert PC.outcome(lo, r) == want


def _records(tmp_path, pass_status=None):
    rec = tmp_path / 'records'
    (rec / 'gates').mkdir(parents=True)
    if pass_status is not None:
        (rec / 'pass_rule.json').write_text(json.dumps(dict(status=pass_status)))
    return rec


@pytest.mark.parametrize('status', [None, 'fail', 'parked'])
@pytest.mark.parametrize('command', ['picks', 'cells'])
def test_stage_2_commands_refuse_unless_the_pass_rule_says_pass(tmp_path, status, command):
    rec = _records(tmp_path, status)
    with pytest.raises(RuntimeError, match='pass_rule'):
        getattr(PC, f'run_{command}')('siglip1', data=FakeData(), records=rec, gates=rec / 'gates')
    assert not list(rec.glob('photo_*.json'))


def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True, capture_output=True)


def test_cells_refuse_without_the_picks_record_and_without_its_commit(tmp_path, monkeypatch):
    rec = _records(tmp_path, 'pass')
    monkeypatch.setattr(PC, 'pick_readout', fake_pick())
    with pytest.raises(RuntimeError, match='photo_picks_siglip1.json'):
        PC.run_cells('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS)
    _git(tmp_path, 'init', '-q')
    path = PC.run_picks('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS, workers=1)
    assert path == rec / 'photo_picks_siglip1.json'
    with pytest.raises(RuntimeError, match='committed'):                      # written, not committed
        PC.run_cells('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS)
    _git(tmp_path, 'add', '-f', 'records/photo_picks_siglip1.json')
    _git(tmp_path, 'commit', '-q', '-m', 'picks')
    with pytest.raises(RuntimeError, match='gate v_photo not run'):           # committed: the next refusal is the gate record
        PC.run_cells('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS)
    (rec / 'photo_picks_siglip1.json').write_text(path.read_text() + ' ')     # edited after the commit
    with pytest.raises(RuntimeError, match='committed'):
        PC.run_cells('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS)


def test_picks_record_is_written_once(tmp_path, monkeypatch):
    rec = _records(tmp_path, 'pass')
    monkeypatch.setattr(PC, 'pick_readout', fake_pick())
    PC.run_picks('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS, workers=1)
    with pytest.raises(FileExistsError):
        PC.run_picks('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS, workers=1)


# ------------------------------------------------------------------------------------------------ picks and ranges
def test_picks_read_validation_only_and_use_r_E_for_emoset_and_own_range_for_affection(monkeypatch):
    calls = []
    data = FakeData(forbid=('test',))                       # a test row read would fail the run
    body = picks_body(monkeypatch, data, calls)
    assert {s for s, _ in data.asked} == set(KS) and {sp for _, sp in data.asked} == {'val'}
    by = {c['set']: c for c in calls}
    assert by['emoset']['r_map'] == pytest.approx(0.0069460867449697705, abs=1e-15)   # r14's committed r_E (D19), 0.00695
    assert round(by['emoset']['r_map'], 5) == 0.00695
    own = body['sets']['affection']['map_range']
    assert by['affection']['r_map'] == own['r'] and own['r'] > 0 and sorted(own['per_seed']) == ['0', '1', '2', '3', '4']
    assert by['affection']['seeds'] == list(SEEDS) and by['emoset']['n'] == 80
    for s, k in KS.items():                                 # AMI range at the class count: max - min of the five seed means
        a = body['sets'][s]['ami_range']
        v = list(a['per_seed'].values())
        assert a['k'] == k and a['r'] == pytest.approx(max(v) - min(v), abs=1e-15) and len(v) == 5
        assert body['sets'][s]['readout']['pick'] == 'none+k0'
    assert body['sets']['emoset']['r_map_source'].startswith('r14 stage2_emoset.json')
    assert body['sets']['emoset']['val_files']['3'] == 'sha-emoset-val-3'


def test_readout_picks_by_the_rule_of_each_set(monkeypatch):
    """Affection: r12's rule (`pf_stage2.readout` with the sorted keys); EmoSet: r14 part c step 2 (`bs_emoset_score.readout_doc`)."""
    seen = {}
    monkeypatch.setattr(PC.pf_stage2, 'readout', lambda xs, r, labels=None, keys=None, workers=8:
                        seen.update(affection=(sorted(xs), r, keys)) or dict(pick='a', scores={}))
    monkeypatch.setattr(PC.bs_emoset_score, 'readout_doc', lambda xs, labels, r, workers=8:
                        seen.update(emoset=(sorted(xs), r)) or dict(pick='e', scores={}))
    monkeypatch.setattr(PC.bs_emoset_score, 'check_order', lambda keys: seen.update(order=list(keys)))
    xs = {s: np.zeros((4, 3), dtype=np.float32) for s in SEEDS}
    assert PC.pick_readout('affection', xs, np.arange(4), ['k0', 'k1', 'k2', 'k3'], 0.01, 1)['pick'] == 'a'
    assert PC.pick_readout('emoset', xs, np.arange(4), ['k0', 'k1', 'k2', 'k3'], 0.007, 1)['pick'] == 'e'
    assert seen['affection'] == (list(SEEDS), 0.01, ['k0', 'k1', 'k2', 'k3']) and seen['emoset'] == (list(SEEDS), 0.007)
    assert seen['order'] == ['k0', 'k1', 'k2', 'k3']        # EmoSet's sha256 order is asserted, not assumed


# ------------------------------------------------------------------------------------------------ the cells
IDS = [f'{s}/siglip1/siglip1/emotion/{m}/{c}' for s, ms in (('affection', ('ami9', 'map')), ('emoset', ('ami8', 'map')))
       for m in ms for c in ('plain', 'teacher')]


def make_cells(monkeypatch, data=None, comps=None, parked=(), **kw):
    body = picks_body(monkeypatch, FakeData())
    return PC.score_cells(data or FakeData(), body, comps or COMPS, parked=parked, draws=dict(ami=30, map=60), workers=1, **kw)


def test_eight_cells_in_ticket_06_shape_and_readable_by_the_ladder(monkeypatch):
    cells = make_cells(monkeypatch)
    assert sorted(c['id'] for c in cells) == sorted(IDS)
    for c in cells:
        key, comp, cell = L.normalize(c, 'siglip1', 'siglip1', 'SC6', 'photo')     # ticket 08 accepts every cell
        assert key[0] == c['set'] and comp == ('P1' if c['comparator'] == 'plain' else 'Q1') and cell['outcome'] == c['outcome']
        assert c['lo'] <= c['d'] <= c['hi'] and c['d'] == pytest.approx(c['value'] - c['comparator_value'], abs=1e-12)
        assert c['encoder'] == 'siglip1' and c['arm'] == 'siglip1' and c['aspect'] == 'emotion'
    assert {c['k'] for c in cells if c['metric'] == 'ami'} == {8, 9}


def test_level_units_come_from_the_picks_and_emoset_map_uses_r_E(monkeypatch):
    body = picks_body(monkeypatch)
    cells = {c['id']: c for c in PC.score_cells(FakeData(), body, COMPS, draws=dict(ami=30, map=60), workers=1)}
    for s in KS:
        for comp in ('plain', 'teacher'):
            assert cells[f'{s}/siglip1/siglip1/emotion/ami{KS[s]}/{comp}']['r'] == body['sets'][s]['ami_range']['r']
    for comp in ('plain', 'teacher'):
        assert cells[f'affection/siglip1/siglip1/emotion/map/{comp}']['r'] == body['sets']['affection']['map_range']['r']
        assert cells[f'emoset/siglip1/siglip1/emotion/map/{comp}']['r'] == pytest.approx(0.0069460867449697705, abs=1e-15)


def test_a_strong_model_wins_and_a_weak_one_is_below(monkeypatch):
    strong = {c['id']: c['outcome'] for c in make_cells(monkeypatch)}
    assert all(strong[i] == 'WIN' for i in IDS if i.endswith('/plain'))
    weak_data = FakeData(f1_noise=6.0, plain_noise=0.3, teacher_noise=0.3)       # F1 far worse than both comparators
    body = picks_body(monkeypatch, weak_data)
    weak = {c['id']: c['outcome'] for c in PC.score_cells(weak_data, body, COMPS, draws=dict(ami=30, map=60), workers=1)}
    assert set(weak.values()) == {'BELOW'}


def test_comparators_are_scored_at_their_recorded_picks(monkeypatch):
    """The comparator pick is the older round's recorded one; a pick that is not a read-out name is refused."""
    comps = {k: dict(v, map_pick='no-such-readout') for k, v in COMPS.items()}
    with pytest.raises(Exception):
        make_cells(monkeypatch, comps=comps)
    cells = make_cells(monkeypatch)
    assert {c['comparator_readout'] for c in cells if c['metric'] == 'map'} == {'none+k0'}
    assert {c['candidate_readout'] for c in cells if c['metric'] == 'map'} == {'none+k0'}


def test_a_failed_rebuild_parks_only_the_cells_that_use_it(monkeypatch):
    cells = make_cells(monkeypatch, parked={'affection/map/teacher'})
    by = {c['id']: c for c in cells}
    parked = by['affection/siglip1/siglip1/emotion/map/teacher']
    assert parked['outcome'] == 'INCONCLUSIVE' and parked['value'] is None and 'affection/map/teacher' in parked['reason']
    assert [i for i, c in by.items() if c['outcome'] == 'INCONCLUSIVE'] == [parked['id']]
    L.normalize(parked, 'siglip1', 'siglip1', 'SC6', 'photo')                        # the ladder takes it as inconclusive
    assert len(cells) == 8


def test_cells_run_reads_the_gate_record_for_parked_units(tmp_path, monkeypatch):
    rec = _records(tmp_path, 'pass')
    _git(tmp_path, 'init', '-q')
    monkeypatch.setattr(PC, 'pick_readout', fake_pick())
    PC.run_picks('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS, workers=1)
    _git(tmp_path, 'add', '-f', 'records/photo_picks_siglip1.json')
    _git(tmp_path, 'commit', '-q', '-m', 'picks')
    units = {'emoset/map/plain': dict(ok=False), 'emoset/ami/plain': dict(ok=False), 'affection/map/plain': dict(ok=True)}
    (rec / 'gates' / 'v_photo.json').write_text(json.dumps(dict(gate='v_photo', status='fail', tries=1, details=dict(units=units))))
    path = PC.run_cells('siglip1', data=FakeData(), records=rec, gates=rec / 'gates', comps=COMPS, draws=dict(ami=30, map=60), workers=1)
    out = json.loads(path.read_text())
    assert out['arm'] == 'siglip1' and out['encoder'] == 'siglip1' and len(out['cells']) == 8
    assert {c['id'] for c in out['cells'] if c['outcome'] == 'INCONCLUSIVE'} == {
        'emoset/siglip1/siglip1/emotion/map/plain', 'emoset/siglip1/siglip1/emotion/ami8/plain'}
    assert out['picks_sha256'] and out['test_files']['affection']['0'] == 'sha-affection-test-0'


# ------------------------------------------------------------------------------------------------ gate records
def test_gate_status_rule_and_the_fourth_try(tmp_path):
    folder = tmp_path / 'gates'
    statuses = [PC.write_gate('v_photo', False, dict(x=i), folder)['status'] for i in range(3)]
    assert statuses == ['fail', 'fail', 'parked']                                   # fail before try 3, parked at try 3
    assert json.loads((folder / 'v_photo.json').read_text())['tries'] == 3
    with pytest.raises(AssertionError):                                              # a fourth try is refused, nothing written
        PC.write_gate('v_photo', True, {}, folder)
    assert json.loads((folder / 'v_photo.json').read_text())['status'] == 'parked'
    rec = PC.write_gate('vii_photo', True, dict(x=1), folder)
    assert rec['status'] == 'pass' and rec['tries'] == 1 and rec['gate'] == 'vii_photo'


def test_a_gate_that_failed_once_and_passes_next_time_is_pass_with_two_tries(tmp_path):
    PC.write_gate('v_photo', False, {}, tmp_path)
    rec = PC.write_gate('v_photo', True, {}, tmp_path)
    assert (rec['status'], rec['tries']) == ('pass', 2)


def test_reading_an_absent_gate_raises_gate_not_run(tmp_path):
    with pytest.raises(RuntimeError, match='gate v_photo not run'):
        PC.read_gate(tmp_path, 'v_photo')
    PC.write_gate('v_photo', True, dict(units={}), tmp_path)
    assert PC.read_gate(tmp_path, 'v_photo')['status'] == 'pass'


# ------------------------------------------------------------------------------------------------ the gate comparisons
def test_unit_comparison_is_within_1e_9_and_names_the_cell():
    ok = PC.compare_unit('emoset/map/plain', rebuilt=0.3638545779 + 9e-10, recorded=0.3638545779, required=True)
    bad = PC.compare_unit('emoset/map/plain', rebuilt=0.3638545779 + 2e-9, recorded=0.3638545779, required=True)
    assert ok['ok'] and not bad['ok'] and bad['cell'] == 'emoset/map/plain' and bad['diff'] == pytest.approx(2e-9, abs=1e-12)
    assert not PC.compare_unit('x/map/plain', rebuilt=float('nan'), recorded=0.1, required=True)['ok']


def test_gate_status_needs_every_unit_and_names_the_parked_cells():
    units = {n: PC.compare_unit(n, 0.5, 0.5, True) for n in ('a/map/plain', 'b/map/teacher')}
    units['c/ami/plain'] = PC.compare_unit('c/ami/plain', 0.5, 0.6, False)
    ok, parked = PC.judge_units(units)
    assert not ok and parked == ['c/ami/plain']                 # a non-required (extra) unit that differs also parks its cell


def test_the_recorded_comparator_values_are_the_briefs_to_10_decimals():
    comps = PC.comparator_records()
    assert round(comps[('affection', 'teacher')]['map_value'], 10) == 0.2945474692      # r13 teacher_artwork
    assert round(comps[('emoset', 'teacher')]['map_value'], 10) == 0.4518753521         # r14 SC10 comparator
    assert round(comps[('affection', 'plain')]['map_value'], 10) == 0.2545391742        # r14 SC5 SigLIP 1 comparator
    assert round(comps[('emoset', 'plain')]['map_value'], 10) == 0.3638545779           # r28 photos.json cell c
    assert round(comps[('emoset', 'plain')]['ami_value'], 10) == 0.2672607188
    assert all(c['map_pick'] == 'center+k10' for c in comps.values())
    r, source = PC.read_r_E()
    assert round(r, 5) == 0.00695 and 'stage2_emoset.json' in source


# ------------------------------------------------------------------------------------------------ the export reader
def _export(tmp_path, keys, x16, *, set_name='affection', split='val', seed=0, **over):
    folder = tmp_path / 'exports' / set_name / 'photo_siglip1'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{split}__s{seed}.npz'
    np.savez(path, keys=np.array(keys), emb=x16)
    side = dict(dataset=set_name, arm='photo_siglip1', split=split, seed=seed, rows=len(keys), sha256=rs_vh.file_sha(path),
                checkpoint=str(tmp_path / 'photo/siglip1' / f's{seed}' / 'emotion.pt'))
    side.update(over)
    path.with_suffix('.json').write_text(json.dumps(side))
    return path


def _rows16(n, seed=0):
    return _unit(np.random.default_rng(seed).normal(size=(n, W))).astype(np.float16)


def test_export_rows_are_joined_by_key_not_position(tmp_path):
    keys = ['a', 'b', 'c', 'd']
    x = _rows16(4)
    perm = [2, 0, 3, 1]
    path = _export(tmp_path, [keys[i] for i in perm], x[perm])
    got = PC.read_export(path, 'affection', 'val', 0, keys, set(keys))
    assert np.array_equal(got, x) and got.dtype == np.float16


@pytest.mark.parametrize('what, kw', [
    ('sidecar sha', dict(sha256='0' * 64)), ('arm', dict(arm='photo_qwen')), ('split', dict(split='test')),
    ('seed', dict(seed=3)), ('dataset', dict(dataset='emoset'))])
def test_export_sidecar_must_describe_the_file(tmp_path, what, kw):
    keys = ['a', 'b', 'c']
    path = _export(tmp_path, keys, _rows16(3), **kw)
    with pytest.raises(AssertionError):
        PC.read_export(path, 'affection', 'val', 0, keys, set(keys))


def test_export_key_set_must_equal_the_split_files_and_rows_must_be_unit(tmp_path):
    keys = ['a', 'b', 'c']
    path = _export(tmp_path, keys + ['z'], _rows16(4))
    with pytest.raises(AssertionError, match='key set'):
        PC.read_export(path, 'affection', 'val', 0, keys, set(keys))               # an extra key
    path = _export(tmp_path, keys, _rows16(3) * np.float16(1.5), seed=1)
    with pytest.raises(AssertionError, match='unit'):
        PC.read_export(path, 'affection', 'val', 1, keys, set(keys))


def test_export_must_come_from_the_expected_checkpoint(tmp_path):
    keys = ['a', 'b']
    path = _export(tmp_path, keys, _rows16(2), checkpoint='/elsewhere/s0/emotion.pt')
    PC.read_export(path, 'affection', 'val', 0, keys, set(keys))                  # no expectation: accepted (r14's exports, gate (vii))
    with pytest.raises(AssertionError, match='checkpoint'):
        PC.read_export(path, 'affection', 'val', 0, keys, set(keys), checkpoint=tmp_path / 'photo/siglip1/s0/emotion.pt')


def test_f1_runs_must_have_been_trained_on_the_arms_caption_source(tmp_path):
    for s in SEEDS:
        d = tmp_path / 'photo' / 'siglip1' / f's{s}'
        d.mkdir(parents=True)
        (d / 'run.json').write_text(json.dumps(dict(cap_source='siglip1', cap_width=1152, cap_store_sha256='abc')))
    ident = PC.f1_identity(tmp_path, 'siglip1')
    assert ident['cap_store_sha256'] == 'abc' and ident['cap_width'] == 1152
    with pytest.raises(AssertionError, match='cap_source'):
        PC.f1_identity(tmp_path, 'siglip1_prefix')
    (tmp_path / 'photo/siglip1/s2/run.json').write_text(json.dumps(dict(cap_source='siglip1', cap_width=1152, cap_store_sha256='zzz')))
    with pytest.raises(AssertionError, match='cap_store_sha256'):
        PC.f1_identity(tmp_path, 'siglip1')


# ------------------------------------------------------------------------------------------------ the command line
def test_cli_names_are_the_ones_ticket_15_calls():
    p = PC.build_parser()
    assert p.parse_args(['picks', '--arm', 'siglip1']).cmd == 'picks'
    assert p.parse_args(['cells', '--arm', 'siglip1_prefix']).arm == 'siglip1_prefix'
    for cmd in ('gate-v-photo', 'gate-vii-photo'):
        assert p.parse_args([cmd, '--no-record']).no_record
    with pytest.raises(SystemExit):
        p.parse_args(['cells'])                                                     # --arm is required
