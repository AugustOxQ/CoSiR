"""D13 wait rule of the runner (ticket 09). The rest of the runner is proved by the smoke run."""
from datetime import datetime

import hashlib
import json
import pytest

import tcl_data
import tcl_paths as P
import tcl_run as R
from tcl_run import heavy_wait_seconds as wait
from test_tcl_score import repo  # noqa: F401

H = 3600


def test_weekday_before_ten_and_fits():
    assert wait(datetime(2026, 11, 9, 8, 0), 60) == 0                  # Monday 08:00 to 09:00


def test_weekday_before_ten_and_overlaps():
    assert wait(datetime(2026, 11, 9, 9, 30), 45) == 9.5 * H          # ends 10:15: wait to 19:00


def test_weekday_inside_window():
    assert wait(datetime(2026, 11, 11, 14, 0), 20) == 5 * H           # Wednesday 14:00: wait to 19:00


def test_weekday_after_nineteen_ends_in_small_hours():
    assert wait(datetime(2026, 11, 10, 19, 0), 60) == 0
    assert wait(datetime(2026, 11, 10, 23, 30), 60) == 0


def test_saturday_and_sunday_are_free():
    assert wait(datetime(2026, 11, 7, 11, 0), 60) == 0
    assert wait(datetime(2026, 11, 8, 15, 0), 60) == 0


@pytest.mark.parametrize('failed', ['art', 'mm', None])
def test_gate_iv_records_each_group_independently(tmp_path, monkeypatch, failed):
    monkeypatch.setattr(P, 'OUT', tmp_path)

    def check(group):
        assert group != failed, f'{group} failed'
        return {f'{group}_size': 1}

    monkeypatch.setattr(tcl_data, 'gate_iv_art', lambda: check('art'))
    monkeypatch.setattr(tcl_data, 'gate_iv_mm', lambda: check('mm'))
    assert R.cmd_gate_iv() == (1 if failed else 0)
    doc = json.loads(R.gate_path('gate_iv').read_text())
    assert {g: doc[g] for g in R.GROUPS} == {g: g != failed for g in R.GROUPS}
    assert doc['sizes'] == {f'{g}_size': 1 for g in R.GROUPS if g != failed}


def test_decide_rerun_removes_old_verdicts_and_blocks_stage3(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(P, 'RECORDS', tmp_path / 'records')
    monkeypatch.setattr(P, 'SMOKE', 0)
    monkeypatch.setattr(R, 'refuse_unless_committed', lambda: [])
    verdicts = P.record_path('verdicts.json')
    verdicts.parent.mkdir()
    verdicts.write_text('{"old": true}')
    folder = tmp_path / 'stage2'
    folder.mkdir()
    for name in ('outcomes.json', 'cells.json'):
        (folder / name).write_text('{"old": true}')
    called = []

    def decide(stage, name, argv, outputs, ok_rcs):
        assert (stage, name) == ('stage2', 'decide')
        assert not verdicts.exists()
        assert all(not path.exists() for path in outputs)
        for path in outputs:
            path.write_text('{"new": true}')
        called.append(name)
        return dict(failed=False)

    monkeypatch.setattr(R, 'run_step', decide)
    assert R.main(['stage2', '--only', 'decide']) == 0
    assert called == ['decide'] and not verdicts.exists()
    assert R.main(['stage3', '--only', 'diag']) == 2
    assert called == ['decide']


@pytest.fixture
def c13_world(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    folder = tmp_path / 'stage2'
    folder.mkdir()
    (folder / 'outcomes.json').write_bytes(b'{"verdicts": {}, "voided": []}\n')
    (folder / 'cells.json').write_bytes(b'[]\n')
    record = {'pass': True, 'sha256': {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ('stage2/outcomes.json', 'stage2/cells.json')}}
    return (*repo, record)


@pytest.mark.parametrize('fault', ['no_outcomes', 'no_c13', 'uncommitted', 'pass_false',
                                 'wrong_outcomes_hash', 'wrong_cells_hash',
                                 'missing_outcomes_hash', 'missing_cells_hash'])
def test_verdicts_refuses_without_current_committed_passing_c13(c13_world, fault):
    write, commit, records, rec = c13_world
    if fault == 'no_outcomes':
        (P.OUT / 'stage2/outcomes.json').unlink()
    if fault not in ('no_outcomes', 'no_c13'):
        if fault == 'pass_false':
            rec['pass'] = False
        for file in ('outcomes', 'cells'):
            name = f'stage2/{file}.json'
            if fault == f'wrong_{file}_hash':
                rec['sha256'][name] = 'deadbeef'
            if fault == f'missing_{file}_hash':
                del rec['sha256'][name]
        write('c13.json', json.dumps(rec))
        if fault != 'uncommitted':
            commit('c13.json')
    if fault != 'no_outcomes':
        assert R.c13_ok()
    assert R.run_verdicts() == 2
    assert not (records / 'verdicts.json').exists()


def test_verdicts_copies_exact_outputs_only_while_committed_c13_is_current(c13_world):
    write, commit, records, rec = c13_world
    write('c13.json', json.dumps(rec))
    commit('c13.json')
    assert R.c13_ok() == ''
    assert R.run_verdicts() == 0
    outcomes = P.OUT / 'stage2/outcomes.json'
    verdicts = records / 'verdicts.json'
    original = outcomes.read_bytes()
    assert verdicts.read_bytes() == original
    outcomes.write_bytes(b'{"verdicts": {"changed": 1}, "voided": []}\n')
    assert R.c13_ok()
    assert R.run_verdicts() == 2
    assert verdicts.read_bytes() == original
