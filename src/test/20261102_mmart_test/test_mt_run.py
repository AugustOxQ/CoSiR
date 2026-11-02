"""Runner guards and D17 wiring; no real data or scoring subprocesses."""
import hashlib
import json

import pytest

import mt_paths as MP
import mt_run as R


C13_FILES = ('stage2/outcomes.json', 'stage2/cells_clu.json', 'stage2/cells_ret.json')


@pytest.fixture
def c13(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    monkeypatch.setattr(MP, 'OUT', tmp_path)
    monkeypatch.setattr(MP, 'SMOKE', 1)
    record = {'pass': True, 'sha256': {}}
    for name in C13_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'checked': name}))
        record['sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    R.jdump(MP.record_path('c13.json'), record)
    return record


def test_c13_passes_with_current_hashes(c13):
    assert R.c13_ok() == ''


@pytest.mark.parametrize('value', [False, 1, None])
def test_c13_refuses_without_true_pass(c13, value):
    c13['pass'] = value
    R.jdump(MP.record_path('c13.json'), c13)
    assert '"pass" is not true' in R.c13_ok()


@pytest.mark.parametrize('name', C13_FILES)
def test_c13_refuses_missing_sha256_entry(c13, name):
    del c13['sha256'][name]
    R.jdump(MP.record_path('c13.json'), c13)
    assert f'missing sha256 entry for {name}' in R.c13_ok()


@pytest.mark.parametrize('hashes', [None, [], 'invalid'])
def test_c13_refuses_invalid_sha256_map(c13, hashes):
    c13['sha256'] = hashes
    R.jdump(MP.record_path('c13.json'), c13)
    assert 'missing sha256 entry for stage2/outcomes.json' in R.c13_ok()


@pytest.mark.parametrize('name', C13_FILES)
def test_c13_refuses_changed_output(c13, name):
    (R.OUT / name).write_text('{"changed_after_c13": true}')
    assert f'sha256 mismatch for {name}' in R.c13_ok()


def test_c13_refuses_missing_record(tmp_path, monkeypatch):
    monkeypatch.setattr(MP, 'OUT', tmp_path)
    monkeypatch.setattr(MP, 'SMOKE', 1)
    assert 'c13.json does not exist' in R.c13_ok()


def test_c13_refuses_unreadable_output(c13, monkeypatch):
    # Point at an empty directory; leave every checked file in place.
    monkeypatch.setattr(R, 'OUT', R.OUT / 'absent')
    assert 'C13 checked file stage2/outcomes.json cannot be read' in R.c13_ok()


def test_c13_refuses_uncommitted_record(c13, monkeypatch):
    def uncommitted(name):
        assert name == 'c13.json'
        raise RuntimeError('c13.json is not committed unchanged in HEAD')
    monkeypatch.setattr(MP, 'read_committed_record', uncommitted)
    assert 'c13.json is not committed unchanged in HEAD' in R.c13_ok()


@pytest.fixture
def status():
    return dict(gates=dict(i_nearcopy=R.OK, iii=R.OK, vi=R.OK,
                           i_default=dict.fromkeys(R.ENCS, R.OK), ii=dict.fromkeys(R.ENCS, R.OK),
                           iv=dict.fromkeys(R.ENCS, R.OK)),
                stage1=dict(ranges=dict.fromkeys(R.ENCS, R.OK), picks=dict.fromkeys(R.ENCS, R.OK),
                            teacher_pick=R.OK))


@pytest.mark.parametrize('teacher_ok', [True, False])
def test_teacher_record_required_only_for_ok_pick(status, monkeypatch, teacher_ok):
    status['stage1']['teacher_pick'] = R.OK if teacher_ok else R.FAILED
    seen = []

    def read(name):
        seen.append(name)
        if name == 'stage1_teacher.json':
            raise FileNotFoundError(name)
        return status if name == 'status.json' else {}
    monkeypatch.setattr(MP, 'read_committed_record', read)
    problems = R.refuse_unless_committed(R.ENCS)
    assert bool(problems) == teacher_ok
    assert ('stage1_teacher.json' in seen) == teacher_ok
    tasks, skipped = R.stage2_tasks(R.ENCS, status, None, None)
    assert not skipped and len(tasks) == 2 * len(R.ENCS)
    for task in tasks:
        scs = task['argv'][task['argv'].index('--scs') + 1:]
        expected = (['SC2', 'SC4'] if teacher_ok else ['SC2']) if '_ret_' in task['name'] else ['SC1', 'SC3', 'SC4']
        assert scs == expected


@pytest.mark.parametrize('step,gate', [('export', 'ii'), ('matched_export', 'iv')])
def test_failed_export_maps_to_gate_and_score_plan(status, monkeypatch, step, gate):
    monkeypatch.setattr(R, '_load', lambda path: {'pass': True} if path.name.startswith(('gate_ii_', 'gate_iv_')) else None)
    monkeypatch.setattr(R, 'last_step', lambda stage, name: {'rc': 1, 'failed': True} if name == f'{step}_siglip2' else None)
    evidence, gates = R.collect_gates()
    assert gates[gate]['siglip2'] == R.FAILED
    assert evidence[gate]['siglip2']['export_step_failed'] is True
    assert all(gates[gate][e] == R.OK for e in R.ENCS if e != 'siglip2')
    status['gates'][gate] = gates[gate]
    assert R.void_plan(status, 'siglip2') == (None if gate == 'ii' else dict(clu=['SC1', 'SC4'], ret=['SC2', 'SC4']))
