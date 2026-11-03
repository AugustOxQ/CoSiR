"""Only runner gate/status/record boundaries; no expensive scorer replays."""
import json
import pytest
import pp_paths as PP
import pp_run as R
from pp_outcome import check_status


def test_wrong_pass_key_is_refused():
    with pytest.raises(AssertionError, match='passed'):
        R.flag({'pass': True}, 'passed')
    assert not R.flag({'passed': False, 'values_ok': True}, 'passed')


def test_exact_status_schema_and_missing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    evidence, status = R.collect_gates()
    assert check_status(status) == status
    assert status['gates']['i_store']['clip_b32'] == {'mmart': 'failed', 'affection': 'failed'}
    assert 'semart' not in status['gates']['i_store']['clip_b32']
    status['extra'] = 'ok'
    with pytest.raises(AssertionError, match='top level'):
        check_status(status)


def test_passed_not_rc_or_values_and_node_status(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    for p in ('r14', 'r15', 'semart'):
        R.jdump(tmp_path / f'stage0/gate_iv_{p}.json', dict(passed=True))
    for kind in ('cstar', 'photo'):
        R.jdump(tmp_path / f'node/{kind}_clip_b32.json', dict(status='ok', seeds=list(PP.SEEDS)))
    _, st = R.collect_gates()
    assert st['gates']['iv'] == 'ok' and st['runs']['clip_b32']['photo'] == 'ok'
    R.jdump(tmp_path / 'stage0/gate_iv_r15.json', dict(passed=False, values_ok=True))
    R.jdump(tmp_path / 'node/cstar_clip_b32.json', dict(status='failed', seeds=list(PP.SEEDS)))
    _, st = R.collect_gates()
    assert st['gates']['iv'] == 'failed' and st['runs']['clip_b32']['photo'] == 'failed'


def test_merge_preserves_encoder_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    monkeypatch.setattr(PP, 'SMOKE', 1)
    R.jdump(tmp_path / 'stage1/mmart_clip_b32.json', {'picks': {'a': 1}})
    R.merge_records(('clip_b32', 'clip_b16'))
    assert R.jload(PP.record_path('stage1_clip_b32.json'))['mmart'] == {'picks': {'a': 1}}
    assert 'mmart' not in R.jload(PP.record_path('stage1_clip_b16.json'))


def test_semart_missing_store_is_refused():
    with pytest.raises(AssertionError, match='missing SemArt stores'):
        R.assert_semart_stores({'stores_checked': []})
    checked = [str(p.relative_to(PP.SEMART_PRE)) for (d, _), p in PP.NEW_STORE.items() if d == 'semart']
    R.assert_semart_stores({'stores_checked': checked})


def test_failed_loaded_probe_cannot_become_a_void(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    R.jdump(tmp_path / 'stage0/gate_viii.json', {'pass': False})
    R.jdump(tmp_path / 'probe/batches/qwen__e2e.json', {'cpu_load_before': [30, 25, 20], 'cpu_load_after': [22, 20, 19]})
    with pytest.raises(AssertionError, match='rerun quiet'):
        R.collect_gates()
    R.jdump(tmp_path / 'probe/batches/qwen__e2e.json', {'cpu_load_before': [10, 12, 13], 'cpu_load_after': [12, 13, 14]})
    assert R.collect_gates()[1]['gates']['viii'] == 'failed'


def test_stale_gate_is_not_accepted_after_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    gate = tmp_path / 'stage0/gate_iv_r15.json'
    R.jdump(gate, {'passed': True})
    # Exit 1 is a legitimate gate result ONLY with fresh evidence; this crash writes nothing.
    rec = R.run_step('stage0', 'gate_iv_r15', [R.sys.executable, '-c', 'raise SystemExit(1)'], [gate], (0, 1))
    assert rec['failed'] and rec['attempt'] == 2 and 'stale' in rec['problem']
    assert R.collect_gates()[1]['gates']['iv'] == 'failed'
