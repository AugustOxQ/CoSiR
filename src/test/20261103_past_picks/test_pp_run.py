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


def test_semart_missing_store_is_listed_not_refused():
    required = sorted(str(p.relative_to(PP.SEMART_PRE)) for (d, _), p in PP.NEW_STORE.items() if d == 'semart')
    assert R.missing_semart_stores({'stores_checked': []}) == required
    assert R.missing_semart_stores({'stores_checked': required}) == []


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


def _clean_status(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    def ok(x):
        return {k: ok(v) for k, v in x.items()} if isinstance(x, dict) else 'ok'
    status = R.collect_gates()[1]
    return {'gates': ok(status['gates']), 'runs': ok(status['runs'])}


def test_stage2_skips_only_the_voided_path(tmp_path, monkeypatch):
    clean = _clean_status(tmp_path, monkeypatch)
    encs = ('clip_b32', 'clip_b16')
    ts = R.tasks(2, encs)
    names = [t['name'] for t in ts]
    assert [t['name'] for t in R.skip_voided(ts, encs, clean)] == names
    bad = json.loads(json.dumps(clean))
    bad['gates']['i_store']['clip_b32']['affection'] = 'failed'
    kept = [t['name'] for t in R.skip_voided(ts, encs, bad)]
    assert set(names) - set(kept) == {'score_affection_clip_b32'}
    assert 'Skipped affected path' in (tmp_path / 'stage2/logs/score_affection_clip_b32.log').read_text()


# ------------------------------------------------------------------------------------------------ fix wave B
def test_stage3_argv_parses_in_pp_probe_outside_smoke_and_in_smoke(monkeypatch):
    import pp_probe as P
    seen = []
    monkeypatch.setattr(P, 'cmd_cells', lambda encs: seen.append(encs) or 0)
    monkeypatch.setattr(PP, 'SMOKE', 0)
    (t,) = R.tasks(3, PP.ENCODERS)
    assert '--enc' not in t['argv']
    assert P.main([str(a) for a in t['argv'][2:]]) == 0 and seen == [None]      # the real parser, as the run chat calls it
    monkeypatch.setattr(PP, 'SMOKE', 1)
    (t,) = R.tasks(3, ('clip_b32', 'clip_l14'))
    assert P.main([str(a) for a in t['argv'][2:]]) == 0 and seen[-1] == ['clip_b32', 'clip_l14']


def test_semart_set_writes_gate_and_set_first_and_records_missing_stores(tmp_path, monkeypatch, capsys):
    import pp_semart
    monkeypatch.setattr(R, 'OUT', tmp_path)
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    monkeypatch.setattr(PP, 'SMOKE', 1)
    required = sorted(str(p.relative_to(PP.SEMART_PRE)) for (d, _), p in PP.NEW_STORE.items() if d == 'semart')
    doc = dict(gate={'pass': True}, n_set=3, artists_set=2, stores_checked=required[1:])
    monkeypatch.setattr(pp_semart, 'record', lambda: dict(doc))
    assert R.internal('semart-set') == 0                       # no assert: the missing store is data, not a crash
    assert R.jload(tmp_path / 'stage0/gate_vi.json') == {'pass': True}
    written = R.jload(PP.record_path('semart_set.json'))
    assert written['missing_stores'] == required[:1]
    assert required[0] in capsys.readouterr().out
    doc['stores_checked'] = required
    R.internal('semart-set')
    assert R.jload(PP.record_path('semart_set.json'))['missing_stores'] == []


def test_a_missing_semart_store_fails_its_store_gate_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    st = R.collect_gates()[1]['gates']
    assert st['i_store']['dinov2_large']['semart'] == 'failed' and st['i_semart_genre_word'] == 'failed'


def test_a_gate_step_with_a_crash_and_an_old_named_output_is_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    out = tmp_path / 'stage3/cells_cost.json'
    R.jdump(out, [])
    rec = R.run_step('stage3', 'cells_cost', [R.sys.executable, '-c', 'raise SystemExit(1)'], [out], (0, 1))
    assert rec['failed'] and 'stale' in rec['problem']
    fresh = [R.sys.executable, '-c', f'open({str(out)!r}, "w").write("[]"); raise SystemExit(1)']
    assert not R.run_step('stage3', 'cells_cost', fresh, [out], (0, 1))['failed']   # exit 1, fresh output: a result


def test_r14_gate_vii_is_read_per_path(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    R.jdump(tmp_path / 'stage0/gate_vii_r14.json', dict(passed=False, artelingo=dict(passed=True), affection=dict(passed=False)))
    vii = R.collect_gates()[1]['gates']['vii']
    assert vii['artelingo'] == 'ok' and vii['affection'] == 'failed'
    R.jdump(tmp_path / 'stage0/gate_vii_r14.json', dict(passed=True))              # no per-path result: fails closed
    vii = R.collect_gates()[1]['gates']['vii']
    assert vii['artelingo'] == 'failed' and vii['affection'] == 'failed'
    R.jdump(tmp_path / 'stage0/gate_vii_r14.json', dict(passed=True, artelingo=dict(passed=True), affection=dict(passed=True)))
    vii = R.collect_gates()[1]['gates']['vii']
    assert vii['artelingo'] == vii['affection'] == 'ok'


def test_gate_iii_parts_are_read_from_their_own_files(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    R.jdump(tmp_path / 'stage0/gate_iii_r10.json', dict(part='r10', **{'pass': True}))
    iii = R.collect_gates()[1]['gates']['iii']
    assert iii == {'r10': 'ok', 'r14': 'failed'}                 # a part that wrote nothing is failed, not hidden
    (tmp_path / 'stage0/gate_iii_r10.json').unlink()
    R.jdump(tmp_path / 'stage0/gate_iii_r14.json', dict(part='r14', **{'pass': True}))
    assert R.collect_gates()[1]['gates']['iii'] == {'r10': 'failed', 'r14': 'ok'}


def test_printed_commit_command_carries_no_session_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    monkeypatch.setattr(PP, 'SMOKE', 0)
    monkeypatch.setattr(PP, 'record_path', lambda name: tmp_path / 'records' / name)
    monkeypatch.setattr(PP, 'REPO', tmp_path)
    R.merge_records(('clip_b32',))
    out = capsys.readouterr().out
    assert 'git -C' in out and 'Claude-Session' not in out
