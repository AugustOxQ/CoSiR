"""Dry tests of the overnight queue (ticket 13 fixes): fake tasks, a temp results root, no lock, no GPU, nothing under /data."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

import run_r8_queue as Q


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv('R8_RESULTS', str(tmp_path / 'res'))
    c = Q.Cfg(True)
    c.logs.mkdir(parents=True)
    return c


def engine(cfg, queue, tasks):
    e = Q.Engine(cfg, queue, tasks, False)
    e.poll = 0.05
    return e


def test_cond_raising_before_input_exists_does_not_crash(cfg):
    """(a) the engine checks ready() before cond(), and an exception in cond() means "not ready yet"."""
    inp = cfg.root / 'input.json'
    ran = []

    def cond():
        return None if json.loads(inp.read_text())['ok'] else ('skip', 'no')       # FileNotFoundError before the input exists
    t = Q.Task('late', 'light', fn=lambda: ran.append(1), cond=cond, outputs=[cfg.root / 'late.out'])
    t2 = Q.Task('guarded', 'light', fn=lambda: ran.append(2), ready=Q.exists(inp), cond=cond)
    t.fn = lambda: (ran.append(1), (cfg.root / 'late.out').write_text('x'))
    e = engine(cfg, 'cpu', [t, t2])
    threading.Timer(0.4, lambda: inp.write_text(json.dumps({'ok': True}))).start()
    line, code = e.run()
    assert code == 0 and sorted(ran) == [1, 2], line
    assert t.state == 'ok' and t2.state == 'ok'
    assert sum('cond() raised' in l for l in (cfg.logs / 'queue_cpu.log').read_text().splitlines()) == 1     # logged once


def test_exception_after_gpu_done_fails_only_that_step(cfg):
    Q.write_state(cfg, 'done')
    bad = Q.Task('bad', 'light', fn=lambda: None, cond=lambda: json.loads((cfg.root / 'nope.json').read_text()))
    good = Q.Task('good', 'light', fn=lambda: None)
    never = Q.Task('never', 'light', fn=lambda: None, ready=lambda: 1 / 0)
    e = engine(cfg, 'cpu', [bad, good, never])
    e.t_start -= 1000                      # the state file counts (older than this queue's start, past the grace time)
    line, code = e.run()
    assert (bad.state, good.state, never.state) == ('failed', 'ok', 'failed')
    assert 'FileNotFoundError' in bad.why and 'ZeroDivisionError' in never.why and code == 1


def test_state_file_is_written_atomically(cfg):
    Q.write_state(cfg, 'running')
    Q.write_state(cfg, 'done')
    assert json.loads(cfg.gpu_state().read_text())['state'] == 'done'
    assert [p.name for p in cfg.logs.iterdir() if '.tmp' in p.name] == []


def test_c13_disagreement_is_failed_on_resume(cfg):
    out = cfg.root / 'rederive' / 'trigger_rederive.json'
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({'status': 'disagree'}))
    tasks = Q.cpu_tasks(cfg)
    tc = next(t for t in tasks if t.name == 'trigger_c13')
    assert tc.verify() and 'disagree' in tc.verify()
    (cfg.root / 'rederive' / 'trigger_rederive.json').write_text(json.dumps({'status': 'agree'}))
    assert tc.verify() is None


def write_nbc_state(cfg, how):
    """Trigger fired; NBC dropped by 07 (how='07'), 08 ('08') or 09 ('09')."""
    R = cfg.root
    (R / 'groupings').mkdir(parents=True, exist_ok=True)
    (R / 'heads').mkdir(exist_ok=True)
    (R / 'readers').mkdir(exist_ok=True)
    (R / 'probe_trigger.json').write_text(json.dumps({'triggered': True, 'strength': 1.0}))
    g07 = {'groupings': {}, 'dropped': {}}
    if how == '07':
        g07['dropped'] = {'NBC': 'fewer than 2 groups', 'ZNBC': 'no control: NBC is dropped (fewer than 2 groups)'}
    (R / 'groupings' / 'r8_groupings_nbc.json').write_text(json.dumps(g07))
    if how == '08':
        (R / 'heads' / 'head_NBC.json').write_text(json.dumps({'dropped': True, 'reason': 'a group is empty in a half'}))
    if how == '09':
        (R / 'readers' / 'reader_NBC.json').write_text(json.dumps({'dropped': True, 'reason': 'unfillable bank'}))


@pytest.mark.parametrize('how', ['07', '08', '09'])
def test_nbc_dropped_branch(cfg, how):
    """(b) NBC dropped after the trigger fired: nbc_score is skipped, drops_nbc runs after the readers, 10 and 11 get no index_nbc."""
    tasks = {t.name: t for t in Q.cpu_tasks(cfg)}
    write_nbc_state(cfg, how)
    for n in ('nbc_taus', 'nbc_score'):
        kind, why = tasks[n].cond()
        assert kind == 'skip' and 'dropped under M16' in why and how in why
    assert tasks['drops_nbc'].deps == ['nbc_readers']
    Q.write_drops(cfg, 'nbc', ['NBC', 'ZNBC'], {}, [cfg.root / 'groupings' / 'r8_groupings_nbc.json'])
    d = json.loads((cfg.root / 'groupings' / 'r8_drops_nbc.json').read_text())['dropped']
    assert d == {} if how == '07' else set(d) == {'NBC', 'ZNBC'}       # 07 already listed both in its own file
    for name in ('gate_pending', 'gate_c13'):
        cmd = tasks[name].build(False)['cmd']
        assert 'index_nbc.json' not in ' '.join(cmd) and str(cfg.root / 'seed42' / 'index.json') in cmd
        assert str(cfg.root / 'groupings' / 'r8_drops_nbc.json') in cmd


def test_nbc_not_dropped_keeps_index_and_scores(cfg):
    tasks = {t.name: t for t in Q.cpu_tasks(cfg)}
    (cfg.root).mkdir(parents=True, exist_ok=True)
    (cfg.root / 'probe_trigger.json').write_text(json.dumps({'triggered': True, 'strength': 1.0}))
    assert tasks['nbc_score'].cond() is None and tasks['nbc_taus'].cond() is None
    assert any(str(c).endswith('index_nbc.json') for c in tasks['gate_pending'].build(False)['cmd'])
    (cfg.root / 'seed42').mkdir()
    (cfg.root / 'seed42' / 'arrays_nbc.npz').write_text('x')
    assert tasks['nbc_readers'].cond()[0] == 'skip'                 # read_13 #10


def fake_wrap(self, t, cmd, env, plan=False):
    return cmd


def test_load_cap_path_exits_4(cfg, monkeypatch, capsys):
    """(c) the guard returns 4 at the cap and does not start the run; the GPU queue stops with "stopped: load cap" and exit 4."""
    monkeypatch.setattr(Q, 'read_load', lambda: 9.5)
    monkeypatch.setattr(Q.time, 'sleep', lambda s: None)
    monkeypatch.setenv('R8_LOAD_WAIT_MAX', '3')
    monkeypatch.setenv('R8_LOAD_POLL', '1')
    monkeypatch.setenv('R8_LOAD_CAP_FILE', str(cfg.load_cap_file()))
    monkeypatch.setattr(Q.subprocess, 'run', lambda *a, **k: pytest.fail('the FCMI run must not start'))
    assert Q.loadwait(['--', 'echo', 'x']) == 4
    assert 'LOAD CAP' in capsys.readouterr().out and json.loads(cfg.load_cap_file().read_text())['load'] == 9.5
    monkeypatch.undo()
    # the queue: a fake FCMI step that ends with exit 4 (as the guard does), then steps that must not run
    monkeypatch.setattr(Q.Engine, 'wrap', fake_wrap)
    cap_cmd = [sys.executable, '-c', f'import sys; open({str(cfg.load_cap_file())!r},"w").write(\'{{"load": 9.5}}\'); sys.exit(4)']
    ran = []
    T = [Q.Task('start', 'light', fn=lambda: Q.write_state(cfg, 'running')),
         Q.Task('FC3', 'fcmi', build=lambda plan: {'cmd': cap_cmd}, after=['start']),
         Q.Task('enc', 'light', fn=lambda: ran.append(1), after=['FC3']),
         Q.Task('done', 'light', fn=lambda: Q.write_state(cfg, 'done'), after=['enc'])]
    e = engine(cfg, 'gpu', T)
    line, code = e.run()
    assert code == 4 and 'stopped: load cap' in line and not ran
    assert json.loads(cfg.gpu_state().read_text())['state'] == Q.STATE_CAP
    assert [t.state for t in T] == ['ok', 'failed', 'blocked', 'blocked']
    # the CPU queue counts that state as "GPU done"; its waiting step fails with a clear message, the rest goes on
    w = Q.Task('wait', 'light', fn=lambda: None, ready=lambda: False, ready_doc='some gpu output')
    ok = Q.Task('ok', 'light', fn=lambda: None)
    e2 = engine(cfg, 'cpu', [w, ok])
    e2.t_start -= 1000
    line2, code2 = e2.run()
    assert w.state == 'failed' and 'GPU queue is done' in w.why and ok.state == 'ok'


def test_rule_covered_exit_does_not_make_exit_code_1(cfg, monkeypatch):
    monkeypatch.setattr(Q.Engine, 'wrap', fake_wrap)
    rec = cfg.root / 'rec.json'
    cmd = [sys.executable, '-c', f'open({str(rec)!r},"w").write("{{}}"); raise SystemExit(1)']
    t = Q.Task('views', 'gpu', build=lambda plan: {'cmd': cmd}, outputs=[rec], rule_check=lambda: 'M4 view check failed')
    cfg.root.mkdir(parents=True, exist_ok=True)
    line, code = engine(cfg, 'gpu', [t]).run()
    assert t.state == 'failed' and code == 0 and 'dropped under the rule: views' in line and 'all ok' in line


def test_stop_marker_stops_cpu_queue(cfg):
    cfg.stop_marker().write_text('{}')
    t = Q.Task('a', 'light', fn=lambda: None)
    line, code = engine(cfg, 'cpu', [t]).run()
    assert t.state == 'blocked' and code == 3


def test_write_drops_dumps_before_writing(cfg, monkeypatch):
    (cfg.root / 'groupings').mkdir(parents=True)
    g = cfg.root / 'groupings' / 'g.json'
    g.write_text(json.dumps({'groupings': {}, 'dropped': {}}))
    monkeypatch.setattr(Q.json, 'dumps', lambda *a, **k: (_ for _ in ()).throw(ValueError('boom')))
    with pytest.raises(ValueError):
        Q.write_drops(cfg, 'nbc', ['NBC'], {'NBC': 'x'}, [g])
    assert not (cfg.root / 'groupings' / 'r8_drops_nbc.json').exists()
