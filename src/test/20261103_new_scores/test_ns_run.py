"""Runner regressions using synthetic steps and temporary outputs only."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as P  # noqa: E402
import ns_outcome  # noqa: E402
import ns_ret  # noqa: E402
import ns_run  # noqa: E402


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(ns_run, 'OUT', tmp_path)
    monkeypatch.setattr(P, 'SMOKE', 1)
    return tmp_path


def test_cmd_cells_reports_missing_claim_cells_and_writes_rest(outputs, monkeypatch, capsys):
    cells = [dict(c) for c in ns_outcome.claim_cells()
             if c['set'] == 'mmart' and c['encoder'] == 'clip_l14' and c['task'] == 'ret' and c['sc'] != 'SC7']
    monkeypatch.setattr(ns_ret, 'score_cells', lambda *args: cells)
    args = SimpleNamespace(set='mmart', enc='clip_l14', task='ret', draws=20)
    assert ns_run.cmd_cells(args) == 0
    assert '3 claim cells not produced' in capsys.readouterr().out
    assert json.loads((outputs / 'stage2/cells/ret__mmart__clip_l14.json').read_text()) == cells
    assert ns_run.read_step('cells__ret__mmart__clip_l14')['status'] == 'ok'


@pytest.mark.parametrize('smoke_run', [False, True])
@pytest.mark.parametrize('smoke_step', [False, True])
def test_step_status_rejects_smoke_evidence_outside_smoke(outputs, monkeypatch, smoke_run, smoke_step):
    monkeypatch.setattr(P, 'SMOKE', smoke_run)
    nested = dict(metric={m: 'ok' for m in P.CLASS_METRICS},
                  count={f'{s}/{a}': 'ok' for s, a in P.CLASS_COUNT})
    for name, status in [('ii_scorer', 'ok'), ('iii', nested),
                         ('iv', {e: 'ok' for e in P.ENCODERS}), ('v', {e: 'ok' for e in P.ENCODERS})]:
        P.write_json(ns_run.step_path(name), dict(status=status, smoke=smoke_step))
    expected = 'failed' if smoke_step and not smoke_run else 'ok'
    assert ns_run._step_status('ii_scorer') == expected
    assert ns_run._step_status('ii_crl') == 'missing'
    status = ns_run.build_status(write=False)['gates']
    assert status['ii_scorer'] == expected
    for gate in ('iii_metric', 'iii_count', 'iv', 'v'):
        assert set(status[gate].values()) == {expected}


@pytest.mark.parametrize('status, exit_code', [
    ('ok', 0), ('failed', 1),
    ({'qwen': 'ok', 'clip_l14': 'failed'}, 1),
    ({'metric': {'ami': 'ok'}, 'count': {'mmart/emotion': 'failed'}}, 1),
    ({'metric': {'ami': 'ok'}, 'count': {'mmart/emotion': 'ok'}}, 0),
])
def test_cmd_gate_exit_code_walks_status(outputs, monkeypatch, status, exit_code):
    monkeypatch.setattr(ns_run, 'run_gate', lambda name: (status, {}))
    monkeypatch.setattr(ns_run, 'build_status', lambda: None)
    assert ns_run.cmd_gate(SimpleNamespace(name='iii')) == exit_code
    assert ns_run.read_step('iii')['status'] == status


def test_cmd_gate_exception_still_fails(outputs, monkeypatch):
    def fail(name):
        raise RuntimeError('synthetic failure')

    monkeypatch.setattr(ns_run, 'run_gate', fail)
    monkeypatch.setattr(ns_run, 'build_status', lambda: None)
    assert ns_run.cmd_gate(SimpleNamespace(name='iii')) == 1
    assert ns_run.read_step('iii')['status'] == 'failed'
