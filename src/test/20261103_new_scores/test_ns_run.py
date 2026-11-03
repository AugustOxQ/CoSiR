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
