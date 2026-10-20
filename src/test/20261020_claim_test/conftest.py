"""Shared V1 gate fixtures for the module-CLI tests."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def gate_passes(monkeypatch):
    """Stand-in pass for tests that reach a gated CLI body; the gate itself stays untouched."""
    import ct_stage0
    monkeypatch.setattr(ct_stage0, 'require_pass', lambda *args: None)


@pytest.fixture(params=['missing', 'failed'])
def gate_refuses(request, tmp_path, monkeypatch):
    """Point the real require_pass defaults at a missing or failed report; returns the expected error."""
    import ct_stage0
    path = tmp_path / 'stage0.json'
    if request.param == 'failed':
        path.write_text(json.dumps(dict(schema=1, stage=0, smoke=False, status='FAIL',
                                        missing=[], failed=['regression/d2/emotion'], items={})))
    monkeypatch.setattr(ct_stage0.require_pass, '__defaults__', (path, 'cuda'))
    return (FileNotFoundError, 'stage0.json') if request.param == 'missing' else (AssertionError, 'did not pass')
