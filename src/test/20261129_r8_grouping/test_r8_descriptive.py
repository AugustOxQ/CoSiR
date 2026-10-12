"""Ticket 12: the one guard test. Label-based diagnostics and the H<X> arm refuse before gate_verdict.json exists."""
import json
import subprocess

import pytest

import run_r8_descriptive as D


def test_refuses_before_gate_verdict(tmp_path, monkeypatch):
    # An empty results tree: no verdict, no labels, no scores, no heads. Each part must stop at the verdict check.
    def no_subprocess(*a, **k):
        raise AssertionError("the H<X> arm started a subprocess before the verdict")

    def no_frame():
        raise AssertionError("the guard must come before any data is read")
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(D, "frame_from_ctx", no_frame)
    with pytest.raises(AssertionError, match="gate_verdict.json does not exist"):
        D.require_verdict(tmp_path, smoke=True)
    with pytest.raises(AssertionError, match="gate_verdict.json does not exist"):
        D.full(tmp_path, smoke=True)
    with pytest.raises(AssertionError, match="gate_verdict.json does not exist"):
        D.harm(tmp_path, smoke=True)
    assert list(tmp_path.iterdir()) == []
    # A verdict of the wrong role (real file in a smoke run) is refused too.
    (tmp_path / "gate_verdict.json").write_text(json.dumps({"smoke": False, "main": {}, "screen": {}}))
    with pytest.raises(AssertionError, match="smoke role differs"):
        D.require_verdict(tmp_path, smoke=True)
