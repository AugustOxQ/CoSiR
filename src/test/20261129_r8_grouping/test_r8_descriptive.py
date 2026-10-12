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


@pytest.mark.parametrize("x, gname", [("NB100", "r8_groupings"), ("NBC", "r8_groupings_nbc")])
def test_harm_smoke_passes_the_painting_count_of_the_groupings_file(tmp_path, monkeypatch, x, gname):
    """Smoke: 08 asserts `--smoke N` equals the groupings file's painting count; real mode passes no N.
    NBC sits in its own grouping file, which 08 finds only by --groupings-name."""
    calls = []
    monkeypatch.setattr(D, "require_verdict", lambda root, smoke: {"main": {"head_target": {"grouping": x}}})
    monkeypatch.setattr(D, "run", lambda cmd, log: calls.append(cmd))
    monkeypatch.setattr(D.R, "SMOKE", tmp_path)
    (tmp_path / "groupings").mkdir()
    (tmp_path / "groupings" / f"{gname}.json").write_text(json.dumps({"n_paintings": 1234}))
    D.harm(tmp_path.resolve(), True)
    place = next(c for c in calls if "r8_place.py" in c)
    assert place[-5:] == ["--h-only", "--smoke", "1234", "--groupings-name", gname]
