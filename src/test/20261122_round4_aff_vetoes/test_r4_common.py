"""Tests of r4_common.py: rule asserts, tampered SHA, seed guard (guard function only, no bundle build)."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4_common as R4  # noqa: E402


def test_rule_asserts_pass():
    R4.assert_rule()


def test_inputs_pass():
    out = R4.assert_inputs(list(R4.INPUTS) + ["20261117_reader_fix_csd/results/rc_tau.json"])
    assert len(out) == len(R4.INPUTS) + 1


def test_tampered_sha_raises(monkeypatch):
    monkeypatch.setattr(R4, "RULE_SHA", "0" * 64)
    with pytest.raises(SystemExit):
        R4.assert_rule()
    monkeypatch.setitem(R4.INPUTS, "20261121_round3_affect_gate/r3_stats.py", "0" * 64)
    monkeypatch.setattr(R4, "_CHECKED", {})
    with pytest.raises(SystemExit):
        R4.assert_inputs(["20261121_round3_affect_gate/r3_stats.py"])


def test_seed_guard():
    assert R4.R3.TEST_SEEDS == (52, 53, 54)
    assert R4.R3.EARLIER_SEEDS == (42, 43, 45, 47, 48)
    for s in (42, 52, 53, 54):
        R4.RB3._check_seed(s, False)
    R4.RB3._check_seed(9001, True)
    for s in (49, 55):
        with pytest.raises(ValueError):
            R4.RB3._check_seed(s, False)
    with pytest.raises(ValueError):
        R4.RB3._check_seed(52, True)


def test_constants():
    assert R4.IMGABST_TARGETS["cells"]["fused"] == (117, 119)
    assert set(R4.READS_CSD) == set(R4.CANDIDATES)
