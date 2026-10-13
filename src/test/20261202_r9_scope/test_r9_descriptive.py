"""Ticket 23 review fixes: temporary real-mode records and descriptive labels.

No held array, label or feature is loaded by these tests. Agreement and pass schemas
use the apply-rule test's hand-built synthetic pass helper.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R
import r9_descriptive as D
import run_r9_descriptive as RD
from test_r9_apply_rule import make_pass


@pytest.mark.parametrize("read", tuple(RD.RH.READ_BACKBONES))
def test_bind_read_real_agreement_directory_and_missing(tmp_path, monkeypatch, read):
    results = tmp_path / "results"
    results.mkdir()
    agreement_dir = tmp_path / "phase2_custom_location"
    agreement_dir.mkdir()
    monkeypatch.setattr(RD.AR, "AGREEMENT_DIR", agreement_dir)
    names = RD.RH.read_names(read)
    rec = make_pass(read, [0] * 7)
    rec.update(read=read, kind="held", backbone=RD.RH.READ_BACKBONES[read])
    pass_path = results / names.pass_
    pass_path.write_text(json.dumps(rec))
    agreement = {"phase": 2, "smoke": False, "agreement": True, "read": read,
                 "pass_file": names.pass_, "held_pass_sha256": R.sha256_file(pass_path),
                 "items": [{"key": "checks", "equal": True}], "compared": 1, "failed": 0,
                 "rederived_counts": {c: rec["checks"][c]["n"] for c in RD.AR.P_CHECKS}}
    agreement_path = agreement_dir / f"rd9_{read}_agreement.json"
    agreement_path.write_text(json.dumps(agreement))
    verdict = {"status": "run", "verdict": "GO" if read in ("H6", "H7") else "full claim",
               "rule_sha256": R.RULE_SHA256, "read": read, "backbone": rec["backbone"], "mode": "held",
               "variant": None, "fix_pass": False, "pass_file": names.pass_, "seeds": rec["seeds"],
               "held_pass_sha256": R.sha256_file(pass_path), "agreement_file": agreement_path.name,
               "agreement_sha256": R.sha256_file(agreement_path),
               "agreement": {k: agreement[k] for k in ("phase", "smoke", "agreement", "read", "pass_file",
                                                       "held_pass_sha256", "compared")}}
    (results / names.verdict).write_text(json.dumps(verdict))
    bound = RD.bind_read(results, read, False)
    assert bound.sources["agreement"]["file"] == str(agreement_path)
    assert bound.rec == rec
    agreement_path.unlink()  # File created by this test, never a source input.
    with pytest.raises(RD.Refused, match="agreement missing or SHA mismatch"):
        RD.bind_read(results, read, False)


def test_paired_aff_labels_separate_other_and_difference():
    cl = np.tile(np.arange(2), 3)
    pi = np.repeat(np.arange(3), 2)

    def inputs(value):
        return [{"cl": cl, "pair_index": pi,
                 "pa": {D.AFF: {m: np.full(len(cl), value) for m in D.ROW_METRICS}}}]

    paired = D.compare_backbones(inputs(.75), inputs(.25))
    assert set(paired["labels"]) == {"other_aff", "aff_minus"}
    for scope in [paired["pooled"], *paired["per_pair"].values()]:
        assert not set(D.ROW_METRICS) & set(scope)
        for m in D.ROW_METRICS:
            assert scope["other_aff"][m]["point"] == 25
            assert scope["aff_minus"][m]["point"] == 50


@pytest.mark.parametrize("kind", ("boundary", "refused", "apply", "upstream"))
def test_console_only_exposes_own_boundary_names(monkeypatch, capsys, kind):
    def fail(*args, **kw):
        if kind == "boundary":
            D.require(False, "arrays dtype: cosine__r1")
        if kind == "refused":
            RD.refuse_unless(False, "verdict identity")
        if kind == "apply":
            raise RD.AR.Stop(5, "unsafe count 123456")
        raise AssertionError("unsafe upstream count 987654")

    monkeypatch.setattr(RD, "run", fail)
    assert RD.main(["--read", "H6"]) == (4 if kind == "refused" else 5)
    output = capsys.readouterr().out
    assert "123456" not in output and "987654" not in output
    if kind == "boundary":
        assert "arrays dtype: cosine__r1" in output
    if kind == "refused":
        assert "verdict identity" in output
