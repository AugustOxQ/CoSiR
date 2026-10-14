"""Final review A2b: the tests proposed for F2, F3a and F3e, tried on 8db7ceb (not part of the suite: no test_ prefix).

Run: python -m pytest -q final_review/fr_a2b_probe_tests.py  (from the round folder). Synthetic temp files only.
The F2/F3a cases monkeypatch the verdict's backbone check out of the way by writing the code name that the current
bind_read wants (F1 open); after F1's fix they should write AR.READS[read]['backbone'] instead.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402
import run_r9_descriptive as RD  # noqa: E402
from test_r9_apply_rule import make_pass  # noqa: E402


def staged(tmp_path, monkeypatch, read):
    """test_r9_descriptive's real-mode fixture, unchanged except that it returns its paths."""
    results = tmp_path / "results"
    results.mkdir()
    adir = tmp_path / "agreements"
    adir.mkdir()
    monkeypatch.setattr(RD.AR, "AGREEMENT_DIR", adir)
    names = RD.RH.read_names(read)
    rec = make_pass(read, [0] * 7)
    rec.update(read=read, kind="held", backbone=RD.RH.READ_BACKBONES[read])
    ppath = results / names.pass_
    ppath.write_text(json.dumps(rec))
    agr = {"phase": 2, "smoke": False, "agreement": True, "read": read, "pass_file": names.pass_,
           "held_pass_sha256": R.sha256_file(ppath), "items": [{"key": "checks", "equal": True}], "compared": 1,
           "failed": 0, "rederived_counts": {c: rec["checks"][c]["n"] for c in RD.AR.P_CHECKS}}
    apath = adir / f"rd9_{read}_agreement.json"
    apath.write_text(json.dumps(agr))
    verdict = {"status": "run", "verdict": "GO" if read in ("H6", "H7") else "full claim",
               "rule_sha256": R.RULE_SHA256, "read": read, "backbone": rec["backbone"], "mode": "held",
               "variant": None, "fix_pass": False, "pass_file": names.pass_, "seeds": rec["seeds"],
               "held_pass_sha256": R.sha256_file(ppath), "agreement_file": apath.name,
               "agreement_sha256": R.sha256_file(apath),
               "agreement": {k: agr[k] for k in ("phase", "smoke", "agreement", "read", "pass_file",
                                                 "held_pass_sha256", "compared")}}
    vpath = results / names.verdict
    vpath.write_text(json.dumps(verdict))
    return SimpleNamespace(results=results, rec=rec, ppath=ppath, verdict=verdict, vpath=vpath)


@pytest.fixture
def no_arrays(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an array or score was loaded")
    monkeypatch.setattr(RD.D, "load_arrays", boom)
    monkeypatch.setattr(RD.S, "score_seed", boom)


def test_f2_foreign_verdict(tmp_path, monkeypatch, no_arrays):
    s = staged(tmp_path, monkeypatch, "H6")
    s.vpath.write_text(json.dumps({**s.verdict, "read": "H7", "backbone": RD.RH.READ_BACKBONES["H7"]}))
    with pytest.raises(RD.Refused, match="verdict identity"):
        RD.bind_read(s.results, "H6", False)


def test_f2_null_verdict(tmp_path, monkeypatch, no_arrays):
    s = staged(tmp_path, monkeypatch, "H6")
    v = {k: s.verdict[k] for k in ("read", "backbone", "mode", "variant", "rule_sha256")}
    s.vpath.write_text(json.dumps({**v, "verdict": None, "kind": None, "status": "not run: switched off",
                                   "claim": None}))
    with pytest.raises(RD.Refused, match="read has no run verdict"):
        RD.bind_read(s.results, "H6", False)


@pytest.mark.parametrize("read", ["H6", "H8"])
def test_f3a_pass_changed_after_verdict(tmp_path, monkeypatch, no_arrays, read):
    s = staged(tmp_path, monkeypatch, read)
    assert RD.bind_read(s.results, read, False).rec == s.rec
    s.ppath.write_text(json.dumps({**s.rec, "runtime_s": 12345}))
    with pytest.raises(RD.Refused, match="pass SHA binding"):
        RD.bind_read(s.results, read, False)


@pytest.mark.parametrize("change", [{"rule_sha256": "0" * 64}, {"scored": False}])
def test_f3e_stage0b_identity(tmp_path, change):
    rec = {"rule_sha256": R.RULE_SHA256, "scored": True, "case": "x", "passed": True,
           "item3": {"chain": {"changed_episodes": {"n_episodes": 4, "any_a3_fed": 1, "per_scorer": {"aff": 1}},
                               "cpu_changed_episodes": 1}}}
    bound = SimpleNamespace(smoke=True, out=tmp_path)
    (tmp_path / "stage0b.json").write_text(json.dumps(rec))
    assert RD.stage0b(bound)["chains"]["chain"]["cpu_changed_episodes"] == 1
    (tmp_path / "stage0b.json").write_text(json.dumps({**rec, **change}))
    with pytest.raises(RD.D.BoundaryError, match="stage0b record"):
        RD.stage0b(bound)
