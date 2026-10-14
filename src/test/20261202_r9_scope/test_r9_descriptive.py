"""Ticket 23 review fixes: temporary real-mode records and descriptive labels.

No held array, label or feature is loaded by these tests. Agreement and pass schemas
use the apply-rule test's hand-built synthetic pass helper.
"""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R
import r9_bundle as B
import r9_descriptive as D
import r9_score as S
import run_r9_descriptive as RD
from test_r9_apply_rule import ALL0, edit_pass, stage
import test_r9_score as TS


def staged(tmp_path, monkeypatch, read, results=None):
    """A real-mode results folder whose verdict was written by AR.apply on a staged pass (never by hand).

    The agreement sits in a custom folder, the apply step's canonical place patched to tmp_path."""
    d = stage(tmp_path, read, ALL0)
    edit_pass(d, read, lambda r: r.update(kind="held"))           # the held runner's pass carries its variant kind
    adir = tmp_path / "phase2_custom_location"
    adir.mkdir(exist_ok=True)
    apath = adir / f"rd9_{read}_agreement.json"
    shutil.move(str(d / apath.name), str(apath))
    monkeypatch.setattr(RD.AR, "AGREEMENT_DIR", adir)
    vpath, verdict = RD.AR.apply(read, results=d, agreement_dir=adir, ledger=d / "ledger.md")
    return SimpleNamespace(results=d, rec=json.loads((d / f"held_pass_{read}.json").read_text()),
                           ppath=d / f"held_pass_{read}.json", verdict=verdict, vpath=vpath, apath=apath)


@pytest.fixture
def no_arrays(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an array or score was loaded")
    monkeypatch.setattr(RD.D, "load_arrays", boom)
    monkeypatch.setattr(RD.S, "score_seed", boom)


@pytest.mark.parametrize("read", tuple(RD.RH.READ_BACKBONES))
def test_bind_read_real_agreement_directory_and_missing(tmp_path, monkeypatch, no_arrays, read):
    s = staged(tmp_path, monkeypatch, read)
    assert s.verdict["backbone"] == RD.AR.READS[read]["backbone"]        # the ledger name, not the code name
    assert s.verdict["backbone"] == RD.RH.LEDGER_NAMES[RD.RH.READ_BACKBONES[read]]
    bound = RD.bind_read(s.results, read, False)
    assert bound.sources["agreement"]["file"] == str(s.apath)
    assert bound.rec == s.rec
    assert bound.backbone == RD.RH.READ_BACKBONES[read]                  # the pass keeps the code name
    s.apath.unlink()  # File created by this test, never a source input.
    with pytest.raises(RD.Refused, match="agreement missing or SHA mismatch"):
        RD.bind_read(s.results, read, False)


def test_bind_read_agreement_of_wrong_bytes(tmp_path, monkeypatch, no_arrays):
    s = staged(tmp_path, monkeypatch, "H6")
    RD.bind_read(s.results, "H6", False)
    s.apath.write_text(s.apath.read_text() + " ")                       # the file exists but is not the decided one
    with pytest.raises(RD.Refused, match="agreement missing or SHA mismatch"):
        RD.bind_read(s.results, "H6", False)


def test_bind_read_h8_binding_of_the_two_other_a28_reads(tmp_path, monkeypatch, no_arrays):
    """H9 and H10 bind H8 (a28_report) and each other (paired_reads) in one results folder."""
    staged_reads = [staged(tmp_path, monkeypatch, read) for read in ("H8", "H9", "H10")]
    d = staged_reads[0].results
    for read in ("H8", "H9", "H10"):
        assert RD.bind_read(d, read, False).rec["read"] == read


def test_bind_read_refuses_another_reads_verdict(tmp_path, monkeypatch, no_arrays):
    s = staged(tmp_path, monkeypatch, "H6")
    s.vpath.write_text(json.dumps({**s.verdict, "read": "H7", "backbone": RD.AR.READS["H7"]["backbone"]}))
    with pytest.raises(RD.Refused, match="verdict identity"):
        RD.bind_read(s.results, "H6", False)


@pytest.mark.parametrize("field, value", [("read", "H7"), ("backbone", "CLIP ViT-B/32"), ("mode", "smoke"),
                                          ("variant", "reserve"), ("rule_sha256", "0" * 64),
                                          ("backbone", "qwen3vl_emb_2b")])   # the last: the code name, not the ledger's
def test_bind_read_verdict_identity_field_by_field(tmp_path, monkeypatch, no_arrays, field, value):
    s = staged(tmp_path, monkeypatch, "H6")
    s.vpath.write_text(json.dumps({**s.verdict, field: value}))
    with pytest.raises(RD.Refused, match="verdict identity"):
        RD.bind_read(s.results, "H6", False)


def test_bind_read_refuses_a_not_run_verdict(tmp_path, monkeypatch, no_arrays):
    s = staged(tmp_path, monkeypatch, "H6")
    v = {k: s.verdict[k] for k in ("read", "backbone", "mode", "variant", "rule_sha256")}
    s.vpath.write_text(json.dumps({**v, "verdict": None, "kind": None, "status": "not run: switched off",
                                   "claim": None}))
    with pytest.raises(RD.Refused, match="read has no run verdict"):
        RD.bind_read(s.results, "H6", False)


@pytest.mark.parametrize("read", ["H6", "H8"])
def test_bind_read_refuses_a_pass_changed_after_the_verdict(tmp_path, monkeypatch, no_arrays, read):
    s = staged(tmp_path, monkeypatch, read)
    assert RD.bind_read(s.results, read, False).rec == s.rec
    s.ppath.write_text(json.dumps({**s.rec, "runtime_s": 12345}))
    with pytest.raises(RD.Refused, match="pass SHA binding"):
        RD.bind_read(s.results, read, False)


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


# ---------------------------------------------------------------- the checks that stood alone (final review A2b F3)

def saved_arrays(n_seed=5, seeds=2):
    rng = np.random.default_rng(3)
    n = n_seed * seeds
    return {"cosine__r1": rng.random(n), "cl": rng.integers(0, 9, n).astype(np.int64),
            "seed_index": np.repeat(np.arange(seeds), n_seed).astype(np.int64),
            "aff_gate__open": rng.random((4, n)).astype(np.float32)}


def test_check_rebuilt_accepts_equal_arrays_and_refuses_each_difference():
    arrays = saved_arrays()
    sl = slice(5, 10)                                    # the second seed
    got = {"cosine__r1": arrays["cosine__r1"][sl].copy(), "cl": arrays["cl"][sl].copy(),
           "seed_index": np.zeros(5, dtype=np.int64), "aff_gate__open": arrays["aff_gate__open"][:, sl].copy()}
    D.check_rebuilt(got, arrays, sl, 5)
    changed = {"cosine__r1": lambda x: x + np.eye(1, 5, 2)[0] * 1e-12,           # one value, by 1e-12
               "cl": lambda x: x + (np.arange(5) == 0),
               "seed_index": lambda x: x + 1,
               "aff_gate__open": lambda x: np.where(np.arange(x.shape[1]) == 1, x + np.float32(1e-3), x)}
    for key, edit in changed.items():
        with pytest.raises(D.BoundaryError, match="rebuilt primary/raw array: " + key):
            D.check_rebuilt({**got, key: edit(got[key])}, arrays, sl, 5)
    nan = got["cosine__r1"].copy()
    nan[0] = np.nan
    with pytest.raises(D.BoundaryError, match="rebuilt primary/raw array: cosine__r1"):
        D.check_rebuilt({**got, "cosine__r1": nan}, arrays, sl, 5)
    with pytest.raises(D.BoundaryError, match="rebuilt primary/raw array: cl"):                  # dtype, equal values
        D.check_rebuilt({**got, "cl": got["cl"].astype(np.int32)}, arrays, sl, 5)
    with pytest.raises(D.BoundaryError, match="rebuilt primary/raw array: cosine__r1"):          # shape
        D.check_rebuilt({**got, "cosine__r1": got["cosine__r1"][:4]}, arrays, sl, 5)


def test_check_directions_on_a_real_score_and_with_one_row_changed(tmp_path):
    verdict = tmp_path / "held_verdict_H6.json"
    verdict.write_text("{}")
    rows, collect = D.direction_rows()
    sc = S.score_seed(TS.make_bundle(), TS.NESTED_PICKS, TS.lambdas_all(), B.load_readers(), include_pm=True,
                      verdict=verdict, on_scores=collect)
    assert set(rows) == set(S.ALL_SCORERS)
    D.check_directions(sc, rows)
    for scorer, metric in ((S.ALL_SCORERS[0], "r1"), (S.ALL_SCORERS[-1], "swap")):    # first and last cell checked
        saved = rows[scorer]["t2i"][metric]
        rows[scorer]["t2i"][metric] = saved + 1e-12
        with pytest.raises(D.BoundaryError, match="direction averages reproduce core"):
            D.check_directions(sc, rows)
        rows[scorer]["t2i"][metric] = saved
    D.check_directions(sc, rows)


def near_record(tmp_path, monkeypatch, content=None, pin=True):
    path = tmp_path / "near_duplicates_a28.json"
    path.write_text(json.dumps(content or {"cos_099": 3, "cos_098": 7, "cos_092": 20}))
    sha = R.sha256_file(path)
    if pin:
        monkeypatch.setattr(RD, "NEAR_DUPLICATES_SHA256", sha)
    return path, {"near_duplicates_source": {"file": str(path), "sha256": sha},
                  "near_duplicates": {"cos_099": 3, "cos_098": 7}}


def test_near_counts_reads_cos_092_from_the_pinned_file(tmp_path, monkeypatch):
    path, rec = near_record(tmp_path, monkeypatch)
    got_path, near = RD.near_counts(rec)
    assert got_path == path and near["cos_092"] == 20


def test_the_near_duplicates_pin_is_the_stage_1_files_sha():
    assert RD.NEAR_DUPLICATES_SHA256.startswith("ba92530d") and len(RD.NEAR_DUPLICATES_SHA256) == 64
    real = R.MAIN / "src/test/20261202_r9_scope/results/near_duplicates_a28.json"
    assert R.sha256_file(real) == RD.NEAR_DUPLICATES_SHA256


def test_near_counts_refuses_a_file_that_is_not_the_one_the_pass_names(tmp_path, monkeypatch):
    path, rec = near_record(tmp_path, monkeypatch)
    path.write_text(json.dumps({"cos_099": 3, "cos_098": 7, "cos_092": 21}))         # changed after the pass
    with pytest.raises(D.BoundaryError, match="near-duplicates source binding"):
        RD.near_counts(rec)


def test_near_counts_refuses_a_pass_that_names_another_sha(tmp_path, monkeypatch):
    _, rec = near_record(tmp_path, monkeypatch)                      # the file is the pinned one
    rec["near_duplicates_source"]["sha256"] = "f" * 64
    with pytest.raises(D.BoundaryError, match="near-duplicates source binding"):
        RD.near_counts(rec)


def test_near_counts_refuses_a_file_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    _, rec = near_record(tmp_path, monkeypatch, pin=False)          # the pass names the file's own SHA, the pin differs
    with pytest.raises(D.BoundaryError, match="near-duplicates source binding"):
        RD.near_counts(rec)


@pytest.mark.parametrize("content", [
    {"cos_099": 3, "cos_098": 7, "cos_092": 6},                      # cos_092 below cos_098
    {"cos_099": 3, "cos_098": 7, "cos_092": 747},                    # above the 746 paintings
    {"cos_099": 3, "cos_098": 7, "cos_092": 20.0},                   # not an int
    {"cos_099": 3, "cos_098": 8, "cos_092": 20},                     # cos_098 differs from the pass
    {"cos_099": 2, "cos_098": 7, "cos_092": 20}])                    # cos_099 differs from the pass
def test_near_counts_refuses_bad_counts(tmp_path, monkeypatch, content):
    path, rec = near_record(tmp_path, monkeypatch, content)
    rec["near_duplicates"] = {"cos_099": 3, "cos_098": 7}
    with pytest.raises(D.BoundaryError, match="near-duplicate counts"):
        RD.near_counts(rec)


@pytest.mark.parametrize("change", [{"rule_sha256": "0" * 64}, {"scored": False}])
def test_stage0b_identity(tmp_path, change):
    rec = {"rule_sha256": R.RULE_SHA256, "scored": True, "case": "x", "passed": True,
           "item3": {"chain": {"changed_episodes": {"n_episodes": 4, "any_a3_fed": 1, "per_scorer": {"aff": 1}},
                               "cpu_changed_episodes": 1}}}
    bound = SimpleNamespace(smoke=True, out=tmp_path)
    (tmp_path / "stage0b.json").write_text(json.dumps(rec))
    assert RD.stage0b(bound)["chains"]["chain"]["cpu_changed_episodes"] == 1
    (tmp_path / "stage0b.json").write_text(json.dumps({**rec, **change}))
    with pytest.raises(D.BoundaryError, match="stage0b record"):
        RD.stage0b(bound)


# ---------------------------------------------------------------- the --smoke-tag refusals (23 fix re-review LOW-B)

@pytest.mark.parametrize("smoke, tag", [(False, "t1"), (True, "a/b"), (True, "a.b"), (True, "../x"), (True, ""),
                                         (True, "a b")])
def test_smoke_tag_refusals(tmp_path, smoke, tag):
    with pytest.raises(RD.Refused, match="smoke tag requires smoke mode and a plain name"):
        RD.run("H6", smoke=smoke, smoke_dir=str(tmp_path) if smoke else None, smoke_tag=tag)


def test_smoke_tag_never_replaces_an_earlier_file(tmp_path, monkeypatch):
    monkeypatch.setattr(RD.R, "RESULTS", tmp_path / "results")
    target = tmp_path / "results/smoke/descriptive/descriptive_H6_t1.json"
    target.parent.mkdir(parents=True)
    target.write_text("earlier")
    with pytest.raises(RD.Refused, match="output exists"):
        RD.run("H6", smoke=True, smoke_dir=str(tmp_path), smoke_tag="t1")
    assert target.read_text() == "earlier"
    with pytest.raises(RD.Refused, match="--smoke requires --smoke-dir"):
        RD.run("H6", smoke=True, smoke_tag="t1")
