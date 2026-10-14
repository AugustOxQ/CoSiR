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
import r9_stats as ST
import run_r9_descriptive as RD
from test_r9_apply_rule import ALL0, edit_pass, stage
import test_r9_score as TS


def staged(tmp_path, monkeypatch, read, results=None):
    """A real-mode results folder whose verdict was written by AR.apply on a staged pass (never by hand).

    The agreement sits in a custom folder, the apply step's canonical place patched to tmp_path."""
    d = stage(tmp_path, read, ALL0)
    edit_pass(d, read, lambda r: r.update(kind="held", coef_sha256={}))  # held pass schema
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


def smoke_on_clip(tmp_path, read):
    import run_r9_preread as preread
    d = stage(tmp_path, read, ALL0, smoke=True, agree=False)
    ppath = d / f"held_pass_{read}.json"
    rec = json.loads(ppath.read_text())
    rec.update(backbone="clip_b32", kind="smoke", coef_sha256={})
    ppath.write_text(json.dumps(rec))
    preread.smoke_agreement(d, read, "clip")
    vpath, verdict = RD.AR.apply(read, smoke=True, results=d)
    return SimpleNamespace(results=d, rec=rec, ppath=ppath, vpath=vpath, verdict=verdict)


@pytest.mark.parametrize("read", ["H6", "H7"])
def test_clip_smoke_binding_and_descriptive_backbone(tmp_path, monkeypatch, read):
    s = smoke_on_clip(tmp_path, read)
    bound = RD.bind_read(s.results, read, True)
    assert bound.backbone == "clip_b32"
    assert bound.verdict == s.verdict
    assert bound.sources["verdict"]["sha256"] == R.sha256_file(s.vpath)
    # Keep the real binding and report writer; numerical reporting is covered separately.
    monkeypatch.setattr(RD, "R", SimpleNamespace(**{**vars(R), "RESULTS": tmp_path / "reports"}))
    monkeypatch.setattr(RD, "load_bound_inputs", lambda b: SimpleNamespace(checks={}, eps=[]))
    monkeypatch.setattr(RD, "score_inputs", lambda *a: ([], None, None, {"cells": {}}))
    monkeypatch.setattr(RD.D, "describe", lambda *a: {})
    monkeypatch.setattr(RD, "paired_reads", lambda *a: {})
    monkeypatch.setattr(RD, "a28_report", lambda *a: {})
    monkeypatch.setattr(RD, "stage0b", lambda *a: {})
    path, sources = RD.run(read, smoke=True, smoke_dir=s.results, smoke_tag="clip")
    report = json.loads(path.read_text())
    assert report["backbone"] == s.rec["backbone"] == "clip_b32"
    assert s.verdict["backbone"] == RD.RH.LEDGER_NAMES[report["backbone"]]
    assert "on frozen CLIP features" in s.verdict["claim"]
    assert sources["verdict"]["file"] == str(s.vpath)


@pytest.mark.parametrize("field,value,boundary", [
    ("backbone", "Qwen3-VL-Embedding-2B", "smoke verdict identity"),
    ("read", "H7", "smoke verdict identity"),
    ("mode", "held", "smoke verdict identity"),
    ("rule_sha256", "0" * 64, "smoke verdict identity"),
    ("status", "not run: switched off", "smoke verdict identity"),
    ("variant", "reserve", "smoke verdict identity"),
    ("pass_file", "held_pass_H7.json", "smoke verdict identity"),
    ("fix_pass", True, "smoke verdict identity"),
    ("held_pass_sha256", "0" * 64, "pass SHA binding"),
    ("seeds", [52, 53, 54], "verdict seeds"),
])
def test_smoke_binding_refuses_inconsistent_verdict(tmp_path, field, value, boundary):
    s = smoke_on_clip(tmp_path, "H6")
    s.vpath.write_text(json.dumps({**s.verdict, field: value}))
    with pytest.raises((RD.Refused, AssertionError), match=boundary):
        RD.bind_read(s.results, "H6", True)


def test_smoke_binding_without_a_verdict_remains_supported(tmp_path):
    s = smoke_on_clip(tmp_path, "H6")
    s.vpath.unlink()  # File created by this test in its tmp_path.
    assert RD.bind_read(s.results, "H6", True).backbone == "clip_b32"


@pytest.mark.parametrize("read,smoke", [("H6", True), ("H9", True),
                                      ("H6", False), ("H9", False),
                                      ("H7", False), ("H10", False)])
def test_report_writes_infinite_frozen_lambda(tmp_path, monkeypatch, read, smoke):
    if smoke:
        import run_r9_preread as preread
        out = stage(tmp_path, read, ALL0, smoke=True, agree=False)
        ppath = out / f"held_pass_{read}.json"
        ppath.write_text(json.dumps({**json.loads(ppath.read_text()), "kind": "smoke", "coef_sha256": {}}))
        preread.smoke_agreement(out, read, "inf")
        RD.AR.apply(read, smoke=True, results=out)
    else:
        out = staged(tmp_path, monkeypatch, read).results
    # Use the actual recipe parser: scoring must retain the numeric infinity.
    lambdas = {"wang": RD.P.as_lambda_pick({"0": 16.0, "1": "inf"}, "wang")}
    frozen = {"picks": {"B": {0: [0.0, 0.0], 1: [1.0, 0.0]}},
              "lambdas": lambdas, "cells": {"aff": {"fused": (1, 2)}},
              "taus": (-1.0, 0.0, .5, .9)}
    monkeypatch.setattr(RD, "R", SimpleNamespace(**{**vars(R), "RESULTS": out}))
    monkeypatch.setattr(RD, "load_bound_inputs", lambda b: SimpleNamespace(checks={}, eps=[]))
    monkeypatch.setattr(RD, "score_inputs", lambda *a: ([], None, None, frozen))
    monkeypatch.setattr(RD.D, "describe", lambda *a: {})
    monkeypatch.setattr(RD, "paired_reads", lambda *a: {})
    monkeypatch.setattr(RD, "a28_report", lambda *a: {})
    monkeypatch.setattr(RD, "stage0b", lambda *a: {})
    before = {p: p.read_bytes() for p in out.rglob("*.json")}
    path, _ = RD.run(read, smoke=smoke, smoke_dir=out if smoke else None)
    report = json.loads(path.read_text(), parse_constant=lambda x: pytest.fail(x))
    assert report["frozen"]["lambdas"]["wang"] == {"0": 16.0, "1": "inf"}
    assert report["frozen"]["picks"]["B"]["1"] == [1.0, 0.0]
    assert lambdas["wang"][1] == float("inf")
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_report_plain_handles_all_nonfinite_shapes_and_keeps_finite_bytes():
    nonfinite = {1: (np.float64(np.inf), np.float32(-np.inf), np.array([np.nan, 16.0])),
                 "literal": "Infinity"}
    assert RD.report_plain(nonfinite) == {"1": ["inf", "-inf", ["nan", 16.0]], "literal": "Infinity"}
    finite = {0: np.array([1.25, -0.0]), "nested": (np.int64(2), True, None, "inf")}
    assert json.dumps(RD.report_plain(finite), indent=2, allow_nan=False) == json.dumps(RD.RH._plain(finite), indent=2, allow_nan=False)


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


# ---------------------------------------------------------------- H5's copied loader and pass check (rr fixwave 1, M1)
# A small synthetic held_arrays.npz with a pass file made by r9_stats.pass_record from the same arrays.

H5_SEEDS = list(R.SMOKE_SEEDS)
H5_NPP = R.N_SMOKE
H5_NSEED = len(R.PAIRS) * H5_NPP


def h5_scores():
    rng = np.random.default_rng(11)
    cl = np.tile(np.arange(6, dtype=np.int64), H5_NSEED // 6)
    out = []
    for _ in H5_SEEDS:
        s = {"cl": cl}
        for key in RD.H5_CORE:
            m = {k: rng.integers(0, 5, H5_NSEED) / 4 for k in RD.D.METRICS}
            if key == "aff_cf":
                m["gain"] = np.zeros(H5_NSEED)
            s[key] = m
        out.append(s)
    return out


@pytest.fixture(scope="module")
def h5_made():
    scores = h5_scores()
    extra = {"episodes_sha256": {s: {p: "a" * 64 for p in R.PAIR_NAMES} for s in H5_SEEDS},
             "runner_sha256": "b" * 64, "module_sha256": {"r9_stats.py": "c" * 64}}
    rec = ST.pass_record(scores, "smoke", H5_SEEDS, extra, secondary=True)
    arrays = {"seed_index": np.repeat(np.arange(len(H5_SEEDS), dtype=np.int64), H5_NSEED),
              "pair_index": np.tile(np.repeat(np.arange(len(R.PAIRS), dtype=np.int64), H5_NPP), len(H5_SEEDS)),
              "cl": np.concatenate([s["cl"] for s in scores])}
    for key in RD.H5_CORE:
        for m in RD.D.METRICS:
            arrays[f"{key}__{m}"] = np.concatenate([np.asarray(s[key][m], dtype=np.float64) for s in scores])
    return rec, arrays


def h5_write(tmp_path, arrays):
    path = tmp_path / "held_arrays.npz"
    np.savez(path, **arrays)
    return path


def h5_core(tmp_path, arrays):
    return RD.h5_load_core(h5_write(tmp_path, arrays), H5_SEEDS, H5_NPP)


def test_h5_arrays_are_accepted_as_built(tmp_path, h5_made):
    rec, arrays = h5_made
    core = h5_core(tmp_path, arrays)
    assert sorted(core) == H5_SEEDS
    assert set(core[H5_SEEDS[0]]["pa"]) == set(RD.H5_CORE)
    RD.h5_check_pass(core, rec)                      # raises on any difference


def test_h5_load_core_refuses_a_missing_key(tmp_path, h5_made):
    _, arrays = h5_made
    with pytest.raises(D.BoundaryError, match="H5 arrays keys"):
        h5_core(tmp_path, {k: v for k, v in arrays.items() if k != "aff_fused__r1"})


def test_h5_load_core_refuses_an_extra_key(tmp_path, h5_made):
    _, arrays = h5_made
    with pytest.raises(D.BoundaryError, match="H5 arrays keys"):
        h5_core(tmp_path, {**arrays, "unexpected": arrays["cl"]})


@pytest.mark.parametrize("key, bad, match", [
    ("cosine__r1", lambda x: x.astype(np.float32), "H5 arrays float64"),
    ("pair_index", lambda x: x.astype(np.int32), "H5 arrays index dtype"),
    ("seed_index", lambda x: x.astype(np.int32), "H5 arrays index dtype"),
    ("cl", lambda x: x.astype(np.int32), "H5 arrays index dtype"),
    ("B__gain", lambda x: np.where(np.arange(len(x)) == 3, np.nan, x), "H5 arrays float64"),
    ("B__swap", lambda x: x[:-1], "H5 arrays shape"),
    ("seed_index", lambda x: x[::-1].copy(), "H5 arrays seed order"),
    ("pair_index", lambda x: x[::-1].copy(), "H5 arrays pair order"),
])
def test_h5_load_core_refuses_a_bad_array(tmp_path, h5_made, key, bad, match):
    _, arrays = h5_made
    with pytest.raises(D.BoundaryError, match=match):
        h5_core(tmp_path, {**arrays, key: bad(arrays[key])})


def test_h5_load_core_refuses_other_seeds_or_pair_size(tmp_path, h5_made):
    _, arrays = h5_made
    path = h5_write(tmp_path, arrays)
    with pytest.raises(D.BoundaryError, match="H5 arrays shape"):
        RD.h5_load_core(path, H5_SEEDS, H5_NPP + 1)
    with pytest.raises(D.BoundaryError, match="H5 arrays shape"):
        RD.h5_load_core(path, H5_SEEDS[:2], H5_NPP)


def h5_changed(rec, section, check, **change):
    out = json.loads(json.dumps(rec))
    out[section][check].update(change)
    return out


@pytest.mark.parametrize("change", [{"n_episodes": 1}, {"n_clusters": 1}])
def test_h5_check_pass_refuses_a_changed_count(tmp_path, h5_made, change):
    rec, arrays = h5_made
    with pytest.raises(D.BoundaryError, match="H5 arrays episode counts"):
        RD.h5_check_pass(h5_core(tmp_path, arrays), {**rec, **change})


@pytest.mark.parametrize("section, check", [("checks", "P1"), ("checks", "P6"), ("secondary", "S1")])
@pytest.mark.parametrize("field, value", [("n", 1), ("point", 0.123456), ("ci95", [-0.5, 0.5]),
                                          ("quantity", "R@1, AFF - something else")])
def test_h5_check_pass_refuses_each_changed_item(tmp_path, h5_made, section, check, field, value):
    rec, arrays = h5_made
    core = h5_core(tmp_path, arrays)
    assert rec[section][check][field] != value
    with pytest.raises(D.BoundaryError, match="H5 arrays reproduce pass: " + check):
        RD.h5_check_pass(core, h5_changed(rec, section, check, **{field: value}))


def test_h5_check_pass_refuses_changed_arrays(tmp_path, h5_made):
    rec, arrays = h5_made
    bumped = arrays["aff_fused__r1"].copy()
    bumped[:] = np.clip(bumped + 0.5, 0, 1)
    with pytest.raises(D.BoundaryError, match="H5 arrays reproduce pass|H5 arrays episode counts"):
        RD.h5_check_pass(h5_core(tmp_path, {**arrays, "aff_fused__r1": bumped}), rec)


def test_h5_inputs_binds_smoke_pass_arrays_and_episodes(tmp_path, monkeypatch, h5_made):
    rec, arrays = h5_made
    smoke = tmp_path / "results" / "smoke"
    smoke.mkdir(parents=True)
    monkeypatch.setattr(RD.R, "R6D", tmp_path)
    assert RD.h5_inputs(True) is None                # no pass file yet: the read is absent
    rec = {**rec, "rule_sha256": R.R6_RULE_SHA256}   # H5's pass file carries round 6's rule
    (smoke / "held_pass.json").write_text(json.dumps(rec))
    np.savez(smoke / "held_arrays.npz", **arrays)
    for seed in H5_SEEDS:
        (smoke / f"held_episodes_seed{seed}.npz").write_bytes(b"x")
    pair_index = arrays["pair_index"][:H5_NSEED]
    monkeypatch.setattr(RD.E, "load_episodes", lambda path: SimpleNamespace(
        seed=int(Path(path).stem.split("seed")[1]), n=H5_NSEED, pair_index=pair_index,
        sha=rec["episodes_sha256"][Path(path).stem.split("seed")[1]]))
    got = RD.h5_inputs(True)
    assert [x["seed"] for x in got.inputs] == H5_SEEDS and set(got.inputs[0]["pa"]) == set(RD.H5_CORE)
    # a changed point in the pass file stops the read
    bad = json.loads(json.dumps(rec))
    bad["checks"]["P2"]["point"] += 0.01
    (smoke / "held_pass.json").write_text(json.dumps(bad))
    with pytest.raises(D.BoundaryError, match="H5 arrays reproduce pass: P2"):
        RD.h5_inputs(True)
