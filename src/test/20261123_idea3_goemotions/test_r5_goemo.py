"""Tests of r5_goemo.py (rule DECISION_RULE.md of this folder: D2, D3, §5 item 2, §10 list A item 2).
Synthetic only, with a deterministic stub in place of the GoEmotions model: no real caption reaches a model, no
EvalContext is built, nothing is written outside tmp_path.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_goemo.py
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_goemo as G  # noqa: E402

N_ALL, N_TRAIN_FULL, N_SEL_FULL = 308_723, 183_694, 32_413


def fake_probs(texts, loaded=None, batch_size=None, max_length=None, **kw):
    """Deterministic 28 sigmoid-like probabilities per text (float32), records every call."""
    fake_probs.calls.append({"texts": list(texts), "batch_size": batch_size, "max_length": max_length})
    out = np.empty((len(texts), 28), dtype=np.float32)
    for i, t in enumerate(texts):
        seed = int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "little")
        out[i] = np.random.default_rng(seed).uniform(0.001, 0.999, 28).astype(np.float32)
    return out


fake_probs.calls = []


class StubTok:
    def __call__(self, texts, truncation=False, padding=False, **kw):
        assert truncation is False
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


def loaded_cpu():
    return StubTok(), torch.nn.Linear(1, 1)


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    fake_probs.calls = []
    monkeypatch.setattr(G, "goemotions_probabilities", fake_probs)


def synth(n_all=1000, n_train=600, n_sel=250, seed=0):
    """Synthetic data: shuffled sample ids, annotations by id, a split of the rows, scorer-train probabilities."""
    rng = np.random.default_rng(seed)
    sample_ids = rng.permutation(n_all).astype(np.int64)
    annotations = [{"caption": f"caption number {i} of the set"} for i in range(n_all)]
    rows = np.arange(n_all)
    scorer_train, selection, held = rows[:n_train], rows[n_train:n_train + n_sel], rows[n_train + n_sel:]
    ctx = SimpleNamespace(selection=selection, data=SimpleNamespace(sample_ids=sample_ids))
    splits = SimpleNamespace(selection=selection, scorer_train=scorer_train, held=held, val=held[:0])
    caps = [annotations[int(sample_ids[r])]["caption"] for r in scorer_train]
    affect = fake_probs(caps)
    fake_probs.calls = []
    return ctx, annotations, splits, affect


# ---------------------------------------------------------------- D2 item 1: rows and captions

def test_selection_captions_ok():
    ctx, ann, sp, _ = synth()
    rows, sids, caps = G.selection_captions(ctx, annotations=ann, splits=sp, n_expected=250)
    assert rows.dtype == np.int64 and sids.dtype == np.int64
    assert np.array_equal(rows, ctx.selection) and np.array_equal(sids, ctx.data.sample_ids[rows])
    assert list(caps) == [ann[int(s)]["caption"] for s in sids] and len(caps) == 250


def test_selection_captions_full_size_defaults():
    n_all = N_ALL
    ctx, ann, sp, _ = synth(n_all, N_TRAIN_FULL, N_SEL_FULL)
    rows, _, caps = G.selection_captions(ctx, annotations=ann, splits=sp)
    assert len(rows) == len(caps) == 32_413


@pytest.mark.parametrize("mutate", ["unsorted", "duplicate", "short", "overlap_train", "overlap_held", "not_split"])
def test_selection_row_assertions_fire(mutate):
    ctx, ann, sp, _ = synth()
    sel = ctx.selection.copy()
    if mutate == "unsorted":
        sel[[3, 4]] = sel[[4, 3]]
    elif mutate == "duplicate":
        sel[5] = sel[4]
    elif mutate == "short":
        sel = sel[:-1]
    elif mutate == "overlap_train":
        sel[0] = sp.scorer_train[7]
        sel = np.sort(sel)
    elif mutate == "overlap_held":
        sel[-1] = sp.held[3]
        sel = np.sort(sel)
    elif mutate == "not_split":
        sp = SimpleNamespace(**{**vars(sp), "selection": sel[::-1].copy() + 0})
    ctx = SimpleNamespace(selection=sel, data=ctx.data)
    if mutate in ("overlap_train", "overlap_held"):
        sp = SimpleNamespace(**{**vars(sp), "selection": sel})      # isolate the disjointness assertion
    with pytest.raises(AssertionError):
        G.selection_captions(ctx, annotations=ann, splits=sp, n_expected=250)


def test_selection_empty_caption_refused():
    ctx, ann, sp, _ = synth()
    ann[int(ctx.data.sample_ids[ctx.selection[2]])] = {"caption": "  "}
    with pytest.raises((ValueError, AssertionError)):
        G.selection_captions(ctx, annotations=ann, splits=sp, n_expected=250)


# ---------------------------------------------------------------- D2 item 2: snapshot and annotation asserts

def _hub(tmp_path, ref):
    hub = tmp_path / "models--x"
    (hub / "refs").mkdir(parents=True)
    (hub / "refs" / "main").write_text(ref + "\n")
    return hub


def test_snapshot_ref_ok_and_wrong(tmp_path):
    snap = Path(R5.SNAP.rstrip("/")).name
    G.assert_snapshot_ref(_hub(tmp_path, snap))
    with pytest.raises(SystemExit):
        G.assert_snapshot_ref(_hub(tmp_path / "b", "0" * 40))


def test_annotation_env_asserts(monkeypatch):
    monkeypatch.delenv("COSIR_ARTELINGO_ANNOTATIONS", raising=False)
    monkeypatch.setattr(G, "ANNOTATIONS_PATH", Path("/data/PDD/artelingo/artelingo_train.json"))
    G.assert_annotation_source()
    monkeypatch.setenv("COSIR_ARTELINGO_ANNOTATIONS", "/x.json")
    with pytest.raises(SystemExit):
        G.assert_annotation_source()
    monkeypatch.delenv("COSIR_ARTELINGO_ANNOTATIONS")
    monkeypatch.setattr(G, "ANNOTATIONS_PATH", Path("/other.json"))
    with pytest.raises(SystemExit):
        G.assert_annotation_source()


# ---------------------------------------------------------------- D3: positions, order, tolerance

def test_regression_positions_and_order_full_size():
    ctx, ann, sp, _ = synth(N_ALL, N_TRAIN_FULL, N_SEL_FULL)
    pos, caps = G.regression_sample(ctx, annotations=ann, splits=sp)
    expect = np.sort(np.random.default_rng(5).choice(183_694, size=2_048, replace=False))
    assert np.array_equal(pos, expect) and len(caps) == 2048 and np.all(np.diff(pos) > 0)
    want = [ann[int(ctx.data.sample_ids[sp.scorer_train[p]])]["caption"] for p in pos]
    assert list(caps) == want


def test_regression_refuses_wrong_scorer_train_size():
    ctx, ann, sp, _ = synth()
    with pytest.raises(AssertionError):
        G.regression_sample(ctx, annotations=ann, splits=sp)         # 600 rows, not 183,694


def test_compare_tolerance():
    base = np.full((10, 28), 0.5, np.float32)
    for delta, ok in ((0.9e-4, True), (1.1e-4, False)):
        r = G.compare(base + np.float32(delta), base, 1e-4)
        assert r["passed"] is ok
    r = G.compare(base + np.float32(2e-5), base, 1e-4)
    assert r["n_above_1e-5"] == 280 and r["max_abs"] == pytest.approx(2e-5, rel=1e-2)
    assert r["mean_abs"] == pytest.approx(2e-5, rel=1e-2)


def test_compare_shape_and_nonfinite():
    a = np.zeros((4, 28), np.float32)
    with pytest.raises(AssertionError):
        G.compare(a, a[:3], 1e-4)
    b = a.copy()
    b[0, 0] = np.nan
    assert G.compare(b, a, 1e-4)["passed"] is False


# ---------------------------------------------------------------- run

def _run(tmp_path, affect=None, device="cpu", loaded=None, **kw):
    ctx, ann, sp, aff = synth()
    return G.run(device, loaded or loaded_cpu(), tmp_path, ctx, annotations=ann, splits=sp,
                 affect_probs=aff if affect is None else affect, n_expected=250, n_train=600, sample=(5, 40),
                 file_shas={"model.safetensors": "ab"}, **kw), (ctx, ann, sp, aff)


def test_run_pass_writes_files_and_orders_calls(tmp_path):
    rec, (ctx, ann, sp, _) = _run(tmp_path)
    assert rec["item2"]["passed"] is True
    assert (fake_probs.calls[0]["texts"] != fake_probs.calls[1]["texts"]) and len(fake_probs.calls) == 2
    assert len(fake_probs.calls[0]["texts"]) == 40 and len(fake_probs.calls[1]["texts"]) == 250
    assert all(c["batch_size"] == 256 and c["max_length"] == 64 for c in fake_probs.calls)
    z = np.load(tmp_path / "r5_goemotions_selection.npz")
    assert z["probs"].dtype == np.float32 and z["probs"].shape == (250, 28)
    assert z["rows"].dtype == np.int64 and z["sample_ids"].dtype == np.int64
    assert np.array_equal(z["rows"], ctx.selection)
    assert np.array_equal(z["sample_ids"], ctx.data.sample_ids[ctx.selection])
    j = json.loads((tmp_path / "r5_goemotions_selection.json").read_text())
    for k in ("device", "versions", "model_snapshot", "model_file_sha256", "batch_size", "max_length", "n_rows",
              "n_captions", "max_token_count", "n_captions_over_64_tokens", "item2", "written", "npz_sha256",
              "probs_sha256", "rule_sha256"):
        assert k in j, k
    assert j["device"] == "cpu" and j["batch_size"] == 256 and j["max_length"] == 64
    assert j["item2"]["sample_rng"] == 5 and j["item2"]["tol"] == 1e-4
    assert j["item2"]["passed"] is True and j["n_rows"] == j["n_captions"] == 250
    assert j["npz_sha256"] == R5.sha256_file(tmp_path / "r5_goemotions_selection.npz")
    assert j["probs_sha256"] == hashlib.sha256(z["probs"].tobytes()).hexdigest()
    assert j["max_token_count"] == 8 and j["n_captions_over_64_tokens"] == 0


def test_run_item2_fields_use_rule_values(tmp_path):
    rec, _ = _run(tmp_path)
    assert rec["item2"]["sample_rng"] == R5.REG_SAMPLE[0] and rec["item2"]["tol"] == R5.GOEMO_TOL


def test_shifted_sample_fails_and_no_selection_caption_reaches_model(tmp_path):
    ctx, ann, sp, aff = synth()
    rec, _ = _run(tmp_path, affect=np.roll(aff, 1, axis=0))
    assert rec["item2"]["passed"] is False
    assert len(fake_probs.calls) == 1 and len(fake_probs.calls[0]["texts"]) == 40
    sel_caps = {ann[int(s)]["caption"] for s in ctx.data.sample_ids[ctx.selection]}
    assert not (set(fake_probs.calls[0]["texts"]) & sel_caps)
    assert not (tmp_path / "r5_goemotions_selection.npz").exists()
    assert not (tmp_path / "r5_goemotions_selection.json").exists()
    assert (tmp_path / "r5_goemotions_item2_failure.json").exists()


def test_device_tag_and_refusal(tmp_path):
    assert G.device_tag(loaded_cpu()) == "cpu"
    with pytest.raises(SystemExit):
        G.assert_device(loaded_cpu(), "cuda")
    G.assert_device(loaded_cpu(), "cpu")
    with pytest.raises(SystemExit):
        _run(tmp_path, device="cuda")
    assert fake_probs.calls == []


def test_no_overwrite_and_no_model_call(tmp_path):
    _run(tmp_path)
    fake_probs.calls = []
    with pytest.raises(SystemExit):
        _run(tmp_path)
    assert fake_probs.calls == []


def test_failure_record_blocks_rerun(tmp_path):
    ctx, ann, sp, aff = synth()
    _run(tmp_path, affect=np.roll(aff, 1, axis=0))
    fake_probs.calls = []
    with pytest.raises(SystemExit):
        _run(tmp_path)
    assert fake_probs.calls == []


def test_probs_validation():
    G.check_probs(np.full((2, 28), 0.5, np.float32), 2)
    for bad in (np.full((2, 28), 1.5, np.float32), np.full((2, 28), np.nan, np.float32),
                np.full((2, 28), 0.5, np.float64), np.full((3, 28), 0.5, np.float32)):
        with pytest.raises(AssertionError):
            G.check_probs(bad, 2)


def test_token_stats():
    tok = StubTok()
    mx, over = G.token_stats(tok, ["a b", " ".join(["w"] * 70), "x"], limit=64)
    assert mx == 72 and over == 1
