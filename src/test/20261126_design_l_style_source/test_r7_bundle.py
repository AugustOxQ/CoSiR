"""Ticket 07: seed guards, L14 comparators and posterior/feature boundaries (§7.3, G4).

Only the frozen A0/A1 configurations are evaluated on seed 42. Head archives use pytest's temporary folders.
Run from this folder:
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --rootdir=. test_r7_bundle.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_common as R  # noqa: E402
import r7_bundle as B  # noqa: E402


class RuleReached(Exception):
    """Stop admitted builds before reading any data."""


@pytest.mark.parametrize("seed,smoke", [(42, False), (55, False), (56, False), (57, False),
                                        (9001, True), (9002, True), (9003, True)])
def test_build_admits_round7_seeds(seed, smoke, monkeypatch):
    def stop():
        raise RuleReached
    monkeypatch.setattr(R, "assert_rule", stop)
    with pytest.raises(RuleReached):
        B.build_bundle(seed, smoke)


@pytest.mark.parametrize("seed,smoke", [(49, False), (49, True), (55, True), (9001, False)])
def test_build_refuses_other_seeds_before_reading(seed, smoke, monkeypatch):
    def no_reads():
        pytest.fail("a refused seed reached the rule/input reader")
    monkeypatch.setattr(R, "assert_rule", no_reads)
    with pytest.raises(ValueError, match="seed"):
        B.build_bundle(seed, smoke)


@pytest.mark.parametrize("damage", [None, "shape", "mask", "rows", "finite", "dtype", "keys",
                                  "class_count", "record_keys"])
def test_extra_posterior_boundary(tmp_path, monkeypatch, damage):
    ctx = SimpleNamespace(groups=np.zeros(8), selection=np.array([1, 3, 5, 7], dtype=np.int64))
    post = np.full((8, 2), np.nan, dtype=np.float32)
    post[ctx.selection] = 0.5
    arrays = {"img": post.copy(), "txt": post.copy(), "rows": ctx.selection.copy()}
    if damage == "shape":
        arrays["img"] = arrays["img"][:-1]
    elif damage == "mask":
        arrays["txt"][0] = 0.5
    elif damage == "rows":
        arrays["rows"] = arrays["rows"][::-1]
    elif damage == "finite":
        arrays["img"][1, 0] = np.nan
    elif damage == "dtype":
        arrays["txt"] = arrays["txt"].astype(np.float64)
    elif damage == "keys":
        del arrays["rows"]
    (tmp_path / "heads").mkdir()
    np.savez(tmp_path / "heads/post_V1.npz", **arrays)
    record = {} if damage == "record_keys" else {"n_groups": 3 if damage == "class_count" else 2}
    (tmp_path / "heads/head_V1.json").write_text(json.dumps(record))
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    if damage is not None:
        with pytest.raises(AssertionError):
            B.load_posteriors(ctx, "V1")
    else:
        got = B.load_posteriors(ctx, "V1")
        assert all(np.array_equal(got[s], post, equal_nan=True) for s in ("img", "txt"))


def synthetic_bundle():
    n = 6
    selection = np.array([1, 3, 5, 7], dtype=np.int64)
    p = np.full((8, 2), np.nan, dtype=np.float32)
    p[selection] = [[1, 0], [1, 0], [0, 1], [0, 1]]
    ep = SimpleNamespace(anchor=np.full(n, 1), candidates=np.tile([3, 5, 7, 3, 5, 7, 3, 5, 7, 3, 5, 7, 3], (n, 1)),
                         pairs_a_img=np.full((n, 4), 1), pairs_a_txt=np.full((n, 4), 3),
                         pairs_b_img=np.full((n, 4), 1), pairs_b_txt=np.full((n, 4), 5))
    ep.condition = lambda c: ((ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt, None)
                              if c == "a" else
                              (ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt, None))
    post = {h: {"img": p.copy(), "txt": p.copy()} for h in ("affect", "image", "caption", "V1")}
    post["V1"]["txt"][selection] = [[0, 1], [0, 1], [1, 0], [1, 0]]
    zero = {c: {d: np.zeros((n, 13), dtype=np.float32) for d in ("i2t", "t2i")} for c in ("a", "b")}
    return SimpleNamespace(n=n, ctx=SimpleNamespace(pooled=ep, groups=np.zeros(8), selection=selection),
                           post=post, cos=zero, t_n1u=zero, parity=np.arange(n) % 2)


@pytest.mark.parametrize("parts", [("affect", "image", "caption"), ("affect", "image", "caption", "V1"),
                                  ("V1", "caption", "affect", "image")])
def test_stack_and_features_respect_parts_order_and_layout(parts):
    stack, features = B.stack_and_features(synthetic_bundle(), parts)
    same = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    opposite = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    expected_stack = np.tile([opposite if h == "V1" else same for h in parts], (6, 1, 1))
    assert all(np.array_equal(stack[d], expected_stack) for d in ("i2t", "t2i"))
    fa = np.concatenate([[0, 1, -1, 0, 0, 0] if h == "V1" else [1, 0, 1, 0, 0, 1] for h in parts])
    fb = np.concatenate([[1, 0, 1, 0, 0, 1] if h == "V1" else [0, 1, -1, 0, 0, 0] for h in parts])
    assert np.array_equal(features["a"], np.tile(fa, (6, 1)))
    assert np.array_equal(features["b"], np.tile(fb, (6, 1)))
    assert all(x.dtype == np.float64 for x in features.values())


def test_condition_free_ties_choose_first_nested_cell():
    b = synthetic_bundle()
    for post in b.post.values():
        for x in post.values():
            x[b.ctx.selection] = 0.5
    per_anchor, picks = B.condition_free(b, ("affect", "image", "caption", "V1"))
    assert picks == {0: [0.0, 0.0], 1: [0.0, 0.0]}
    assert all(np.array_equal(x, np.zeros(b.n)) for x in per_anchor.values())


@pytest.fixture(scope="module")
def frozen_heads(tmp_path_factory):
    """Package the frozen CSD head at the R7 file boundary; never fit or write into an old round."""
    path = R.TEST / "20261116_grouping_step1_style/results/step1_heads_style.npz"
    # Frozen hash from R6's EXTRA_INPUTS; importing its scoring guards is unnecessary here.
    assert R.sha_file(path) == "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b", \
        "frozen CSD head hash differs"
    root = tmp_path_factory.mktemp("frozen_heads")
    (root / "heads").mkdir()
    with np.load(path) as z:
        rows = z["selection"]
        assert rows.dtype == np.int64 and rows.ndim == 1, "frozen head selection rows differ"
        arrays = {"rows": rows}
        for side in ("img", "txt"):
            values = z[f"style_csd__{side}"]
            assert values.shape == (len(rows), 17) and values.dtype == np.float32, "frozen CSD shape/dtype differs"
            x = np.full((308723, 17), np.nan, dtype=np.float32)
            x[rows] = values
            arrays[side] = x
    np.savez(root / "heads/post_R0.npz", **arrays)
    # Synthetic extra used ONLY on smoke seeds; seed 42 always builds frozen A0/A1.
    np.savez(root / "heads/post_V1.npz", **arrays)
    (root / "heads/head_V1.json").write_text(json.dumps({"n_groups": 17}))
    return root


@pytest.fixture(scope="module")
def seed42(frozen_heads):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "RESULTS", frozen_heads)
        return B.build_bundle(42, False)


def test_seed42_condition_free_regressions(seed42):
    assert 100 * float(seed42.pB["r1"].mean()) == 18.341064453125, "G4 B differs"
    assert 100 * float(seed42.pBp["r1"].mean()) == 18.436686197916664, "G4 B0 differs"
    b0, _ = B.condition_free(seed42, B.A0)
    assert all(np.array_equal(b0[m], seed42.pBp[m]) for m in b0), "B0 differs per anchor"
    b1, p1 = B.condition_free(seed42, B.A1)
    assert 100 * float(b1["r1"].mean()) == 18.804931640625, "G4 B1 differs"
    bx, px = B.condition_free(seed42, B.A0 + ("R0",))
    assert px == p1 and all(np.array_equal(bx[m], b1[m]) for m in b1), "BX[R0] differs from B1"
    for pa in (b0, b1, bx):
        assert bool((pa["gain"] == 0).all()), "condition-free gain must be zero"
    # G4's independently stored per-episode AFF-minus-B1 difference.
    paths = ("20261121_round3_affect_gate/results/seed42_arrays.npz",
             "20261125_artelingo_held_test/results/seed42_per_episode.npz")
    for path in paths:
        assert R.sha_file(R.path_of(path)) == R.INPUTS[path], "G4 input hash differs"
    with np.load(R.path_of(paths[0])) as r3, np.load(R.path_of(paths[1])) as r6:
        assert np.array_equal(seed42.cl, r6["cl"]) and np.array_equal(seed42.pair_index, r6["pair_index"]), \
            "G4 reference alignment differs"
        assert np.array_equal(r3["aff_fused__r1"] - b1["r1"], r6["diff__S1"]), "G4 per-anchor B1 differs"


@pytest.mark.parametrize("explicit_root", [False, True])
def test_smoke_extra_reads_smoke_or_explicit_root(frozen_heads, tmp_path, explicit_root):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "RESULTS", tmp_path / "unused_real")
        mp.setattr(R, "SMOKE_CHAIN", tmp_path / "unused_smoke" if explicit_root else frozen_heads)
        if explicit_root:
            b = B.build_bundle(9001, True, extra=("V1",), root=str(frozen_heads))
        else:
            b = B.build_bundle(9001, True, extra=("V1",))
    assert set(b.post) == {"affect", "image", "caption", "R0", "V1"}, "R0 or requested X missing"
    assert str(frozen_heads / "heads/post_R0.npz") in b.input_sha256, "R0 read outside requested root"
    assert str(frozen_heads / "heads/post_V1.npz") in b.input_sha256, "X read outside requested root"
    assert str(frozen_heads / "heads/head_V1.json") in b.input_sha256, "X record read outside requested root"
    assert b.readers["feature_names"] == R.load("rb_features").feature_names(B.A0), "A0 reader layout changed"
    parts = B.A0 + ("V1",)
    stack, features = B.stack_and_features(b, parts)
    assert stack["i2t"].shape == (b.n, 4, 13) and features["a"].shape == (b.n, 24), "X feature layout differs"
    bx, picks = B.condition_free(b, parts)
    b1, p1 = B.condition_free(b, B.A1)
    assert picks == p1 and all(np.array_equal(bx[m], b1[m]) for m in bx), "same posteriors must yield same BX"
