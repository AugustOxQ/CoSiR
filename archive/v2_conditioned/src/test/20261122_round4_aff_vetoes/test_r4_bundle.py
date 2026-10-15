"""Tests of the round-4 bundle (r4_bundle.py; rule DECISION_RULE.md §4 item 1, D2 to D5, §5 item 1, §10: T1-1 and the
seed guard). Run from this folder, output redirected (rule §10: the log is read only with grep for PASS/FAIL,
passed/failed, Error and Traceback):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r4_bundle.py \
        > results/smoke/test_r4_bundle.log 2>&1

Two kinds of test:
  synthetic  the seed guard (Review focus 1) and the build's refusal of other seeds before anything is read; the guard
             functions under constructed violations (T1-1: CSD-head selection, finiteness on selection rows, parity,
             round 3's head identity); validate_a1's refusals; the cache (round trip, refusals, pair binding);
             compare_a1_with_round1 on a synthetic seed-42 pair (prints nothing, flags each mismatch, propagates round
             1's exceptions, uses a round-1 bundle passed in).
  smoke      the real builder on smoke seed 9001: shapes and dtypes, D2 to D5's identities, no seed-42 file opened,
             the real A1 readers and heads used (never smoke ones), round 3's fields equal a pure round-3 build, each
             bundle guard fires through the build under a constructed violation (T1-1), the extension's own
             unchanged-fields guard fires when a step mutates a round-3 field, the cache round trip.
Every assertion on real data asserts a plain bool with a message, so a failure prints no array or metric value.
"""
import builtins
import contextlib
import copy
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r4_common as R4  # noqa: E402
import r4_bundle as R4B  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores  # noqa: E402

R3, RB3, C = R4.R3, R4.RB3, R4.C
A0, A1 = R3.A0, R4.A1
SMOKE_SEED = 9001
SMOKE_DIR = R4.res_dir(True)


# ---------------------------------------------------------------- synthetic helpers

def _cf_scores(rng, n, k=13):
    base = {d: rng.normal(size=(n, k)).astype(np.float32) for d in DIRECTIONS}
    return {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}


def _per_anchor(rng, n):
    r1 = (rng.integers(0, 5, size=n) / 4.0).astype(np.float64)
    return {"r1": r1, "gain": np.zeros(n), "other": r1.copy(), "swap": np.zeros(n), "strict": r1 * 0.0}


def _features18(rng, n):
    """A0 features with Delta^b = -Delta^a and the image supports and contrasts swapped under b (D3, D5)."""
    Fa, Fb = rng.normal(size=(n, 18)), rng.normal(size=(n, 18))
    for j in range(3):
        Fb[:, 6 * j + 2] = -Fa[:, 6 * j + 2]
    Fb[:, 6], Fb[:, 7] = Fa[:, 7], Fa[:, 6]
    return {"a": Fa, "b": Fb}


def fake_bundle(n=10, seed=SMOKE_SEED, smoke=True, rng_seed=0):
    """A synthetic extended bundle (round 3's fields plus the A1 fields) that passes validate_a1."""
    rng = np.random.default_rng(rng_seed)
    F = _features18(rng, n)
    F1 = {c: np.concatenate([F[c], rng.normal(size=(n, 6))], axis=1) for c in CONDITIONS}
    F1["b"][:, 20] = -F1["a"][:, 20]
    Bp1 = _cf_scores(rng, n)
    return SimpleNamespace(
        seed=seed, smoke=smoke, n=n, ctx=None, cl=rng.integers(0, 4, size=n).astype(np.int64),
        parity=(np.arange(n) % 2).astype(np.int64), pair_index=np.repeat(np.arange(3), -(-n // 3))[:n].astype(np.int64),
        anchor=rng.integers(0, 1000, size=n).astype(np.int64), cos=_cf_scores(rng, n), B=_cf_scores(rng, n),
        pB=_per_anchor(rng, n), Bp=_cf_scores(rng, n), pBp=_per_anchor(rng, n), t_n1u=_cf_scores(rng, n),
        post=None, stack={d: rng.random(size=(n, 3, 13)).astype(np.float32) for d in DIRECTIONS}, F=F,
        affect_head={"n_classes": 41, "draw_rows_sha256": "x"}, readers=None, readers_record=None,
        episodes_sha256={"emotion__style": "y"}, input_sha256={"z": "w"}, checks={"synthetic": True},
        F1=F1, v=np.minimum(F["a"][:, 6], F["a"][:, 7]), Bp1=Bp1, pBp1=per_anchor(Bp1), readers_a1=None,
        readers_a1_record=None, checks_a1={"synthetic": True}, input_sha256_a1={"q": "r"})


# ---------------------------------------------------------------- the seed guard (Review focus 1; no build)

def test_round3_seed_guard_admits_round4_seeds():
    assert R3.TEST_SEEDS == (52, 53, 54)
    for s in (42, 52, 53, 54):
        RB3._check_seed(s, False)
        R4B._check_seed(s, False)
    for s in R4.SMOKE_SEEDS:
        RB3._check_seed(s, True)
        R4B._check_seed(s, True)
    for s, smoke in ((49, False), (50, False), (51, False), (55, False), (43, False), (52, True), (42, True),
                     (9004, True), (9001, False)):
        with pytest.raises(ValueError):
            RB3._check_seed(s, smoke)
        with pytest.raises(ValueError):
            R4B._check_seed(s, smoke)


def test_guard_refuses_when_round3_seeds_are_not_round4s(monkeypatch):
    monkeypatch.setattr(R3, "TEST_SEEDS", (49, 50, 51))
    for s in (52, 49):
        with pytest.raises(ValueError):
            R4B._check_seed(s, False)
    monkeypatch.setattr(R3, "TEST_SEEDS", (52, 53, 54))
    monkeypatch.setattr(R3, "SMOKE_SEEDS", (9001, 9002, 9003, 9004))
    with pytest.raises(ValueError):
        R4B._check_seed(9001, True)


def test_build_refuses_other_seeds_before_reading():
    def boom(*a, **k):
        raise AssertionError("read before the seed guard")

    with mock.patch.object(np, "load", boom), mock.patch.object(RB3, "build_bundle", boom), \
            mock.patch.object(R4B, "_load_csd_heads", boom):
        for seed, smoke in ((49, False), (50, False), (51, False), (55, False), (43, False), (52, True),
                            (42, True), (9004, True)):
            with pytest.raises(ValueError):
                R4B.build_bundle(seed, smoke)


# ---------------------------------------------------------------- the guard functions (T1-1, synthetic)

def test_heads_selection_guard():
    sel = np.array([3, 5, 8, 13, 21], dtype=np.int64)
    R4B.check_heads_selection(sel.copy(), sel)
    for bad in (sel[::-1].copy(), sel[:-1].copy(), sel + 1, np.array([3, 5, 8, 13, 22], dtype=np.int64)):
        with pytest.raises(AssertionError):
            R4B.check_heads_selection(bad, sel)


def test_finiteness_guard_on_selection_rows_only():
    rows, sel = 12, np.array([1, 4, 7, 10])
    rng = np.random.default_rng(0)

    def post():
        out = {}
        for h, k in zip(A1, (5, 6, 6, 4)):
            out[h] = {}
            for side in ("img", "txt"):
                x = np.full((rows, k), np.nan, dtype=np.float32)
                x[sel] = rng.random(size=(len(sel), k)).astype(np.float32)
                out[h][side] = x
        return out

    R4B.check_finite_on_selection(post(), sel, A1)                    # NaN outside selection rows: allowed
    for h in A1:
        for side in ("img", "txt"):
            p = post()
            p[h][side][7, 1] = np.nan                                   # a selection row
            with pytest.raises(AssertionError):
                R4B.check_finite_on_selection(p, sel, A1)
            p = post()
            p[h][side][4, 0] = np.inf
            with pytest.raises(AssertionError):
                R4B.check_finite_on_selection(p, sel, A1)


def test_parity_guard():
    n = 9
    R4B.check_parity(np.arange(n) % 2, n)
    for bad in (1 - np.arange(n) % 2, np.zeros(n, np.int64), (np.arange(n) % 2)[:-1], np.arange(n) % 3):
        with pytest.raises(ValueError):
            R4B.check_parity(bad, n)
    b = fake_bundle(n=n)
    b.parity = 1 - b.parity                                          # round 3's validate: the same guard
    with pytest.raises(ValueError):
        RB3.validate(b)
    with pytest.raises(ValueError):
        R4B.validate_a1(b)


# ---------------------------------------------------------------- validate_a1 (synthetic)

def test_validate_a1_passes_a_consistent_bundle():
    checks = R4B.validate_a1(fake_bundle(n=12))
    assert checks and all(v is True for v in checks.values())


@pytest.mark.parametrize("what", [
    "first18", "F1_float32", "F1_shape", "F1_nonfinite", "delta_csd", "v_value", "v_dtype", "v_b", "Bp1_cond",
    "Bp1_dtype", "pBp1_gain", "pBp1_not_per_anchor", "F1_keys"])
def test_validate_a1_refusals(what):
    b = fake_bundle(n=12, rng_seed=4)
    if what == "first18":
        b.F1["a"][3, 5] += 1.0
    elif what == "F1_float32":
        b.F1["a"] = b.F1["a"].astype(np.float32)
    elif what == "F1_shape":
        b.F1 = {c: b.F1[c][:, :23] for c in CONDITIONS}
    elif what == "F1_nonfinite":
        b.F1["b"][2, 22] = np.nan
    elif what == "delta_csd":
        b.F1["b"][2, 20] += 1.0
    elif what == "v_value":
        b.v = b.v.copy()
        b.v[2] += 1e-3
    elif what == "v_dtype":
        b.v = b.v.astype(np.float32)
    elif what == "v_b":                                            # v^b != v^a: image S and C of b both lowered
        low = min(b.F["a"][5, 6], b.F["a"][5, 7]) - 1.0
        for X in (b.F["b"], b.F1["b"]):
            X[5, 6] = X[5, 7] = low
    elif what == "Bp1_cond":
        b.Bp1["b"]["i2t"] = b.Bp1["b"]["i2t"] + np.float32(1.0)
    elif what == "Bp1_dtype":
        b.Bp1 = {c: {d: b.Bp1[c][d].astype(np.float64) for d in DIRECTIONS} for c in CONDITIONS}
    elif what == "pBp1_gain":
        b.pBp1["gain"] = b.pBp1["gain"].copy()
        b.pBp1["gain"][3] = 0.25
    elif what == "pBp1_not_per_anchor":
        b.pBp1["r1"] = 1.0 - b.pBp1["r1"]
    elif what == "F1_keys":
        b.F1 = {"a": b.F1["a"]}
    with pytest.raises(ValueError):
        R4B.validate_a1(b)


# ---------------------------------------------------------------- the cache (synthetic)

def _cache_paths(stem):
    p = SMOKE_DIR / f"{stem}.npz"
    return p, R4B.a1_path(p)


def _unlink(*paths):
    for p in paths:
        Path(p).unlink(missing_ok=True)


def test_cache_round_trip_synthetic():
    b = fake_bundle(n=14)
    p, p1 = _cache_paths("test_r4_bundle_cache_synthetic")
    _unlink(p, p1)
    try:
        shas = R4B.save_bundle(b, p)
        assert set(shas) == {"r3", "a1"} and shas["r3"] == R4.sha_file(p) and shas["a1"] == R4.sha_file(p1)
        got = R4B.load_bundle_cache(p, b.seed, True, shas, readers=False)
        eq = R4B.bundles_equal(b, got)
        assert eq and all(eq.values()), [k for k, v in eq.items() if not v]
        assert got.from_cache is True and got.ctx is None and got.post is None
        assert got.checks_a1 == b.checks_a1 and got.input_sha256_a1 == b.input_sha256_a1
        assert got.v.dtype == np.float64 and got.F1["a"].dtype == np.float64 and got.Bp1["a"]["i2t"].dtype == np.float32
    finally:
        _unlink(p, p1)


def test_cache_refusals_synthetic(monkeypatch):
    b = fake_bundle(n=9)
    p, p1 = _cache_paths("test_r4_bundle_cache_refusals")
    _unlink(p, p1)
    try:
        shas = R4B.save_bundle(b, p)
        with pytest.raises(SystemExit):
            R4B.load_bundle_cache(p, b.seed + 1, True, shas, readers=False)                 # another seed
        with pytest.raises(SystemExit):
            R4B.load_bundle_cache(p, b.seed, False, shas, readers=False)                    # another smoke flag
        with pytest.raises(SystemExit):
            R4B.load_bundle_cache(p, b.seed, True, {**shas, "a1": "0" * 64}, readers=False)  # A1 file changed
        with pytest.raises(SystemExit):
            R4B.load_bundle_cache(p, b.seed, True, {**shas, "r3": "0" * 64}, readers=False)  # round-3 file changed
        for bad in (None, shas["r3"], {"r3": shas["r3"]}):
            with pytest.raises(ValueError):
                R4B.load_bundle_cache(p, b.seed, True, bad, readers=False)                  # both SHA-256s required
        with mock.patch.object(R4, "RULE_SHA", "f" * 64):                                    # another round-4 rule
            with pytest.raises(SystemExit):
                R4B.load_bundle_cache(p, b.seed, True, shas, readers=False)
        # pair binding: a different round-3 cache under the same name is refused by the A1 file's record
        b2 = fake_bundle(n=9, rng_seed=7)
        sha_r3_other = RB3.save_bundle(b2, p)
        with pytest.raises(SystemExit):
            R4B.load_bundle_cache(p, b.seed, True, {"r3": sha_r3_other, "a1": shas["a1"]}, readers=False)
        # a missing A1 file
        p1.unlink()
        with pytest.raises((SystemExit, FileNotFoundError)):
            R4B.load_bundle_cache(p, b.seed, True, {"r3": sha_r3_other, "a1": shas["a1"]}, readers=False)
    finally:
        _unlink(p, p1)


def test_cache_never_overwrites_non_smoke_and_refuses_bad_bundles():
    p, p1 = _cache_paths("test_r4_bundle_cache_nonsmoke")
    _unlink(p, p1)
    try:
        b = fake_bundle(n=9, seed=42, smoke=False)
        shas = R4B.save_bundle(b, p)
        with pytest.raises(SystemExit):
            R4B.save_bundle(fake_bundle(n=9, seed=42, smoke=False, rng_seed=3), p)
        assert R4.sha_file(p) == shas["r3"] and R4.sha_file(p1) == shas["a1"]
        p.unlink()                                                    # only the A1 file left: still refused,
        with pytest.raises(SystemExit):                               # and the round-3 file is not written
            R4B.save_bundle(fake_bundle(n=9, seed=42, smoke=False, rng_seed=3), p)
        assert not p.exists()
        _unlink(p, p1)
        bad = fake_bundle(n=9)
        bad.F1["a"][1, 2] += 1.0                                     # fails validate_a1: nothing written
        with pytest.raises(ValueError):
            R4B.save_bundle(bad, p)
        assert not p.exists() and not p1.exists()
    finally:
        _unlink(p, p1)


# ---------------------------------------------------------------- compare_a1_with_round1 (synthetic seed-42 pair)

def mini_seed42_pair(n=6, rows=40, seed=0):
    """A synthetic extended seed-42 bundle and an identical stand-in for round 1's load_bundle(), built with the real
    round-1 feature code on small arrays. Every A1 comparison passes except B'(A1)'s mean against the rule's
    18.804931640625, which a synthetic bundle cannot equal."""
    from src.eval.aspect_episodes import AspectEpisodes
    rng = np.random.default_rng(seed)

    def idx(*shape):
        return rng.integers(0, rows, size=shape).astype(np.int64)

    def soft(k):
        x = rng.random(size=(rows, k)).astype(np.float32)
        x = (x / x.sum(axis=1, keepdims=True)).astype(np.float32)
        x[rows - 1] = np.nan                                          # one row outside selection
        return x

    ep = AspectEpisodes("mixed", "mixed", idx(n), idx(n, 13), idx(n, 4), idx(n, 4), idx(n, 4), idx(n, 4))
    ep.anchor[:] = np.minimum(ep.anchor, rows - 2)
    for f in ("candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"):
        getattr(ep, f)[:] = np.minimum(getattr(ep, f), rows - 2)
    post = {h: {"img": soft(k), "txt": soft(k)} for h, k in zip(A1 + ("rand",), (5, 6, 6, 4, 4))}
    ctx = SimpleNamespace(pooled=ep)
    F, _ = R3.rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), A0)
    F1, _ = R3.rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), A1)
    Bp1 = _cf_scores(rng, n)
    b = fake_bundle(n=n, seed=42, smoke=False, rng_seed=seed + 1)
    b.ctx, b.anchor, b.F, b.F1 = ctx, ep.anchor, F, F1
    b.post = {h: post[h] for h in A1}
    b.v = np.minimum(F["a"][:, 6], F["a"][:, 7])
    b.Bp1, b.pBp1 = Bp1, per_anchor(Bp1)

    def copy_scores(x):
        return {c: {d: np.array(x[c][d], copy=True) for d in DIRECTIONS} for c in CONDITIONS}

    ep1 = AspectEpisodes("mixed", "mixed", *(np.array(getattr(ep, f), copy=True) for f in
                                             ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img",
                                              "pairs_b_txt")))
    r1 = SimpleNamespace(
        ctx=SimpleNamespace(pooled=ep1), post={h: {s: post[h][s].copy() for s in ("img", "txt")} for h in post},
        Bp={"A1": copy_scores(Bp1)}, pBp={"A1": {m: v.copy() for m, v in b.pBp1.items()}},
        checks={"stand_in": True})
    return b, r1


def _noisy_stand_in(r1):
    def load_bundle(smoke=False):
        print("  told: margin R@1 1.23 [0.45, 6.78]")
        C.log("B'(A1) mean R@1 18.8")
        print("reader margin 0.35", file=sys.stderr)
        return r1
    return load_bundle


MEAN_CHECK = "Bprime_A1_mean_r1_equals_rule"


def test_compare_a1_flags_only_the_mean_on_a_synthetic_pair_and_prints_nothing(capfd):
    b, r1 = mini_seed42_pair()
    with mock.patch.object(C, "load_bundle", _noisy_stand_in(r1)):
        with pytest.raises(RB3.BundleMismatch) as e:
            R4B.compare_a1_with_round1(b)
    res = e.value.result
    failed = {k for k, v in res["checks"].items() if v is not True}
    assert failed == {MEAN_CHECK}, failed
    names = set(res["checks"])
    for want in ("post.csd.img", "post.csd.txt", "features_A1.a", "features_A1.b", "features_A1.a.first18_equal_A0",
                 "Bprime_A1.a.i2t", "Bprime_A1.b.t2i", "pBprime_A1.r1", "pBprime_A1.gain"):
        assert want in names, want
    out, err = capfd.readouterr()
    assert out == "" and err == "", "compare_a1_with_round1 let round 1's printing through"


def test_compare_a1_all_pass_when_the_mean_is_the_rules(monkeypatch):
    b, r1 = mini_seed42_pair(seed=2)
    monkeypatch.setattr(R4, "BPA1_MEAN_42", 100 * float(np.mean(b.pBp1["r1"])))
    with mock.patch.object(C, "load_bundle", side_effect=AssertionError("must use the r1 passed in")):
        res = R4B.compare_a1_with_round1(b, r1)
    assert res["all_pass"] is True and all(v is True for v in res["checks"].values())
    assert res["n_checks"] == len(res["checks"])


@pytest.mark.parametrize("what,expect", [
    ("Bp", {"Bprime_A1.a.i2t"}),
    ("pBp", {"pBprime_A1.r1"}),
    ("post_csd", {"post.csd.img", "features_A1.a", "features_A1.b"}),
    ("features", {"features_A1.b"}),
    ("first18", {"features_A1.a", "features_A1.a.first18_equal_A0"}),
    ("dtype", {"Bprime_A1.b.t2i"}),
    ("r1_checks", {"round1_load_bundle_own_checks"}),
])
def test_compare_a1_names_each_mismatch(monkeypatch, what, expect):
    b, r1 = mini_seed42_pair(seed=3)
    monkeypatch.setattr(R4, "BPA1_MEAN_42", 100 * float(np.mean(b.pBp1["r1"])))
    if what == "Bp":
        r1.Bp["A1"]["a"]["i2t"][2, 5] += np.float32(1.0)
    elif what == "pBp":
        r1.pBp["A1"]["r1"][1] += 0.25
    elif what == "post_csd":
        r1.post["csd"]["img"][:-1] = r1.post["csd"]["img"][:-1][::-1].copy()
    elif what == "features":
        b.F1["b"] = b.F1["b"].copy()
        b.F1["b"][0, 23] += 0.5
    elif what == "first18":
        b.F1["a"] = b.F1["a"].copy()
        b.F1["a"][0, 1] += 0.5
    elif what == "dtype":
        r1.Bp["A1"]["b"]["t2i"] = r1.Bp["A1"]["b"]["t2i"].astype(np.float64)
    elif what == "r1_checks":
        r1.checks["stand_in"] = False
    with pytest.raises(RB3.BundleMismatch) as e:
        R4B.compare_a1_with_round1(b, r1)
    failed = {k for k, v in e.value.result["checks"].items() if v is not True}
    assert failed == expect, failed


def test_compare_a1_propagates_round1_exceptions_silently(capfd):
    b, _ = mini_seed42_pair()

    def failing(smoke=False):
        print("told: margin R@1 9.99")
        raise SystemExit("regression check failed: 3 step-1 arrays differ from step1_eval_style.npz")

    with mock.patch.object(C, "load_bundle", failing):
        with pytest.raises(SystemExit, match="regression check failed"):
            R4B.compare_a1_with_round1(b)
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_compare_a1_refuses_other_bundles():
    for kw in ({"seed": 52, "smoke": False}, {"seed": 42, "smoke": True}, {"seed": SMOKE_SEED, "smoke": True}):
        b, r1 = mini_seed42_pair()
        b.seed, b.smoke = kw["seed"], kw["smoke"]
        with pytest.raises(ValueError):
            R4B.compare_a1_with_round1(b, r1)
    b, r1 = mini_seed42_pair()
    del b.F1                                                         # not extended
    with pytest.raises(ValueError):
        R4B.compare_a1_with_round1(b, r1)
    b, r1 = mini_seed42_pair()
    b.ctx = None                                                     # from a cache
    with pytest.raises(ValueError):
        R4B.compare_a1_with_round1(b, r1)


def test_check_v75_on_constructed_values(monkeypatch):
    b = fake_bundle(n=8, seed=42, smoke=False)
    b.v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    q = float(np.percentile(b.v, 75))
    monkeypatch.setattr(R4, "V75", q)
    monkeypatch.setattr(R4, "V75_KEEP_42", int((b.v < q).sum()))
    monkeypatch.setattr(R4B, "N_SEED42", 8)
    res = R4B.check_v75(b)
    assert res["all_pass"] is True
    monkeypatch.setattr(R4, "V75_KEEP_42", int((b.v < q).sum()) + 1)
    with pytest.raises(RB3.BundleMismatch) as e:
        R4B.check_v75(b)
    assert {k for k, v in e.value.result["checks"].items() if v is not True} == {"v75_keep_count_equals_rule"}
    monkeypatch.setattr(R4, "V75", q + 1e-12)
    with pytest.raises(RB3.BundleMismatch):
        R4B.check_v75(b)
    b.seed = SMOKE_SEED
    with pytest.raises(ValueError):
        R4B.check_v75(b)


# ---------------------------------------------------------------- the real builder on smoke seed 9001

SEED42_ARRAYS = {"step1_eval_style.npz", "per_anchor_n6c_gate.npz", "bs_cache.npz", "n6_seed42.json",
                 "per_anchor_sweep.npz"}
ALLOWED_SEED42_NAMES = {"A3_seed42.pt"}   # the method-A checkpoint (training seed 42), the same for every episode seed


def is_seed42_file(p) -> bool:
    name = Path(p).name
    if name in ALLOWED_SEED42_NAMES:
        return False
    return "seed42" in name or name in SEED42_ARRAYS or name.startswith("cand_")


class Recorder:
    """Records every path opened through builtins.open / io.open (with the calling function's name) and every np.load
    path while active."""

    def __init__(self):
        self.opens, self.loads = [], []
        self._open, self._load = builtins.open, np.load

    def _rec_open(self, file, *a, **k):
        if isinstance(file, (str, bytes, Path)) or hasattr(file, "__fspath__"):
            caller = sys._getframe(1).f_code.co_name
            self.opens.append((str(Path(file).resolve()), caller))
        return self._open(file, *a, **k)

    def _rec_load(self, file, *a, **k):
        if isinstance(file, (str, Path)) or hasattr(file, "__fspath__"):
            self.loads.append(str(Path(file).resolve()))
        return self._load(file, *a, **k)

    def __enter__(self):
        self._ps = [mock.patch.object(builtins, "open", self._rec_open), mock.patch.object(io, "open", self._rec_open),
                    mock.patch.object(np, "load", self._rec_load)]
        for p in self._ps:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._ps):
            p.stop()
        return False


@pytest.fixture(scope="module")
def smoke_build():
    eps = R3.AB / "results" / "smoke" / f"episodes_seed{SMOKE_SEED}.npz"
    if not eps.exists():
        pytest.fail(f"smoke seed {SMOKE_SEED} is not built: run run_baselines.py --smoke --episodes-seed {SMOKE_SEED}")
    out, err = io.StringIO(), io.StringIO()
    with Recorder() as rec, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        bundle = R4B.build_bundle(SMOKE_SEED, True)
    rec.stdout, rec.stderr = out.getvalue(), err.getvalue()
    return bundle, rec


def r3_view(b, **changes):
    """A pure round-3 bundle that shares b's round-3 fields (post restricted to A0), for extension tests."""
    v = SimpleNamespace(**{k: getattr(b, k) for k in R4B.R3_FIELDS})
    v.post = {h: b.post[h] for h in A0}
    for k, x in changes.items():
        setattr(v, k, x)
    return v


DECIMAL = re.compile(r"\d\.\d")


def test_smoke_build_prints_no_value(smoke_build):
    _, rec = smoke_build
    lines = [ln for ln in rec.stdout.splitlines() if ln.strip()]
    foreign = sum(("[r3_bundle]" not in ln and "[r4_bundle]" not in ln) for ln in lines)
    decimals = sum(bool(DECIMAL.search(ln)) for ln in lines)
    own = sum("[r4_bundle]" in ln for ln in lines)
    assert foreign == 0, f"{foreign} stdout lines of the build are not r3_bundle's or r4_bundle's own"
    assert decimals == 0, f"{decimals} stdout lines of the build carry a decimal number"
    assert own >= 1, "r4_bundle logged no progress line"
    assert rec.stderr.strip() == "", f"the build wrote {len(rec.stderr.splitlines())} lines to stderr"


def test_smoke_shapes_and_dtypes(smoke_build):
    b, _ = smoke_build
    n = b.n
    assert b.seed == SMOKE_SEED and b.smoke is True and n == 3 * 64 and b.ctx is not None, "seed, smoke flag or n"
    assert tuple(b.post) == A1, "post must hold A1's groupings in A1 order"
    n_rows = len(b.ctx.groups)
    for side in ("img", "txt"):
        x = b.post["csd"][side]
        assert x.shape == (n_rows, 17) and x.dtype == np.float32, f"post csd/{side} shape or dtype"
        assert bool(np.isfinite(x[b.ctx.selection]).all()), f"post csd/{side} not finite on selection rows"
        assert bool(np.isnan(x[~b.ctx.in_sel]).all()), f"post csd/{side} must be NaN outside selection rows"
    for c in CONDITIONS:
        x = b.F1[c]
        assert x.shape == (n, 24) and x.dtype == np.float64, f"F1/{c} shape or dtype"
        assert bool(np.isfinite(x).all()), f"F1/{c} not finite"
    assert b.v.shape == (n,) and b.v.dtype == np.float64, "v shape or dtype"
    for c in CONDITIONS:
        for d in DIRECTIONS:
            x = b.Bp1[c][d]
            assert x.shape == (n, 13) and x.dtype == np.float32, f"Bp1/{c}/{d} shape or dtype"
            assert bool(np.isfinite(x).all()), f"Bp1/{c}/{d} not finite"
    assert set(b.pBp1) == set(METRICS) and all(b.pBp1[m].shape == (n,) for m in METRICS), "pBp1 metrics or shapes"
    assert b.readers_a1["config"] == "A1" and tuple(b.readers_a1["groupings"]) == A1, "A1 readers"
    assert list(b.readers_a1["feature_names"]) == R3.rf.feature_names(A1), "A1 reader feature names"
    assert len(b.readers_a1["halves"]) == 2, "two A1 half-readers"
    assert b.readers_a1_record == {"rule_sha256": C.RULE_SHA, "smoke": False}, "A1 readers record"
    assert b.checks_a1 and all(v is True for v in b.checks_a1.values()), "an A1 build check is not True"
    assert set(b.input_sha256_a1) >= set(R4B.A1_INPUTS), "A1 inputs not all hashed"


def test_smoke_identities_D2_to_D5(smoke_build):
    b, _ = smoke_build
    ctx, ep = b.ctx, b.ctx.pooled
    for c in CONDITIONS:
        assert bool(np.array_equal(b.F1[c][:, :18], b.F[c])), f"F1/{c}: first 18 columns differ from F (D3)"
    for j in range(4):
        assert bool(np.array_equal(b.F1["b"][:, 6 * j + 2], -b.F1["a"][:, 6 * j + 2])), f"Delta^b != -Delta^a ({j})"
    assert bool(np.array_equal(b.v, np.minimum(b.F["a"][:, 6], b.F["a"][:, 7]))), "v != min(S_image^a, C_image^a)"
    assert bool(np.array_equal(np.minimum(b.F1["b"][:, 6], b.F1["b"][:, 7]), b.v)), "v^b != v^a (D5)"
    # D2: post["csd"] is step 1's heads through run_step1.full_post
    with np.load(C.INPUT_FILES["step1_heads_style"]) as z:
        sel_ok = bool(np.array_equal(z["selection"], ctx.selection))
        img = np.asarray(z["style_csd__img"], np.float32)
        txt = np.asarray(z["style_csd__txt"], np.float32)
    assert sel_ok, "heads selection differs from the context's"
    assert bool(np.array_equal(b.post["csd"]["img"][ctx.selection], img)), "post csd/img is not the stored head"
    assert bool(np.array_equal(b.post["csd"]["txt"][ctx.selection], txt)), "post csd/txt is not the stored head"
    # D3: the csd columns are S, C, Delta of the csd agreements (spot check against a direct computation)
    si, st, ci, ct = C.condition_sets(ep, "a")
    sup = np.einsum("nsc,nsc->ns", b.post["csd"]["img"][si], b.post["csd"]["txt"][st])
    con = np.einsum("nsc,nsc->ns", b.post["csd"]["img"][ci], b.post["csd"]["txt"][ct])
    assert bool(np.array_equal(b.F1["a"][:, 18], sup.mean(axis=1).astype(np.float64))), "csd S^a"
    assert bool(np.array_equal(b.F1["a"][:, 19], con.mean(axis=1).astype(np.float64))), "csd C^a"
    # D4: B'(A1) with key csd equals step 1's T_6u with key style_csd, through the same cross-fit
    post_s1 = {"affect": b.post["affect"], "image": b.post["image"], "caption": b.post["caption"],
               "style_csd": b.post["csd"]}
    t6u = uniform_probe_scores(post_s1, ep, ("affect", "image", "caption", "style_csd"))
    want = crossfit_condition_free(ctx.cos, b.t_n1u, t6u, ctx.parity)[0]
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert bool(np.array_equal(b.Bp1[c][d], want[c][d])), f"Bp1/{c}/{d} differs from step 1's recipe"
    for d in DIRECTIONS:
        assert bool(np.array_equal(b.Bp1["a"][d], b.Bp1["b"][d])), f"Bp1/{d} not condition-free"
    pa = per_anchor(b.Bp1)
    assert all(bool(np.array_equal(b.pBp1[m], pa[m])) for m in METRICS), "pBp1 is not per_anchor(Bp1)"
    assert bool((b.pBp1["gain"] == 0).all()), "B'(A1) condition gain is not 0 on every episode"


def test_smoke_no_seed42_file_opened(smoke_build):
    _, rec = smoke_build
    bad_loads = [p for p in rec.loads if is_seed42_file(p)]
    bad_opens = [(p, f) for p, f in rec.opens if is_seed42_file(p) and f != "sha_file"]
    assert not bad_loads, f"seed-42 arrays loaded: {bad_loads}"
    assert not bad_opens, f"seed-42 files opened (not for hashing): {bad_opens}"
    eps = str((R3.AB / "results" / "smoke" / f"episodes_seed{SMOKE_SEED}.npz").resolve())
    assert eps in rec.loads, "the smoke episodes were not loaded through np.load"


def test_smoke_uses_real_heads_and_readers(smoke_build):
    _, rec = smoke_build
    paths = {p for p, _ in rec.opens} | set(rec.loads)
    smoke_dirs = tuple(str(p.resolve()) for p in (
        R3.R1 / "results" / "smoke", R3.TEST / "20261116_grouping_step1_style" / "results" / "smoke",
        R3.TEST / "20261101_aspect_factor_gonogo" / "checkpoints" / "smoke",
        R3.TEST / "20261108_new_method_quick_checks" / "results" / "smoke"))
    bad = [p for p in paths if p.startswith(smoke_dirs)]
    assert not bad, f"smoke checkpoint, heads or readers opened: {bad}"
    for want in (R3.R1 / "results" / "rb_reader_A1.pkl", R3.R1 / "results" / "rb_reader_A0.pkl"):
        assert str(want.resolve()) in paths, f"{want.name} was not read"
    heads = str((R3.TEST / "20261116_grouping_step1_style" / "results" / "step1_heads_style.npz").resolve())
    assert heads in rec.loads, "the stored step-1 heads were not loaded"


def test_smoke_round3_fields_equal_a_pure_round3_build(smoke_build):
    b, _ = smoke_build
    with contextlib.redirect_stdout(io.StringIO()):
        b3 = RB3.build_bundle(SMOKE_SEED, True)
    eq = RB3.bundles_equal(b, b3)
    assert all(eq.values()), f"round-3 arrays differ: {[k for k, v in eq.items() if not v]}"
    for h in A0:
        for side in ("img", "txt"):
            ok = bool(np.array_equal(b.post[h][side], b3.post[h][side], equal_nan=True))
            assert ok, f"post {h}/{side} differs from round 3's"
    assert tuple(b3.post) == A0, "round 3's own post gained a key"
    assert b.checks == b3.checks and b.affect_head == b3.affect_head, "round 3's checks or head differ"
    assert set(vars(b3)) == set(R4B.R3_FIELDS), "R3_FIELDS is not round 3's field list"


def test_smoke_extension_refuses_an_extended_bundle(smoke_build):
    b, _ = smoke_build
    with pytest.raises(ValueError):
        R4B.extend_a1(b)


# ---------------------------------------------------------------- T1-1: each bundle guard fires through the build

def _heads_variant(change):
    real = R4B._load_csd_heads

    def load():
        z = {k: np.array(v, copy=True) for k, v in real().items()}
        change(z)
        return z
    return load


def test_t1_1_heads_selection_fires_in_the_build(smoke_build):
    b, _ = smoke_build

    def other_selection(z):
        z["selection"] = z["selection"][::-1].copy()

    with mock.patch.object(R4B, "_load_csd_heads", _heads_variant(other_selection)):
        with pytest.raises(AssertionError, match="selection"):
            R4B.extend_a1(r3_view(b))


def test_t1_1_csd_finiteness_fires_in_the_build(smoke_build):
    b, _ = smoke_build

    def nan_head(z):
        z["style_csd__txt"][3, 2] = np.nan

    with mock.patch.object(R4B, "_load_csd_heads", _heads_variant(nan_head)):
        with pytest.raises(AssertionError, match="non-finite"):
            R4B.extend_a1(r3_view(b))


def test_t1_1_A0_finiteness_fires_in_the_extension(smoke_build):
    b, _ = smoke_build
    img = b.post["image"]["img"].copy()
    img[b.ctx.selection[5], 0] = np.nan
    v = r3_view(b)
    v.post = {**v.post, "image": {"img": img, "txt": b.post["image"]["txt"]}}
    with pytest.raises(AssertionError, match="non-finite"):
        R4B.extend_a1(v)


def test_t1_1_parity_fires_in_the_extension(smoke_build):
    b, _ = smoke_build
    with pytest.raises(ValueError, match="parity"):
        R4B.extend_a1(r3_view(b, parity=1 - b.parity))


def test_t1_1_context_parity_fires_in_the_extension(smoke_build):
    """The cross-fit of B'(A1) reads ctx.parity: a context whose parity is not the episode-index parity is refused
    even when the bundle's own parity field is right (round 3's validate checks only the latter)."""
    b, _ = smoke_build
    ctx2 = copy.copy(b.ctx)
    ctx2.parity = 1 - b.ctx.parity
    with pytest.raises(ValueError, match="parity"):
        R4B.extend_a1(r3_view(b, ctx=ctx2))


def test_t1_1_head_identity_fires(smoke_build):
    b, _ = smoke_build
    M = RB3.modules()
    assert RB3._assert_head_identity(M, b.affect_head) is True
    bad = copy.deepcopy(b.affect_head)
    key = sorted(bad)[0]
    bad[key] = "tampered"
    with pytest.raises(SystemExit):
        RB3._assert_head_identity(M, bad)
    with pytest.raises(SystemExit):
        R4B.extend_a1(r3_view(b, affect_head=bad))


def test_t1_1_round3_inline_finiteness_fires_in_the_build(smoke_build):
    """Round 3's own check of the A0 posteriors (inline in r3_bundle.build_bundle) fires through round 4's build
    when the affect heads carry a NaN on a selection row (the fixture's heads are reused from this process)."""
    real = RB3._affect_heads

    def nan_heads(M, ctx, scorer_train):
        post, prov, reused = real(M, ctx, scorer_train)
        img = np.array(post["img"], copy=True)
        img[ctx.selection[0], 0] = np.nan
        return {"img": img, "txt": post["txt"]}, prov, reused

    with mock.patch.object(RB3, "_affect_heads", nan_heads), contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(AssertionError, match="non-finite"):
            R4B.build_bundle(SMOKE_SEED, True)


def test_extension_guard_fires_when_a_round3_field_changes(smoke_build):
    b, _ = smoke_build
    v = r3_view(b, B={c: {d: np.array(b.B[c][d], copy=True) for d in DIRECTIONS} for c in CONDITIONS})
    real = R4B.crossfit_condition_free

    def mutating(cos, t_u, t_c, parity):
        v.B["a"]["i2t"][0, 0] = np.float32(v.B["a"]["i2t"][0, 0] + 1.0)
        return real(cos, t_u, t_c, parity)

    with mock.patch.object(R4B, "crossfit_condition_free", mutating):
        with pytest.raises(AssertionError, match="round 3"):
            R4B.extend_a1(v)


def test_smoke_extension_of_a_view_reproduces_the_build(smoke_build):
    b, _ = smoke_build
    e = R4B.extend_a1(r3_view(b))
    for c in CONDITIONS:
        assert bool(np.array_equal(e.F1[c], b.F1[c])), f"F1/{c}"
        for d in DIRECTIONS:
            assert bool(np.array_equal(e.Bp1[c][d], b.Bp1[c][d])), f"Bp1/{c}/{d}"
    assert bool(np.array_equal(e.v, b.v)), "v"


# ---------------------------------------------------------------- the cache on the real smoke bundle

def test_smoke_cache_round_trip(smoke_build):
    b, _ = smoke_build
    p, p1 = _cache_paths(f"test_r4_bundle_cache_seed{SMOKE_SEED}")
    _unlink(p, p1)
    try:
        shas = R4B.save_bundle(b, p)
        got = R4B.load_bundle_cache(p, SMOKE_SEED, True, shas)
        eq = R4B.bundles_equal(b, got)
        assert all(eq.values()), f"cache round trip differs: {[k for k, v in eq.items() if not v]}"
        assert got.readers["config"] == "A0" and got.readers_a1["config"] == "A1", "readers not reloaded"
        assert got.checks_a1 == json.loads(json.dumps(b.checks_a1)), "checks_a1"
        RB3.load_external(got)
    finally:
        _unlink(p, p1)
