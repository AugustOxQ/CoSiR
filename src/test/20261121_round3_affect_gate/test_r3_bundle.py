"""Tests of the round-3 bundle builder (r3_bundle.py; rule DECISION_RULE.md §4 items 1 and 2, D7, §6.4, §10).
Run from /project/CoSiR, output redirected (rule §10: the log is read only with grep for PASS/FAIL/passed/failed,
Error and Traceback):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/20261121_round3_affect_gate/test_r3_bundle.py -q \
        -p no:cacheprovider > src/test/20261121_round3_affect_gate/results/smoke/test_r3_bundle.log 2>&1

Two kinds of test:
  synthetic  redundancy (D7) against a direct numpy Pearson, the seed guard, the per-seed cache (save, load, refusals);
  smoke      the real builder on smoke seed 9001 (built first with run_baselines.py --smoke --episodes-seed 9001):
             shapes and dtypes, B and B' condition-free, Delta^b = -Delta^a, no seed-42 file opened (open, io.open and
             np.load recorded during the build), the real A3 checkpoint and readers used (never the smoke ones),
             load_external's alignment assertions (a shuffled per_anchor file fails), the cache round trip.
Every assertion on real data asserts a plain bool with a message, so a failure prints no array or metric value.
"""
import builtins
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_bundle as RB  # noqa: E402
import r3_common as R3  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402

SMOKE_SEED = 9001
SMOKE_DIR = R3.RES / "smoke"
A0 = R3.A0


# ---------------------------------------------------------------- synthetic helpers

def _cf_scores(rng, n, k=13):
    """Condition-free {c: {d: (n, k) float32}}."""
    base = {d: rng.normal(size=(n, k)).astype(np.float32) for d in DIRECTIONS}
    return {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}


def _per_anchor(rng, n):
    """Per-anchor arrays of a condition-free scorer (gain 0 on every episode)."""
    r1 = (rng.integers(0, 5, size=n) / 4.0).astype(np.float64)
    return {"r1": r1, "gain": np.zeros(n), "other": r1.copy(), "swap": np.zeros(n), "strict": r1 * 0.0}


def _features(rng, n):
    """(n, 18) per condition with Delta^b = -Delta^a (columns 2, 8, 14), as D3 requires."""
    Fa, Fb = rng.normal(size=(n, 18)), rng.normal(size=(n, 18))
    for j in range(3):
        Fb[:, 6 * j + 2] = -Fa[:, 6 * j + 2]
    return {"a": Fa, "b": Fb}


def fake_bundle(n=10, seed=9001, smoke=True, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    return SimpleNamespace(
        seed=seed, smoke=smoke, n=n, ctx=None, cl=rng.integers(0, 4, size=n).astype(np.int64),
        parity=(np.arange(n) % 2).astype(np.int64), pair_index=np.repeat(np.arange(3), -(-n // 3))[:n].astype(np.int64),
        anchor=rng.integers(0, 1000, size=n).astype(np.int64), cos=_cf_scores(rng, n), B=_cf_scores(rng, n),
        pB=_per_anchor(rng, n), Bp=_cf_scores(rng, n), pBp=_per_anchor(rng, n), t_n1u=_cf_scores(rng, n),
        post=None, stack={d: rng.random(size=(n, 3, 13)).astype(np.float32) for d in DIRECTIONS},
        F=_features(rng, n), affect_head={"n_classes": 41, "draw_rows_sha256": "x"},
        episodes_sha256={"emotion__style": "y"}, input_sha256={"z": "w"}, checks={"synthetic": True})


def direct_redundancy(stack, B):
    """Mean over rows of np.corrcoef(s_h row, B row), rows where either side is constant left out."""
    out = {}
    for j, h in enumerate(A0):
        out[h] = {}
        for d in DIRECTIONS:
            vals = []
            for s_row, b_row in zip(stack[d][:, j].astype(np.float64), B["a"][d].astype(np.float64)):
                if np.ptp(s_row) == 0 or np.ptp(b_row) == 0:
                    continue
                vals.append(np.corrcoef(s_row, b_row)[0, 1])
            out[h][d] = float(np.mean(vals))
    return out


# ---------------------------------------------------------------- synthetic tests (fast)

def test_redundancy_equals_direct_pearson():
    rng = np.random.default_rng(3)
    n = 50
    b = fake_bundle(n=n)
    # make grouping scores partly correlated with B, with a constant s-row and a constant B-row
    for d in DIRECTIONS:
        noise = rng.normal(size=(n, 3, 13)).astype(np.float32)
        b.stack[d] = (np.asarray(b.B["a"][d])[:, None, :] * np.array([0.2, 1.0, -0.5], np.float32)[None, :, None]
                      + noise).astype(np.float32)
        b.stack[d][4, 0] = 0.25                                   # constant s_affect row: left out for affect
    for c in CONDITIONS:
        b.B[c]["i2t"][7] = 1.5                                     # constant B row: left out in i2t
    got = RB.redundancy(b)
    want = direct_redundancy(b.stack, b.B)
    assert set(got) == set(A0) and all(set(got[h]) == set(DIRECTIONS) for h in A0)
    for h in A0:
        for d in DIRECTIONS:
            assert isinstance(got[h][d], float)
            assert abs(got[h][d] - want[h][d]) < 1e-6, (h, d, got[h][d], want[h][d])


def test_redundancy_extremes():
    b = fake_bundle(n=20)
    for d in DIRECTIONS:
        b.stack[d][:, 0] = b.B["a"][d]
        b.stack[d][:, 1] = -b.B["a"][d]
        b.stack[d][:, 2] = 2.0 * b.B["a"][d] + 1.0
    got = RB.redundancy(b)
    for d in DIRECTIONS:
        assert abs(got["affect"][d] - 1.0) < 1e-6
        assert abs(got["image"][d] + 1.0) < 1e-6
        assert abs(got["caption"][d] - 1.0) < 1e-6


def test_redundancy_refuses_condition_dependent_B():
    b = fake_bundle(n=12)
    b.B["b"]["t2i"] = b.B["b"]["t2i"] + np.float32(1.0)
    with pytest.raises(ValueError):
        RB.redundancy(b)


def test_row_corr_is_the_brainstorm_definition():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [3.0, 1.0, 2.0]])
    y = np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    # row 0: +1; row 1: x constant -> left out; row 2: corr([3,1,2],[1,2,3]) = -0.5
    assert RB.row_corr(x, y) == pytest.approx((1.0 - 0.5) / 2, abs=1e-15)


def test_build_bundle_refuses_other_seeds_before_reading():
    with mock.patch.object(np, "load", side_effect=AssertionError("np.load called")):
        for seed, smoke in ((43, False), (45, False), (48, False), (42, True), (9004, True), (49, True)):
            with pytest.raises(ValueError):
                RB.build_bundle(seed, smoke)


def test_cache_round_trip_synthetic():
    b = fake_bundle(n=14)
    path = SMOKE_DIR / "test_r3_bundle_cache_synthetic.npz"
    try:
        sha = RB.save_bundle(b, path)
        assert sha == R3.sha_file(path)
        got = RB.load_bundle_cache(path, seed=b.seed, smoke=True, sha256=sha, readers=False)
        assert_bundles_equal(b, got)
        assert got.ctx is None and got.post is None and got.from_cache is True
        assert got.affect_head == b.affect_head and got.episodes_sha256 == b.episodes_sha256
    finally:
        path.unlink(missing_ok=True)


def test_cache_refusals():
    b = fake_bundle(n=9)
    path = SMOKE_DIR / "test_r3_bundle_cache_refusals.npz"
    try:
        sha = RB.save_bundle(b, path)
        with pytest.raises(SystemExit):
            RB.load_bundle_cache(path, seed=b.seed + 1, readers=False)          # another seed
        with pytest.raises(SystemExit):
            RB.load_bundle_cache(path, smoke=False, readers=False)              # another smoke flag
        with pytest.raises(SystemExit):
            RB.load_bundle_cache(path, sha256="0" * 64, readers=False)          # file changed
        with mock.patch.object(R3, "RULE_SHA", "f" * 64):                       # written under another rule
            with pytest.raises(SystemExit):
                RB.load_bundle_cache(path, readers=False)
        # a non-smoke cache is never overwritten (rule §10)
        b2 = fake_bundle(n=9, seed=49, smoke=False)
        with pytest.raises(SystemExit):
            RB.save_bundle(b2, path)
        assert R3.sha_file(path) == sha
    finally:
        path.unlink(missing_ok=True)


def test_cache_refuses_missing_or_bad_arrays():
    bad = SMOKE_DIR / "test_r3_bundle_cache_bad.npz"
    bad.unlink(missing_ok=True)                                                 # a stale file from an earlier run
    try:
        _cache_refusals_of_bad_arrays(bad)
    finally:
        bad.unlink(missing_ok=True)


def _cache_refusals_of_bad_arrays(bad):
    b = fake_bundle(n=9)
    b.B["b"]["i2t"] = b.B["b"]["i2t"] + np.float32(0.5)                       # B no longer condition-free
    with pytest.raises(ValueError):
        RB.save_bundle(b, bad)
    assert not bad.exists()
    b = fake_bundle(n=9)
    b.stack = {d: b.stack[d][:, :2] for d in DIRECTIONS}                       # wrong grouping count
    with pytest.raises(ValueError):
        RB.save_bundle(b, bad)
    b = fake_bundle(n=9)
    b.pBp["gain"] = b.pBp["gain"].copy()
    b.pBp["gain"][3] = 0.25                                                     # B' with a condition gain
    with pytest.raises(ValueError):
        RB.save_bundle(b, bad)
    b = fake_bundle(n=9)
    b.F["b"][2, 8] += 1.0                                                       # Delta^b != -Delta^a (image)
    with pytest.raises(ValueError):
        RB.save_bundle(b, bad)
    b = fake_bundle(n=9)
    b.F["a"] = b.F["a"].astype(np.float32)                                      # features must be float64
    with pytest.raises(ValueError):
        RB.save_bundle(b, bad)
    assert not bad.exists()


def assert_bundles_equal(a, b):
    for k in ("seed", "smoke", "n"):
        assert getattr(a, k) == getattr(b, k), k
    for k in ("cl", "parity", "pair_index", "anchor"):
        ok = bool(np.array_equal(getattr(a, k), getattr(b, k)) and getattr(a, k).dtype == getattr(b, k).dtype)
        assert ok, f"{k} differs after the cache round trip"
    for k in ("cos", "B", "Bp", "t_n1u"):
        for c in CONDITIONS:
            for d in DIRECTIONS:
                x, y = getattr(a, k)[c][d], getattr(b, k)[c][d]
                ok = bool(np.array_equal(x, y) and np.asarray(x).dtype == np.asarray(y).dtype)
                assert ok, f"{k}/{c}/{d} differs after the cache round trip"
    for k in ("pB", "pBp"):
        for m in METRICS:
            x, y = getattr(a, k)[m], getattr(b, k)[m]
            ok = bool(np.array_equal(x, y) and np.asarray(x).dtype == np.asarray(y).dtype)
            assert ok, f"{k}/{m} differs after the cache round trip"
    for d in DIRECTIONS:
        ok = bool(np.array_equal(a.stack[d], b.stack[d]) and a.stack[d].dtype == b.stack[d].dtype)
        assert ok, f"stack/{d} differs after the cache round trip"
    for c in CONDITIONS:
        ok = bool(np.array_equal(a.F[c], b.F[c]) and a.F[c].dtype == b.F[c].dtype)
        assert ok, f"F/{c} differs after the cache round trip"


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
    """Records every path opened through builtins.open / io.open (with the calling function's name) and every
    np.load path while active."""

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
    with Recorder() as rec:
        bundle = RB.build_bundle(SMOKE_SEED, True)
    return bundle, rec


def test_smoke_shapes_and_dtypes(smoke_build):
    b, _ = smoke_build
    n = b.n
    ok = (b.seed == SMOKE_SEED and b.smoke is True and n == 3 * 64 and b.ctx is not None and b.ctx.n == n)
    assert ok, "seed, smoke flag or episode count"
    for k in ("cl", "parity", "pair_index", "anchor"):
        assert getattr(b, k).shape == (n,), f"{k} shape"
    assert bool(np.array_equal(b.parity, np.arange(n) % 2)), "parity is not the episode-index parity"
    assert bool(np.array_equal(b.pair_index, np.repeat(np.arange(3), 64))), "pair index"
    assert bool(np.array_equal(b.cl, b.ctx.groups[b.ctx.pooled.anchor])), "anchor paintings"
    for k in ("cos", "B", "Bp", "t_n1u"):
        for c in CONDITIONS:
            for d in DIRECTIONS:
                x = getattr(b, k)[c][d]
                assert x.shape == (n, 13) and x.dtype == np.float32, f"{k}/{c}/{d} shape or dtype"
                assert bool(np.isfinite(x).all()), f"{k}/{c}/{d} not finite"
    for k in ("pB", "pBp"):
        assert set(getattr(b, k)) == set(METRICS), f"{k} metrics"
        for m in METRICS:
            assert getattr(b, k)[m].shape == (n,), f"{k}/{m} shape"
    assert tuple(b.post) == A0, "post must hold A0's groupings in A0 order"
    n_rows = len(b.ctx.groups)
    for h, k_groups in zip(A0, (41, 64, 64)):
        for side in ("img", "txt"):
            x = b.post[h][side]
            assert x.shape == (n_rows, k_groups) and x.dtype == np.float32, f"post {h}/{side} shape or dtype"
            assert bool(np.isfinite(x[b.ctx.selection]).all()), f"post {h}/{side} not finite on selection rows"
    for d in DIRECTIONS:
        assert b.stack[d].shape == (n, 3, 13) and b.stack[d].dtype == np.float32, f"stack/{d} shape or dtype"
    for c in CONDITIONS:
        assert b.F[c].shape == (n, 18) and b.F[c].dtype == np.float64, f"F/{c} shape or dtype"
        assert bool(np.isfinite(b.F[c]).all()), f"F/{c} not finite"
    assert b.affect_head["n_classes"] == 41, "affect head classes"
    assert all(v is True for v in b.checks.values()), "a build check is not True"
    assert b.readers["config"] == "A0" and tuple(b.readers["groupings"]) == A0, "readers"


def test_smoke_B_and_Bprime_condition_free(smoke_build):
    b, _ = smoke_build
    for k, pk in (("B", "pB"), ("Bp", "pBp"), ("cos", None)):
        for d in DIRECTIONS:
            assert bool(np.array_equal(getattr(b, k)["a"][d], getattr(b, k)["b"][d])), f"{k}/{d} not condition-free"
        if pk:
            assert bool((getattr(b, pk)["gain"] == 0).all()), f"{pk}: condition gain is not 0 on every episode"


def test_smoke_delta_b_is_minus_delta_a(smoke_build):
    b, _ = smoke_build
    for j in range(3):
        ok = bool(np.array_equal(b.F["b"][:, 6 * j + 2], -b.F["a"][:, 6 * j + 2]))
        assert ok, f"Delta^b != -Delta^a for grouping {A0[j]}"
    assert b.checks["delta_b_equals_minus_delta_a"] is True


def test_smoke_stack_is_the_grouping_scores(smoke_build):
    b, _ = smoke_build
    st = R3.C.grouping_stack(b.post, b.ctx.pooled, A0)
    for d in DIRECTIONS:
        assert bool(np.array_equal(b.stack[d], st[d])), f"stack/{d} differs from common.grouping_stack"
    # D4: s_h(q, k) = p_h(q) . p_h(k), query from its own modality's head
    ep = b.ctx.pooled
    i = 5
    for j, h in enumerate(A0):
        want = b.post[h]["img"][ep.anchor[i]] @ b.post[h]["txt"][ep.candidates[i]].T
        assert bool(np.allclose(b.stack["i2t"][i, j], want, rtol=1e-6, atol=1e-7)), f"stack i2t {h}"


def test_smoke_no_seed42_file_opened(smoke_build):
    _, rec = smoke_build
    bad_loads = [p for p in rec.loads if is_seed42_file(p)]
    bad_opens = [(p, f) for p, f in rec.opens if is_seed42_file(p) and f != "sha_file"]
    assert not bad_loads, f"seed-42 arrays loaded: {bad_loads}"
    assert not bad_opens, f"seed-42 files opened (not for hashing): {bad_opens}"
    # the recorder saw the build: the smoke episodes were loaded
    eps = str((R3.AB / "results" / "smoke" / f"episodes_seed{SMOKE_SEED}.npz").resolve())
    assert eps in rec.loads, "the smoke episodes were not loaded through np.load"


def test_smoke_uses_real_checkpoint_heads_and_readers(smoke_build):
    _, rec = smoke_build
    paths = {p for p, _ in rec.opens} | set(rec.loads)
    gonogo_smoke = str((R3.TEST / "20261101_aspect_factor_gonogo" / "checkpoints" / "smoke").resolve())
    r1_smoke = str((R3.R1 / "results" / "smoke").resolve())
    qc_smoke = str((R3.TEST / "20261108_new_method_quick_checks" / "results" / "smoke").resolve())
    bad = [p for p in paths if p.startswith((gonogo_smoke, r1_smoke, qc_smoke))]
    assert not bad, f"smoke checkpoint, heads or readers opened: {bad}"
    a3 = str((R3.TEST / "20261101_aspect_factor_gonogo" / "checkpoints" / "A3_seed42.pt").resolve())
    assert a3 in paths, "the real A3 checkpoint was not read"
    rdr = str((R3.R1 / "results" / "rb_reader_A0.pkl").resolve())
    assert rdr in paths, "the real A0 readers were not read"
    post = str((R3.TEST / "20261108_new_method_quick_checks" / "results" / "n6_posteriors.npz").resolve())
    assert post in rec.loads, "the stored n6 posteriors were not loaded"


def test_smoke_load_external_passes(smoke_build):
    b, _ = smoke_build
    ext = RB.load_external(b)
    assert set(ext) == {"cosine", "rca"}
    for k in ("cosine", "rca"):
        assert set(ext[k]) == set(METRICS), k
        for m in METRICS:
            assert ext[k][m].shape == (b.n,) and ext[k][m].dtype == np.float64, f"{k}/{m} shape or dtype"


def _write_variant(src, dst, change):
    z = dict(np.load(src))
    change(z)
    np.savez(dst, **z)


def test_smoke_load_external_fails_on_shuffled_file(smoke_build):
    b, _ = smoke_build
    src = R3.AB / "results" / "smoke" / f"per_anchor_seed{SMOKE_SEED}.npz"
    dst = SMOKE_DIR / "test_r3_bundle_shuffled_per_anchor.npz"
    perm = np.random.default_rng(0).permutation(b.n)

    def shuffle_all(z):                                   # every per-anchor array in another episode order
        for k in list(z):
            z[k] = z[k][perm]

    def shuffle_cosine(z):                                # only the cosine arrays
        for m in METRICS:
            z[f"cosine__{m}"] = z[f"cosine__{m}"][perm]

    def other_pairs(z):                                   # pair index changed
        z["pair_index"] = z["pair_index"][::-1].copy()

    def other_paintings(z):                               # anchor paintings changed
        z["anchor_group"] = z["anchor_group"][perm]

    try:
        for change in (shuffle_all, shuffle_cosine, other_pairs, other_paintings):
            _write_variant(src, dst, change)
            with pytest.raises(AssertionError):
                RB.load_external(b, path=dst)
        _write_variant(src, dst, lambda z: None)          # an unchanged copy passes
        RB.load_external(b, path=dst)
    finally:
        dst.unlink(missing_ok=True)


def test_smoke_cache_round_trip(smoke_build):
    b, _ = smoke_build
    path = SMOKE_DIR / f"test_r3_bundle_cache_seed{SMOKE_SEED}.npz"
    try:
        sha = RB.save_bundle(b, path)
        got = RB.load_bundle_cache(path, seed=SMOKE_SEED, smoke=True, sha256=sha)
        assert_bundles_equal(b, got)
        assert got.readers["config"] == "A0", "readers not reloaded"
        assert got.affect_head == json.loads(json.dumps(b.affect_head))
        assert got.episodes_sha256 == b.episodes_sha256
        # the cached bundle still passes load_external's alignment assertions
        RB.load_external(got)
    finally:
        path.unlink(missing_ok=True)


def test_smoke_affect_head_cache_changes_nothing(smoke_build):
    """A second build in the same process reuses the affect heads (cache) and gives identical arrays."""
    b, _ = smoke_build
    b2 = RB.build_bundle(SMOKE_SEED, True)
    assert b2.post["affect"]["img"] is b.post["affect"]["img"], "affect heads were refit"
    assert b2.affect_heads_reused is True and all(v is True for v in b2.checks.values())
    assert_bundles_equal(b, b2)
    for h in A0:
        for side in ("img", "txt"):
            ok = bool(np.array_equal(b.post[h][side], b2.post[h][side], equal_nan=True))
            assert ok, f"post {h}/{side} differs between builds"
