"""Tests of r5_placement.py (rule DECISION_RULE.md, section 5 item 3 and D4; section 10 list A item 3). Synthetic only: no
real GoEmotions file exists, no GE head is fitted on real data.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_placement.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_placement as P  # noqa: E402

K, N, D = 4, 600, 6
KW = dict(n_draw=150, n_check=60, n_classes=K)
pytestmark = pytest.mark.filterwarnings("ignore")


def world(seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    scorer_train = np.sort(rng.choice(N, 400, replace=False))
    rows = np.setdiff1d(np.arange(N), scorer_train)[:100]
    lab = np.full(N, -1, np.int64)
    lab[scorer_train] = rng.integers(0, K, len(scorer_train))
    F = np.full((N, D), np.nan, np.float32)
    cen = rng.normal(size=(K, D))
    for r in np.concatenate([scorer_train, rows]):
        c = lab[r] if lab[r] >= 0 else rng.integers(0, K)
        F[r] = cen[c] + noise * rng.normal(size=D)
    return F, lab, scorer_train, rows


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_place_equals_direct_fit():
    F, lab, st, rows = world()
    res = P.place(F, unit, lab, st, rows, 300, **KW)
    draw = np.random.default_rng(0).choice(st, 150, replace=False)
    rest = np.setdiff1d(st, draw)
    check = np.random.default_rng(1).choice(rest, 60, replace=False)
    clf = LogisticRegression(C=1.0, max_iter=300).fit(unit(F[draw]), lab[draw])
    full = np.full((N, K), np.nan, np.float32)
    full[rows] = clf.predict_proba(unit(F[rows]))
    assert np.array_equal(res["post"], full, equal_nan=True)
    assert res["acc"] == 100 * float(clf.score(unit(F[check]), lab[check]))
    assert np.array_equal(res["draw"], draw) and np.array_equal(res["check"], check)
    assert np.array_equal(res["classes"], np.arange(K)) and res["n_iter"] == int(clf.n_iter_.max())


def test_place_nan_outside_rows_and_float32():
    F, lab, st, rows = world()
    post = P.place(F, P.identity, lab, st, rows, 300, **KW)["post"]
    out = np.ones(N, bool)
    out[rows] = False
    assert post.dtype == np.float32 and post.shape == (N, K)
    assert np.isnan(post[out]).all() and np.isfinite(post[rows]).all()
    assert np.allclose(post[rows].sum(1), 1, atol=1e-5)


def test_classes_not_0_to_K_refused():
    F, lab, st, rows = world()
    lab[st[lab[st] == K - 1]] = 0                     # class K-1 vanishes
    with pytest.raises(P.PlacementError, match="classes_"):
        P.place(F, P.identity, lab, st, rows, 300, **KW)
    F, lab, st, rows = world()
    lab[st] = lab[st] + 1                             # classes 1..K
    with pytest.raises(P.PlacementError, match="classes_"):
        P.place(F, P.identity, lab, st, rows, 300, **KW)


def test_fallback_refits_once_and_records_both_counts():
    F, lab, st, rows = world()
    res, rec = P.fit_ge_head(F, lab, st, rows, caps=(1, 500), **KW)
    assert rec["fallback_used"] and rec["n_iter_first"] == 1
    assert 1 < rec["n_iter_fallback"] < 500 and rec["n_iter"] == rec["n_iter_fallback"]
    direct = P.place(F, P.identity, lab, st, rows, 500, **KW)
    assert np.array_equal(res["post"], direct["post"], equal_nan=True)     # the refit is the head, all else equal
    assert set(rec) >= {"heldout_accuracy", "classes", "n_iter", "fallback_used", "n_iter_fallback",
                        "check_majority_share", "uniform"}


def test_no_fallback_when_converged():
    F, lab, st, rows = world()
    _, rec = P.fit_ge_head(F, lab, st, rows, **KW)
    assert not rec["fallback_used"] and rec["n_iter_fallback"] is None and rec["n_iter"] < 300


def test_second_cap_stops():
    F, lab, st, rows = world()
    with pytest.raises(P.PlacementError, match="fallback cap"):
        P.fit_ge_head(F, lab, st, rows, caps=(1, 2), **KW)


def test_misaligned_mapping_refused():
    F, lab, st, rows = world()
    F2 = F.copy()
    F2[st[:5]] = np.nan                               # features missing on some scorer-train rows
    # draw covers some of them or the check does; with 150 of 400 rows a few missing rows are hit with certainty
    F2[st] = np.nan
    with pytest.raises(P.PlacementError, match="non-finite"):
        P.place(F2, P.identity, lab, st, rows, 300, **KW)
    with pytest.raises(P.PlacementError, match="not a scorer-train row"):
        P.draw_positions(st, np.array([rows[0]]))
    pos = P.draw_positions(st[::-1].copy(), st[10:20])  # an unsorted scorer_train still maps
    assert np.array_equal(st[::-1][pos], st[10:20])


def test_ge_input_mapping_and_refusals(tmp_path, monkeypatch):
    F, lab, st, rows = world()
    ctx = SimpleNamespace(groups=np.zeros(N), selection=rows)
    ap = np.random.default_rng(1).dirichlet(np.ones(28), len(st)).astype(np.float32)
    pr = np.random.default_rng(2).dirichlet(np.ones(28), len(rows)).astype(np.float32)
    z = {"probs": pr, "rows": rows.astype(np.int64)}
    X = P.ge_input(ctx, z, st, ap)
    assert X.dtype == np.float32 and X.shape == (N, 28)
    assert np.array_equal(X[st], ap) and np.array_equal(X[rows], pr)
    assert np.isnan(np.delete(X, np.concatenate([st, rows]), 0)).all()
    with pytest.raises(P.PlacementError, match="affect_probs"):
        P.ge_input(ctx, z, st[:-1], ap)
    with pytest.raises(P.PlacementError, match="selection"):
        P.ge_input(SimpleNamespace(groups=np.zeros(N), selection=rows[::-1].copy()), z, st, ap)
    f = tmp_path / "g.npz"
    np.savez(f, **z)
    with pytest.raises(P.PlacementError, match="GOEMO_FILE_SHA is not set"):
        P.ge_input(ctx, f, st, ap)                    # refused while the constant is None
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", "0" * 64)
    with pytest.raises(P.PlacementError, match="differs"):
        P.ge_input(ctx, f, st, ap)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", R5.sha256_file(f))
    assert np.array_equal(P.ge_input(ctx, f, st, ap)[rows], pr)


def test_write_ge_posterior_once_and_checks(tmp_path):
    n_sel, n_rows, k = 20, 50, 3
    rows = np.sort(np.random.default_rng(0).choice(n_rows, n_sel, replace=False)).astype(np.int64)
    post = np.random.default_rng(1).dirichlet(np.ones(k), n_sel).astype(np.float32)
    kw = dict(n_rows=n_rows, n_sel=n_sel, n_classes=k)
    path = tmp_path / "ge.npz"
    sha = P.write_ge_posterior(path, post, rows, np.arange(k), **kw)
    assert sha == R5.sha256_file(path)
    with np.load(path) as z:
        assert np.array_equal(z["post_sel"], post) and np.array_equal(z["rows"], rows)
    with pytest.raises(SystemExit):
        P.write_ge_posterior(path, post, rows, np.arange(k), **kw)         # never overwritten
    for bad in (post.astype(np.float64), post * 2, np.where(np.arange(n_sel)[:, None] == 0, np.nan, post)):
        with pytest.raises(P.PlacementError):
            P.write_ge_posterior(tmp_path / "b.npz", bad, rows, np.arange(k), **kw)
    with pytest.raises(P.PlacementError):
        P.write_ge_posterior(tmp_path / "b.npz", post, rows[::-1].copy(), np.arange(k), **kw)
    with pytest.raises(P.PlacementError):
        P.write_ge_posterior(tmp_path / "b.npz", post, rows, np.arange(1, k + 1), **kw)
