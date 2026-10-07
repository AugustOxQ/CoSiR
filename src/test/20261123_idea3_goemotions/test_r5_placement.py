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
    f0 = tmp_path / "g0.npz"
    np.savez(f0, **z)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", R5.sha256_file(f0))
    X = P.ge_input(ctx, f0, st, ap)
    assert X.dtype == np.float32 and X.shape == (N, 28)
    assert np.array_equal(X[st], ap) and np.array_equal(X[rows], pr)
    assert np.isnan(np.delete(X, np.concatenate([st, rows]), 0)).all()
    with pytest.raises(P.PlacementError, match="affect_probs"):
        P.ge_input(ctx, f0, st[:-1], ap)
    with pytest.raises(P.PlacementError, match="selection"):
        P.ge_input(SimpleNamespace(groups=np.zeros(N), selection=rows[::-1].copy()), f0, st, ap)
    with pytest.raises(P.PlacementError, match="path"):
        P.ge_input(ctx, z, st, ap)                    # a loaded mapping would skip the SHA-256 check
    f = tmp_path / "g.npz"
    np.savez(f, **z)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", None)
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


def test_unlabeled_rows_refused():
    F, lab, st, rows = world()
    lab[st] = -1                                      # every draw and check row unlabeled
    with pytest.raises(P.PlacementError, match="without a label"):
        P.place(F, P.identity, lab, st, rows, 300, **KW)


def _ge_files(tmp_path, monkeypatch, rows_override=None, probs_bad=False, ap_bad=False):
    F, lab, st, rows = world()
    ctx = SimpleNamespace(groups=np.zeros(N), selection=rows)
    ap = np.random.default_rng(1).dirichlet(np.ones(28), len(st)).astype(np.float32)
    pr = np.random.default_rng(2).dirichlet(np.ones(28), len(rows)).astype(np.float32)
    if probs_bad:
        pr[3, 5] = np.nan
    if ap_bad:
        ap[3, 5] = np.inf
    f = tmp_path / "g.npz"
    np.savez(f, probs=pr, rows=rows.astype(np.int64))
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", R5.sha256_file(f))
    return ctx, f, st, ap


def test_ge_input_overlapping_rows_refused(tmp_path, monkeypatch):
    ctx, f, st, ap = _ge_files(tmp_path, monkeypatch)
    with pytest.raises(P.PlacementError, match="overlap"):
        P.ge_input(ctx, f, np.concatenate([st[:-1], ctx.selection[:1]]), ap)


@pytest.mark.parametrize("kw", [{"probs_bad": True}, {"ap_bad": True}])
def test_ge_input_nonfinite_refused(tmp_path, monkeypatch, kw):
    ctx, f, st, ap = _ge_files(tmp_path, monkeypatch, **kw)
    with pytest.raises(P.PlacementError, match="non-finite"):
        P.ge_input(ctx, f, st, ap)


def test_cap_reached_carries_both_counts():
    F, lab, st, rows = world()
    with pytest.raises(P.CapReached) as e:
        P.fit_ge_head(F, lab, st, rows, caps=(1, 2), **KW)
    assert e.value.n_iter_first == 1 and e.value.n_iter_fallback == 2 and e.value.caps == (1, 2)


# ---------------------------------------------------------------- main() with stubs (no real data, no real fit)

import json  # noqa: E402
import run_r5_placement as RUN  # noqa: E402

NSEL = R5.N_SELECTION


class Stubs:
    def __init__(self, tmp_path, monkeypatch, item3_pass=True, cap_fail=False):
        self.calls = []
        self.tmp = tmp_path
        (tmp_path / "results" / "smoke").mkdir(parents=True)
        (tmp_path / "cache").mkdir()
        monkeypatch.setattr(R5, "RESULTS", tmp_path / "results")
        monkeypatch.setattr(R5, "SMOKE", tmp_path / "results" / "smoke")
        monkeypatch.setattr(RUN, "GE_NPZ", tmp_path / "cache" / "r5_ge_posterior.npz")
        monkeypatch.setattr(R5, "assert_rule", lambda: None)
        monkeypatch.setattr(R5, "assert_modules", lambda: None)
        monkeypatch.setattr(R5, "assert_inputs", lambda names: None)
        sel = np.arange(NSEL, dtype=np.int64)
        ctx = SimpleNamespace(groups=np.zeros(R5.N_ROWS, np.int8), selection=sel)
        monkeypatch.setattr(RUN, "load_world", lambda: (ctx, np.arange(NSEL, NSEL + 10), None))
        monkeypatch.setattr(RUN, "load_affect_probs", lambda: None)
        monkeypatch.setattr(RUN, "item3", lambda *a: {"passed": item3_pass})
        monkeypatch.setattr(RUN.P, "ge_input", lambda *a, **k: self.calls.append("ge_input") or "X")

        def fit(*a, **k):
            self.calls.append("fit_ge_head")
            if cap_fail:
                raise P.CapReached("cap", 300, 3000, (300, 3000))
            post = np.random.default_rng(0).dirichlet(np.ones(R5.N_CLASSES), NSEL).astype(np.float32)
            rec = {"heldout_accuracy": 12.5, "classes": list(range(R5.N_CLASSES)), "n_iter": 5, "fallback_used": False,
                   "n_iter_fallback": None, "check_majority_share": 6.0, "uniform": 2.4}
            return {"post": post, "classes": np.arange(R5.N_CLASSES)}, rec
        monkeypatch.setattr(RUN.P, "fit_ge_head", fit)


def test_main_refuses_without_goemo_sha(tmp_path, monkeypatch):
    st = Stubs(tmp_path, monkeypatch)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", None)
    monkeypatch.setattr(RUN, "load_world", lambda: st.calls.append("load_world"))
    with pytest.raises(SystemExit, match="GOEMO_FILE_SHA"):
        RUN.main([])
    assert st.calls == [] and not list((tmp_path / "results").glob("*.json"))


def test_main_item3_fail_stops_before_ge(tmp_path, monkeypatch, capsys):
    st = Stubs(tmp_path, monkeypatch, item3_pass=False)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", "0" * 64)
    with pytest.raises(SystemExit) as e:
        RUN.main([])
    assert e.value.code == 1
    assert st.calls == []
    rec = json.loads((tmp_path / "results" / "placement.json").read_text())
    assert rec["item3"]["passed"] is False and "ge_head" not in rec
    assert not (tmp_path / "cache" / "r5_ge_posterior.npz").exists()


def test_main_pass_writes_both_once(tmp_path, monkeypatch):
    st = Stubs(tmp_path, monkeypatch)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", "0" * 64)
    RUN.main([])
    assert st.calls == ["ge_input", "fit_ge_head"]
    rec = json.loads((tmp_path / "results" / "placement.json").read_text())
    npz = tmp_path / "cache" / "r5_ge_posterior.npz"
    assert rec["item3"]["passed"] and rec["ge_head"]["heldout_accuracy"] == 12.5
    assert rec["ge_posterior_sha256"] == R5.sha256_file(npz) and rec["rule_sha256"] == R5.RULE_SHA
    with pytest.raises(SystemExit):
        RUN.main([])                                  # a second run never overwrites


def test_main_cap_failure_writes_record_and_no_posterior(tmp_path, monkeypatch):
    Stubs(tmp_path, monkeypatch, cap_fail=True)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", "0" * 64)
    with pytest.raises(SystemExit) as e:
        RUN.main([])
    assert e.value.code == 1
    rec = json.loads((tmp_path / "results" / "placement_failure.json").read_text())
    assert rec["n_iter_first"] == 300 and rec["n_iter_fallback"] == 3000
    assert not (tmp_path / "cache" / "r5_ge_posterior.npz").exists()
    assert not (tmp_path / "results" / "placement.json").exists()


@pytest.mark.parametrize("name", ["results/placement.json", "cache/r5_ge_posterior.npz"])
def test_main_refuses_existing_outputs_before_any_work(tmp_path, monkeypatch, name):
    st = Stubs(tmp_path, monkeypatch)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", "0" * 64)
    (tmp_path / name).write_bytes(b"x")
    with pytest.raises(SystemExit, match="exist"):
        RUN.main([])
    assert st.calls == []
