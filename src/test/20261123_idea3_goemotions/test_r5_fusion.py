"""Tests of r5_fusion.py (rule DECISION_RULE.md of this folder: D6, D7, D11, §5 items 4 and 5, §6.3, §6.4, §10 list A
item 5). Run from this folder, output redirected:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_fusion.py \
        > results/smoke/test_r5_fusion.log 2>&1

  synthetic  test_r5_bundle's synthetic round-4 bundles (real round-1/round-3 code paths for stack, F and B'(A0)), with
             half-readers fitted on the bundle's own features, and synthetic 'ge' placements after a synthetic release:
             G-T reads P, m and pi from F and T from stack_G (a swap of either fires D7's check); G-TF reads both from
             F_G; tau' = rc_core.thresholds of G-TF's margins on seed 42 only, read from the record on every other seed
             and never recomputed; G-T's gates equal AFF's; the counterpart is built from the candidate's own term and
             gates (passing AFF's fires); exact pi' ties (M28); the rho_ctrl criterion by hand (M06); the guard.
  smoke      smoke seed 9001 with the bundle's own CLIP placement: G-T's and G-TF's reader outputs, gates and families
             equal AFF's exactly (rule §5 item 4's path; tau' passed in, as on every seed but 42).
Assertions on real data assert plain bools with a message, so a failure prints no array or metric value.
"""
import contextlib
import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_guard as R5G  # noqa: E402
import r5_bundle as R5B  # noqa: E402
import r5_fusion as R5F  # noqa: E402
from test_r5_bundle import make_ge, release_guard, synth_bundle  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402
from src.eval.aspect_nested import _zdict  # noqa: E402

R3, RB3, RB4, RF3, C, rc_core = R5.R3, R5.RB3, R5.RB4, R5.RF3, R5.C, R5.rc_core
F2 = RF3.F                                     # round 2's fusion pieces
A0 = R3.A0
TAUS = tuple(R3.TAUS)
SMOKE_SEED = 9001
DECIMAL = re.compile(r"\d\.\d")
GIVEN = (0.0, 0.02, 0.05, 0.1)                 # a tau' "read from the record" on a seed other than 42 (synthetic)


@pytest.fixture(autouse=True)
def isolate():
    saved = dict(RB3._HEADS)
    R5G._reset_for_tests()
    yield
    RB3._HEADS.clear()
    RB3._HEADS.update(saved)
    R5G._reset_for_tests()


def _ge(tmp_path, monkeypatch, seed=1, **kw):
    b = synth_bundle(seed=seed, **kw)
    pl = make_ge(tmp_path, monkeypatch, b, seed=100 + seed)
    release_guard(tmp_path)
    return b, pl, R5B.extend(b, pl)


def _aff(b):
    """AFF as round 3 defines it: the readers on F (CLIP placement) with the bundle's stack, AFF's gates at tau."""
    r = RF3.reader(SimpleNamespace(F=b.F, stack=b.stack), readers=b.readers)
    return r, RF3.gates_aff(r["m"], r["pick"], TAUS)


def _same_gates(g, h):
    return len(g) == len(h) and all(np.asarray(g[t][c]).dtype == np.asarray(h[t][c]).dtype
                                    and np.array_equal(g[t][c], h[t][c]) for t in range(len(g)) for c in CONDITIONS)


def _same_T(S, T):
    return all(np.asarray(S[c][d]).dtype == np.asarray(T[c][d]).dtype and np.array_equal(S[c][d], T[c][d])
               for c in CONDITIONS for d in DIRECTIONS)


# ---------------------------------------------------------------- tau'

def test_tau_prime_is_rc_core_thresholds_on_seed42_counts(monkeypatch):
    assert R5F.N_MARGINS_42 == 24_576 and R5F.TAUS == TAUS
    rng = np.random.default_rng(0)
    m = {"a": rng.random(12_288), "b": rng.random(12_288) * 0.5}
    calls = []
    real = rc_core.thresholds

    def spy(margins):
        calls.append(list(margins))
        return real(margins)

    monkeypatch.setattr(rc_core, "thresholds", spy)
    t = R5F.tau_prime(m)
    assert calls == [["a", "b"]]                                             # condition a first
    want = np.percentile(np.concatenate([m["a"], m["b"]]), [0, 25, 50, 75])
    assert type(t) is tuple and len(t) == 4 and all(type(x) is float for x in t)
    assert t == tuple(float(x) for x in want) and t[0] == min(m["a"].min(), m["b"].min())
    for bad in ({"a": m["a"][:-1], "b": m["b"]}, {"a": m["a"], "b": m["b"], "c": m["a"]}, {"a": m["a"]},
                {"a": m["a"].astype(np.float32), "b": m["b"]}):
        with pytest.raises(ValueError):
            R5F.tau_prime(bad)


@pytest.mark.parametrize("bad", [(0.0, 0.1, 0.2), (0.0, 0.3, 0.2, 0.4), (0.0, np.nan, 0.2, 0.3), "0.1",
                                 (0.0, 0.1, 0.2, 0.3, 0.4), None])
def test_check_taus_refuses_malformed_records(bad):
    with pytest.raises(ValueError):
        R5F.check_taus(bad)
    assert R5F.check_taus(list(GIVEN)) == GIVEN


# ---------------------------------------------------------------- D6: the candidates' inputs

def test_g_t_reads_p_m_pi_from_F_and_T_from_stack_G(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=1)
    cand = R5F.candidate("G-T", b, ext)
    assert set(cand) == {"P", "T", "m", "pick", "gates", "taus"}
    aff, g_aff = _aff(b)
    for k in ("P", "m", "pick"):
        for c in CONDITIONS:
            np.testing.assert_array_equal(cand[k][c], aff[k][c])                 # AFF's reader outputs (F)
    assert _same_T(cand["T"], C.expected_term(ext.stack, aff["P"]))            # the term on stack_G
    assert not _same_T(cand["T"], aff["T"])                                     # not AFF's term on the CLIP stack
    assert cand["taus"] == TAUS and _same_gates(cand["gates"], g_aff)          # AFF's tau and gates exactly
    fp = RF3.reader(SimpleNamespace(F=ext.F, stack=ext.stack), readers=b.readers)
    assert any(not np.array_equal(fp["P"][c], aff["P"][c]) for c in CONDITIONS)   # F_G would give other P


def test_g_tf_reads_p_and_T_from_F_G_and_stack_G(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=2)
    cand = R5F.candidate("G-TF", b, ext, tau_prime=GIVEN)
    r = RF3.reader(SimpleNamespace(F=ext.F, stack=ext.stack), readers=b.readers)
    for k in ("P", "m", "pick"):
        for c in CONDITIONS:
            np.testing.assert_array_equal(cand[k][c], r[k][c])
    assert _same_T(cand["T"], C.expected_term(ext.stack, r["P"]))
    assert cand["taus"] == GIVEN and _same_gates(cand["gates"], RF3.gates_aff(r["m"], r["pick"], GIVEN))
    aff, _ = _aff(b)
    assert any(not np.array_equal(cand["P"][c], aff["P"][c]) for c in CONDITIONS)


def test_gates_are_float32_0_1_at_four_tau_indices(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=3)
    for name, kw in (("G-T", {}), ("G-TF", {"tau_prime": GIVEN})):
        g = R5F.candidate(name, b, ext, **kw)["gates"]
        assert len(g) == 4
        for t in range(4):
            for c in CONDITIONS:
                assert g[t][c].dtype == np.float32 and g[t][c].shape == (b.n,)
                assert set(np.unique(g[t][c])) <= {0.0, 1.0}
        assert any(np.any(g[0][c] == 1) for c in CONDITIONS) and any(np.any(g[3][c] == 0) for c in CONDITIONS)


@pytest.mark.parametrize("name, swap", [("G-T", ("bundle", "bundle")), ("G-T", ("ext", "ext")),
                                        ("G-TF", ("bundle", "ext")), ("G-TF", ("ext", "bundle"))])
def test_a_swap_of_the_reader_input_or_the_term_stack_fires_d7(tmp_path, monkeypatch, name, swap):
    """G-T reading T from F's stack, or P from F_G (and G-TF the reverse) fires run_candidate's D7 check."""
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=4)
    kw = {"tau_prime": GIVEN} if name == "G-TF" else {}
    good = R5F.candidate(name, b, ext, **kw)
    R5F.run_candidate(name, b, ext, good, **kw)                                # the correct wiring passes
    monkeypatch.setitem(R5F.READER_INPUTS, name, swap)
    bad = R5F.candidate(name, b, ext, **kw)
    with pytest.raises(AssertionError, match="D7"):
        R5F.run_candidate(name, b, ext, bad, **kw)


def test_hand_made_wrong_candidates_fire_d7(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=5)
    gt = R5F.candidate("G-T", b, ext)
    gtf = R5F.candidate("G-TF", b, ext, tau_prime=GIVEN)
    aff, g_aff = _aff(b)
    wrong = {"G-T with AFF's term on the CLIP stack": ("G-T", dict(gt, T=aff["T"])),
             "G-T labelled from G-TF's reader": ("G-T", gtf),
             "G-TF labelled from G-T's reader": ("G-TF", gt),
             "G-TF with AFF's gates": ("G-TF", dict(gtf, gates=g_aff)),
             "G-TF with AFF's tau": ("G-TF", dict(gtf, taus=TAUS)),
             "G-T with one gate flipped": ("G-T", dict(gt, gates=[{c: (1 - g[c]).astype(np.float32) if t == 1 else g[c]
                                                                    for c in CONDITIONS}
                                                                   for t, g in enumerate(gt["gates"])])),
             "G-T with float64 gates": ("G-T", dict(gt, gates=[{c: g[c].astype(np.float64) for c in CONDITIONS}
                                                               for g in gt["gates"]]))}
    assert not _same_gates(gtf["gates"], g_aff)                                 # precondition: they differ
    for what, (name, cand) in wrong.items():
        kw = {"tau_prime": GIVEN} if name == "G-TF" else {}
        with pytest.raises(AssertionError, match="D7"):
            R5F.run_candidate(name, b, ext, cand, **kw)
    with pytest.raises(ValueError):
        R5F.run_candidate("G-T", b, ext, dict(gt, extra=1))
    with pytest.raises(ValueError):
        R5F.run_candidate("G-T", b, ext, {k: gt[k] for k in ("P", "T", "m", "pick", "gates")})
    with pytest.raises(ValueError):
        R5F.candidate("G-X", b, ext)


# ---------------------------------------------------------------- tau' on seed 42 and elsewhere

def test_tau_prime_on_seed_42_is_computed_from_g_tf_margins(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=6, bundle_seed=42, smoke=False)
    monkeypatch.setattr(R5F, "N_MARGINS_42", 2 * b.n)                          # a small synthetic "seed 42"
    cand = R5F.candidate("G-TF", b, ext)
    r = RF3.reader(SimpleNamespace(F=ext.F, stack=ext.stack), readers=b.readers)
    assert cand["taus"] == tuple(rc_core.thresholds(r["m"])[0])
    assert cand["taus"][0] == min(float(r["m"][c].min()) for c in CONDITIONS)  # tau'_0 is the smallest margin
    for c in CONDITIONS:                                                       # so the tau'_0 gate opens every affect pick
        np.testing.assert_array_equal(cand["gates"][0][c], (r["pick"][c] == 0).astype(np.float32))
    with pytest.raises(ValueError, match="seed 42"):
        R5F.candidate("G-TF", b, ext, tau_prime=cand["taus"])                  # never passed in on seed 42
    fam = R5F.run_candidate("G-TF", b, ext, cand)
    assert fam["taus"] == cand["taus"]
    # T3a-3: a mutated tau' (with gates consistent with it) fires the recomputation on seed 42
    t2 = (cand["taus"][0], cand["taus"][1] + 1e-9, cand["taus"][2], cand["taus"][3])
    mut = dict(cand, taus=t2, gates=RF3.gates_aff(cand["m"], cand["pick"], t2))
    with pytest.raises(AssertionError, match="D7"):
        R5F.run_candidate("G-TF", b, ext, mut)
    with pytest.raises(ValueError):
        R5F.run_candidate("G-TF", b, ext, cand, tau_prime=cand["taus"])


def test_tau_prime_needs_seed42s_count(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=7, bundle_seed=42, smoke=False)
    with pytest.raises(ValueError, match="24,576"):
        R5F.candidate("G-TF", b, ext)                                         # 48 margins on a synthetic seed 42


@pytest.mark.parametrize("bundle_seed, smoke", [(SMOKE_SEED, True), (52, False)])
def test_tau_prime_is_never_recomputed_on_another_seed(tmp_path, monkeypatch, bundle_seed, smoke):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=8, bundle_seed=bundle_seed, smoke=smoke)
    with pytest.raises(ValueError, match="never recomputed"):
        R5F.candidate("G-TF", b, ext)

    def boom(*a, **k):
        raise AssertionError("tau' recomputed")

    monkeypatch.setattr(rc_core, "thresholds", boom)
    monkeypatch.setattr(R5F, "tau_prime", boom)
    cand = R5F.candidate("G-TF", b, ext, tau_prime=GIVEN)
    assert cand["taus"] == GIVEN
    fam = R5F.run_candidate("G-TF", b, ext, cand, tau_prime=GIVEN)
    assert fam["taus"] == GIVEN
    with pytest.raises(ValueError, match="never recomputed"):
        R5F.run_candidate("G-TF", b, ext, cand)                               # the record's tau' is required
    other = (0.0, 0.02, 0.05, 0.11)
    with pytest.raises(AssertionError, match="D7"):
        R5F.run_candidate("G-TF", b, ext, cand, tau_prime=other)              # cand's tau' differs from the record


def test_g_t_takes_no_tau_prime(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=9)
    with pytest.raises(ValueError, match="G-T"):
        R5F.candidate("G-T", b, ext, tau_prime=TAUS)
    cand = R5F.candidate("G-T", b, ext)
    with pytest.raises(ValueError, match="G-T"):
        R5F.run_candidate("G-T", b, ext, cand, tau_prime=TAUS)


# ---------------------------------------------------------------- D7: family, counterpart, stored gates

def test_counterpart_is_built_from_the_candidates_own_term_and_gates(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=10)
    cand = R5F.candidate("G-TF", b, ext, tau_prime=GIVEN)
    aff, g_aff = _aff(b)
    assert not _same_gates(cand["gates"], g_aff)                                # precondition
    seen = []
    real = RF3.run_family

    def spy(bundle, T, gates, fused_only=False):
        seen.append((bundle, T, gates, fused_only))
        return real(bundle, T, gates, fused_only=fused_only)

    monkeypatch.setattr(RF3, "run_family", spy)
    fam = R5F.run_candidate("G-TF", b, ext, cand, tau_prime=GIVEN)
    monkeypatch.undo()
    assert len(seen) == 1 and seen[0][0] is b and seen[0][1] is cand["T"] and seen[0][2] is cand["gates"]
    assert seen[0][3] is False                                                  # the counterpart is cross-fitted
    own = RF3.run_family(b, cand["T"], cand["gates"])
    with_aff = RF3.run_family(b, cand["T"], g_aff)
    for m in METRICS:
        np.testing.assert_array_equal(fam["cf"][m], own["cf"][m])
        np.testing.assert_array_equal(fam["fused"][m], own["fused"][m])
    assert fam["cpick"] == own["cpick"] and fam["fpick"] == own["fpick"] and fam["sigma"] == own["sigma"]
    assert (fam["cpick"] != with_aff["cpick"]
            or any(not np.array_equal(fam["cf"][m], with_aff["cf"][m]) for m in METRICS))
    zB, gated, G = RF3._terms(b, cand["T"], cand["gates"])                     # G_cf from the candidate's own gates
    for t in range(4):
        for d in DIRECTIONS:
            want = (0.5 * (np.asarray(cand["gates"][t]["a"], np.float64)[:, None] * _zdict(cand["T"])["a"][d].numpy()
                           + np.asarray(cand["gates"][t]["b"], np.float64)[:, None]
                           * _zdict(cand["T"])["b"][d].numpy())).astype(np.float32)
            np.testing.assert_allclose(G[t]["a"][d].numpy(), want, rtol=0, atol=1e-6)


def test_the_family_stores_the_gates_it_ran_from(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=11)
    for name, kw in (("G-T", {}), ("G-TF", {"tau_prime": GIVEN})):
        cand = R5F.candidate(name, b, ext, **kw)
        fam = R5F.run_candidate(name, b, ext, cand, **kw)
        assert fam["gates"] is cand["gates"] and fam["taus"] == cand["taus"] and fam["candidate"] == name
        assert set(fam["gate_check"]) == {f"tau_{t}" for t in range(4)} | {"P", "T", "m", "pick", "taus"}
        assert all(fam["gate_check"][f"tau_{t}"] == {"a": True, "b": True} for t in range(4))
        assert all(fam["gate_check"][k] is True for k in ("P", "T", "m", "pick", "taus"))
        for k in ("fpick", "cpick", "sigma", "fused", "cf", "details", "ctrl"):
            assert k in fam and fam[k] is not None


def test_g_t_family_equals_affs_family_on_g_ts_term(tmp_path, monkeypatch):
    """G-T changes only the term: its family is round 3's run_family on stack_G's term with AFF's gates."""
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=12)
    aff, g_aff = _aff(b)
    fam = R5F.run_candidate("G-T", b, ext, R5F.candidate("G-T", b, ext))
    ref = RF3.run_family(b, C.expected_term(ext.stack, aff["P"]), g_aff)
    assert fam["fpick"] == ref["fpick"] and fam["cpick"] == ref["cpick"]
    for m in METRICS:
        np.testing.assert_array_equal(fam["fused"][m], ref["fused"][m])
        np.testing.assert_array_equal(fam["cf"][m], ref["cf"][m])


# ---------------------------------------------------------------- M28 and M06

class StubScaler:
    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


class StubModel:
    classes_ = np.arange(3)

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)

    def predict_proba(self, X):
        assert len(X) == len(self.probs)
        return self.probs


def test_exact_pi_prime_ties_go_to_the_first_grouping_affect_included(tmp_path, monkeypatch):
    """M28: pi' = argmax P' (numpy.argmax), so an exact tie that includes affect is affect and opens G-TF's gate."""
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=13)
    rows = np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.4, 0.2, 0.4], [0.25, 0.375, 0.375], [0.5, 0.25, 0.25],
                     [1 / 3, 1 / 3, 1 / 3]])
    P = np.tile(rows, (b.n // len(rows), 1))
    b.readers = {"halves": [{"scaler": StubScaler(), "model": StubModel(P)}] * 2,
                 "feature_names": R3.rf.feature_names(A0), "config": "A0"}
    # G-TF's tau'_0 = 0 opens every affect pick, ties included; G-T's tau_0 = 3.87e-05 > 0 closes the zero margins
    want_g0 = {"G-TF": [1, 0, 1, 0, 1, 1], "G-T": [0, 0, 0, 0, 1, 0]}
    for name, kw in (("G-TF", {"tau_prime": GIVEN}), ("G-T", {})):
        cand = R5F.candidate(name, b, ext, **kw)
        for c in CONDITIONS:
            assert cand["pick"][c][:6].tolist() == [0, 1, 0, 1, 0, 0]
            assert cand["m"][c][:6].tolist() == [0.0, 0.0, 0.0, 0.0, 0.25, 0.0]
            assert cand["gates"][0][c][:6].tolist() == want_g0[name]
            assert cand["gates"][1][c][:6].tolist() == [0, 0, 0, 0, 1, 0]       # tau_1 0.02 or 0.217 < 0.25
        R5F.run_candidate(name, b, ext, cand, **kw)                            # the D7 recomputation agrees


def test_rho_ctrl_criterion_by_hand_through_run_candidate(tmp_path, monkeypatch):
    """M06: (1 + sigma) z(B) ranks as z(B), so sigma* = 0 and rho_ctrl = the integer 4*R@1 sum of B over each tune half;
    the fused cell maximises min(rho - rho_ctrl, gamma) over the 224 cells, ties to the lowest cell."""
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=14)
    cand = R5F.candidate("G-TF", b, ext, tau_prime=GIVEN)
    fam = R5F.run_candidate("G-TF", b, ext, cand, tau_prime=GIVEN)

    def hit(s, col):
        return (s[:, col] > np.delete(s, col, axis=1).max(axis=1)).astype(np.int64)

    rho = sum(hit(np.asarray(b.B["a"][d]), 0) + hit(np.asarray(b.B["b"][d]), 1) for d in DIRECTIONS)
    parity = np.asarray(b.parity)
    zB, gated, G = RF3._terms(b, cand["T"], cand["gates"])
    fri, fgi, cri = F2.cell_statistics(zB, F2.rank_info(b.B), gated, G, 1)
    for h in (0, 1):
        tune = parity == h
        rho_ctrl = int(rho[tune].sum())
        assert fam["ctrl"][h] == {"sigma": 0.0, "rho_ctrl": rho_ctrl} and fam["sigma"][h] == 0.0
        assert fam["details"]["fused"][str(h)]["rho_ctrl"] == rho_ctrl
        crit = np.minimum(fri[:, tune].sum(axis=1, dtype=np.int64) - rho_ctrl, fgi[:, tune].sum(axis=1, dtype=np.int64))
        assert fam["fpick"][h] == int(np.flatnonzero(crit == crit.max())[0])
    # the criterion itself on a constructed example: ignoring rho_ctrl would choose other cells
    par = np.array([0, 1, 0, 1])
    fri4 = np.array([[3, 2, 3, 2], [2, 2, 2, 2], [3, 3, 2, 3]], np.int8)
    fgi4 = np.array([[1, 1, 0, 1], [2, 1, 1, 1], [1, 1, 1, 0]], np.int8)
    assert F2.select_fused(fri4, fgi4, {0: (0.0, 3), 1: (0.0, 4)}, par) == {0: 2, 1: 2}
    assert F2.select_fused(fri4, fgi4, {0: (0.0, 0), 1: (0.0, 0)}, par) == {0: 1, 1: 0}


# ---------------------------------------------------------------- D11: the guard and the pairing

def test_candidate_and_run_candidate_refuse_a_ge_extension_before_release(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=15)
    gt = R5F.candidate("G-T", b, ext)
    gtf = R5F.candidate("G-TF", b, ext, tau_prime=GIVEN)
    R5G._reset_for_tests()                                                     # the guard closes

    def boom(*a, **k):
        raise AssertionError("computed before the guard")

    monkeypatch.setattr(RF3, "reader", boom)
    monkeypatch.setattr(RF3, "run_family", boom)
    for who, call in (("r5_fusion.candidate G-T", lambda: R5F.candidate("G-T", b, ext)),
                      ("r5_fusion.candidate G-TF", lambda: R5F.candidate("G-TF", b, ext, tau_prime=GIVEN)),
                      ("r5_fusion.run_candidate G-T", lambda: R5F.run_candidate("G-T", b, ext, gt)),
                      ("r5_fusion.run_candidate G-TF", lambda: R5F.run_candidate("G-TF", b, ext, gtf, tau_prime=GIVEN)),
                      ("r5_fusion.expected_d6 G-T", lambda: R5F.expected_d6("G-T", b, ext))):
        with pytest.raises(R5G.GuardError, match=re.escape(who)):          # the function itself refuses (D11)
            call()


def test_an_extension_of_another_bundle_or_placement_is_refused(tmp_path, monkeypatch):
    b, pl, ext = _ge(tmp_path, monkeypatch, seed=16)
    other = synth_bundle(seed=17)
    with pytest.raises(AssertionError):
        R5F.candidate("G-T", other, ext)
    forged = SimpleNamespace(**{**vars(ext), "placement_sha256": "0" * 64})
    with pytest.raises(R5G.GuardError):
        R5F.candidate("G-T", b, forged)
    b.readers = None
    with pytest.raises(ValueError, match="readers"):
        R5F.candidate("G-T", b, ext)


# ---------------------------------------------------------------- smoke seed 9001, CLIP placement (rule §5 item 4's path)

@pytest.fixture(scope="module")
def smoke_build():
    eps = R3.AB / "results" / "smoke" / f"episodes_seed{SMOKE_SEED}.npz"
    if not eps.exists():
        pytest.fail(f"smoke seed {SMOKE_SEED} is not built: run run_baselines.py --smoke --episodes-seed {SMOKE_SEED}")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return RB4.build_bundle(SMOKE_SEED, True)


def test_smoke_clip_candidates_and_families_equal_affs(smoke_build):
    b = smoke_build
    pl = R5G.clip_from_bundle(b)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ext = R5B.extend(b, pl)
        aff, g_aff = _aff(b)
        aff_fam = RF3.run_family(b, aff["T"], g_aff)
        cands = {"G-T": R5F.candidate("G-T", b, ext), "G-TF": R5F.candidate("G-TF", b, ext, tau_prime=TAUS)}
        fams = {"G-T": R5F.run_candidate("G-T", b, ext, cands["G-T"]),
                "G-TF": R5F.run_candidate("G-TF", b, ext, cands["G-TF"], tau_prime=TAUS)}
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    assert not any(DECIMAL.search(ln) for ln in lines), "a decimal number was printed"
    for name, cand in cands.items():
        for k in ("P", "m", "pick"):
            assert all(bool(np.array_equal(cand[k][c], aff[k][c])) and cand[k][c].dtype == aff[k][c].dtype
                       for c in CONDITIONS), f"{name}: {k} differs from AFF's"
        assert _same_T(cand["T"], aff["T"]), f"{name}: T differs from AFF's"
        assert cand["taus"] == TAUS, f"{name}: tau differs from AFF's"
        assert _same_gates(cand["gates"], g_aff), f"{name}: gates differ from AFF's"
        fam = fams[name]
        assert (fam["fpick"], fam["cpick"], fam["sigma"]) == (aff_fam["fpick"], aff_fam["cpick"], aff_fam["sigma"]), \
            f"{name}: cells or sigma* differ from AFF's"
        for part in ("fused", "cf"):
            assert all(bool(np.array_equal(fam[part][m], aff_fam[part][m])) for m in METRICS), \
                f"{name}: {part} per-anchor arrays differ from AFF's"
        assert fam["details"] == aff_fam["details"], f"{name}: the cross-fit details differ from AFF's"


def test_smoke_g_tf_takes_tau_prime_from_the_record_only(smoke_build, monkeypatch):
    b = smoke_build
    ext = R5B.extend(b, R5G.clip_from_bundle(b))
    with pytest.raises(ValueError, match="never recomputed"):
        R5F.candidate("G-TF", b, ext)

    def boom(*a, **k):
        raise AssertionError("tau' recomputed on a smoke seed")

    monkeypatch.setattr(rc_core, "thresholds", boom)
    cand = R5F.candidate("G-TF", b, ext, tau_prime=TAUS)
    assert cand["taus"] == TAUS, "the record's tau' was not used as given"
