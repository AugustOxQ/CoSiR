import math

import numpy as np
import pytest

from src.eval.aspect_episodes import build_aspect_episodes
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_nested import (
    K_SE, NESTED_A, NESTED_U, ceiling_threshold, control_scores, control_sums, crossfit_nested, fit_reading,
    h3_reading, joint_decision, margin_reading, nested_cells, nested_scores, predicted_power, se_from_ci,
)
from src.eval.aspect_scorers import EvalInputs, agreement_term, cosine_scores, fused_scores


def _world(n_paintings=1200, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    a = rng.integers(0, 6, 2 * n_paintings)
    b = np.repeat(rng.integers(0, 6, n_paintings), 2)
    pattern_a, pattern_b = rng.random((6, 6)) ** 3, rng.random((6, 6)) ** 3
    code = np.concatenate([pattern_a[a], pattern_b[b]], axis=1).astype(np.float32)
    img = code + 0.3 * rng.normal(size=code.shape).astype(np.float32)
    txt = code + 0.3 * rng.normal(size=code.shape).astype(np.float32)
    ep = build_aspect_episodes({"a": a, "b": b}, groups, np.arange(len(groups)), "a", "b", 400, seed=1,
                               min_paintings=5)
    inputs = EvalInputs(img, txt, code, code)
    return cosine_scores(inputs, ep), agreement_term(inputs, ep, uniform=True), agreement_term(inputs, ep)


def test_grids_are_the_preregistered_ones():
    assert NESTED_U == (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert NESTED_A == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    cells = nested_cells()
    assert len(cells) == 56 and cells[0] == (0.0, 0.0) and cells[1] == (0.0, 0.25) and cells[8] == (0.5, 0.0)
    sums = control_sums()
    assert len(sums) == 30 and sums[0] == 0.0 and sums[-1] == 32.0 and sums == sorted(sums)


def test_nested_reduces_to_cosine_and_to_1d_fusion():
    cos, tu, ta = _world()
    for c in CONDITIONS:
        for d in DIRECTIONS:
            base = nested_scores(cos, tu, ta, 0.0, 0.0)[c][d]
            assert np.allclose(base, fused_scores(cos, ta, 0.0)[c][d])
            for lam in (0.25, 2.0, 16.0):
                assert np.allclose(nested_scores(cos, tu, ta, 0.0, lam)[c][d], fused_scores(cos, ta, lam)[c][d],
                                   atol=1e-5)
                assert np.allclose(control_scores(cos, tu, lam)[c][d], fused_scores(cos, tu, lam)[c][d], atol=1e-5)


def test_control_is_condition_blind():
    cos, tu, _ = _world()
    for s in (0.5, 4.0, 32.0):
        assert np.allclose(per_anchor(control_scores(cos, tu, s))["gain"], 0.0)


def test_nested_rejects_bad_weights():
    cos, tu, ta = _world()
    for bad in ((-1.0, 0.0), (0.0, float("inf")), (float("nan"), 1.0)):
        with pytest.raises(ValueError):
            nested_scores(cos, tu, ta, *bad)


def test_nested_nonfinite_rows_are_misses_only_when_used():
    cos, tu, ta = _world()
    ta_bad = {c: {d: ta[c][d].copy() for d in DIRECTIONS} for c in CONDITIONS}
    for c in CONDITIONS:
        for d in DIRECTIONS:
            ta_bad[c][d][3] = np.nan
    used = nested_scores(cos, tu, ta_bad, 1.0, 2.0)
    unused = nested_scores(cos, tu, ta_bad, 1.0, 0.0)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert np.isnan(used[c][d][3]).all() and np.isfinite(used[c][d][4]).all()
            assert np.isfinite(unused[c][d][3]).all()
    assert per_anchor(used)["r1"][3] == 0.0


def test_crossfit_nested_returns_grid_picks_and_finite_scores():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    nested, control, picks = crossfit_nested(cos, tu, ta, np.arange(n) % 2)
    assert set(picks) == {0, 1}
    for half in (0, 1):
        assert picks[half]["sigma"] in control_sums()
        assert tuple(picks[half]["cell"]) in nested_cells()
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert np.isfinite(nested[c][d]).all() and np.isfinite(control[c][d]).all()
    assert np.allclose(per_anchor(control)["gain"], 0.0)
    assert per_anchor(nested)["gain"].mean() > 0.05          # the world has aspect-block codes


def test_crossfit_nested_halves_are_independent():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    parity = np.arange(n) % 2
    nested, control, picks = crossfit_nested(cos, tu, ta, parity)
    for half in (0, 1):
        apply = parity != half
        cell, sigma = tuple(picks[half]["cell"]), picks[half]["sigma"]
        for c in CONDITIONS:
            for d in DIRECTIONS:
                assert np.allclose(nested[c][d][apply], nested_scores(cos, tu, ta, *cell)[c][d][apply])
                assert np.allclose(control[c][d][apply], control_scores(cos, tu, sigma)[c][d][apply])
    # changing only half 1's rows must not change half 0's pick (picks are decided on the tuning half alone)
    shuffled = {k: {c: {d: v[c][d].copy() for d in DIRECTIONS} for c in CONDITIONS}
                for k, v in (("cos", cos), ("tu", tu), ("ta", ta))}
    rng = np.random.default_rng(5)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            for key in shuffled:
                rows = shuffled[key][c][d][parity == 1]
                shuffled[key][c][d][parity == 1] = rows + rng.normal(size=rows.shape).astype(np.float32)
    _, _, picks2 = crossfit_nested(shuffled["cos"], shuffled["tu"], shuffled["ta"], parity)
    assert picks2[0] == picks[0]


def test_crossfit_nested_ties_go_to_first_cell_and_smallest_sigma():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    flat = {c: {d: np.zeros_like(cos[c][d]) for d in DIRECTIONS} for c in CONDITIONS}  # every cell and sum ties
    _, _, picks = crossfit_nested(flat, flat, flat, np.arange(n) % 2)
    for half in (0, 1):
        assert picks[half]["sigma"] == 0.0 and picks[half]["cell"] == [0.0, 0.0]


def test_crossfit_nested_validates_parity():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    for bad in (np.zeros(n, int), np.arange(n) % 3, np.arange(n - 1) % 2):
        with pytest.raises(ValueError):
            crossfit_nested(cos, tu, ta, bad)


def test_readings_at_boundaries():
    assert se_from_ci([-1.959963984540054, 1.959963984540054]) == pytest.approx(1.0)
    assert margin_reading(0.0, 0.1, 5.0, 0.1) == "not_promising"
    assert margin_reading(5.0, 0.1, -0.01, 0.1) == "not_promising"
    assert margin_reading(K_SE * 0.1, 0.1, K_SE * 0.2, 0.2) == "promising"
    assert margin_reading(K_SE * 0.1 - 1e-9, 0.1, 1.0, 0.1) == "inconclusive"
    assert ceiling_threshold(0.168, 0.2) == pytest.approx(max(5.6 * 0.168, 2.8 * 0.2))
    assert predicted_power(2 * 1.959963984540054, 1.0) == pytest.approx(0.5)
    assert fit_reading({"point": 0.5, "ci95": [0.001, 1.0]}) == "fits"
    assert fit_reading({"point": 0.5, "ci95": [0.0, 1.0]}) == "inconclusive"
    assert fit_reading({"point": 0.0, "ci95": [-1.0, 1.0]}) == "no_fit"


def test_h3_reading_refuses_unresolved_fit():
    with pytest.raises(ValueError):
        h3_reading({"L3": "inconclusive", "L5": "no_fit", "LT": "no_fit"}, None, 1.0)
    assert h3_reading({"L3": "no_fit", "L5": "no_fit", "LT": "no_fit"}, None, 1.0) == "no_fit"
    assert h3_reading({"L3": "fits", "L5": "no_fit", "LT": "no_fit"}, 0.99, 1.0) == "ceiling_too_low"
    assert h3_reading({"L3": "fits", "L5": "no_fit", "LT": "no_fit"}, 1.0, 1.0) == "ceiling_sufficient"
    with pytest.raises(ValueError):
        h3_reading({"L3": "fits", "L5": "no_fit", "LT": "no_fit"}, None, 1.0)


def test_joint_decision_table():
    for h3 in ("no_fit", "ceiling_too_low", "ceiling_sufficient"):
        assert joint_decision("promising", h3) == "preregister_A3_nested"
    for h1 in ("inconclusive", "not_promising"):
        assert joint_decision(h1, "ceiling_sufficient") == "h2_grid"
        assert joint_decision(h1, "no_fit") == "branch_3"
        assert joint_decision(h1, "ceiling_too_low") == "branch_3"
    with pytest.raises(ValueError):
        joint_decision("maybe", "no_fit")
    with pytest.raises(ValueError):
        joint_decision("promising", "fits")
