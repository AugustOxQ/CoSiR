"""Method A′ (CVPR plan spec §15): the nested test-time score z(cos) + λ_u·z(T_u) + λ_a·z(T_a), its nested uniform
control z(cos) + σ·z(T_u), parity cross-fitting with the min-margin pick rule, and the pre-registered readings of the
method-repair diagnostics stage (src/test/20261105_method_repair_diagnostics/PREREGISTRATION.md)."""

import math

import numpy as np
import torch

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.model.aspect_rule import zscore_rows

NESTED_U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NESTED_A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
K_SE = 2.80
Z975 = 1.959963984540054
H1_READINGS = ("promising", "inconclusive", "not_promising")
H3_READINGS = ("no_fit", "ceiling_too_low", "ceiling_sufficient")


def nested_cells() -> list:
    """The 56 (λ_u, λ_a) cells in row-major order (λ_u outer, λ_a inner, ascending); ties go to the first."""
    return [(u, a) for u in NESTED_U for a in NESTED_A]


def control_sums() -> list:
    """The 30 distinct σ = λ_u + λ_a, ascending; the control's ties go to the smallest."""
    return sorted({u + a for u, a in nested_cells()})


def _zdict(x: dict) -> dict:
    return {c: {d: zscore_rows(torch.as_tensor(np.asarray(x[c][d]), dtype=torch.float32)) for d in DIRECTIONS}
            for c in CONDITIONS}


def _check(lam_u: float, lam_a: float) -> None:
    if not (math.isfinite(lam_u) and math.isfinite(lam_a)) or lam_u < 0 or lam_a < 0:
        raise ValueError(f"nested weights must be finite and >= 0, got ({lam_u}, {lam_a})")


def _combine(zc: dict, zu: dict, za: dict, lam_u: float, lam_a: float) -> dict:
    """Terms with weight 0 are left out, so a non-finite row in an unused term cannot turn a row into a miss."""
    out = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for d in DIRECTIONS:
            s = zc[c][d]
            if lam_u > 0:
                s = s + lam_u * zu[c][d]
            if lam_a > 0:
                s = s + lam_a * za[c][d]
            out[c][d] = s.numpy().astype(np.float32)
    return out


def nested_scores(cos: dict, t_u: dict, t_a: dict, lam_u: float, lam_a: float) -> dict:
    _check(lam_u, lam_a)
    return _combine(_zdict(cos), _zdict(t_u), _zdict(t_a), lam_u, lam_a)


def control_scores(cos: dict, t_u: dict, sigma: float) -> dict:
    return nested_scores(cos, t_u, t_u, sigma, 0.0)


def _means(scores: dict, rows: np.ndarray) -> tuple:
    m = per_anchor({c: {d: scores[c][d][rows] for d in DIRECTIONS} for c in CONDITIONS})
    return float(m["r1"].mean()), float(m["gain"].mean())


def crossfit_nested(cos: dict, t_u: dict, t_a: dict, parity) -> tuple:
    """Spec §15. On each tuning half the control picks σ by R@1; the nested score then picks the cell maximising
    min(R@1 − that control's R@1 on the same half, condition gain). Each half's picks score the other half."""
    n = len(cos["a"]["i2t"])
    parity = np.asarray(parity)
    if parity.shape != (n,) or not np.isin(parity, (0, 1)).all() or not ((parity == 0).any() and (parity == 1).any()):
        raise ValueError(f"parity must be a length-{n} array of 0/1 with both halves non-empty")
    zc, zu, za = _zdict(cos), _zdict(t_u), _zdict(t_a)
    ctrl_cache, nest_cache = {}, {}

    def ctrl(sigma):
        if sigma not in ctrl_cache:
            ctrl_cache[sigma] = _combine(zc, zu, zu, sigma, 0.0)
        return ctrl_cache[sigma]

    def nest(cell):
        if cell not in nest_cache:
            nest_cache[cell] = _combine(zc, zu, za, *cell)
        return nest_cache[cell]

    shape = {c: {d: np.asarray(cos[c][d]).shape for d in DIRECTIONS} for c in CONDITIONS}
    nested = {c: {d: np.empty(shape[c][d], np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    control = {c: {d: np.empty(shape[c][d], np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    picks = {}
    for half in (0, 1):
        tune, apply = parity == half, parity != half
        sigma = max(control_sums(), key=lambda s: _means(ctrl(s), tune)[0])
        r_ctrl = _means(ctrl(sigma), tune)[0]

        def criterion(cell):
            r1, gain = _means(nest(cell), tune)
            return min(r1 - r_ctrl, gain)

        cell = max(nested_cells(), key=criterion)
        picks[half] = {"sigma": float(sigma), "cell": [float(cell[0]), float(cell[1])]}
        for c in CONDITIONS:
            for d in DIRECTIONS:
                nested[c][d][apply] = nest(cell)[c][d][apply]
                control[c][d][apply] = ctrl(sigma)[c][d][apply]
    return nested, control, picks


def se_from_ci(ci95) -> float:
    """SE of a paired difference from its 95% percentile interval (pre-registered approximation)."""
    lo, hi = ci95
    return (hi - lo) / (2 * Z975)


def margin_reading(m_r: float, se_r: float, m_g: float, se_g: float, k: float = K_SE) -> str:
    """H1 pilot: not promising if either margin <= 0; promising if both >= k SE; inconclusive otherwise."""
    if not (math.isfinite(m_r) and math.isfinite(se_r) and math.isfinite(m_g) and math.isfinite(se_g)):
        raise ValueError(f"all of m_r, se_r, m_g, se_g must be finite, got ({m_r}, {se_r}, {m_g}, {se_g})")
    if se_r <= 0 or se_g <= 0:
        raise ValueError(f"se_r and se_g must be positive, got se_r={se_r}, se_g={se_g}")
    if m_r <= 0 or m_g <= 0:
        return "not_promising"
    if m_r >= k * se_r and m_g >= k * se_g:
        return "promising"
    return "inconclusive"


def ceiling_threshold(se_r: float, se_g: float) -> float:
    """g* = max(2·K_SE·SE_R, K_SE·SE_g): the R@1 margin against the control is about gain / 2."""
    if not (math.isfinite(se_r) and math.isfinite(se_g)):
        raise ValueError(f"se_r and se_g must be finite, got se_r={se_r}, se_g={se_g}")
    if se_r < 0 or se_g < 0:
        raise ValueError(f"se_r and se_g must be non-negative, got se_r={se_r}, se_g={se_g}")
    return max(2 * K_SE * se_r, K_SE * se_g)


def predicted_power(margin: float, se: float) -> float:
    """Descriptive: P(lower bound > 0) on a fresh draw if the true margin is half the observed one."""
    if not (math.isfinite(margin) and math.isfinite(se)):
        raise ValueError(f"margin and se must be finite, got margin={margin}, se={se}")
    if se <= 0:
        raise ValueError(f"se must be positive, got se={se}")
    return 0.5 * (1.0 + math.erf((margin / (2 * se) - Z975) / math.sqrt(2.0)))


def fit_reading(result: dict) -> str:
    """H3 fit from a paired compare(X, A3, ..., 'gain') result: fits / inconclusive / no_fit."""
    point = result["point"]
    lo, hi = result["ci95"]
    if not (math.isfinite(point) and math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"point and CI bounds must be finite, got point={point}, ci95=[{lo}, {hi}]")
    if lo > 0:
        return "fits"
    if point <= 0:
        return "no_fit"
    return "inconclusive"


def h3_reading(fit: dict, best_nested_gain, g_star: float) -> str:
    if any(v not in ("fits", "no_fit") for v in fit.values()):
        raise ValueError(f"every LAB fit must be resolved to fits/no_fit first: {fit}")
    if not any(v == "fits" for v in fit.values()):
        return "no_fit"
    if best_nested_gain is None:
        raise ValueError("a fitting LAB run needs its seed-42 nested gain")
    return "ceiling_sufficient" if best_nested_gain >= g_star else "ceiling_too_low"


def joint_decision(h1: str, h3: str) -> str:
    if h1 not in H1_READINGS or h3 not in H3_READINGS:
        raise ValueError(f"unknown readings: h1={h1!r}, h3={h3!r}")
    if h1 == "promising":
        return "preregister_A3_nested"
    return "h2_grid" if h3 == "ceiling_sufficient" else "branch_3"
