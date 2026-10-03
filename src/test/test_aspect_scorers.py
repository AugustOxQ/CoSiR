import numpy as np

from src.eval.aspect_episodes import build_aspect_episodes
from src.eval.aspect_metrics import per_anchor
from src.eval.aspect_scorers import (
    LAMBDA_GRID, EvalInputs, agreement_term, cosine_scores, crossfit_lambda, fixed_beta_scores,
)


def _world(n_paintings=2500, seed=0):
    """Aspect-block codes: every value of an aspect is a different positive pattern over the SAME 6 factors
    (aspect a uses factors 0-5, aspect b factors 6-11). This is the code shape the agreement rule needs."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    a = rng.integers(0, 6, 2 * n_paintings)
    b = np.repeat(rng.integers(0, 6, n_paintings), 2)
    pattern_a, pattern_b = rng.random((6, 6)) ** 3, rng.random((6, 6)) ** 3
    code = np.concatenate([pattern_a[a], pattern_b[b]], axis=1).astype(np.float32)
    img = code + 0.1 * rng.normal(size=code.shape).astype(np.float32)
    txt = code + 0.1 * rng.normal(size=code.shape).astype(np.float32)
    return {"a": a, "b": b}, groups, img, txt, code


def test_cosine_is_condition_blind_and_agreement_is_not():
    labels, groups, img, txt, code = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 300, seed=1)
    inputs = EvalInputs(img, txt, code, code)
    assert np.allclose(per_anchor(cosine_scores(inputs, ep))["gain"], 0.0)
    gain = per_anchor(agreement_term(inputs, ep))["gain"].mean()
    assert gain > 0.15                     # controller's simulation on this world: about 0.32
    assert np.allclose(per_anchor(agreement_term(inputs, ep, uniform=True))["gain"], 0.0)
    assert per_anchor(fixed_beta_scores(inputs, ep, 0.3))["gain"].mean() > 0.05


def test_value_onehot_codes_are_blind_under_value_disjoint_conditions():
    """Pins the mechanism: if each value has its own factor, supports showing OTHER values give the anchor's value
    zero weight, so every candidate ties and the condition gain is exactly 0 (simulation 2026-10-03)."""
    labels, groups, img, txt, _ = _world()
    onehot = np.zeros((len(groups), 12), np.float32)
    onehot[np.arange(len(groups)), labels["a"]] = 1.0
    onehot[np.arange(len(groups)), 6 + labels["b"]] = 1.0
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 200, seed=3)
    m = per_anchor(agreement_term(EvalInputs(img, txt, onehot, onehot), ep))
    assert np.allclose(m["gain"], 0.0) and np.allclose(m["r1"], 0.0)


def test_crossfit_returns_grid_picks_and_valid_scores():
    labels, groups, img, txt, code = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 200, seed=2)
    inputs = EvalInputs(img, txt, code, code)
    scores, picks = crossfit_lambda(cosine_scores(inputs, ep), agreement_term(inputs, ep), np.arange(200) % 2)
    assert set(picks) == {0, 1} and all(p in LAMBDA_GRID + [32.0, 64.0] for p in picks.values())
    assert scores["a"]["i2t"].shape == (200, 13)
