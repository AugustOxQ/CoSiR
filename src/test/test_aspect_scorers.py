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


def test_nonfinite_inputs_are_misses_through_scorers():                # Review Focus 4
    from src.eval.aspect_metrics import first_place
    from src.eval.aspect_scorers import fused_scores
    labels, groups, img, txt, code = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 40, seed=5)
    cos, term = cosine_scores(EvalInputs(img, txt, code, code), ep), agreement_term(EvalInputs(img, txt, code, code), ep)
    for c in ("a", "b"):
        for d in ("i2t", "t2i"):
            cos[c][d] = cos[c][d].copy(); term[c][d] = term[c][d].copy()
            cos[c][d][0, 3] = np.nan                                  # NaN in cos row 0
            term[c][d][1, 3] = np.nan                                 # NaN in term row 1
    out = fused_scores(cos, term, 2.0)
    for c, col in (("a", 0), ("b", 1)):
        for d in ("i2t", "t2i"):
            assert not np.isfinite(out[c][d][:2]).any()
            assert (first_place(out[c][d], col)[:2] == 0).all()
    # NaN codes in a support pair of episode 2
    code_bad = code.copy()
    si = ep.condition("a")[0][2]
    code_bad[si] = np.nan
    t = agreement_term(EvalInputs(img, txt, code_bad, code_bad), ep)
    assert not np.isfinite(t["a"]["i2t"][2]).any()
    assert first_place(t["a"]["i2t"], 0)[2] == 0
    fb = fixed_beta_scores(EvalInputs(img, txt, code_bad, code_bad), ep, 0.3)
    assert not np.isfinite(fb["a"]["i2t"][2]).any() and first_place(fb["a"]["i2t"], 0)[2] == 0


def _synthetic_halves(half0_term_distractor):
    """cos/term score dicts over 40 rows x 13 candidates. Even rows (half 0): cos favours a distractor (col 2) and
    the term the right candidate, so a large lambda wins. Odd rows (half 1): cos is right for a, the term is
    misleading, so lambda 0 wins."""
    n = 40
    cos = {c: {d: np.zeros((n, 13)) for d in ("i2t", "t2i")} for c in ("a", "b")}
    term = {c: {d: np.zeros((n, 13)) for d in ("i2t", "t2i")} for c in ("a", "b")}
    for d in ("i2t", "t2i"):
        for c, col, wrong in (("a", 0, 1), ("b", 1, 0)):
            cos[c][d][0::2, 2] = 1.0
            term[c][d][0::2, col] = 1.0
            term[c][d][0::2, 2] = half0_term_distractor
            cos[c][d][1::2, 0] = 1.0; cos[c][d][1::2, 1] = 0.9
            term[c][d][1::2, wrong] = 1.0
    return cos, term


def test_crossfit_halves_pick_independently_and_apply_to_the_other_half():
    from src.eval.aspect_scorers import fused_scores
    cos, term = _synthetic_halves(0.0)
    parity = np.arange(40) % 2
    scores, picks = crossfit_lambda(cos, term, parity)
    assert picks[0] > 0 and picks[1] == 0.0
    for half in (0, 1):
        rows = parity != half                                          # tuned on `half`, applied to the others
        want = fused_scores(cos, term, picks[half])
        for c in ("a", "b"):
            for d in ("i2t", "t2i"):
                assert np.isfinite(scores[c][d]).all()
                assert np.array_equal(scores[c][d][rows], want[c][d][rows])


def test_crossfit_extends_grid_only_for_the_half_that_picks_16(monkeypatch):
    import src.eval.aspect_scorers as mod
    cos, term = _synthetic_halves(0.9)                                 # half 0 needs lambda >= 16
    real = mod._criterion
    calls = []
    real_fused = mod.fused_scores
    monkeypatch.setattr(mod, "fused_scores", lambda c, t, lam: (calls.append(lam), real_fused(c, t, lam))[1])
    _, picks = crossfit_lambda(cos, term, np.arange(40) % 2)
    assert picks[0] in (16.0, 32.0, 64.0) and picks[1] == 0.0
    assert set(calls) >= {32.0, 64.0}
    # half 1 never evaluated an extended lambda: replay its candidate set
    tune1 = np.arange(40) % 2 == 1
    best1 = max(LAMBDA_GRID, key=lambda lam: real(real_fused(cos, term, lam), tune1))
    assert best1 == picks[1] == 0.0


def test_crossfit_validates_parity():
    import pytest
    cos, term = _synthetic_halves(0.0)
    for bad in (np.arange(39) % 2, np.arange(40) % 3, np.zeros(40, int), np.ones(40, int)):
        with pytest.raises(ValueError):
            crossfit_lambda(cos, term, bad)
