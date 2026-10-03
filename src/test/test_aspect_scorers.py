import numpy as np
import pytest

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


# Row patterns for the per-half extension tests, written for condition a (target column 0, other-aspect column 1);
# condition b swaps columns 0 and 1. Column 2 is a distractor (a negative), so the other-aspect rate stays 0 and the
# cross-fit criterion equals R@1. With z-scores over 13 candidates, the target beats the distractor in an "A" row once
# lambda * eps / std(term row) > 1 / std(cos row), i.e. lambda > about 1.38 / eps:
#   "A12" (eps 0.115): lost at lambda <= 8, won at 16, 32, 64 and inf;
#   "A45" (eps 0.031): lost at lambda <= 32, won at 64 and inf;
#   "B": the term ties the target with the distractor and the cosine prefers the target, so every finite lambda wins
#        and lambda = inf (term alone, a tie) is a miss.
def _row(kind):
    cos, term = np.zeros(13), np.zeros(13)
    if kind == "B":
        cos[0], term[0], term[2] = 1.0, 1.0, 1.0
    else:
        eps = {"A12": 0.115, "A45": 0.031}[kind]
        cos[2], term[0], term[2] = 1.0, 1.0, 1.0 - eps
    return cos, term


def _extension_halves(extending_half):
    """40 rows, parity halves of 20. The extending half (12 "A12" + 8 "B" rows) picks 16 on the base grid: 20 hits
    at 16 against 12 at inf and 8 at lambda <= 8, so its grid is extended. The isolated half (12 "A45" + 8 "B" rows)
    picks inf on the base grid (12 hits against 8 for every finite lambda up to 16), but lambda = 64 would give it 20
    hits. It must keep inf: the extension belongs to the other half only."""
    n = 40
    cos = {c: {d: np.zeros((n, 13)) for d in ("i2t", "t2i")} for c in ("a", "b")}
    term = {c: {d: np.zeros((n, 13)) for d in ("i2t", "t2i")} for c in ("a", "b")}
    for r in range(n):
        half_rank = r // 2                                              # 0..19 inside the row's parity half
        if r % 2 == extending_half:
            kind = "A12" if half_rank < 12 else "B"
        else:
            kind = "A45" if half_rank < 12 else "B"
        c_row, t_row = _row(kind)
        for d in ("i2t", "t2i"):
            cos["a"][d][r], term["a"][d][r] = c_row, t_row
            cos["b"][d][r], term["b"][d][r] = c_row[[1, 0, *range(2, 13)]], t_row[[1, 0, *range(2, 13)]]
    return cos, term


@pytest.mark.parametrize("extending_half", [0, 1])
def test_crossfit_extension_never_leaks_into_the_other_half(monkeypatch, extending_half):
    """extending_half = 0 fails on the earlier shared-dict code (half 1 then chose 64 from half 0's extension);
    extending_half = 1 fails on any variant that extends both halves when either picks 16."""
    import src.eval.aspect_scorers as mod
    cos, term = _extension_halves(extending_half)
    parity = np.arange(40) % 2
    isolated = 1 - extending_half
    real_fused = mod.fused_scores
    tune_iso = parity == isolated
    # the design does what the docstring says: the isolated half prefers 64 to inf, and inf to every base finite lambda
    crit = {lam: mod._criterion(real_fused(cos, term, lam), tune_iso) for lam in LAMBDA_GRID + [32.0, 64.0]}
    assert crit[64.0] > crit[float("inf")] > max(crit[lam] for lam in LAMBDA_GRID[:-1])
    calls = []
    monkeypatch.setattr(mod, "fused_scores", lambda c, t, lam: (calls.append(lam), real_fused(c, t, lam))[1])
    _, picks = crossfit_lambda(cos, term, parity)
    assert set(calls) >= {32.0, 64.0}                                   # the extending half did extend
    assert picks[extending_half] in (16.0, 32.0, 64.0)
    assert picks[isolated] == float("inf")
