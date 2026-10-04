import numpy as np
import pytest
import torch

from src.eval.aspect_episodes import AspectEpisodes
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, first_place, per_anchor
from src.eval.aspect_quick_checks import (
    aspect_deltas, both_in_topk, cascade_scores, centered_agreement_weights, centered_term, code_scale,
    inferred_scores, inferred_weights, kissme_diag_term, told_scores,
)
from src.eval.aspect_scorers import EvalInputs
from src.model.aspect_rule import agreement_weights


def _ep(n_ep=1):
    """Hand-built episodes on 30 rows each: anchor 0, candidates 1..13 (p_a = row 1, p_b = row 2), aspect-a pairs
    (images 14..17, captions 18..21), aspect-b pairs (images 22..25, captions 26..29)."""
    rows = np.tile(np.arange(30), (n_ep, 1)) + 30 * np.arange(n_ep)[:, None]
    return AspectEpisodes("a", "b", rows[:, 0], rows[:, 1:14], rows[:, 14:18], rows[:, 18:22], rows[:, 22:26],
                          rows[:, 26:30])


def _inputs(img_codes, txt_codes):
    feats = np.random.default_rng(0).normal(size=(len(img_codes), 4)).astype(np.float32)
    return EvalInputs(feats, feats.copy(), np.asarray(img_codes, np.float32), np.asarray(txt_codes, np.float32))


def _sc(a):
    """The same score matrix under both conditions and both directions."""
    a = np.asarray(a, dtype=np.float64)
    return {c: {d: a.copy() for d in DIRECTIONS} for c in CONDITIONS}


# ---------------------------------------------------------------- N1: centered agreement rule

def test_centered_weights_reward_covariation_not_mean_activity():
    # factor 0: image and caption codes rise together across the 4 support pairs; factor 1: both high and constant
    sup = np.array([[[0.1, 0.9], [0.2, 0.9], [0.3, 0.9], [0.4, 0.9]]])
    zero = np.zeros((1, 4, 2))
    np.testing.assert_allclose(centered_agreement_weights(sup, sup, zero, zero), [[1.0, 0.0]], atol=1e-6)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    old = agreement_weights(t(sup), t(sup), t(zero), t(zero)).numpy()
    assert old[0, 1] > old[0, 0]          # the uncentered rule prefers the generally active factor


def test_centered_weights_subtract_contrast_covariance():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    sup = np.stack([v, v], axis=-1)[None]                       # both factors covary in the supports
    con = np.stack([v, np.full(4, 0.5)], axis=-1)[None]         # only factor 0 covaries in the contrasts
    np.testing.assert_allclose(centered_agreement_weights(sup, sup, con, con), [[0.0, 1.0]], atol=1e-6)


def test_centered_weights_zero_and_nonfinite_rows():
    flat = np.full((1, 4, 3), 0.5)
    assert (centered_agreement_weights(flat, flat, flat, flat) == 0).all()
    bad = flat.copy()
    bad[0, 2, 1] = np.nan
    assert np.isnan(centered_agreement_weights(bad, flat, flat, flat)).all()


def test_centered_term_centres_the_query_on_its_own_modality_examples():
    img, txt = np.zeros((30, 2)), np.zeros((30, 2))
    v = np.array([0.2, 0.4, 0.6, 0.8])
    img[14:18, 0], txt[18:22, 0] = v, v - 0.5        # condition a's support pairs covary on factor 0
    img[22:26, 0] = 0.5                               # contrast images constant; the 8 example images average 0.5
    img[0, 0] = 0.2                                   # the query image lies below the example images' mean
    txt[1:14, 0] = np.linspace(0.0, 1.0, 13)          # candidate captions; column 0 has the lowest code
    s = centered_term(_inputs(img, txt), _ep())["a"]["i2t"]
    assert s.shape == (1, 13)
    # (q - mu_q) < 0, so the lowest candidate wins; without centring, or centring on the captions (mean 0), the
    # highest candidate (column 12) would win
    assert np.argmax(s[0]) == 0


def test_centered_uniform_term_ignores_the_condition():
    rng = np.random.default_rng(1)
    t = centered_term(_inputs(rng.random((60, 3)), rng.random((60, 3))), _ep(2), uniform=True)
    for d in DIRECTIONS:
        np.testing.assert_array_equal(t["a"][d], t["b"][d])
    assert (per_anchor(t)["gain"] == 0).all()


# ---------------------------------------------------------------- diagonal KISSME on codes

def _kissme_world():
    img, txt = np.zeros((30, 2)), np.zeros((30, 2))
    img[14:18], txt[18:22] = [1.0, 2.0], [1.0, 0.0]    # condition a supports: difference (0, 2)
    img[22:26], txt[26:30] = [2.0, 1.0], [0.0, 1.0]    # condition a contrasts: difference (2, 0)
    img[0] = [1.0, 1.0]
    txt[1] = [1.0, 3.0]                                 # column 0 matches the query on factor 0
    txt[2:14] = [3.0, 1.0]                              # the others match it on factor 1
    return img, txt


def test_kissme_diag_prefers_closeness_where_supports_agree():
    img, txt = _kissme_world()
    t = kissme_diag_term(_inputs(img, txt), _ep(), scale=np.ones(2))
    assert np.argmax(t["a"]["i2t"][0]) == 0             # m = (0.8, -0.8): column 0 scores 3.2, the others -3.2
    assert np.argmax(t["b"]["i2t"][0]) != 0             # roles swap under condition b


def test_kissme_diag_divides_codes_by_the_scale():
    img, txt = _kissme_world()
    s = np.array([2.0, 4.0])
    a = kissme_diag_term(_inputs(img, txt), _ep(), scale=s)["a"]["i2t"]
    b = kissme_diag_term(_inputs(img / s, txt / s), _ep(), scale=np.ones(2))["a"]["i2t"]
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_kissme_diag_is_flat_when_supports_and_contrasts_vary_alike():
    img, txt = np.zeros((30, 2)), np.zeros((30, 2))
    img[14:18], txt[18:22] = [1.0, 2.0], [0.0, 0.0]
    img[22:26], txt[26:30] = [1.0, 2.0], [0.0, 0.0]
    txt[1:14] = np.random.default_rng(2).random((13, 2))
    t = kissme_diag_term(_inputs(img, txt), _ep(), scale=np.ones(2))
    assert np.ptp(t["a"]["i2t"][0]) == 0


def test_code_scale_pools_modalities_and_guards_dead_factors():
    tr = np.array([[0.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(code_scale(tr, tr), [2.0, 1.0])     # std of (0, 4, 0, 4) = 2; dead factor -> 1
    with pytest.raises(ValueError):
        code_scale(np.array([[np.nan, 0.0]]), tr)


# ---------------------------------------------------------------- N2: find-then-select cascade

def test_cascade_reorders_only_the_top_k():
    control = _sc([[0.5, 0.8, 0.9, 0.1, 0.7]])          # control order: 2, 1, 4, 0, 3
    rerank = _sc([[9.0, 1.0, 0.0, 5.0, 2.0]])
    out = cascade_scores(control, rerank, k=3)["a"]["i2t"][0]
    # top 3 by control = {2, 1, 4}, reordered by rerank to 4, 1, 2; columns 0 and 3 stay below in control order,
    # although column 0 has the highest rerank score
    assert list(np.argsort(-out)) == [4, 1, 2, 0, 3]
    assert list(np.argsort(-cascade_scores(control, rerank, k=1)["a"]["i2t"][0])) == [2, 1, 4, 0, 3]


def test_cascade_breaks_rerank_ties_by_control():
    control, rerank = _sc([[0.8, 0.9, 0.1]]), _sc([[0.0, 0.0, 5.0]])
    out = cascade_scores(control, rerank, k=2)["a"]["i2t"]
    # columns 0 and 1 tie on rerank; the control prefers column 1, so column 1 leads (not the lower column index)
    assert list(np.argsort(-out[0])) == [1, 0, 2]
    assert first_place(out, 1)[0] == 1.0                # no tie is created, so the leader is a strict hit


def test_cascade_nonfinite_rows_become_misses():
    control = _sc([[0.9, np.nan, 0.1], [0.9, 0.8, 0.1], [0.9, 0.8, 0.1]])
    rerank = _sc([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [0.0, 1.0, np.nan]])
    out = cascade_scores(control, rerank, k=2)["a"]["i2t"]
    assert np.isnan(out[0]).all() and np.isnan(out[1]).all()
    assert np.isfinite(out[2]).all()                    # a NaN rerank score outside the top k does not matter
    with pytest.raises(ValueError):
        cascade_scores(control, rerank, k=0)
    with pytest.raises(ValueError):
        cascade_scores(control, rerank, k=4)


def test_both_in_topk_counts_rankings_with_both_aspect_candidates():
    s = np.array([[0.9, 0.8, 0.1, 0.0], [0.9, 0.1, 0.8, 0.0]])
    np.testing.assert_array_equal(both_in_topk(_sc(s), 2), [1.0, 0.0])
    np.testing.assert_array_equal(both_in_topk(_sc(s), 3), [1.0, 1.0])


# ---------------------------------------------------------------- D0: told and inferred label-probe scorers

ASP = ("a", "b", "c")


def _d0_world(pairs_agree=True):
    """One episode, aspects (a, b, c) with 5 values, one-hot posteriors (the same probe output for both modalities).
    Anchor: a=0, b=0, c=0. p_a (row 1) shares a only, p_b (row 2) shares b only, negatives share neither."""
    lab = {h: np.full(30, 4) for h in ASP}
    for h in ASP:
        lab[h][0] = 0
    lab["a"][1], lab["b"][2] = 0, 0
    vals = np.array([1, 2, 3, 4])
    lab["a"][14:18], lab["a"][18:22] = vals, (vals if pairs_agree else np.roll(vals, 1))   # aspect-a pairs share a
    lab["b"][14:18], lab["b"][18:22] = 1, 2                                                 # ... and differ on b
    lab["b"][22:26], lab["b"][26:30] = vals, (vals if pairs_agree else np.roll(vals, 1))   # aspect-b pairs share b
    lab["a"][22:26], lab["a"][26:30] = 1, 2                                                 # ... and differ on a
    lab["c"][[*range(14, 18), *range(22, 26)]], lab["c"][[*range(18, 22), *range(26, 30)]] = 1, 2
    post = {}
    for h in ASP:
        p = np.eye(5)[lab[h]]
        post[h] = {"img": p, "txt": p.copy()}
    return post


def test_aspect_deltas_point_at_the_conditioned_aspect():
    post, ep = _d0_world(), _ep()
    np.testing.assert_allclose(aspect_deltas(post, ep, "a", ASP), [[1.0, -1.0, 0.0]])
    np.testing.assert_allclose(aspect_deltas(post, ep, "b", ASP), [[-1.0, 1.0, 0.0]])


def test_told_uses_the_per_episode_aspect():
    post, ep = _d0_world(), _ep()
    told = per_anchor(told_scores(post, ep, np.array([0]), np.array([1]), ASP))
    assert told["r1"][0] == 1.0 and told["gain"][0] == 1.0
    swapped = per_anchor(told_scores(post, ep, np.array([1]), np.array([0]), ASP))
    assert swapped["gain"][0] == -1.0


def test_inferred_hard_and_soft_recover_the_aspect():
    post, ep = _d0_world(), _ep()
    for mode in ("hard", "soft"):
        scores, info = inferred_scores(post, ep, mode, ASP)
        assert per_anchor(scores)["gain"][0] == 1.0
        assert info["a"]["weights"].argmax(axis=1)[0] == 0 and info["b"]["weights"].argmax(axis=1)[0] == 1
        assert not info["a"]["fallback"].any()


def test_soft_falls_back_to_uniform_when_no_aspect_is_shown():
    post, ep = _d0_world(pairs_agree=False), _ep()
    w, fallback = inferred_weights(post, ep, "a", "soft", ASP)
    assert fallback.all()
    np.testing.assert_allclose(w, [[1 / 3, 1 / 3, 1 / 3]])
    with pytest.raises(ValueError):
        inferred_weights(post, ep, "a", "median", ASP)


# ---------------------------------------------------------------- decision rule (DECISION_RULE.md)

from src.eval.aspect_quick_checks import (  # noqa: E402
    CONFIG_ORDER, config_passes, d0_reading, decision_row, n1_stop, n2_reading,
)


def _r(point, lo, hi):
    return {"point": point, "ci95": [lo, hi]}


def test_d0_reading_threshold_is_half_of_told_gain():
    told = _r(10.0, 9.0, 11.0)
    assert d0_reading(told, {"hard": 5.0, "soft": 4.0})["reading"] == "close"       # exactly half counts as close
    assert d0_reading(told, {"hard": 4.99, "soft": 4.0})["reading"] == "far"
    r = d0_reading(told, {"hard": 3.0, "soft": 6.0})
    assert r["variant"] == "soft" and r["gain"] == 6.0 and r["threshold"] == 5.0 and r["reading"] == "close"
    assert d0_reading(told, {"hard": 4.0, "soft": 4.0})["variant"] == "hard"        # ties go to hard


def test_d0_reading_is_unreadable_when_told_has_no_reliable_gain():
    assert d0_reading(_r(1.0, -0.2, 2.0), {"hard": 0.9, "soft": 0.8})["reading"] == "unreadable"
    assert d0_reading(_r(1.0, 0.0, 2.0), {"hard": 0.9, "soft": 0.8})["reading"] == "unreadable"


def test_config_passes_needs_both_lower_bounds_above_zero():
    assert config_passes(_r(0.5, 0.1, 0.9), _r(0.4, 0.05, 0.8))
    assert not config_passes(_r(0.5, 0.1, 0.9), _r(0.4, 0.0, 0.8))
    assert not config_passes(_r(0.5, -0.1, 0.9), _r(0.4, 0.05, 0.8))


def test_n1_stop_rules():
    assert n1_stop(0.3, 26.0, 25.92) == {"stop": False, "reasons": []}
    assert n1_stop(0.0, 26.0, 25.92)["stop"]                     # gain no more than the current rule
    assert n1_stop(0.3, 25.0, 25.92)["stop"]                     # either rate below cosine's
    assert len(n1_stop(-0.1, 20.0, 25.92)["reasons"]) == 2


def test_n2_reading():
    assert n2_reading(9.99, _r(0.2, 0.01, 0.4)) == {"rare": True, "no_gain": False}
    assert n2_reading(10.0, _r(0.2, 0.0, 0.4)) == {"rare": False, "no_gain": True}


def _cfg(name, passes, m_r=0.0, m_g=0.0):
    return {"name": name, "passes": passes, "m_r": m_r, "m_g": m_g}


def test_decision_row_order_and_pick():
    configs = [_cfg("N1-nested-A3", True, 0.3, 0.2), _cfg("N1-nested-C0", True, 0.5, 0.2),
               _cfg("N2-2-agree", True, 0.1, 0.9)]
    d = decision_row(configs, "far")                              # row 1 wins over any D0 reading
    assert d["row"] == 1 and d["config"] == "N1-nested-A3"        # min margins 0.2, 0.2, 0.1: tie -> earlier
    none = [_cfg("N1-nested-A3", False), _cfg("N2-2-agree", False)]
    assert decision_row(none, "close")["row"] == 2
    assert decision_row(none, "far")["row"] == 3
    assert decision_row(none, "unreadable")["row"] is None
    with pytest.raises(ValueError):
        decision_row(none, "maybe")
    with pytest.raises(ValueError):
        decision_row([_cfg("N2-3-agree", False), _cfg("N1-nested-A3", False)], "far")   # out of CONFIG_ORDER
    with pytest.raises(ValueError):
        decision_row([_cfg("N3-nested-A3", False)], "far")                               # unknown name
    assert CONFIG_ORDER[0] == "N1-nested-A3" and len(CONFIG_ORDER) == 9
