import dataclasses

import numpy as np
import pytest

from src.eval.aspect_episodes import build_aspect_episodes
from src.eval.aspect_metrics import per_anchor
from src.eval.aspect_scorers import EvalInputs
from src.eval import pair_metric_baselines as B


def _world(n_paintings=2500, seed=0):
    """Each aspect occupies its own block of feature dimensions (a: 0-23, b: 24-47); each value is a pattern over
    its block. If the aspects are instead mixed across all dimensions, raw-feature pair rules show no gain
    (controller's simulation on 2026-10-03: 0.00 mixed vs 0.70 to 0.95 block); that is exactly what raw CLIP does."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    a = rng.integers(0, 6, 2 * n_paintings)
    b = np.repeat(rng.integers(0, 6, n_paintings), 2)
    pattern_a = rng.normal(size=(6, 24)).astype(np.float32)
    pattern_b = rng.normal(size=(6, 24)).astype(np.float32)
    x = np.concatenate([pattern_a[a], pattern_b[b]], axis=1)
    img = x + 0.4 * rng.normal(size=x.shape).astype(np.float32)
    txt = x + 0.4 * rng.normal(size=x.shape).astype(np.float32)
    return {"a": a, "b": b}, groups, img, txt


@pytest.fixture(scope="module")
def setup():
    labels, groups, img, txt = _world()
    ep = build_aspect_episodes(labels, groups, np.arange(len(groups)), "a", "b", 200, seed=1)
    inputs = EvalInputs(img, txt)
    basis = B.fit_pca_basis(inputs.img, inputs.txt, r=16)
    basis.scaler = B.fit_pair_scaler(inputs.img, inputs.txt)
    return ep, inputs, basis


@pytest.mark.parametrize("name", ["diag", "diag_relu", "kissme", "rca", "xing", "wang", "probe", "tip"])
def test_pair_baselines_use_the_condition(setup, name):
    ep, inputs, basis = setup
    term = {"diag": lambda: B.diag_agreement_term(inputs, ep, relu=False),
            "diag_relu": lambda: B.diag_agreement_term(inputs, ep, relu=True),
            "kissme": lambda: B.kissme_term(inputs, ep, basis),
            "rca": lambda: B.rca_term(inputs, ep, basis),
            "xing": lambda: B.xing_term(inputs, ep, basis),
            "wang": lambda: B.wang_term(inputs, ep),
            "probe": lambda: B.pair_probe_term(inputs, ep, basis.scaler),
            "tip": lambda: B.tip_adapter_term(inputs, ep)}[name]()
    assert term["a"]["i2t"].shape == (200, 13) and np.isfinite(term["a"]["i2t"]).all()
    assert per_anchor(term)["gain"].mean() > 0.05                 # the condition changes the ranking


def test_value_prototype_has_little_gain_on_aspect_episodes(setup):
    ep, inputs, _ = setup
    assert per_anchor(B.value_prototype_term(inputs, ep))["gain"].mean() < 0.05


def test_bilinear_is_condition_dependent_but_weak_under_value_disjoint_supports(setup):
    """After per-modality centring and whitening, the 6 value patterns of an aspect form a near-regular simplex, so
    the value-level quadratic form q'^T (sum_v P_v P_v^T) c' gives the shared value only a small constant boost and is
    close to blind when supports never show the anchor's value (measured gain -0.019 at r=16, 0.116 at r=8).
    The positive control gives the supports the anchor's own value, where the same code must show a clear gain."""
    ep, inputs, basis = setup
    term = B.bilinear_agreement_term(inputs, ep, basis)
    assert term["a"]["i2t"].shape == (200, 13) and np.isfinite(term["a"]["i2t"]).all()
    assert not np.allclose(term["a"]["i2t"], term["b"]["i2t"])
    assert abs(per_anchor(term)["gain"].mean()) < 0.05
    labels, groups, _, _ = _world()
    rng = np.random.default_rng(3)
    pick = lambda key: np.stack([rng.choice(np.flatnonzero(labels[key] == labels[key][i]), 4) for i in ep.anchor])  # noqa: E731
    shared_a, shared_b = pick("a"), pick("b")
    ep2 = dataclasses.replace(ep, pairs_a_img=shared_a, pairs_a_txt=shared_a, pairs_b_img=shared_b, pairs_b_txt=shared_b)
    assert per_anchor(B.bilinear_agreement_term(inputs, ep2, basis))["gain"].mean() > 0.05


def _standardized_diag(inputs, ep, scaler):
    """The Rocchio rule in the probe's own standardized space: w = mean over S of z - mean over C of z, with
    z = (x*y - mean) / std, and score = w . standardize(q*c). This is the closed-form rule a probe could collapse to."""
    out = {}
    for c in ("a", "b"):
        si, st, ci, ct, _ = ep.condition(c)
        w = (((inputs.img[si] * inputs.txt[st] - scaler.mean) / scaler.std).mean(1)
             - ((inputs.img[ci] * inputs.txt[ct] - scaler.mean) / scaler.std).mean(1))
        out[c] = {}
        for d in ("i2t", "t2i"):
            q, cand = ((inputs.img[ep.anchor], inputs.txt[ep.candidates]) if d == "i2t"
                       else (inputs.txt[ep.anchor], inputs.img[ep.candidates]))
            out[c][d] = (((q[:, None] * cand - scaler.mean) / scaler.std) * w[:, None]).sum(-1)
    return out


def test_pair_probe_is_not_the_diag_rule(setup):
    """Ranking differs from both closed-form agreement rules: the fraction of (condition, direction, episode) rows
    whose argmax differs is > 0 against the raw diag rule and > 0.02 against the standardized diag (Rocchio) rule.
    Measured on this world (2026-10-03): 0.32 against raw diag and 0.081 against standardized diag (0.065 to 0.090
    per condition and direction, 800 rankings); a copy of either rule gives exactly 0, and 0.02 is a quarter of the
    measured value. Scale: mean |term| on the first 50 episodes is stable (< 10% change) whether the probe is fit on
    50 episodes or on all 200, because the loss is summed per episode."""
    ep, inputs, basis = setup
    probe = B.pair_probe_term(inputs, ep, basis.scaler)
    diag = B.diag_agreement_term(inputs, ep, relu=False)
    differs = np.mean([(probe[c][d].argmax(1) != diag[c][d].argmax(1)).mean() for c in probe for d in probe[c]])
    assert differs > 0
    rocchio = _standardized_diag(inputs, ep, basis.scaler)
    differs_std = np.mean([(probe[c][d].argmax(1) != rocchio[c][d].argmax(1)).mean() for c in probe for d in probe[c]])
    assert differs_std > 0.02
    sub = dataclasses.replace(ep, **{f.name: getattr(ep, f.name)[:50] for f in dataclasses.fields(ep)})
    small = B.pair_probe_term(inputs, sub, basis.scaler)
    full_abs = np.mean([np.abs(probe[c][d][:50]).mean() for c in probe for d in probe[c]])
    small_abs = np.mean([np.abs(small[c][d]).mean() for c in small for d in small[c]])
    assert abs(small_abs - full_abs) / full_abs < 0.10
