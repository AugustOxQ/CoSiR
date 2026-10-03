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
    return ep, inputs, basis


@pytest.mark.parametrize("name", ["diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip"])
def test_pair_baselines_use_the_condition(setup, name):
    ep, inputs, basis = setup
    term = {"diag": lambda: B.diag_agreement_term(inputs, ep, relu=False),
            "diag_relu": lambda: B.diag_agreement_term(inputs, ep, relu=True),
            "bilinear": lambda: B.bilinear_agreement_term(inputs, ep, basis),
            "kissme": lambda: B.kissme_term(inputs, ep, basis),
            "rca": lambda: B.rca_term(inputs, ep, basis),
            "xing": lambda: B.xing_term(inputs, ep, basis),
            "wang": lambda: B.wang_term(inputs, ep),
            "probe": lambda: B.pair_probe_term(inputs, ep),
            "tip": lambda: B.tip_adapter_term(inputs, ep)}[name]()
    assert term["a"]["i2t"].shape == (200, 13) and np.isfinite(term["a"]["i2t"]).all()
    assert per_anchor(term)["gain"].mean() > 0.05                 # the condition changes the ranking


def test_value_prototype_has_little_gain_on_aspect_episodes(setup):
    ep, inputs, _ = setup
    assert per_anchor(B.value_prototype_term(inputs, ep))["gain"].mean() < 0.05
