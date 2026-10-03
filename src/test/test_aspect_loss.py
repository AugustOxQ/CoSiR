import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.eval.aspect_episodes import build_aspect_episodes
from src.train.aspect_loss import aspect_episode_loss, aspect_episode_scores
from src.train.train_factors import R3_CONFIG, train_factors


def _tiny():
    rng = np.random.default_rng(0)
    n_paint = 600
    groups = np.repeat(np.arange(n_paint), 2)
    a = rng.integers(0, 6, 2 * n_paint)
    b = np.repeat(rng.integers(0, 6, n_paint), 2)
    proto = rng.normal(size=(12, 16)).astype(np.float32)
    img = proto[a] + proto[6 + b] + 0.3 * rng.normal(size=(2 * n_paint, 16)).astype(np.float32)
    txt = proto[a] + proto[6 + b] + 0.3 * rng.normal(size=(2 * n_paint, 16)).astype(np.float32)
    bank = build_aspect_episodes({"a": a, "b": b}, groups, np.arange(2 * n_paint), "a", "b", 256, seed=1,
                                 min_paintings=5)
    nbr = np.arange(2 * n_paint) ^ 1                                  # pair rows of the same painting
    graph = csr_matrix((np.ones(2 * n_paint), (np.arange(2 * n_paint), nbr)), shape=(2 * n_paint, 2 * n_paint))
    return img, txt, groups, bank, graph


def test_loss_is_finite_and_prefers_correct_target():
    s_good = {("a", d): torch.tensor([[2.0] + [0.0] * 12]) for d in ("i2t", "t2i")}
    s_good |= {("b", d): torch.tensor([[0.0, 2.0] + [0.0] * 11]) for d in ("i2t", "t2i")}
    s_bad = {("a", d): torch.tensor([[0.0, 2.0] + [0.0] * 11]) for d in ("i2t", "t2i")}
    s_bad |= {("b", d): torch.tensor([[2.0] + [0.0] * 12]) for d in ("i2t", "t2i")}
    zero = torch.tensor(0.0)
    assert aspect_episode_loss(s_good, zero, 1.0) < aspect_episode_loss(s_bad, zero, 1.0)


def test_train_factors_with_aspect_bank_runs_and_validates():
    img, txt, groups, bank, graph = _tiny()
    import dataclasses
    cfg = dataclasses.replace(R3_CONFIG, epochs=5, batch_size=64, lambda_aspect=1.0, aspect_episodes_per_step=8,
                              num_factors=8, painting_batches=True)
    history = {}
    model, ic, tc = train_factors(img, txt, graph, cfg, device="cpu", group_ids=groups, aspect_bank=bank,
                                  history=history, log_every=1)
    assert np.isfinite(ic).all() and "aspect_loss" in history and len(history["aspect_loss"]) == 5


def test_aspect_bank_required_iff_lambda():
    img, txt, groups, bank, graph = _tiny()
    import dataclasses, pytest
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, dataclasses.replace(R3_CONFIG, epochs=1, lambda_aspect=1.0), device="cpu",
                      group_ids=groups)
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, dataclasses.replace(R3_CONFIG, epochs=1), device="cpu", group_ids=groups,
                      aspect_bank=bank)


def test_fixed_tau_stays_at_step0_value():
    import dataclasses
    img, txt, groups, bank, graph = _tiny()
    cfg = dataclasses.replace(R3_CONFIG, epochs=6, batch_size=64, lambda_aspect=1.0, aspect_episodes_per_step=8,
                              num_factors=8, painting_batches=True, aspect_tau_fixed=True)
    history = {}
    train_factors(img, txt, graph, cfg, device="cpu", group_ids=groups, aspect_bank=bank, history=history,
                  log_every=1)
    taus = history["tau"]
    assert len(taus) == 6 and all(t == taus[0] for t in taus)


def test_learned_tau_moves_by_default():
    import dataclasses
    img, txt, groups, bank, graph = _tiny()
    cfg = dataclasses.replace(R3_CONFIG, epochs=6, batch_size=64, lambda_aspect=1.0, aspect_episodes_per_step=8,
                              num_factors=8, painting_batches=True)
    assert cfg.aspect_tau_fixed is False
    history = {}
    train_factors(img, txt, graph, cfg, device="cpu", group_ids=groups, aspect_bank=bank, history=history,
                  log_every=1)
    assert len(set(history["tau"])) > 1


def test_fixed_tau_needs_the_aspect_loss():
    import dataclasses, pytest
    img, txt, groups, bank, graph = _tiny()
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, dataclasses.replace(R3_CONFIG, epochs=1, aspect_tau_fixed=True),
                      device="cpu", group_ids=groups)
