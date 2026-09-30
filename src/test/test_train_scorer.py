import dataclasses

import numpy as np
import pytest
import torch

from src.train.condition_sources import CommunitySource, FactorComboSource
from src.train.train_scorer import (
    ScorerTrainingConfig, load_scorer_checkpoint, multi_positive_nce, save_scorer_checkpoint, train_scorer,
    episode_logits,
)
from src.train.condition_episodes import mine_condition_episodes, pair_feature_units


def test_multi_positive_nce_matches_cross_entropy_for_one_positive_and_is_zero_when_all_positive():
    logits = torch.randn(5, 7)
    mask = torch.zeros(5, 7, dtype=torch.bool)
    mask[:, 0] = True
    expected = torch.nn.functional.cross_entropy(logits, torch.zeros(5, dtype=torch.long))
    assert torch.allclose(multi_positive_nce(logits, mask), expected, atol=1e-6)
    assert multi_positive_nce(logits, torch.ones(5, 7, dtype=torch.bool)).abs() < 1e-6
    with pytest.raises(ValueError):
        multi_positive_nce(logits, torch.zeros(5, 7, dtype=torch.bool))


def _noisy_factor_world(n=3000, seed=0):
    """Naive is misled: each group's true factor has a small consistent gap; factor 7 is large random noise."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(6), n // 6)
    codes = np.zeros((n, 8), dtype=np.float32)
    codes[np.arange(n), labels] = 0.3                               # consistent, small true signal
    noisy = rng.random(n) < 0.5
    codes[noisy, 7] = rng.exponential(5.0, size=noisy.sum())        # large, inconsistent, uninformative
    img = rng.normal(size=(n, 8)).astype(np.float32)                # CLIP term carries no signal
    txt = rng.normal(size=(n, 8)).astype(np.float32)
    keys = np.arange(n)
    return labels, codes, img, txt, keys


def _r1(scorer, source, img, txt, codes, keys, seed):
    ep = mine_condition_episodes(source, pair_feature_units(img, txt), keys, 256, np.random.default_rng(seed))
    with torch.no_grad():
        logits = episode_logits(scorer, ep.anchor, ep.supports, ep.contrasts, ep.candidates, img, txt, codes, codes, "cpu")
    hits = [float(ep.positive_mask[np.arange(256), logits[d].argmax(dim=1).numpy()].mean()) for d in ("i2t", "t2i")]
    return float(np.mean(hits))


def test_training_beats_the_naive_start_when_naive_is_misled():
    labels, codes, img, txt, keys = _noisy_factor_world()
    source = CommunitySource(labels, np.arange(len(labels)), min_group_rows=50)
    scale = torch.as_tensor(codes.std(axis=0) + 1e-3)
    # beta ~ 0 so the (uninformative) CLIP term cannot be the thing training improves; the interface must.
    config = ScorerTrainingConfig(steps=400, batch_episodes=32, lr=3e-3, beta_init=1e-3, hard_pool=512, seed=0)
    naive, _ = train_scorer(source, img, txt, codes, codes, keys, scale, dataclasses.replace(config, steps=0),
                            device="cpu")
    trained, history = train_scorer(source, img, txt, codes, codes, keys, scale, config, device="cpu")
    assert np.isfinite(history["loss"]).all()
    assert _r1(trained, source, img, txt, codes, keys, 99) > _r1(naive, source, img, txt, codes, keys, 99) + 0.10


def test_swap_training_runs_and_non_swap_source_rejects_swap():
    rng = np.random.default_rng(1)
    codes = np.maximum(0.0, rng.normal(size=(4000, 8)) - 0.2).astype(np.float32)
    img = rng.normal(size=(4000, 8)).astype(np.float32)
    txt = rng.normal(size=(4000, 8)).astype(np.float32)
    keys = np.arange(4000)
    scale = torch.as_tensor(codes.std(axis=0) + 1e-3)
    src = FactorComboSource(codes, np.arange(4000), min_group_rows=50)
    config = ScorerTrainingConfig(steps=5, batch_episodes=8, swap=True, hard_pool=256)
    _, history = train_scorer(src, img, txt, codes, codes, keys, scale, config, device="cpu")
    assert len(history["loss_swap"]) > 0 and np.isfinite(history["loss_swap"]).all()
    community = CommunitySource(np.repeat(np.arange(4), 1000), np.arange(4000), min_group_rows=50)
    with pytest.raises(ValueError):
        train_scorer(community, img, txt, codes, codes, keys, scale, config, device="cpu")


def test_checkpoint_round_trip_and_determinism(tmp_path):
    labels, codes, img, txt, keys = _noisy_factor_world(n=1200)
    source = CommunitySource(labels, np.arange(1200), min_group_rows=50)
    scale = torch.as_tensor(codes.std(axis=0) + 1e-3)
    config = ScorerTrainingConfig(steps=20, batch_episodes=8, hard_pool=256, seed=3)
    a, hist_a = train_scorer(source, img, txt, codes, codes, keys, scale, config, device="cpu")
    b, hist_b = train_scorer(source, img, txt, codes, codes, keys, scale, config, device="cpu")
    assert hist_a["loss"] == hist_b["loss"]
    save_scorer_checkpoint(a, config, tmp_path / "s.pt")
    loaded, loaded_config = load_scorer_checkpoint(tmp_path / "s.pt")
    assert loaded_config == config
    for (name, p), (_, q) in zip(a.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(p, q), name
