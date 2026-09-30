"""Tests for the naive-rule condition-episode loss (factor-learning spec §5)."""

import math

import pytest
import torch

from src.train.factor_condition_loss import naive_episode_loss, naive_episode_scores


def _episode(zero_gap: bool = False):
    """Rows: 0 anchor, 1 support, 2 contrast, 3 positive, 4 negative. D = 2 features, F = 2 factors.

    Support pair code [2, 1], contrast pair code [0, 0.5] -> gap [2, 0.5] -> naive weights [0.8, 0.2].
    With zero_gap the contrast equals the support, so the weights are all zero.
    """
    img_feat = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    txt_feat = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6]])
    contrast = [2.0, 1.0] if zero_gap else [0.0, 0.5]
    img_codes = torch.tensor([[1.0, 1.0], [2.0, 1.0], contrast, [1.0, 0.0], [0.0, 1.0]])
    txt_codes = torch.tensor([[1.0, 1.0], [2.0, 1.0], contrast, [2.0, 0.0], [0.0, 2.0]])
    index = {"anchor": torch.tensor([0]), "supports": torch.tensor([[1]]), "contrasts": torch.tensor([[2]]),
             "candidates": torch.tensor([[3, 4]])}
    return img_feat, txt_feat, img_codes, txt_codes, index


MASK = torch.tensor([[True, False]])


def test_scores_match_hand_computation():
    # i2t: cos [0.6, 0.8] * 0.3 + factors [0.8*1*2, 0.2*1*2] = [1.78, 0.64]
    # t2i: cos [1, 0] * 0.3 + factors [0.8*1*1, 0.2*1*1] = [1.1, 0.2]
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    scores = naive_episode_scores(img_feat, txt_feat, img_codes, txt_codes, **index, beta=0.3)
    assert torch.allclose(scores["i2t"], torch.tensor([[1.78, 0.64]]), atol=1e-6)
    assert torch.allclose(scores["t2i"], torch.tensor([[1.1, 0.2]]), atol=1e-6)


def test_loss_matches_hand_computation_and_uses_tau():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    expected = 0.5 * (math.log1p(math.exp(0.64 - 1.78)) + math.log1p(math.exp(0.2 - 1.1)))
    loss = naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                              log_tau=torch.tensor(0.0))
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    halved = 0.5 * (math.log1p(math.exp((0.64 - 1.78) / 2)) + math.log1p(math.exp((0.2 - 1.1) / 2)))
    loss_tau2 = naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                                   log_tau=torch.tensor(math.log(2.0)))
    assert loss_tau2.item() == pytest.approx(halved, abs=1e-6)


def test_gradients_reach_codes_through_weights_and_scores():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    img_codes.requires_grad_(True)
    txt_codes.requires_grad_(True)
    naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                       log_tau=torch.tensor(0.0)).backward()
    assert img_codes.grad[1].abs().sum() > 0          # support row: through the naive weights
    assert img_codes.grad[2].abs().sum() > 0          # contrast row: through the naive weights
    assert txt_codes.grad[3].abs().sum() > 0          # positive candidate caption (i2t)
    assert img_codes.grad[3].abs().sum() > 0          # positive candidate image (t2i)


def test_all_zero_weights_fall_back_to_clip_only():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode(zero_gap=True)
    scores = naive_episode_scores(img_feat, txt_feat, img_codes, txt_codes, **index, beta=0.3)
    assert torch.allclose(scores["i2t"], torch.tensor([[0.18, 0.24]]), atol=1e-6)
    assert torch.allclose(scores["t2i"], torch.tensor([[0.3, 0.0]]), atol=1e-6)
    loss = naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index, positive_mask=MASK, beta=0.3,
                              log_tau=torch.tensor(0.0))
    assert torch.isfinite(loss)


def test_episode_without_positive_raises():
    img_feat, txt_feat, img_codes, txt_codes, index = _episode()
    with pytest.raises(ValueError, match="positive"):
        naive_episode_loss(img_feat, txt_feat, img_codes, txt_codes, **index,
                           positive_mask=torch.tensor([[False, False]]), beta=0.3, log_tau=torch.tensor(0.0))
