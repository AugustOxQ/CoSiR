"""Behavior tests for the per-factor condition set encoder."""

import pytest
import torch
from torch.nn import functional as F

from src.model.condition import ConditionEncoder


@pytest.mark.parametrize(
    ("num_factors", "num_support", "num_contrast"),
    [(1, 1, 0), (8, 3, 2), (32, 7, 5)],
)
def test_output_shape_and_nonnegativity(num_factors, num_support, num_contrast):
    torch.manual_seed(42)
    encoder = ConditionEncoder()
    support_img = torch.rand(num_support, num_factors)
    support_txt = torch.rand(num_support, num_factors)
    contrast_img = torch.rand(num_contrast, num_factors) if num_contrast else None
    contrast_txt = torch.rand(num_contrast, num_factors) if num_contrast else None

    weights = encoder(support_img, support_txt, contrast_img, contrast_txt)

    assert weights.shape == (num_factors,)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)


def test_pair_permutations_leave_output_exactly_unchanged():
    torch.manual_seed(42)
    encoder = ConditionEncoder()
    support_img = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    support_txt = support_img + 2
    contrast_img = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    contrast_txt = contrast_img + 4
    original = encoder(support_img, support_txt, contrast_img, contrast_txt)

    support_order = torch.tensor([2, 0, 3, 1])
    contrast_order = torch.tensor([1, 2, 0])
    assert torch.equal(
        original,
        encoder(
            support_img[support_order],
            support_txt[support_order],
            contrast_img,
            contrast_txt,
        ),
    )
    assert torch.equal(
        original,
        encoder(
            support_img,
            support_txt,
            contrast_img[contrast_order],
            contrast_txt[contrast_order],
        ),
    )


def test_one_support_pair_without_contrast_is_finite_and_nonnegative():
    encoder = ConditionEncoder()
    weights = encoder(
        torch.tensor([[0.0, 4.0, 0.0]]),
        torch.tensor([[0.0, 3.0, 0.0]]),
    )

    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)


def test_zero_contrast_is_distinct_from_no_contrast():
    encoder = ConditionEncoder(hidden_dim=2)
    with torch.no_grad():
        encoder.head[0].weight.zero_()
        encoder.head[0].bias.zero_()
        encoder.head[0].weight[0, 2] = 1.0
        encoder.head[2].weight.zero_()
        encoder.head[2].bias.zero_()
        encoder.head[2].weight[0, 0] = 1.0

    support = torch.zeros(1, 4)
    zero_contrast = torch.zeros(2, 4)
    absent = encoder(support, support)
    present = encoder(support, support, zero_contrast, zero_contrast)

    assert torch.all(present > absent)
    present.sum().backward()
    assert encoder.has_contrast_flag.grad is not None
    assert encoder.has_contrast_flag.grad.item() > 0


def test_negative_raw_scores_remain_positive_with_gradient():
    encoder = ConditionEncoder(hidden_dim=2)
    with torch.no_grad():
        encoder.head[0].weight.zero_()
        encoder.head[0].bias.zero_()
        encoder.head[2].weight.zero_()
        encoder.head[2].bias.fill_(-3.0)

    support = torch.zeros(1, 3)
    weights = encoder(support, support)
    assert torch.all(weights > 0)
    weights.sum().backward()
    assert encoder.head[2].bias.grad.item() > 0


def test_planted_factor_recovery_generalizes_to_unseen_factor_indices():
    """Validation-only auxiliary objective; stage (d) uses swap-loss training."""
    torch.manual_seed(42)
    num_factors = 8
    encoder = ConditionEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.05)

    def fixture(true_factor_idx):
        support_img = torch.full((3, num_factors), 0.1)
        support_txt = torch.full((3, num_factors), 0.1)
        support_img[:, true_factor_idx] = 5.0
        support_txt[:, true_factor_idx] = 4.0
        contrast_img = torch.full((2, num_factors), 0.2)
        contrast_txt = torch.full((2, num_factors), 0.2)
        return support_img, support_txt, contrast_img, contrast_txt

    train_indices = (0, 2, 4, 6)
    held_out_indices = (1, 3, 5, 7)
    for _ in range(100):
        optimizer.zero_grad()
        # Cross-entropy on w(c) verifies the public output is itself trainable.
        loss = sum(
            F.cross_entropy(
                encoder(*fixture(idx)).unsqueeze(0), torch.tensor([idx])
            )
            for idx in train_indices
        ) / len(train_indices)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        correct = sum(
            encoder(*fixture(idx)).argmax().item() == idx
            for idx in held_out_indices
        )
    assert correct == len(held_out_indices)  # 100% versus 1/8 = 12.5% chance.
