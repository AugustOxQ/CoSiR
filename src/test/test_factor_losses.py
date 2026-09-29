"""Behavior tests for factor-discovery objectives on controlled synthetic codes."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.model.factors import SharedFactorEncoder
from src.train.factors import (
    anti_split_penalty,
    graph_neighbor_consistency_loss,
    paired_agreement_loss,
    reconstruction_loss,
    sparsity_penalty,
    usage_balance_penalty,
)


def test_reconstruction_loss_falls_when_encoder_learns_round_trip():
    torch.manual_seed(42)
    encoder = SharedFactorEncoder(feature_dim=6, num_factors=12, dropout=0.0)
    img_feat = torch.randn(32, 6)
    txt_feat = torch.randn(32, 6)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.03)

    initial = reconstruction_loss(encoder, img_feat, txt_feat).item()
    for _ in range(50):
        loss = reconstruction_loss(encoder, img_feat, txt_feat)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = reconstruction_loss(encoder, img_feat, txt_feat).item()

    assert final < 0.7 * initial, (initial, final)


def test_paired_agreement_prefers_true_rows_over_shuffled_rows():
    img_codes = torch.eye(8)
    txt_codes = 0.9 * img_codes + 0.1 * torch.roll(img_codes, 1, dims=1)
    shuffled = txt_codes[torch.randperm(8, generator=torch.Generator().manual_seed(42))]

    matched_loss = paired_agreement_loss(img_codes, txt_codes)
    shuffled_loss = paired_agreement_loss(img_codes, shuffled)

    assert matched_loss.ndim == 0
    assert matched_loss.item() < 0.1
    assert shuffled_loss.item() > matched_loss.item() + 0.5


def test_graph_consistency_prefers_codes_on_actual_neighbor_edges():
    # Four disconnected pairs; each true neighbor pair shares a unique factor.
    left = np.arange(0, 8, 2)
    right = left + 1
    graph = csr_matrix(
        (np.ones(8), (np.r_[left, right], np.r_[right, left])), shape=(8, 8)
    )
    codes = torch.eye(4).repeat_interleave(2, dim=0).requires_grad_()
    sample_idx = np.array([7, 0, 4, 1, 6, 2, 5, 3])
    permuted = codes[[0, 2, 4, 6, 1, 3, 5, 7]]

    aligned_loss = graph_neighbor_consistency_loss(codes, graph, sample_idx)
    permuted_loss = graph_neighbor_consistency_loss(permuted, graph, sample_idx)
    aligned_loss.backward()

    assert aligned_loss.ndim == 0
    assert permuted_loss.item() > aligned_loss.item() + 0.5
    assert codes.grad is not None and codes.grad.abs().sum() > 0


def test_sparsity_penalty_distinguishes_dense_from_sparse_equal_amplitudes():
    dense = torch.full((8, 8), 0.8)
    sparse = torch.eye(8) * 0.8

    dense_loss = sparsity_penalty(dense)
    sparse_loss = sparsity_penalty(sparse)

    torch.testing.assert_close(dense_loss, torch.tensor(6.4))
    torch.testing.assert_close(sparse_loss, torch.tensor(0.8))


def test_anti_split_penalty_detects_modality_private_factors():
    # Each factor fires on two of 16 rows, with small background activation.
    # The split fixtures retain the same image and text sample structure but
    # systematically suppress opposite halves of the dictionary by modality.
    base = torch.eye(8).repeat(2, 1) + 0.05
    shared_img = base
    shared_txt = 0.9 * base
    split_img = (base * torch.tensor([1.0] * 4 + [0.1] * 4)).requires_grad_()
    split_txt = (base * torch.tensor([0.1] * 4 + [1.0] * 4)).requires_grad_()
    moderate_img = base * torch.tensor([1.0] * 4 + [0.3] * 4)
    moderate_txt = base * torch.tensor([0.3] * 4 + [1.0] * 4)

    shared_loss = anti_split_penalty(shared_img, shared_txt)
    split_loss = anti_split_penalty(split_img, split_txt)
    moderate_loss = anti_split_penalty(moderate_img, moderate_txt)
    split_loss.backward()

    print(
        f"anti_split shared={shared_loss.item():.6f} "
        f"split={split_loss.item():.6f} moderate={moderate_loss.item():.6f}"
    )
    assert shared_loss.item() < 0.1
    assert split_loss.item() > 0.75
    assert split_loss.item() > shared_loss.item() + 0.65
    assert moderate_loss.item() > 0.45
    assert split_img.grad is not None and split_img.grad.abs().sum() > 0
    assert split_txt.grad is not None and split_txt.grad.abs().sum() > 0
    # Gradient descent reduces dominant activations and raises suppressed ones.
    assert (split_img.grad[:, :4] > 0).all()
    assert (split_txt.grad[:, :4] < 0).all()


def test_usage_balance_penalty_detects_two_factor_mass_concentration():
    uniform = torch.ones(16, 8)
    concentrated = torch.tensor([[7.0, 7.0] + [0.1] * 6] * 16).requires_grad_()

    uniform_loss = usage_balance_penalty(uniform, uniform)
    concentrated_loss = usage_balance_penalty(concentrated, concentrated)
    concentrated_loss.backward()

    assert uniform_loss.item() == pytest.approx(-np.log(8), abs=1e-5)
    assert concentrated_loss.item() > uniform_loss.item() + 1.1
    assert concentrated.grad is not None and concentrated.grad.abs().sum() > 0
