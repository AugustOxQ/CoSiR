"""Behavior tests for independent image and text factor projections."""

import torch
from torch.nn import functional as F

from src.model.factors import SharedFactorEncoder


def test_factor_encoder_shapes_and_nonnegative_codes():
    torch.manual_seed(42)
    encoder = SharedFactorEncoder(feature_dim=12, num_factors=8, dropout=0.0)
    img_feat = torch.randn(5, 12)
    txt_feat = torch.randn(5, 12)

    img_codes = encoder.encode_image(img_feat)
    txt_codes = encoder.encode_text(txt_feat)

    assert img_codes.shape == (5, 8)
    assert txt_codes.shape == (5, 8)
    assert torch.all(img_codes >= 0)
    assert torch.all(txt_codes >= 0)
    assert encoder.reconstruct_image(img_codes).shape == (5, 12)
    assert encoder.reconstruct_text(txt_codes).shape == (5, 12)


def test_factor_encoder_round_trips_are_finite_and_nontrivial():
    torch.manual_seed(42)
    encoder = SharedFactorEncoder(feature_dim=12, num_factors=8, dropout=0.0)

    for encode, reconstruct in (
        (encoder.encode_image, encoder.reconstruct_image),
        (encoder.encode_text, encoder.reconstruct_text),
    ):
        reconstructed = reconstruct(encode(torch.randn(16, 12)))
        assert torch.isfinite(reconstructed).all()
        assert torch.count_nonzero(reconstructed) > 0


def test_reconstruction_losses_reach_each_modality_and_all_parameters():
    torch.manual_seed(42)
    encoder = SharedFactorEncoder(feature_dim=12, num_factors=8, dropout=0.0)
    img_feat = torch.randn(16, 12)
    txt_feat = torch.randn(16, 12)
    img_loss = F.mse_loss(
        encoder.reconstruct_image(encoder.encode_image(img_feat)), img_feat
    )
    txt_loss = F.mse_loss(
        encoder.reconstruct_text(encoder.encode_text(txt_feat)), txt_feat
    )

    img_grads = torch.autograd.grad(
        img_loss, tuple(encoder.image_encoder.parameters())
        + tuple(encoder.image_decoder.parameters()), retain_graph=True
    )
    txt_grads = torch.autograd.grad(
        txt_loss, tuple(encoder.text_encoder.parameters())
        + tuple(encoder.text_decoder.parameters()), retain_graph=True
    )
    for grad in (*img_grads, *txt_grads):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0

    (img_loss + txt_loss).backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_image_and_text_heads_have_independent_parameters():
    torch.manual_seed(42)
    encoder = SharedFactorEncoder(feature_dim=12, num_factors=8, dropout=0.0)
    image_parameters = tuple(encoder.image_encoder.parameters())
    text_parameters = tuple(encoder.text_encoder.parameters())
    assert {id(parameter) for parameter in image_parameters}.isdisjoint(
        id(parameter) for parameter in text_parameters
    )
    assert encoder.image_encoder.weight.data_ptr() != encoder.text_encoder.weight.data_ptr()
    assert encoder.image_decoder.weight.data_ptr() != encoder.text_decoder.weight.data_ptr()

    features = torch.randn(5, 12)
    original_text_codes = encoder.encode_text(features).clone()
    with torch.no_grad():
        encoder.image_encoder.weight.add_(1.0)
    torch.testing.assert_close(encoder.encode_text(features), original_text_codes)
