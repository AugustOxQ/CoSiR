import torch

from src.model.cosirmodel import CoSiRModel


def test_free_vector_mode_has_no_prototype_bank():
    model = CoSiRModel(conditioning_mode="free_vector", label_dim=16)
    assert model.prototype_bank is None


def test_prototype_pooled_mode_constructs_prototype_bank_with_right_dims():
    model = CoSiRModel(
        conditioning_mode="prototype_pooled",
        label_dim=16,
        num_prototypes=8,
    )
    assert model.prototype_bank is not None
    assert model.prototype_bank.num_prototypes == 8
    assert model.prototype_bank.condition_dim == 16
    assert model.prototype_bank.query_proj.in_features == model.feature_dim


def test_invalid_conditioning_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        CoSiRModel(conditioning_mode="not_a_real_mode", label_dim=16)
