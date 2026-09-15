import math
import torch
import pytest

from src.model.prototype_bank import PrototypeBank


def test_prototype_bank_forward_shape():
    bank = PrototypeBank(num_prototypes=8, condition_dim=16, query_dim=512)
    q = torch.randn(4, 512)
    out = bank(q)
    assert out.shape == (4, 16)


def test_prototype_bank_gradient_flows_to_keys_values_and_query_proj():
    bank = PrototypeBank(num_prototypes=8, condition_dim=16, query_dim=512)
    q = torch.randn(4, 512, requires_grad=True)
    out = bank(q)
    out.sum().backward()
    assert bank.keys.grad is not None and bank.keys.grad.abs().sum() > 0
    assert bank.values.grad is not None and bank.values.grad.abs().sum() > 0
    assert bank.query_proj.weight.grad is not None
    assert q.grad is not None


def test_seed_from_communities_overwrites_keys_and_values():
    bank = PrototypeBank(num_prototypes=4, condition_dim=3, query_dim=5)
    means = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bank.seed_from_communities(means)
    assert torch.allclose(bank.keys.data[:2], means)
    assert torch.allclose(bank.values.data[:2], means)


def test_seed_from_communities_raises_when_too_many_communities():
    bank = PrototypeBank(num_prototypes=2, condition_dim=3, query_dim=5)
    means = torch.randn(3, 3)
    with pytest.raises(ValueError):
        bank.seed_from_communities(means)


def test_usage_entropy_uniform_attention_near_max_entropy():
    bank = PrototypeBank(num_prototypes=8, condition_dim=16, query_dim=512)
    with torch.no_grad():
        bank.query_proj.weight.zero_()
        bank.query_proj.bias.zero_()
    q = torch.randn(4, 512)
    bank(q)
    assert bank.usage_entropy().item() == pytest.approx(math.log(8), abs=1e-4)
