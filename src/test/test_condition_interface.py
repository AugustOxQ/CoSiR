import torch

from src.model.condition_interface import (
    EVIDENCE_FEATURES, ConditionalScorer, ResidualConditionInterface, factor_evidence,
)
from src.model.conditioning import conditional_score, naive_condition_weights


def _pairs(seed=0, e=32, f=8):
    g = torch.Generator().manual_seed(seed)
    s = torch.relu(torch.randn(e, 4, f, generator=g))
    c = torch.relu(torch.randn(e, 4, f, generator=g))
    return s, c


def test_step_zero_interface_equals_naive_rule_exactly():
    s, c = _pairs()
    interface = ResidualConditionInterface(torch.rand(8) + 0.1)
    assert torch.equal(interface(s, c), naive_condition_weights(s, c))


def test_weights_are_nonnegative_and_sum_to_one_or_zero():
    s, c = _pairs(1)
    interface = ResidualConditionInterface(torch.ones(8))
    with torch.no_grad():
        interface.mlp[-1].bias.fill_(0.3)                           # a non-zero correction
    w = interface(s, c)
    total = w.sum(dim=-1)
    assert (w >= 0).all() and torch.all((total - 1).abs() < 1e-5)
    zero = interface(torch.zeros(2, 4, 8), torch.ones(2, 4, 8) * 5)  # every corrected gap still negative
    assert torch.equal(zero, torch.zeros(2, 8))


def test_evidence_features_have_the_documented_layout():
    s = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
    c = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
    ev = factor_evidence(s, c, torch.tensor([2.0, 1.0]))
    assert len(EVIDENCE_FEATURES) == 6 and ev.shape == (1, 2, 6)
    assert torch.allclose(ev[0, 0], torch.tensor([1.0, 1.0, 0.0, 0.5, 1.0, 0.0]))   # gap 2/2, mean 2/2, std 1/2
    assert torch.allclose(ev[0, 1], torch.tensor([-2.0, 0.0, 2.0, 0.0, 0.0, 1.0]))


def test_interface_is_equivariant_to_factor_permutation():
    s, c = _pairs(2)
    scale = torch.rand(8) + 0.1
    interface = ResidualConditionInterface(scale)
    with torch.no_grad():
        for p in interface.mlp.parameters():
            p.normal_(0, 0.5)
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(3))
    permuted = ResidualConditionInterface(scale[perm])
    permuted.load_state_dict({**interface.state_dict(), "factor_scale": scale[perm]})
    assert torch.allclose(interface(s, c)[:, perm], permuted(s[..., perm], c[..., perm]), atol=1e-6)


def test_scorer_uses_conditional_score_with_learned_beta_and_gradients_reach_the_mlp():
    s, c = _pairs(4, e=6)
    scorer = ConditionalScorer(ResidualConditionInterface(torch.ones(8)), beta_init=0.3)
    assert abs(scorer.beta.item() - 0.3) < 1e-6
    q, k = torch.randn(6, 5), torch.randn(6, 7, 5)
    qc, kc = torch.rand(6, 8), torch.rand(6, 7, 8)
    w = scorer.weights(s, c)
    out = scorer.score(q, k, qc, kc, w)
    assert torch.allclose(out, conditional_score(q, k, qc, kc, w, scorer.beta), atol=1e-6)
    (out / scorer.tau).sum().backward()
    assert scorer.interface.mlp[-1].weight.grad is not None
    assert scorer.interface.mlp[-1].weight.grad.abs().sum() > 0
    scorer.set_tau(0.25)
    assert abs(scorer.tau.item() - 0.25) < 1e-6
