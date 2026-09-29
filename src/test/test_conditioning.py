import torch

from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes


def test_naive_weights_are_l1_normalized_relu_gap_and_zero_stays_zero():
    support = torch.tensor([[[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]]])
    contrast = torch.tensor([[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]])
    w = naive_condition_weights(support, contrast)
    assert torch.allclose(w, torch.tensor([[0.5, 0.0, 0.5]]))
    assert torch.equal(naive_condition_weights(support, support), torch.zeros(1, 3))


def test_top_k_keeps_only_the_largest_gaps():
    support = torch.tensor([[[3.0, 2.0, 1.0, 0.5]]])
    w = naive_condition_weights(support, None, top_k=2)
    assert torch.allclose(w, torch.tensor([[0.6, 0.4, 0.0, 0.0]]))


def test_zero_weights_reduce_score_to_scaled_clip_cosine():
    torch.manual_seed(0)
    q, c = torch.randn(2, 5), torch.randn(2, 4, 5)
    qc, cc = torch.rand(2, 3), torch.rand(2, 4, 3)
    score = conditional_score(q, c, qc, cc, torch.zeros(2, 3), beta=0.3)
    expected = 0.3 * torch.nn.functional.cosine_similarity(q[:, None, :], c, dim=-1)
    assert torch.allclose(score, expected, atol=1e-6)


def test_beta_zero_ranking_is_invariant_to_weight_scale():
    torch.manual_seed(1)
    q, c = torch.randn(3, 5), torch.randn(3, 6, 5)
    qc, cc, w = torch.rand(3, 4), torch.rand(3, 6, 4), torch.rand(3, 4)
    a = conditional_score(q, c, qc, cc, w, beta=0.0).argsort(dim=1)
    b = conditional_score(q, c, qc, cc, 7.0 * w, beta=0.0).argsort(dim=1)
    assert torch.equal(a, b)


def test_pair_codes_is_the_modality_mean():
    assert torch.equal(pair_codes(torch.ones(2, 3), torch.zeros(2, 3)), torch.full((2, 3), 0.5))
