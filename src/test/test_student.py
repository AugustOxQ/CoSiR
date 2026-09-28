"""Behavior tests for the content-only attention student."""

import torch

from src.model.student import AttentionFusionStudent


def test_student_output_shape_and_unit_norm():
    torch.manual_seed(7)
    student = AttentionFusionStudent(feature_dim=12, output_dim=8, dropout=0.0)
    img_feat = torch.randn(5, 12)
    txt_feat = torch.randn(5, 12)

    embedding = student(img_feat, txt_feat)

    assert embedding.shape == (5, 8)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(5), atol=1e-6, rtol=0)


def test_student_backpropagates_to_every_parameter():
    torch.manual_seed(11)
    student = AttentionFusionStudent(feature_dim=12, output_dim=8, dropout=0.0)
    img_feat = torch.randn(5, 12)
    txt_feat = torch.randn(5, 12)
    target = torch.randn(5, 8)

    loss = (student(img_feat, txt_feat) * target).sum()
    loss.backward()

    for name, parameter in student.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_student_initial_weights_repeat_with_seed_42():
    torch.manual_seed(42)
    first = AttentionFusionStudent(feature_dim=12)
    torch.manual_seed(42)
    second = AttentionFusionStudent(feature_dim=12)

    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


def test_student_uses_two_token_self_attention():
    torch.manual_seed(13)
    student = AttentionFusionStudent(feature_dim=12, output_dim=8, dropout=0.0)
    attention_outputs = []
    handle = student.attn.register_forward_hook(
        lambda _module, _inputs, output: attention_outputs.append(output)
    )
    try:
        student(torch.randn(5, 12), torch.randn(5, 12))
    finally:
        handle.remove()

    assert len(attention_outputs) == 1
    attended, weights = attention_outputs[0]
    assert attended.shape == (5, 2, 8)
    assert weights.shape == (5, 2, 2)
    assert not torch.allclose(weights, torch.full_like(weights, 0.5))
