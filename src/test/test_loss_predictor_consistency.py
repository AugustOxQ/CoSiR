"""Tests for src.metrics.loss.predictor_consistency_loss (Experiment 11.3,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md).

Verifies the actual autograd claim the spec's "What" section makes: with
stopgrad=True (today's default), only the predictor receives gradient from this
term (one-way distillation). With stopgrad=False, gradient also flows into the
condition table -- the mechanism Experiment 11.3 tests.

Run: python src/test/test_loss_predictor_consistency.py
"""
import torch

from src.metrics.loss import predictor_consistency_loss


def test_stopgrad_true_blocks_gradient_into_table():
    torch.manual_seed(0)
    label_embeddings = torch.randn(4, 8, requires_grad=True)
    pred_cond = torch.randn(4, 8, requires_grad=True)

    loss = predictor_consistency_loss(pred_cond, label_embeddings, stopgrad=True)
    loss.backward()

    assert pred_cond.grad is not None, "predictor must receive gradient regardless of stopgrad"
    assert not torch.allclose(pred_cond.grad, torch.zeros_like(pred_cond.grad)), "predictor gradient must be nonzero"
    assert label_embeddings.grad is None, "stopgrad=True must block gradient into the condition table"
    print("PASS: stopgrad=True blocks gradient into label_embeddings")


def test_stopgrad_false_allows_gradient_into_table():
    torch.manual_seed(0)
    label_embeddings = torch.randn(4, 8, requires_grad=True)
    pred_cond = torch.randn(4, 8, requires_grad=True)

    loss = predictor_consistency_loss(pred_cond, label_embeddings, stopgrad=False)
    loss.backward()

    assert pred_cond.grad is not None, "predictor must receive gradient regardless of stopgrad"
    assert not torch.allclose(pred_cond.grad, torch.zeros_like(pred_cond.grad)), "predictor gradient must be nonzero"
    assert label_embeddings.grad is not None, "stopgrad=False must let gradient flow into the condition table"
    assert not torch.allclose(label_embeddings.grad, torch.zeros_like(label_embeddings.grad)), "table gradient must be nonzero"
    print("PASS: stopgrad=False lets gradient flow into label_embeddings")


def test_stopgrad_default_is_true():
    torch.manual_seed(0)
    label_embeddings = torch.randn(4, 8, requires_grad=True)
    pred_cond = torch.randn(4, 8, requires_grad=True)

    loss = predictor_consistency_loss(pred_cond, label_embeddings)  # no stopgrad= arg
    loss.backward()

    assert pred_cond.grad is not None, "predictor must receive gradient by default"
    assert label_embeddings.grad is None, "default must block gradient into the condition table (backward-compat)"
    print("PASS: default (no stopgrad= arg) blocks gradient into label_embeddings, matching stopgrad=True")


def main():
    test_stopgrad_true_blocks_gradient_into_table()
    test_stopgrad_false_allows_gradient_into_table()
    test_stopgrad_default_is_true()


if __name__ == "__main__":
    main()
