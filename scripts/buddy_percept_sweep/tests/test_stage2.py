import numpy as np
import torch

from scripts.buddy_percept_sweep.stage2 import (
    ParameterizedAttentionPoolingMapper, auc_summary, evaluate_auc, train_stage2,
)


def test_mapper_forward_shape_across_capacity_options():
    for num_queries in (1, 2, 4):
        for mlp_head in ("linear", "one_hidden"):
            mapper = ParameterizedAttentionPoolingMapper(
                n_topics=5, num_queries=num_queries, mlp_head=mlp_head, d_model=16
            )
            patches = torch.randn(3, 50, 16)
            logits = mapper(patches)
            assert logits.shape == (3, 5)


def test_train_stage2_reduces_loss_on_learnable_synthetic_task():
    torch.manual_seed(0)
    mapper = ParameterizedAttentionPoolingMapper(n_topics=2, num_queries=1, mlp_head="linear", d_model=8)
    patches = torch.randn(20, 50, 8)
    targets = torch.zeros(20, 2)
    targets[:10, 0] = 1.0
    targets[10:, 1] = 1.0
    train_labels = np.array([0] * 10 + [1] * 10)
    with torch.no_grad():
        initial_loss = torch.nn.functional.binary_cross_entropy_with_logits(mapper(patches), targets).item()
    train_stage2(
        mapper, patches, targets, mapper_lr=1e-2, mapper_epochs=50, weight_decay=0.0,
        class_balanced=False, train_labels_for_weighting=train_labels, seed=42,
    )
    with torch.no_grad():
        final_loss = torch.nn.functional.binary_cross_entropy_with_logits(mapper(patches), targets).item()
    assert final_loss < initial_loss


def test_evaluate_auc_skips_degenerate_topic():
    scores = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5]])
    targets = np.array([[1, 1], [0, 1], [1, 1]])  # topic 1 is all-positive -> skip
    aucs, skipped = evaluate_auc(scores, targets)
    assert 1 in skipped
    assert 0 in aucs
    summary = auc_summary(aucs)
    assert 0.0 <= summary["macro"] <= 1.0
