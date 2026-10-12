"""Synthetic checks for the verdict computations of decision-rule M5/M6."""
import numpy as np
import pytest
import torch
import math

from r8_head import batch_draws, embed, info_nce, neighbours, train_head


def test_neighbours_exclude_content_and_break_ties_by_painting_index():
    # p=0 cannot pick the identical same-group vector at 1; 2 and 3 tie.
    x = np.array([[1., 0.], [1., 0.], [.6, .8], [.6, .8], [-1., 0.]])
    content = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    for block_size in (1, 3, 10):
        n = neighbours(x, content, block_size=block_size)
        np.testing.assert_array_equal(n, [2, 2, 0, 0, 2])
        assert np.all(content[n] != content)


def test_neighbours_compare_in_float64():
    x = np.array([[1., 0.], [.6, .8], [.6 + 1e-9, .8]])
    content = np.array([0, 1, 1], dtype=np.int64)
    assert neighbours(x, content)[0] == 2


@pytest.mark.parametrize('n_own', [128, 86, 42])
def test_negative_sets_have_exact_counts_without_query_or_positive(n_own):
    content = np.repeat(np.arange(3, dtype=np.int64), 130)
    n = (np.arange(390, dtype=np.int64) + 130) % 390
    batch = np.array([0, 129, 130, 259, 389])
    views, negative = batch_draws(np.random.default_rng(0), batch, content, n, n_own)
    assert views.shape == (5,) and np.all((views >= 0) & (views < 5))
    assert negative.shape == (5, 128)
    for p, row in zip(batch, negative):
        assert len(np.unique(row)) == 128
        assert p not in row and n[p] not in row
        assert np.all(content[row[:n_own]] == content[p])
        assert np.all(content[row[n_own:]] != content[p])


def test_m6_rng_order_after_permutation_matches_fixed_seed_zero_draws():
    # Fixed fixture calculated from M6's ordered calls, not from batch_draws.
    content = np.repeat(np.arange(3, dtype=np.int64), 130)
    n = (np.arange(390, dtype=np.int64) + 130) % 390
    rng = np.random.default_rng(0)
    batch = rng.permutation(390)[:3]
    views, negative = batch_draws(rng, batch, content, n, 86)
    np.testing.assert_array_equal(views, [0, 0, 2])
    np.testing.assert_array_equal(negative[:, [0, 1, 2, 3, 86, 87, 88, 89]],
                                 [[319, 305, 358, 298, 52, 148, 4, 17],
                                  [263, 327, 344, 375, 103, 115, 113, 214],
                                  [295, 377, 354, 291, 12, 234, 103, 142]])
    assert rng.integers(0, 1000000) == 472467


def test_info_nce_matches_hand_computation_and_keeps_negative_gradient():
    query = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64, requires_grad=True)
    positive = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([[[0., 1.], [-1., 0.]],
                             [[0., 1.], [0., -1.]]], dtype=torch.float64, requires_grad=True)
    # Logits: [10,0,-10] and [0,10,-10]; positive is column zero.
    expected = (math.log1p(math.exp(-10) + math.exp(-20))
                + math.log(1 + math.exp(10) + math.exp(-10))) / 2
    loss = info_nce(query, positive, negative)
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    loss.backward()
    assert all(t.grad is not None and torch.any(t.grad != 0)
               for t in (query, positive, negative))


@pytest.mark.parametrize('view_positive', [False, True])
def test_seed_zero_training_is_identical_including_partial_batch(view_positive):
    rng = np.random.default_rng(123)
    x = rng.normal(size=(390, 512)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    content = np.repeat(np.arange(3, dtype=np.int64), 130)
    n = (np.arange(390, dtype=np.int64) + 130) % 390
    views = np.repeat(x[:, None, :], 5, axis=1) if view_positive else None
    outputs, records = [], []
    for _ in range(2):
        head, record = train_head(x, content, n, n_own=86, views=views,
                                  epochs=2, batch_size=3, anchors=np.arange(5))
        outputs.append(embed(head, x[:7]))
        records.append(record)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert outputs[0].dtype == np.float32 and outputs[0].shape == (7, 128)
    np.testing.assert_allclose(np.linalg.norm(outputs[0], axis=1), 1, atol=1e-6)
    assert records[0]['epoch_losses'] == records[1]['epoch_losses']
    assert records[0]['batches_per_epoch'] == 2
    assert records[0]['last_epoch_mean_loss'] == records[0]['epoch_losses'][-1]
