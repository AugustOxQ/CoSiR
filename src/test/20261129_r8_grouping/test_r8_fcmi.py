"""CPU verdict seams for M7 / D21; no data files or GPU required."""
from types import SimpleNamespace
from pathlib import Path
import uuid

import numpy as np
import pytest
import torch

import r8_fcmi as fcmi


def test_loaders_keep_common_protected_groups_in_painting_order():
    x = np.zeros((4, 512), dtype=np.float32)
    x[np.arange(4), [8, 2, 9, 1]] = 1
    context = SimpleNamespace(n_paintings=4, content=np.array([12, 3, 12, 0], dtype=np.int64))
    train, test = fcmi.painting_loaders(x, context, batch_size=2)
    batches = list(test)
    assert torch.cat([b[1] for b in batches]).tolist() == [12, 3, 12, 0]
    assert torch.cat([b[3] for b in batches]).tolist() == [0, 1, 2, 3]
    assert torch.cat([b[2] for b in batches]).tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(torch.cat([b[0] for b in batches]).numpy(), x)
    shuffled = list(train)
    for xb, groups, _, indices in shuffled:
        np.testing.assert_array_equal(groups.numpy(), context.content[indices.numpy()])
        np.testing.assert_array_equal(xb.numpy(), x[indices.numpy()])


def test_d21_rejects_nonfinite_step_and_one_cluster():
    guard = fcmi.D21()
    guard.step(torch.tensor(1.25), epoch=0, batch=0)
    with pytest.raises(FloatingPointError):
        guard.step(torch.tensor(float('nan')), epoch=1, batch=2)
    record = guard.record(exit_code=1, labels=np.array([0, 1]), wall_seconds=1)
    assert record['tests'] == dict(process=False, finite_loss=False, clusters=True, wall_time=True)
    assert record['passed'] is False
    assert guard.losses[-1] == dict(epoch=1, batch=2, total_loss=None, finite=False)
    collapsed = fcmi.D21()
    collapsed.step(torch.tensor(0.5), epoch=0, batch=0)
    record = collapsed.record(exit_code=0, labels=np.array([4, 4, 4]), wall_seconds=3)
    assert record['tests'] == dict(process=True, finite_loss=True, clusters=False, wall_time=True)
    assert record['passed'] is False


def test_d21_uses_strict_sixty_minute_limit_and_exit_status():
    guard = fcmi.D21()
    guard.step(torch.tensor(-0.4), epoch=0, batch=0)
    labels = np.array([0, 1])
    assert guard.record(0, labels, 3600)['passed'] is True
    assert guard.record(0, labels, 3600.01)['tests']['wall_time'] is False
    assert guard.record(9, labels, 2)['tests']['process'] is False


def test_native_labels_are_encode_class_argmax_with_empty_clusters_removed():
    # Identity encoder and literal centres make raw winners 3, 1, 3, 1.
    # Compact labels must preserve ascending centre order, hence 1, 0, 1, 0.
    args = fcmi.default_args(0.2, device='cpu')
    net = fcmi.NetFCN(input_dim=512, class_num=5, group_num=2, args=args)
    net.encoder = torch.nn.ModuleList([torch.nn.Identity()])
    net.encoder_linear = torch.nn.ModuleList([torch.nn.Identity()])
    net.cluster_centers = torch.tensor([[0., 0.], [0., 1.], [-1., 0.], [1., 0.], [0., -1.]])
    x = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    data = torch.utils.data.TensorDataset(x, torch.tensor([1, 0, 1, 0]),
                                         torch.zeros(4, dtype=torch.long), torch.arange(4))
    labels = fcmi.native_labels(net, torch.utils.data.DataLoader(data, batch_size=2))
    np.testing.assert_array_equal(labels, [1, 0, 1, 0])
    assert labels.dtype == np.int64 and not net.training


@pytest.mark.parametrize('fault', ['reconstruction', 'fairness'])
def test_copied_training_loop_checks_total_loss_before_backward(fault):
    # Fault injection at the input boundary: inference is finite on the separate
    # test paintings, but reconstruction of a NaN training vector must fail D21.
    args = fcmi.default_args(0.2, device='cpu')
    args.train_epoch = 3  # first checkpoint is epoch 2; the bad first step writes none
    if fault == 'fairness':
        args.WarmAll = 0
        args.InfoFairLoss = float('nan')
    # The original WarmAll=0 NpPoints schedule writes before the fault; keep that
    # tiny artifact inside this worktree, never in pytest's /tmp or the main checkout.
    args.output_dir = str(Path(__file__).parent / 'results/smoke/test_r8_fcmi' / uuid.uuid4().hex)
    net = fcmi.NetFCN(input_dim=512, class_num=2, group_num=2, args=args)
    x = torch.randn(12, 512)
    g = torch.arange(12) % 2
    data = lambda values: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        values, g, torch.zeros(12, dtype=torch.long), torch.arange(12)), batch_size=12)
    bad_x = x.clone()
    if fault == 'reconstruction':
        bad_x[0, 0] = float('nan')
    guard = fcmi.D21()
    with pytest.raises(FloatingPointError, match='epoch 0, batch 0'):
        net.run(3, data(bad_x), data(x), args, guard)
    assert guard.losses == [dict(epoch=0, batch=0, total_loss=None, finite=False)]
