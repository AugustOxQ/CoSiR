"""CPU verdict seams for M7 / D21; no data files or GPU required."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

import r8_fcmi as fcmi


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    monkeypatch.setenv('R8_RESULTS', str(tmp_path))
    monkeypatch.setattr(fcmi.C, 'RESULTS', tmp_path)
    monkeypatch.setattr(fcmi.C, 'SMOKE', tmp_path / 'smoke')
    monkeypatch.setattr(fcmi, 'CHECKPOINT_ROOT', tmp_path / 'ssd', raising=False)
    return tmp_path


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


def test_loader_drops_partial_training_batch_and_tests_every_painting():
    x = np.zeros((5, 512), dtype=np.float32)
    x[:, 0] = 1
    context = SimpleNamespace(n_paintings=5, content=np.arange(5, dtype=np.int64))
    train, test = fcmi.painting_loaders(x, context, batch_size=2)
    assert len(train) == 2
    assert sum(len(b[0]) for b in train) == 4
    assert test.batch_size == 200
    assert torch.cat([b[3] for b in test]).tolist() == [0, 1, 2, 3, 4]


def test_smoke_enters_fairness_without_changing_real_defaults():
    real = fcmi.default_args(0.2, device='cpu')
    smoke = fcmi.default_args(0.2, device='cpu', smoke=True)
    assert (real.train_epoch, real.WarmAll, real.VisualFreq) == (300, 20, 20)
    assert (smoke.train_epoch, smoke.WarmAll, smoke.VisualFreq) == (2, 1, 2)


def test_d21_rejects_nonfinite_step_and_one_cluster():
    guard = fcmi.D21()
    guard.step(torch.tensor(1.25), epoch=0, batch=0)
    with pytest.raises(FloatingPointError):
        guard.step(torch.tensor(float('nan')), epoch=1, batch=2)
    record = guard.record(exit_code=1, labels=np.array([0, 1]), wall_seconds=1)
    assert record['tests'] == dict(process=False, finite_loss=False, clusters=True, wall_time=True)
    assert record['pass'] is False
    assert guard.losses[-1] == dict(epoch=1, batch=2, total_loss=None, finite=False)
    collapsed = fcmi.D21()
    collapsed.step(torch.tensor(0.5), epoch=0, batch=0)
    record = collapsed.record(exit_code=0, labels=np.array([4, 4, 4]), wall_seconds=3)
    assert record['tests'] == dict(process=True, finite_loss=True, clusters=False, wall_time=True)
    assert record['pass'] is False


def test_d21_uses_strict_sixty_minute_limit_and_exit_status():
    guard = fcmi.D21()
    guard.step(torch.tensor(-0.4), epoch=0, batch=0)
    labels = np.array([0, 1])
    assert guard.record(0, labels, 3600)['pass'] is True
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
def test_copied_training_loop_checks_total_loss_before_backward(fault, results_root):
    # Fault injection at the input boundary: inference is finite on the separate
    # test paintings, but reconstruction of a NaN training vector must fail D21.
    args = fcmi.default_args(0.2, device='cpu')
    args.train_epoch = 3  # first checkpoint is epoch 2; the bad first step writes none
    if fault == 'fairness':
        args.WarmAll = 0
        args.InfoFairLoss = float('nan')
    args.output_dir = str(results_root / 'smoke' / 'training_fault')
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


@pytest.mark.parametrize('weight,smoke,collapsed,child_exit', [
    (0.2, False, False, 0), (0.2, False, True, 0), (0.2, False, False, 7),
    (0.2, True, False, 0), (0.2, True, True, 0), (0.4, False, False, 0),
])
def test_build_publishes_bool_pass_and_failure_exit_with_fake_worker(
        weight, smoke, collapsed, child_exit, results_root, monkeypatch):
    n = 2048 if smoke else 36518
    context = SimpleNamespace(painting_ids=np.array([f'p{i}' for i in range(36518)]))
    images = np.arange(36518, dtype=np.int32)
    monkeypatch.setattr(fcmi.C, 'ctx', lambda: context)
    monkeypatch.setattr(fcmi.C, 'painting_image', lambda: images)
    paths = fcmi.run_paths(weight, smoke)
    if weight != 0.2:
        fcmi.C.write_json_once(results_root / 'stage1_fcmi.json', dict(
            **{'pass': True}, tests=dict(process=True, finite_loss=True, clusters=True, wall_time=True),
            smoke=False, weight=0.2, device='cuda'))

    def fake_worker(command, stdout, stderr):
        assert '--worker' in command
        epochs, batches = (2, 4) if smoke else (300, 71)
        with paths['losses'].open('w') as f:
            for epoch in range(epochs):
                for batch in range(batches):
                    f.write(json.dumps(dict(epoch=epoch, batch=batch, total_loss=0.5, finite=True)) + '\n')
        labels = np.zeros(n, dtype=np.int64) if collapsed else np.arange(n, dtype=np.int64) % 2
        fcmi.C.write_npz_once(paths['labels'], labels=labels, painting_index=np.arange(n),
                             painting_ids=context.painting_ids[:n], painting_image=images[:n])
        fcmi.C.write_json_once(paths['labels_done'], dict(monotonic=2., epoch_seconds=[1.] * epochs))
        fcmi.C.write_json_once(paths['status'], dict(wall_seconds=2., started=0., error=None))
        return SimpleNamespace(poll=lambda: child_exit, returncode=child_exit)

    monkeypatch.setattr(fcmi.subprocess, 'Popen', fake_worker)
    result = fcmi.build(weight, device='cpu' if smoke else 'cuda', smoke=smoke)
    record = json.loads(paths['verdict'].read_text())
    assert isinstance(record.get('pass'), bool)
    assert 'passed' not in record
    expected = not collapsed and child_exit == 0
    assert record['pass'] is (expected and not smoke)
    if smoke:
        assert record['wiring_pass'] is expected
        assert record['settings']['WarmAll'] == 1
        assert record['settings']['VisualFreq'] == 2
    assert result == (child_exit if child_exit else (0 if expected else 1))


def test_full_painting_timing_mode_has_no_stage_or_space(results_root, monkeypatch, capsys):
    args = fcmi.default_args(0.2, device='cpu', time_epochs=3)
    assert (args.train_epoch, args.WarmAll, args.VisualFreq) == (3, 1, 20)
    paths = fcmi.run_paths(0.2, smoke=False, time_epochs=3)
    assert paths['directory'].is_relative_to(results_root / 'smoke')
    assert paths['checkpoint'].is_relative_to(fcmi.CHECKPOINT_ROOT / 'smoke')

    def fake_worker(command, stdout, stderr):
        assert command[command.index('--time-epochs') + 1] == '3'
        assert '--smoke' not in command
        with paths['losses'].open('w') as f:
            for epoch in range(3):
                for batch in range(71):
                    f.write(json.dumps(dict(epoch=epoch, batch=batch, total_loss=0.5, finite=True)) + '\n')
        fcmi.C.write_json_once(paths['status'], dict(wall_seconds=9., started=0., error=None))
        fcmi.C.write_json_once(paths['timing'], dict(epoch_seconds=[2., 3., 4.], n_paintings=36518,
                                                  settings=vars(args), smoke=True))
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(fcmi.subprocess, 'Popen', fake_worker)
    assert fcmi.build(0.2, device='cpu', time_epochs=3) == 0
    assert not paths['verdict'].exists()
    assert not paths['space'].exists()
    assert not paths['labels'].exists()
    assert not (results_root / 'stage1_fcmi.json').exists()
    output = capsys.readouterr().out
    for epoch, seconds in [(1, 2), (2, 3), (3, 4)]:
        assert f'epoch={epoch} epoch_wall_seconds={seconds:.6f}' in output


def _old_loaders(x, c, batch_size):
    """Upstream-style loaders: plain TensorDataset, default collate (what M7 was written against)."""
    from torch.utils.data import DataLoader, TensorDataset
    n = len(x)
    ds = TensorDataset(torch.from_numpy(x.copy()), torch.from_numpy(c.copy()),
                       torch.zeros(n, dtype=torch.long), torch.arange(n))
    train = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(0))
    return train, DataLoader(ds, batch_size=batch_size * 100, shuffle=False)


def _assert_same_epochs(old, new, epochs=3):
    (o_train, o_test), (n_train, n_test) = old, new
    for _ in range(epochs):
        ob, nb = list(o_train) + list(o_test), list(n_train) + list(n_test)
        assert len(ob) == len(nb)
        for a, b in zip(ob, nb):
            assert type(a) is type(b) and len(a) == len(b) == 4
            for u, v in zip(a, b):
                assert u.dtype == v.dtype and u.shape == v.shape and torch.equal(u, v)


def _synthetic(n=3001, batch_size=64):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n, 512)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    c = rng.integers(0, 64, n).astype(np.int64)
    return x, c, SimpleNamespace(n_paintings=n, content=c), batch_size


def test_batched_loaders_give_the_same_batches_as_the_plain_loaders_for_several_epochs():
    x, c, context, bs = _synthetic()
    assert len(x) % bs != 0
    old = _old_loaders(x, c, bs)
    new = fcmi.painting_loaders(x, context, batch_size=bs)
    assert isinstance(new[0].dataset, fcmi.BatchedTensorDataset)
    assert new[0].collate_fn is fcmi.identity_collate
    _assert_same_epochs(old, new, epochs=4)


@pytest.mark.parametrize('fault', ['no_drop_last', 'other_seed'])
def test_same_batches_check_catches_a_changed_drop_last_or_generator_seed(fault):
    from torch.utils.data import DataLoader
    x, c, context, bs = _synthetic()
    good_train, good_test = fcmi.painting_loaders(x, context, batch_size=bs)
    kw = dict(drop_last=fault != 'no_drop_last')
    seed = 1 if fault == 'other_seed' else 0
    bad_train = DataLoader(good_train.dataset, batch_size=bs, shuffle=True, collate_fn=fcmi.identity_collate,
                           generator=torch.Generator().manual_seed(seed), **kw)
    with pytest.raises(AssertionError):
        _assert_same_epochs(_old_loaders(x, c, bs), (bad_train, good_test), epochs=3)
