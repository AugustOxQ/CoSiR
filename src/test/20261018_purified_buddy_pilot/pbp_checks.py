"""CPU contract checks, synthetic graph/loss tests, and strict resume verification."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

from pbp_graphs import (HERE, SMOKE, atomic_json, check_sets, derive, pair_array,
                        sha_file, source_pairs)
from pbp_train import (adjacency, direction_logits, hard_logits, pair_loss,
                       sample_negatives, verify_output)
from pbp_score import aggregate, bar_numbers
from itp_train import CONFIG


def raises(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Expected rejection')


def graph_check():
    e = {(0, 1), (1, 2), (2, 3)}
    d = {(0, 1), (0, 3)}
    i = {(1, 2), (1, 3)}
    g = derive(e, d, i)
    check_sets(g, 4)
    assert g['P'] == {(2, 3)} and g['N'] == {(0, 3), (1, 3)}
    assert g['EC'] == {(0, 1), (1, 2)}
    for bad in [(0, 0), (3, 0), (0, 4)]:
        broken = copy.deepcopy(g)
        broken['E'].add(bad)
        raises(lambda: check_sets(broken, 4))
    xx = np.random.default_rng(0).normal(size=(64, 32)).astype(np.float32)
    ps = source_pairs(xx)
    xx /= np.linalg.norm(xx, axis=1, keepdims=True)
    sim = xx @ xx.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :20]
    expected = {(a, int(b)) for a, row in enumerate(nn) for b in row if a < b and a in nn[b]}
    assert ps == expected and len(ps) > 0
    print(f'PASS graph sets/source symmetry: synthetic reciprocal pairs={len(ps)}')


def logits_check():
    torch.manual_seed(0)
    x = F.normalize(torch.randn(8, 5), dim=1)
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 5)
        def forward(self, value):
            return F.normalize(self.linear(value), dim=1)
    model = Small()
    pairs = np.asarray([[0, 1], [2, 3], [4, 5]])
    ns = adjacency(pair_array({(0, 6), (0, 7), (2, 6), (1, 7)}), 8)
    draws = sample_negatives(pairs[:, 0], ns, np.random.default_rng(4))
    assert draws.shape == (3, 4) and np.array_equal((draws >= 0).sum(1), [2, 1, 0])
    for anchor, row in zip(pairs[:, 0], draws):
        valid = row[row >= 0]
        assert len(set(valid)) == len(valid) and set(valid) <= set(ns[anchor])
    z = model(x[torch.as_tensor(pairs[:, 0])])
    extra = hard_logits(model, x, z, draws)
    assert extra.shape == (3, 4)
    assert torch.isneginf(extra[torch.as_tensor(draws < 0)]).all()
    for row, col in zip(*np.nonzero(draws >= 0)):
        expected = (z[row] * model(x[draws[row, col]:draws[row, col] + 1])[0]).sum() / CONFIG['temperature']
        torch.testing.assert_close(extra[row, col], expected)
    base = z @ model(x[torch.as_tensor(pairs[:, 1])]).T / CONFIG['temperature']
    logits = direction_logits(base, extra)
    assert logits.shape == (3, 7)
    torch.testing.assert_close(logits[:, :3], base)
    labs = torch.arange(3)
    # An all-masked anchor contributes exactly the unextended CE.
    torch.testing.assert_close(F.cross_entropy(logits[2:3], labs[2:3]),
                               F.cross_entropy(base[2:3], labs[2:3]))
    pure = pair_loss(model, x, pairs)
    expected = (F.cross_entropy(base, labs) + F.cross_entropy(base.T, labs)) / 2
    torch.testing.assert_close(pure, expected)
    pn = pair_loss(model, x, pairs, ns, np.random.default_rng(4))
    assert torch.isfinite(pn) and pn >= pure - 1e-6
    pn.backward()
    assert model.linear.weight.grad is not None and torch.isfinite(model.linear.weight.grad).all()
    # Both directions independently sample their own anchors, including missing rows.
    assert (sample_negatives(pairs[:, 1], ns, np.random.default_rng(4)) >= 0).sum() == 1
    many = adjacency(pair_array({(0, j) for j in range(1, 8)}), 8)
    assert len(set(sample_negatives([0], many, np.random.default_rng(0))[0])) == 4
    print(f'PASS hard negatives: logits=(3,7), valid extras=[2,1,0], masking/gradient; losses={pure.item():.6f}/{pn.item():.6f}')


def resume_check():
    folder = SMOKE / 'checks'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'resume.npy'
    assert verify_output(folder / 'absent.npy', {}) is None
    a = np.zeros((5000, 2048), dtype=np.float16)
    a[:, 0] = 1
    np.save(path, a)
    cfg = dict(max_epochs=2, seed=0, loss='synthetic')
    info = dict(config=cfg, best_epoch=1, epochs_run=2, best_dev_loss=1., output_sha256=sha_file(path))
    atomic_json(path.with_suffix('.json'), info)
    assert verify_output(path, cfg) == info
    raises(lambda: verify_output(path, dict(cfg, seed=1)))
    a[0, 0] = 0
    np.save(path, a)
    raises(lambda: verify_output(path, cfg))
    info['output_sha256'] = sha_file(path)
    atomic_json(path.with_suffix('.json'), info)
    raises(lambda: verify_output(path, cfg))  # Unit-norm failure even with a matching checksum.
    a[0, 0] = 1
    np.save(path, a)
    info['output_sha256'] = sha_file(path)
    atomic_json(path.with_suffix('.json'), info)
    assert verify_output(path, cfg) == info
    print('PASS resume: identical skips; config, checksum and norm mismatches rejected')


def scoring_check():
    assert bar_numbers(dict(mean=.13, sd=.001), dict(mean=.10, sd=.002))['pass_bar']
    assert not bar_numbers(dict(mean=.119, sd=0.), dict(mean=.10, sd=0.))['pass_bar']
    assert not bar_numbers(dict(mean=.125, sd=.025), dict(mean=.10, sd=.03))['pass_bar']
    views = {f'S_P__s{i}': dict(complete=True, painting_ami=dict(emotion=dict(mean=v, sd=.9)))
             for i, v in enumerate([.1, .2, .3])}
    ag = aggregate(views)['S_P']
    np.testing.assert_allclose(ag['metrics']['painting_ami.emotion.mean']['mean'], .2)
    np.testing.assert_allclose(ag['metrics']['painting_ami.emotion.mean']['sd'], np.std([.1, .2, .3]))
    print('PASS aggregation/bar: population student sd and strict margin')


def entrypoints_check():
    for module in ('pbp_graphs.py', 'pbp_train.py', 'pbp_score.py'):
        done = subprocess.run([sys.executable, str(HERE / module), '--help'], env=os.environ,
                              capture_output=True, text=True, timeout=40)
        assert done.returncode == 0, done.stderr
    subprocess.run(['bash', '-n', str(HERE / 'pbp_queue.sh')], check=True, timeout=5)
    assert not torch.cuda.is_available(), 'Expected CPU-only container'
    assert 'src.data.feature_extract' not in sys.modules
    print('PASS entrypoints and bash syntax; CUDA unavailable; no Qwen model module loaded')


def main():
    torch.set_num_threads(8)
    graph_check()
    logits_check()
    resume_check()
    scoring_check()
    entrypoints_check()
    print('PBP_CHECKS PASS (5 groups)')


if __name__ == '__main__':
    main()
