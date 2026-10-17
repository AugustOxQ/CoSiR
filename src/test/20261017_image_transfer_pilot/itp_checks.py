"""Bounded CPU validation helpers; never construct an embedding encoder."""
import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def imports():
    here = Path(__file__).resolve().parent
    for path in sorted(here.glob('itp_*.py')):
        importlib.import_module(path.stem)
        if path.stem != 'itp_data':
            result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            assert 'usage:' in result.stdout
    assert 'src.data.feature_extract' not in sys.modules, 'model implementation imported on import/help path'
    result = subprocess.run(['bash', '-n', str(here / 'itp_queue.sh')], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    print('All itp module imports, script --help and bash -n passed; model implementation not imported.')


def synthetic():
    from itp_score import neighbour_overlap, top20_neighbours
    from itp_data import l2, load_frozen, split
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(83, 37)).astype(np.float32), rng.normal(size=(83, 37)).astype(np.float32)
    assert neighbour_overlap(x, x) == 1.0
    expected = []
    for arr in (x, y):
        sim = l2(arr) @ l2(arr).T
        np.fill_diagonal(sim, -np.inf)
        expected.append(np.argsort(-sim, axis=1)[:, :20])
    nn = top20_neighbours(x)
    assert np.array_equal(np.sort(nn, axis=1), np.sort(expected[0], axis=1))
    brute = np.mean([len(set(a) & set(b)) / 20 for a, b in zip(*expected)])
    actual = neighbour_overlap(x, y)
    assert abs(actual - brute) < 1e-12
    s = load_frozen()
    all_train, fit, dev = split(s)
    assert (len(all_train), len(fit), len(dev)) == (14889, 13400, 1489)
    assert np.array_equal(dev, np.random.default_rng(0).permutation(all_train)[:1489])
    assert set(fit) | set(dev) | set(s['subset_pid']) == set(range(19889))
    assert not set(dev) & set(s['subset_pid'])
    print(f'Synthetic overlap passed: identical=1.0; random={actual:.8f}, brute={brute:.8f}. Fixed split passed.')
    # Exercise the prespecified bar and aggregation without using real labels.
    from itp_score import aggregate, bars
    r = dict(views={})
    for target, scores in [('emotion_focus', [0.13, 0.14, 0.15]), ('default', [0.10, 0.11, 0.12])]:
        for seed, value in enumerate(scores):
            r['views'][f'default__{target}__s{seed}'] = dict(complete=True,
                painting_ami={label: dict(mean=value, sd=0.001) for label in ('emotion', 'genre', 'style')})
    r['views']['default_image'] = dict(painting_ami={label: dict(mean=0.09, sd=0.001) for label in ('emotion', 'genre', 'style')})
    r['aggregates'] = aggregate(r)
    assert bars(r)['default']['pass_bar']
    r['aggregates']['default__emotion_focus']['metrics']['painting_ami.emotion.mean']['mean'] = 0.119
    assert not bars(r)['default']['pass_bar']
    print('Bar PASS/FAIL boundary and seed aggregation passed.')
    from itp_score import consistency
    old = {label: dict(mean=0.1, sd=0.001) for label in ('emotion', 'genre', 'style')}
    previous = dict(views=dict(default=dict(painting_ami=dict(image=old))))
    current = {label: dict(mean=0.102, sd=0.02) for label in old}
    assert consistency('default_image', current, previous)['ok']
    current['emotion']['mean'] = 0.104
    assert not consistency('default_image', current, previous)['ok']
    print('Consistency mean-AMI tolerance passed; sd deltas remain diagnostic.')


def join():
    from itp_data import default_images, load_frozen
    x, gate = default_images(load_frozen())
    assert x.shape == (19889, 2048) and x.dtype == np.float32
    assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < 1e-5
    print(f'Real join / finite / unit-norm checks passed: {gate}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['imports', 'synthetic', 'join'], required=True)
    args = p.parse_args()
    start = time.monotonic()
    {'imports': imports, 'synthetic': synthetic, 'join': join}[args.mode]()
    print(f'CHECK {args.mode} passed in {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
