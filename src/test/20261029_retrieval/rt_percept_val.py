"""Export validation probabilities with the held PercepT P3 mapper (amendments 1 and 2: the full-scale mapper).

Run on the training CUDA device, under the caller's GPU lock, for example:
  flock -o -w 3600 /tmp/gpu0.lock python src/test/20261029_retrieval/rt_percept_val.py

Frozen inputs come from the main checkout; exports go to RT_OUTPUTS/percept_val_full
(a new folder; the earlier sample-scale percept_val/ exports stay untouched).
An existing array or sidecar is retained only when its content agrees exactly.
"""
import argparse
import json
from unittest.mock import patch

import numpy as np

import rt_paths
import rt_data
import rs_vh
import ct_percept


def _publish(path, prob, side):
    """Check both existing files before creating either; never overwrite a file."""
    metadata = path.with_suffix('.json')
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            agrees = set(z.files) == {'prob'}
            if agrees:
                stored = z['prob']
                agrees = stored.dtype == prob.dtype and np.array_equal(stored, prob)
        if not agrees:
            raise FileExistsError(f'Refusing to overwrite {path}: different content')
    if metadata.exists() and json.loads(metadata.read_text()) != side:
        raise FileExistsError(f'Refusing to overwrite {metadata}: different content')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('xb') as handle:
            np.savez_compressed(handle, prob=prob)
    if not metadata.exists():
        with metadata.open('x') as handle:
            handle.write(json.dumps(side, indent=2, allow_nan=False) + '\n')


def export(seed):
    """Infer one seed on validation, check its held mapper hash, and seal the pair."""
    assert seed in rt_paths.SEEDS, seed
    ns = rt_data.names('val')
    assert len(ns) == len(set(ns)) == rt_paths.N_VAL and ns == sorted(ns), 'validation names/order'
    source = rt_paths.PERCEPT_OUT / f'percept/core/stage2/k40/P3__s{seed}.npz'
    held = rt_paths.PERCEPT_OUT / f'held/percept/core__k40/P3__s{seed}.json'
    original = json.loads(source.with_suffix('.json').read_text())
    comparator = json.loads(held.read_text())
    for record in (original, comparator):
        config = record['config']
        assert (config['settings'], config['cut'], config['arm'], config['seed']) == ('core', 'k40', 'P3', seed)
        assert not config['smoke'] and record['K'] == 40
    device = original['config']['device']
    assert device == comparator['inference_device'] == comparator['config']['device'] == 'cuda', 'training device'
    weights_sha = rs_vh.file_sha(source.with_suffix('.pt'))
    assert weights_sha == original['weights_sha256'] == comparator['weights_sha256'], 'held mapper weights hash'

    # ct_percept defaults to its importing checkout's outputs. Redirect only its
    # read-only input root, and let infer_stage2 enforce the training device.
    with patch.object(ct_percept, 'OUTPUTS', rt_paths.PERCEPT_OUT):
        assert ct_percept.stage2_path('core', 'k40', 'P3', seed) == source
        prob = ct_percept.infer_stage2('core', 'k40', 'P3', seed, ns)
    assert prob.shape == (rt_paths.N_VAL, 40) and prob.dtype == np.float32, 'validation probabilities shape/dtype'
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(), 'validation probability range'
    assert rs_vh.file_sha(source.with_suffix('.pt')) == weights_sha, 'mapper weights changed during inference'
    side = dict(rows=len(ns), val_sha256=rs_vh.names_sha(ns), weights_sha256=weights_sha,
                array_sha256s={'prob': rs_vh.array_sha(prob)}, device=device, seed=seed)
    path = rt_paths.OUTPUTS / rt_paths.PERCEPT_VAL_DIR / f'P3__s{seed}.npz'
    _publish(path, prob, side)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    for seed in rt_paths.SEEDS:
        print(export(seed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
