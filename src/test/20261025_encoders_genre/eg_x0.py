"""X0 (rule §4.1, C12): the changed extractor and the extractor at eg_paths.X0_OLD_COMMIT re-extract the first 512
images (store order) of the artelingo_full clip_b32 and siglip2_so400m stores with default batch settings; rows must
be equal bit for bit. Each one's difference from the stored rows is reported and decides nothing.

    python eg_x0.py --outputs <dir> --device cuda      # GPU: run under the GPU lock
Writes <outputs>/stores/x0.json; exit 1 when not equal. Scratch under <outputs>/stores/x0/; an existing
destination stops the run (fresh destinations only).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import eg_paths

REPO = Path(__file__).resolve().parents[3]
DATASET, BACKBONES, N = 'artelingo_full', ('clip_b32', 'siglip2_so400m'), 512
# The old extractor imports src.utils (FeatureManager), so the old copy carries src/utils and src/__init__.py too.
OLD_PATHS = ('scripts/extract_features.py', 'src/data', 'src/utils', 'src/__init__.py')


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def extract(script_root, backbone, out_root, device):
    cmd = [sys.executable, str(script_root / 'scripts/extract_features.py'), '--dataset', DATASET,
           '--backbone', backbone, '--limit', str(N), '--out-root', str(out_root), '--device', device]
    subprocess.run(cmd, check=True, cwd=script_root)
    d = Path(out_root) / '_smoke' / DATASET / backbone
    idx = json.load(open(d / 'index.json'))
    img = np.load(d / 'img.npy')
    assert img.dtype == np.float16 and img.shape[0] == N and idx['n_images'] == N, (img.dtype, img.shape)
    assert len(idx['images']) == N
    return img, idx['images']


def diff(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {'max_abs': float(d.max()), 'rows_affected': int((d.max(axis=1) > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    eg_paths.check_imports()
    head = git('rev-parse', 'HEAD')
    dirty = bool(git('status', '--porcelain', '--', 'scripts/extract_features.py', 'src/data'))
    base = a.outputs / 'stores/x0'
    if base.exists():
        raise SystemExit(f'{base} exists: X0 writes to fresh destinations only')
    scratch = base / 'old_checkout'
    scratch.mkdir(parents=True)
    tar = subprocess.run(['git', '-C', str(REPO), 'archive', eg_paths.X0_OLD_COMMIT, *OLD_PATHS], check=True,
                         capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(scratch)], input=tar, check=True)
    res = {'pass': True, 'code_commit': head, 'working_tree_dirty': dirty, 'old_commit': eg_paths.X0_OLD_COMMIT,
           'n_images': N, 'stores': {}}
    for bb in BACKBONES:
        store = eg_paths.PRE / DATASET / bb
        sidx = json.load(open(store / 'index.json'))
        stored = np.asarray(np.load(store / 'img.npy', mmap_mode='r')[:N])
        assert stored.dtype == np.float16 and stored.shape[0] == N
        new, new_names = extract(REPO, bb, base / 'new', a.device)
        old, old_names = extract(scratch, bb, base / 'old', a.device)
        assert new_names == old_names == sidx['images'][:N], f'{bb}: names differ from the store order'
        assert new.dtype == old.dtype == np.float16 and new.shape == old.shape == stored.shape
        equal = bool(np.array_equal(new.view(np.uint16), old.view(np.uint16)))
        res['stores'][bb] = {'equal': equal, 'dtype': str(new.dtype), 'shape': list(new.shape),
                             'new_vs_old': diff(new, old),
                             'vs_stored': {'new': diff(new, stored), 'old': diff(old, stored)}}
        res['pass'] &= equal
        print(bb, json.dumps(res['stores'][bb]), flush=True)
    (a.outputs / 'stores').mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.outputs / 'stores/x0.json', 'w'), indent=1)
    print('X0', 'pass' if res['pass'] else 'FAIL')
    raise SystemExit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
