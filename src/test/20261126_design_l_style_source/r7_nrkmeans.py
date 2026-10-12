"""NrKmeans reference N (DECISION_RULE L1, L2, L6, L7; C5/C6).

CPU only, no episodes or label AMIs. Install clustpy==0.0.3 with --no-deps.
--limit selects the first sorted scorer-train paintings for a trial, requiring
--out under this worktree's results/ticket04_trial. Full runs belong to prep.
"""
import r7_common as C

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

import r7_groupings as G

CLUSTPY_VERSION = '0.0.3'
TIMEOUT_SECONDS = 30 * 60


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def worker(directory):
    """Only the disposable child imports ClustPy and fits the default model."""
    directory = Path(directory)
    started = time.monotonic()
    rec = {'dropped': True, 'reason': None}
    try:
        assert package_version('clustpy') == CLUSTPY_VERSION, 'ClustPy version differs from pin'
        from clustpy.alternative import NrKmeans
        x = np.load(directory / 'vectors.npy', allow_pickle=False)
        assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 768, 'CSD vectors shape/dtype'
        assert len(x) >= 17 and np.isfinite(x).all(), 'CSD vectors count/finite'
        labels = NrKmeans(n_clusters=[17, 17], random_state=0).fit(x).labels_
        assert labels.shape == (len(x), 2) and labels.dtype.kind == 'i', 'NrKmeans labels shape/dtype'
        assert (labels >= 0).all(), 'NrKmeans negative label'
        np.save(directory / 'labels.npy', labels.astype(np.int64), allow_pickle=False)
        rec.update(dropped=False)
    except (Exception, SystemExit) as exc:
        rec['reason'] = f'{type(exc).__name__}: {exc}'
    rec['fit_seconds'] = time.monotonic() - started
    (directory / 'worker.json').write_text(json.dumps(rec))


def fit_child(x, out, timeout_seconds=TIMEOUT_SECONDS):
    """Bound import and fit together; translate child failures into a drop record."""
    started = time.monotonic()
    rec = {'dropped': True, 'reason': None, 'timeout_seconds': timeout_seconds,
           'clustpy_version': package_version('clustpy'), 'clustpy_version_pin': CLUSTPY_VERSION,
           'torch_version': package_version('torch'), 'extra_installs': [],
           'parameters': {'n_clusters': [17, 17], 'random_state': 0},
           'other_parameters': 'ClustPy defaults'}
    labels = None
    # All temporary files are inside the output folder; nothing reused is written.
    with tempfile.TemporaryDirectory(prefix='nrkmeans_', dir=out) as tmp:
        directory = Path(tmp)
        np.save(directory / 'vectors.npy', x, allow_pickle=False)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
        try:
            with open(directory / 'child.log', 'w+') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                         '--worker', str(directory)], env=env,
                                        stdout=log, stderr=log, timeout=timeout_seconds, check=False)
                log.seek(0)
                rec['child_log_tail'] = log.read()[-8000:]
            rec['child_returncode'] = result.returncode
            assert result.returncode == 0, f'child exited with code {result.returncode}'
            status = json.loads((directory / 'worker.json').read_text())
            assert {'dropped', 'reason', 'fit_seconds'} <= status.keys(), 'worker record keys'
            rec.update(status)
            if not rec['dropped']:
                labels = np.load(directory / 'labels.npy', allow_pickle=False)
                assert labels.dtype == np.int64 and labels.shape == (len(x), 2), 'child label alignment'
                assert (labels >= 0).all(), 'child negative label'
        except subprocess.TimeoutExpired:
            rec.update(dropped=True, reason=f'NrKmeans exceeded {timeout_seconds} seconds')
        except (Exception, SystemExit) as exc:
            labels = None
            rec.update(dropped=True, reason=f'{type(exc).__name__}: {exc}')
    rec['run_seconds'] = time.monotonic() - started
    return labels, rec


def select_and_merge(labels, x, g, content, rs, rec):
    """Choose lower painting-content AMI (ties first), then row-weighted L6 merge."""
    clusterings = [np.unique(labels[:, k], return_inverse=True)[1].astype(np.int64) for k in (0, 1)]
    scores = [float(ami(content, p)) for p in clusterings]
    chosen = 0 if scores[0] <= scores[1] else 1
    # L7 does not say rows or paintings: N is chosen at painting level; the row-level AMIs are recorded beside it
    rows = [float(ami(content[g], p[g])) for p in clusterings]
    rec.update(ami_content_paintings_both=scores, ami_content_rows_both=rows, chosen_clustering=chosen,
               choice_level='paintings', choice_row_level_would_pick=0 if rows[0] <= rows[1] else 1,
               raw_sizes_paintings_both=[np.bincount(p).tolist() for p in clusterings],
               raw_sizes_rows_both=[np.bincount(p[g]).tolist() for p in clusterings],
               merge_min_rows=200)
    try:
        xp = rs.rc.unit(np.asarray(x, dtype=np.float32))
        assert np.isfinite(xp).all(), 'L6 merge vectors finite'
        local, moves = rs.rto.merge_small(clusterings[chosen][g], xp[g], 200)
        rec.update(G.describe(local, g, content), merge_moves=moves)
        assert np.bincount(local).min() >= 200, 'L6 merge min rows'
        if rec['n_groups'] < 2:
            rec.update(dropped=True, reason='fewer than 2 groups after merge')
            return None
        return local
    except (Exception, SystemExit) as exc:
        rec.update(dropped=True, reason=f'{type(exc).__name__}: {exc}')
        return None


def build(out=None, limit=None, timeout_seconds=TIMEOUT_SECONDS):
    C.assert_rule()
    C.assert_threads()
    assert 0 < timeout_seconds <= TIMEOUT_SECONDS, 'timeout must be within 30 minutes'
    assert limit is not None or timeout_seconds == TIMEOUT_SECONDS, 'full run requires L7 timeout'
    if limit is not None:
        assert 17 <= limit < 36518, 'trial painting limit'
        assert out is not None, 'trial requires explicit output directory'
        assert Path(out).resolve().is_relative_to(C.HERE / 'results/ticket04_trial'), 'trial output location'
    out = Path(out) if out is not None else C.RESULTS / 'groupings'
    targets = [out / f'r7_grouping_N.{ext}' for ext in ('npz', 'json')]
    assert all(not p.exists() or C.in_smoke(p) for p in targets), 'output exists (write once)'
    rs = C.load('run_step1')
    data, _, st_full, _, g_full = rs.base_data()
    assert st_full.dtype == np.int64 and st_full.shape == (183694,), 'scorer-train shape/dtype'
    assert (np.diff(st_full) > 0).all() and len(G.first_rows(g_full)) == 36518, 'scorer-train order/count'
    source, row_images, provenance = G.read_source(G.STYLE, data, rs)
    mask = g_full < limit if limit is not None else np.ones(len(g_full), dtype=bool)
    st, g = st_full[mask], g_full[mask]
    x, provenance['painting_vectors'] = G.painting_vectors(source, row_images, st, g)
    key = '20261031_pseudo_partitions/results/partitions.npz'
    path = C.path_of(key)
    assert C.sha_file(path) == C.INPUTS[key], 'E2 SHA'
    with np.load(path, allow_pickle=False) as z:
        assert {'image', 'local_groups'} <= set(z.files), 'E2 keys'
        assert np.array_equal(z['local_groups'], g_full), 'E2 alignment'
        assert z['image'].dtype == np.int64 and z['image'].shape == st_full.shape, 'content shape/dtype'
        content_full = G.painting_labels(z['image'], g_full)
        assert len(np.unique(content_full)) == 64, 'L2 content count'
        content = content_full[:len(x)]
    out.mkdir(parents=True, exist_ok=True)
    print(f'NrKmeans: {len(x)} paintings, timeout {timeout_seconds}s', flush=True)
    labels, rec = fit_child(x, out, timeout_seconds)
    local = select_and_merge(labels, x, g, content, rs, rec) if labels is not None else None
    arrays = {'scorer_train': st, 'g_st': g}
    if local is not None:
        arrays['N'] = local
    C.write_npz_once(targets[0], **arrays)
    result = C.write_json_once(targets[1], {'trial': limit is not None, 'limit': limit,
        'requested': ['N'], 'groupings': {'N': rec}, 'sources': {'CSD': provenance},
        'scorer_train_sha256': G.sha_array(st), 'g_st_sha256': G.sha_array(g),
        'n_rows': len(st), 'n_paintings': len(x)})
    print(f'N dropped={rec["dropped"]}: {rec["reason"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--timeout-seconds', type=int, default=TIMEOUT_SECONDS,
                        help='shorter timeout allowed only for trials')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
    else:
        build(args.out, args.limit, args.timeout_seconds)


if __name__ == '__main__':
    main()
