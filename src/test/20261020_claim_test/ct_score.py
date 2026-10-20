"""Fixed held partitions and rule §3 paired painting/model-seed statistics (CPU).

Run after ticket 09's V1 queue gate. R-COL's JSON supplies
``eligible_seeds: {B2: [...], B3: [...], B4: [...], B5: [...]}``.
The computed outcome requires external V2 agreement before it can be reported.
"""
import argparse
import hashlib
import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_mutual_info_score as ami
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ARM_ORDER = ('K1', 'B1', 'B2', 'B3', 'B4', 'B5')
HELD_ROWS = 12281
SEEDS = list(range(5))
OUTPUTS = HERE / 'outputs'
HELD_STEMS = dict(K1='d2/emotion', B1='percept/core__k40/P3',
                  B2='percept/paper__strict/P1', B3='percept/paper__strict/P4',
                  B4='percept/paper__k40/P1', B5='percept/paper__k40/P4')
for folder in ('20261017_aspect_premise_checks', '20261017_prompted_aspect_views'):
    sys.path.insert(0, str(HERE.parent / folder))
from apc_data import l2
from pav_score import partitions
import apc_data
import pav_score


def load_held_output(path, arm, names, held_sha256):
    """Validate saved order via its list hash; non-finite paper runs collapse (V4)."""
    assert arm in ARM_ORDER
    assert len(names) == HELD_ROWS and names == sorted(set(names))
    assert hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest() == held_sha256
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['held_sha256'] == held_sha256 and meta['rows'] == HELD_ROWS, path
    with np.load(path, allow_pickle=False) as z:
        x = z['img' if arm == 'K1' else 'prob']
        if 'painting_names' in z:
            assert z['painting_names'].tolist() == names, path
    assert x.ndim == 2 and len(x) == HELD_ROWS and x.shape[1] > 0, path
    if arm == 'K1':
        assert x.shape[1] == 2048 and x.dtype == np.float16, path
        assert len(meta['weights_sha256']) == 64, path
    else:
        assert x.dtype == np.float32, path
        if arm in ('B1', 'B4', 'B5'):
            assert x.shape[1] == 40, path
    x = x.astype(np.float32)
    if arm not in ('K1', 'B1') and not np.isfinite(x).all():
        return None
    assert np.isfinite(x).all(), path
    if arm == 'K1':
        assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3), path
    else:
        assert ((x >= 0) & (x <= 1)).all(), path
    return x


def output_partitions(x, kind):
    """Cluster all rows once; PercepT also has its lowest-index top-topic reading."""
    x = np.asarray(x, dtype=np.float32)
    assert kind in ('k1', 'percept')
    assert x.ndim == 2 and x.shape[1] > 0 and np.isfinite(x).all()
    if kind == 'k1':
        assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3)
    else:
        assert ((x >= 0) & (x <= 1)).all()
    with threadpool_limits(limits=8):
        r20 = np.asarray(partitions(l2(x) if kind == 'percept' else x, 20))
    result = dict(r20=r20)
    if kind == 'percept':
        result['own'] = x.argmax(axis=1)[None, :]
    return result


def score_dev(x, labels, kind='k1'):
    """Sheet regression seam: {label name: three r20 AMIs} on development vectors.

    The two-argument form takes the sheet's vector input without normalization.
    Use kind='percept' for raw float32 sigmoid probabilities; they are normalized
    exactly as the core scorer. labels maps aspect names to integer arrays.
    """
    assert len(x) == 5000
    assert labels
    assert all(y.shape == (len(x),) and np.issubdtype(y.dtype, np.integer)
               and (y >= 0).any() for y in labels.values())
    ps = output_partitions(x, kind)['r20']
    return {name: [float(ami(y[y >= 0], p[y >= 0])) for p in ps] for name, y in labels.items()}


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fixed_partitions(x, kind, path, held_sha256, source_sha256):
    """Persist/reuse one all-held partition set, with inputs and hashes for V2."""
    assert len(x) == HELD_ROWS
    path = Path(path)
    identity = dict(held_sha256=held_sha256, source_sha256=source_sha256, kind=kind,
                    rows=len(x), sklearn=sklearn.__version__,
                    code_sha256={Path(p).name: _sha_file(p) for p in
                                 (__file__, pav_score.__file__, apc_data.__file__)})
    if path.exists():
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['identity'] == identity, 'fixed partition inputs changed'
        assert meta['partitions_sha256'] == _sha_file(path)
        with np.load(path, allow_pickle=False) as z:
            result = {k: z[k] for k in z.files}
    else:
        result = output_partitions(x, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.pending.npz')
        np.savez_compressed(tmp, **result)
        tmp.replace(path)
        _atomic_json(path.with_suffix('.json'), dict(identity=identity, partitions_sha256=_sha_file(path)))
    assert set(result) == ({'r20'} if kind == 'k1' else {'r20', 'own'})
    for reading, ps in result.items():
        assert ps.shape == (3 if reading == 'r20' else 1, HELD_ROWS)
        assert np.issubdtype(ps.dtype, np.integer)
    return result


def prepare_held(names, held_sha256, rcol_eligible, output_root=OUTPUTS):
    """Validate every saved run before scoring; V4 only drops non-finite paper runs.

    rcol_eligible maps B2..B5 to ascending uncollapsed seeds from the saved
    R-COL file. Missing eligible outputs fail; finite collapsed outputs keep
    partitions for the later diagnostics but never enter the bar.
    """
    assert set(rcol_eligible) == set(ARM_ORDER[2:])
    assert all(ss == sorted(set(ss)) and set(ss) <= set(SEEDS) for ss in rcol_eligible.values())
    eligible = dict(K1=SEEDS.copy(), B1=SEEDS.copy(), **{k: list(v) for k, v in rcol_eligible.items()})
    nonfinite = {arm: [] for arm in ARM_ORDER[2:]}
    arrays, sources = {}, {}
    root = Path(output_root) / 'held'
    for arm in ARM_ORDER:
        arrays[arm], sources[arm] = {}, {}
        for seed in SEEDS:
            path = root / f'{HELD_STEMS[arm]}__s{seed}.npz'
            if not path.exists():
                assert seed not in eligible[arm], f'missing eligible output: {path}'
                sources[arm][str(seed)] = dict(status='rcol_collapsed_no_output')
                continue
            x = load_held_output(path, arm, names, held_sha256)
            sources[arm][str(seed)] = dict(path=str(path), output_sha256=_sha_file(path),
                                           sidecar_sha256=_sha_file(path.with_suffix('.json')),
                                           status='finite')
            if x is None:
                nonfinite[arm].append(seed)
                eligible[arm] = [s for s in eligible[arm] if s != seed]
                sources[arm][str(seed)]['status'] = 'nonfinite_held'
            else:
                arrays[arm][seed] = x
    # No AMI is possible until every required output has passed V3.
    fixed = {arm: {} for arm in ARM_ORDER}
    for arm, runs in arrays.items():
        for seed, x in runs.items():
            path = root / 'partitions' / f'{arm}__s{seed}.npz'
            fixed[arm][seed] = fixed_partitions(x, 'k1' if arm == 'K1' else 'percept', path,
                                               held_sha256, sources[arm][str(seed)]['output_sha256'])
            sources[arm][str(seed)].update(partitions=str(path), partitions_sha256=_sha_file(path))
    eligibility = dict(eligible_seeds=eligible, rcol_eligible_seeds=rcol_eligible, nonfinite_held=nonfinite)
    return fixed, eligibility, sources


def score_indices(fixed, labels, rows, seeds):
    """A_K1, every A_e and d for a paired row/seed draw of fixed partitions."""
    y = labels[rows]
    mask = y >= 0
    per_seed, means = {}, {}
    for arm, chosen in seeds.items():
        if not len(chosen):
            continue
        per_seed[arm] = {
            int(seed): {reading: [float(ami(y[mask], p[rows][mask])) for p in ps]
                        for reading, ps in fixed[arm][seed].items()}
            for seed in np.unique(chosen)}
        means[arm] = {
            reading: float(np.mean([np.mean(per_seed[arm][int(s)][reading]) for s in chosen]))
            for reading in per_seed[arm][int(chosen[0])]}
    a_k1 = means.pop('K1')['r20']
    entries = {f'{arm}-{reading}': value for arm, readings in means.items()
               for reading, value in readings.items()}
    return dict(d=a_k1 - max(entries.values()), A_K1=a_k1, A_e=entries, per_seed=per_seed)


def draw_bootstrap(n_rows, eligible, n_resamples=2000):
    """Draw the entire RNG transcript before any bootstrap AMIs are scheduled."""
    assert set(eligible) <= set(ARM_ORDER)
    assert n_rows > 0 and n_resamples > 0
    assert all(list(ss) == sorted(set(ss)) for ss in eligible.values())
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n_rows, size=n_rows)
        seeds = {arm: rng.choice(eligible[arm], size=len(eligible[arm]), replace=True)
                 for arm in ARM_ORDER if len(eligible.get(arm, []))}
        draws.append((rows, seeds))
    return draws


def _init_worker(fixed, labels):
    global _worker_fixed, _worker_labels
    _worker_fixed, _worker_labels = fixed, labels
    threadpool_limits(limits=1)


def _draw_difference(draw):
    rows, seeds = draw
    return score_indices(_worker_fixed, _worker_labels, rows, seeds)['d']


def paired_bootstrap(fixed, labels, eligible, n_resamples=2000, workers=8):
    """Point estimate and percentile interval; draws are independent of scheduling."""
    assert 1 <= workers <= 8
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
    for arm, seeds in eligible.items():
        for seed in seeds:
            readings = fixed[arm][seed]
            assert set(readings) == ({'r20'} if arm == 'K1' else {'r20', 'own'})
            for reading, ps in readings.items():
                assert ps.shape == (3 if reading == 'r20' else 1, len(labels))
                assert np.issubdtype(ps.dtype, np.integer)
    draws = draw_bootstrap(len(labels), eligible, n_resamples)
    point = score_indices(fixed, labels, np.arange(len(labels)), eligible)
    if workers == 1:
        differences = [score_indices(fixed, labels, rows, seeds)['d'] for rows, seeds in draws]
    else:
        with multiprocessing.Pool(workers, initializer=_init_worker, initargs=(fixed, labels)) as pool:
            differences = pool.map(_draw_difference, draws)
    point.update(interval=np.percentile(differences, [2.5, 97.5]).tolist(), bootstrap_d=differences)
    return point


def stage1_outcome(d, lower):
    """Rule §4 outcome ID, decided on full precision values."""
    assert np.isfinite(d) and np.isfinite(lower)
    if d >= .01:
        return 'S1-GO' if lower > 0 else 'S1-MM'
    return 'S1-IN' if d > 0 else 'S1-NO'


def write_verdict(path, result, eligibility, held_sha256, labels, code_commit, sources):
    """Write the statistics before diagnostics; V1/V2 are checked by the run chat.

    The saved outcome is computed, not licensed for reporting until independent
    V2 reproduction agrees (rule §4). This module computes no diagnostics.
    """
    counts = {}
    for name, y in labels.items():
        values, ns = np.unique(y[y >= 0], return_counts=True)
        counts[name] = dict(classes={str(v): int(n) for v, n in zip(values, ns)},
                            missing=int((y < 0).sum()))
    verdict = dict(schema=1, stage=1, **result, eligibility=eligibility,
                   held_sha256=held_sha256, code_commit=code_commit, sources=sources,
                   label_counts=counts, outcome_id=stage1_outcome(result['d'], result['interval'][0]),
                   bootstrap=dict(resamples=len(result['bootstrap_d']), rng='numpy.random.default_rng(0)',
                                  arm_order=list(ARM_ORDER), percentiles=[2.5, 97.5], method='linear'),
                   reporting_requires='V1 queue gate and independent V2 agreement')
    # JSON makes integer seed keys strings; return exactly the persisted value.
    verdict = json.loads(json.dumps(verdict, allow_nan=False))
    _atomic_json(path, verdict)
    return verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = p.parse_args()
    # Kept local so the pure scorer can be tested before ticket 01 merges.
    import ct_held

    rcol_path = OUTPUTS / 'percept/rcol_stage1.json'
    rcol = json.loads(rcol_path.read_text())
    names, held_sha256 = ct_held.read_held_list()
    assert names == ct_held.held_names()
    assert ct_held.names_sha256(names) == held_sha256
    labels = ct_held.painting_labels(names)
    assert set(labels) == {'emotion', 'style', 'genre'}
    assert all(y.shape == (HELD_ROWS,) and np.issubdtype(y.dtype, np.integer) for y in labels.values())
    assert (labels['emotion'] >= 0).any()
    fixed, eligibility, sources = prepare_held(names, held_sha256, rcol['eligible_seeds'])
    result = paired_bootstrap(fixed, labels['emotion'], eligibility['eligible_seeds'], workers=args.workers)
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    verdict_path = HERE / 'verdict_stage1.json'
    write_verdict(verdict_path, result, eligibility, held_sha256, labels, code_commit,
                  dict(held_outputs=sources, rcol=dict(path=str(rcol_path), sha256=_sha_file(rcol_path),
                                                     record=rcol)))
    print(f'Wrote {verdict_path}; V2 agreement is required before reporting the outcome.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
