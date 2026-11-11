"""r25 AMI-at-20 cells using r17's scorer unchanged (CoSiR environment).

    python pb_score.py [--only {vlm2vec,bertopic}] [--record]
    python pb_score.py --gate-only

PB_SMOKE=1 PB_OUTPUTS=<scratch> uses r17's 40 smoke draws and refuses --record.
All three candidate gates pass before scoring; cells are written together only after every requested cell succeeds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_paths as P
from pb_io import array_sha, file_sha, names_sha, write_json

NS = P.add_scorer_paths()
import ns_clu
import ns_metrics

COMPARATORS = ('vlm2vec', 'bertopic')


def check_names(got, names, path):
    if list(got) != list(names):
        raise ValueError(f'{path}: names must exactly match r17 held names in order')


def check_topics(topics, n, path, k=P.K):
    t = np.asarray(topics)
    if t.shape != (n,) or not np.issubdtype(t.dtype, np.integer):
        raise ValueError(f'{path}: need integer topics [{n}], got {t.shape} {t.dtype}')
    if (t < 0).any() or (t >= k).any():
        raise ValueError(f'{path}: topics outside [0, {k})')
    if len(np.unique(t)) != k:
        raise ValueError(f'{path}: expected {k} distinct topics, got {len(np.unique(t))}')
    return t.astype(np.int32)


def load_vlm2vec(store, aspect, names):
    root = Path(store) / aspect
    path, key_path = root / 'emb.npy', root / 'keys.json'
    for p in (path, key_path):
        if not p.is_file():
            raise FileNotFoundError(f'vlm2vec/{aspect}: required comparator input missing: {p}')
    keys = json.loads(key_path.read_text())
    # Side extractors may store just the ordered list, or a named list with provenance.
    check_names(keys['names'] if isinstance(keys, dict) else keys, names, key_path)
    rows = np.load(path, allow_pickle=False)
    if (rows.ndim != 2 or rows.shape[0] != len(names) or rows.shape[1] < 1
            or not np.issubdtype(rows.dtype, np.floating) or not np.isfinite(rows).all()):
        raise ValueError(f'{path}: need finite floating embedding rows [n, d], got {rows.shape} {rows.dtype}')
    deviation = float(np.abs(np.linalg.norm(rows.astype(np.float64), axis=1) - 1).max())
    if not deviation < NS.TOL_UNIT:
        raise ValueError(f'{path}: rows not unit norm (max deviation {deviation}, tolerance {NS.TOL_UNIT})')
    hashes = {str(p): file_sha(p) for p in (path, key_path)}
    if (root / 'meta.json').is_file():
        hashes[str(root / 'meta.json')] = file_sha(root / 'meta.json')
    return rows.astype(np.float32), hashes


def load_bertopic(root, names):
    parts, hashes = [], {}
    for seed in P.UMAP_SEEDS:
        path = Path(root) / f'topics_seed{seed}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'bertopic: required comparator input missing: {path}')
        with np.load(path, allow_pickle=False) as z:
            check_names(z['names'].tolist(), names, path)
            parts.append(check_topics(z['topics'], len(names), path))
        hashes[str(path)] = file_sha(path)
        side = path.with_suffix('.json')
        if side.is_file():
            meta = json.loads(side.read_text())
            if meta['npz_sha256'] != hashes[str(path)]:
                raise ValueError(f'{path}: sidecar npz_sha256 mismatch')
            if meta.get('smoke'):
                raise ValueError(f'{path}: smoke partition cannot be used for held scoring')
            hashes[str(side)] = file_sha(side)
    return np.stack(parts).astype(np.int32), hashes


def candidate_inputs():
    import ns_data
    import ns_clu_cells
    names = ns_data.keys(P.SET, P.SPLIT)
    if len(names) != P.N_HELD or len(set(names)) != P.N_HELD:
        raise ValueError(f'candidate: expected {P.N_HELD} unique held names')
    data = {}
    for aspect in P.ASPECTS:
        y = ns_data.labels(P.SET, P.SPLIT, aspect)
        if y.dtype != np.int64 or y.shape != (len(names),) or (y < -1).any():
            raise ValueError(f'candidate/{aspect}: invalid labels')
        if aspect == 'genre' and int((y >= 0).sum()) != 9949:
            raise ValueError('candidate/genre: expected 9,949 labelled rows')
        cand = ns_clu_cells._cand_parts(P.SET, P.SPLIT, P.CSTAR_ENC, aspect, P.K)
        if cand.dtype != np.int32 or cand.shape != (len(NS.SEEDS), len(NS.KMEANS_STATES), len(names)):
            raise ValueError(f'candidate/{aspect}: expected int32 [5, 3, n], got {cand.shape} {cand.dtype}')
        if (cand < 0).any() or (cand >= P.K).any():
            raise ValueError(f'candidate/{aspect}: invalid cluster ids')
        data[aspect] = (cand, y)
    return names, data


def candidate_gate(data, reference, tol=P.R18_CAND_TOL):
    """All aspects checked as one gate, with exactly r17's metric reduction."""
    by_id = {}
    for c in reference['cells']:
        if c['id'] in by_id:
            raise ValueError(f'duplicate r18 cell {c["id"]}')
        by_id[c['id']] = c
    gate = {}
    for aspect in P.ASPECTS:
        cand, y = data[aspect]
        row = ns_clu.metric_table('cstar', aspect, cand, y, P.K, metrics=('ami',))[0]
        key = f'SC1/{P.SET}/{P.CSTAR_ENC}/{aspect}/ami/{P.K}/tpips'
        expected = float(by_id[key]['cand'])
        difference = abs(row['mean'] - expected)
        if not np.isfinite(expected) or not difference <= tol:
            raise ValueError(f'candidate gate FAILED for {aspect}: {row["mean"]:.17g} vs r18 {expected:.17g}; '
                             f'abs difference {difference:.3g} > {tol}')
        gate[aspect] = dict(cand=row['mean'], r18_cand=expected, abs_difference=difference,
                            tolerance=tol, per_seed=row['per_seed'])
    return gate


def score_cell(comparator, aspect, cand, comp, y, r, gate, input_sha256s):
    draws = (NS.SMOKE_DRAWS if P.SMOKE else NS.N_DRAWS)['clu']
    res = ns_clu.paired_cell(cand, comp, y, P.K, r={'ami': r}, metrics=('ami',),
                              draws=draws, rng_seed=NS.BOOT_SEED)['ami']
    if abs(res['cand'] - gate['cand']) > P.R18_CAND_TOL:
        raise ValueError(f'{aspect}: paired_cell candidate disagrees with passed gate')
    keep = y >= 0
    per_state = [ns_metrics.ami(y[keep], p[keep]) for p in comp]
    return dict(res, id=f'{comparator}/{aspect}/ami/{P.K}', label=res['outcome'], candidate='cstar',
                comparator=comparator, set=P.SET, split=P.SPLIT, encoder=P.CSTAR_ENC, aspect=aspect,
                metric='ami', k=P.K, n_rows=len(y), n_labelled=int(keep.sum()),
                cand_per_seed_ami=gate['per_seed'], comp_per_state_ami=per_state,
                comparator_states=list(P.UMAP_SEEDS if comparator == 'bertopic' else NS.KMEANS_STATES),
                gate=gate, smoke=bool(P.SMOKE), bootstrap_seed=NS.BOOT_SEED,
                r_source=f'{P.RANGES}:{P.SET}/{P.CSTAR_ENC}/{aspect}/ami/{P.K}:r',
                input_sha256s=dict(input_sha256s, candidate_partitions_array=array_sha(cand),
                                   comparator_partitions_array=array_sha(comp), labels_array=array_sha(y)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', choices=COMPARATORS)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--gate-only', action='store_true')
    a = ap.parse_args(argv)
    if P.SMOKE and a.record:
        ap.error('PB_SMOKE never writes RECORDS; omit --record')
    if P.SMOKE and (P.OUT.resolve() == P.DEFAULT_OUT.resolve() or P.RECORDS.resolve() in P.OUT.resolve().parents
                    or P.OUT.resolve() == P.RECORDS.resolve()):
        ap.error('PB_SMOKE requires PB_OUTPUTS pointing to a scratch directory outside RECORDS')
    if a.gate_only and a.record:
        ap.error('--gate-only cannot write --record')
    start = time.monotonic()
    reference = json.loads(P.R18_LABELS.read_text())
    names, data = candidate_inputs()
    gate = candidate_gate(data, reference)
    print(json.dumps(dict(gate='PASS', aspects=gate, seconds=time.monotonic() - start)), flush=True)
    if a.gate_only:
        return 0
    ranges = json.loads(P.RANGES.read_text())
    common_hashes = {str(p): file_sha(p) for p in (P.RANGES, P.R18_LABELS)}
    common_hashes['names_newline_utf8'] = names_sha(names)
    comparators = (a.only,) if a.only else COMPARATORS
    bertopic = load_bertopic(P.BERTOPIC_OUT, names) if 'bertopic' in comparators else None
    cells = []
    for comparator in comparators:
        for aspect in P.ASPECTS:
            cand, y = data[aspect]
            if comparator == 'bertopic':
                comp, hashes = bertopic
            else:
                rows, hashes = load_vlm2vec(P.VLM2VEC_STORE, aspect, names)
                comp = ns_metrics.partitions(rows, P.K, P.SET)
            key = f'{P.SET}/{P.CSTAR_ENC}/{aspect}/ami/{P.K}'
            r = float(ranges[key]['r'])
            if not np.isfinite(r) or r < 0:
                raise ValueError(f'{P.RANGES}:{key}: r must be finite and nonnegative')
            cells.append(score_cell(comparator, aspect, cand, comp, y, r, gate[aspect], dict(common_hashes, **hashes)))
    path = P.OUT / 'score' / 'cells.json'
    write_json(path, cells)
    if a.record:
        write_json(P.RECORDS / 'cells.json', cells)
    print(f'{len(cells)} cells -> {path}; {time.monotonic() - start:.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
