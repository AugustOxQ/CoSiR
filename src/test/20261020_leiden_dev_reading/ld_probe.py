"""CPU-only development Leiden probe; existing artifacts are read-only.

Run from /project/CoSiR with the CoSiR Python environment. Singleton target and
image_encoder vectors have model_seed=null and are scored once, irrespective of
--seeds. --seeds selects model seeds only; clustering always uses seeds 0,1,2.
The image_encoder comparator is default Qwen for every aspect (the sheet's
style bar instead uses CLIP). Output files must be new: reruns need a new --out
and a directory without ld_table.md. No existing file is overwritten.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
for folder in ('20261017_image_transfer_pilot', '20261018_purified_buddy_pilot',
               '20261018_aspect_graph_sheet', '20261018_percept_core',
               '20261019_model_sheet'):
    sys.path.insert(0, str(TEST / folder))

import igraph as ig
import leidenalg
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits
import ms_score as sc

ROWS = ('D2', 'T1', 'percept', 'target', 'image_encoder')
GRAPHS = dict(emotion='Gres', style='Fres05', genre='R0o')
CLUSTER_SEEDS = (0, 1, 2)


def normalize(x):
    x = np.array(x, dtype=np.float32, order='C', copy=True)
    if x.ndim != 2 or len(x) != 5000 or not np.isfinite(x).all():
        raise ValueError('Expected 5000 finite vectors')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError('Zero vector cannot be L2-normalized')
    x /= norms
    return x


def union_knn(x, k=20, block=256):
    """Exact cosine neighbors; union of directed choices, unweighted/simple.

    Ties at the boundary use NumPy argpartition's deterministic choice for the
    installed NumPy version. All paintings participate, including missing labels.
    """
    n = len(x)
    edges = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = x[start:stop] @ x.T
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        neighbors = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        source = np.repeat(np.arange(start, stop), k)
        edges.extend(zip(source.tolist(), neighbors.ravel().tolist()))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.simplify(multiple=True, loops=True)
    assert g.is_simple() and not g.is_directed() and 'weight' not in g.es.attributes()
    return g


def ami_values(partitions, y):
    mask = y >= 0
    return [float(adjusted_mutual_info_score(y[mask], np.asarray(p)[mask]))
            for p in partitions]


def network(s, row, aspect, seed):
    path = sc.OUTPUTS / row / f'{aspect}__{GRAPHS[aspect]}__s{seed}.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    c = meta['config']
    assert (c['name'], c['aspect'], c['seed'], c['graph_row']) == (
        row, aspect, seed, GRAPHS[aspect])
    assert c['sample_sha256'] == sc.fingerprint(s)
    assert sc.sha_file(path) == meta['output_sha256']
    with np.load(path, allow_pickle=False) as z:
        assert z['img'].shape == (5000, 2048) and z['img'].dtype == np.float16
        return sc.check_view(z['img']), str(path)


def vector_sets(s, rows, aspects, seeds):
    # Avoid loading all comparator data for a network-only test.
    bars = sc.bar_vectors(s) if any(r in rows for r in ROWS[2:]) else {}
    if 'image_encoder' in rows:
        qwen = sc.encoder_vectors(s, 'qwen_default')
        bars['image_encoder'] = {a: qwen for a in aspects}
    for row in rows:
        for aspect in aspects:
            if row in ('D2', 'T1'):
                for seed in seeds:
                    x, source = network(s, row, aspect, seed)
                    yield row, aspect, seed, x, source
            elif row == 'percept':
                for seed in seeds:
                    yield row, aspect, seed, bars[row][aspect][seed], f'PercepT {sc.BEST[aspect]}'
            else:
                yield row, aspect, None, bars[row][aspect], (
                    'qwen_default' if row == 'image_encoder' else sc.PROMPTS[aspect])


def sheet_reference(sheet, row, aspect, seed):
    if row in ('D2', 'T1'):
        return sheet['runs'][f'{row}/{aspect}/s{seed}']['scores']['img'][aspect]['ami']['mean']
    if row == 'target' or (row == 'image_encoder' and aspect != 'style'):
        return sheet['bars'][row][aspect]['ami']['mean']
    if row == 'percept':
        recorded = json.loads((sc.PC / 'pc_results.json').read_text())
        return recorded['arms'][sc.BEST[aspect]]['seeds'][str(seed)]['img'][aspect]['mean']
    return None


def score(x, y):
    x = normalize(x)
    k20 = ami_values(sc.pav_score.partitions(x, 20), y)
    graph = union_knn(x)
    leiden = []
    for seed in CLUSTER_SEEDS:
        p = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, seed=seed)
        sizes = np.asarray(p.sizes())
        leiden.append(dict(seed=seed, communities=len(sizes),
                           communities_ge50=int((sizes >= 50).sum()),
                           largest_share=float(sizes.max() / len(x)),
                           ami=ami_values([p.membership], y)[0]))
    c = int(round(float(np.mean([p['communities'] for p in leiden]))))
    kc = k20 if c == 20 else ami_values(sc.pav_score.partitions(x, c), y)
    return dict(k20_ami_mean=float(np.mean(k20)), k20_ami_values=k20,
                leiden=leiden, leiden_ami_mean=float(np.mean([p['ami'] for p in leiden])),
                leiden_count_mean=float(np.mean([p['communities'] for p in leiden])),
                c=c, kmeans_at_c_ami_mean=float(np.mean(kc)), kmeans_at_c_ami_values=kc,
                graph_vertices=graph.vcount(), graph_edges=graph.ecount())


def table(records):
    lines = ['# Leiden development probe', '',
             'Mean ± population SD over model-seed means; singleton comparators have SD 0. '
             'Leiden AMI/count first average over clustering seeds 0,1,2. '
             'Image encoder: default Qwen for all aspects. All inputs L2-normalized float32.', '',
             '| Row | Aspect | Model seeds | k20 AMI | Leiden AMI | Leiden count | k-means at c AMI |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in ROWS:
        for aspect in sc.ASPECTS:
            group = [r for r in records if (r['row'], r['aspect']) == (row, aspect)]
            if not group:
                continue
            cells = []
            for key in ('k20_ami_mean', 'leiden_ami_mean', 'leiden_count_mean', 'kmeans_at_c_ami_mean'):
                values = [r[key] for r in group]
                cells.append(f'{np.mean(values):.6f} ± {np.std(values, ddof=0):.6f}')
            lines.append(f'| {row} | {aspect} | {len(group)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', choices=ROWS, default=list(ROWS))
    parser.add_argument('--aspects', nargs='+', choices=sc.ASPECTS, default=list(sc.ASPECTS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=CLUSTER_SEEDS, default=list(CLUSTER_SEEDS))
    parser.add_argument('--out', type=Path, default=HERE / 'outputs/ld_results.json')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(HERE):
        parser.error('--out must stay inside the probe folder')
    md = args.out.parent / 'ld_table.md'
    if args.out.exists() or md.exists() or args.out == md:
        parser.error('output files must be new; select a fresh output directory')
    for key in ('rows', 'aspects', 'seeds'):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    s = sc.load_frozen()
    labels = sc.label_vectors(s, s['subset_pid'])
    sheet_path = sc.OUTPUTS / 'ms_results.json'
    sheet = json.loads(sheet_path.read_text()) if sheet_path.exists() else None
    if sheet is not None:
        assert sheet['sample_sha256'] == sc.fingerprint(s)
    records = []
    for row, aspect, seed, x, source in vector_sets(s, args.rows, args.aspects, args.seeds):
        tick = time.monotonic()
        record = dict(row=row, aspect=aspect, model_seed=seed, source=source)
        record.update(score(x, labels[aspect]))
        expected = sheet_reference(sheet, row, aspect, seed) if sheet is not None else None
        record['sheet_k20_ami_mean'] = expected
        record['sheet_k20_abs_difference'] = (abs(record['k20_ami_mean'] - expected)
                                               if expected is not None else None)
        record['seconds'] = time.monotonic() - tick
        records.append(record)
    differences = [r['sheet_k20_abs_difference'] for r in records
                   if r['sheet_k20_abs_difference'] is not None]
    result = dict(records=records, sample_sha256=sc.fingerprint(s),
                  total_seconds=time.monotonic() - start,
                  max_abs_difference_vs_sheet=max(differences) if differences else None,
                  protocol=dict(knn=20, graph='union cosine, undirected unweighted simple',
                                leiden='ModularityVertexPartition (resolution 1)',
                                clustering_seeds=list(CLUSTER_SEEDS), threads=8,
                                normalization='L2 float32; sheet only casts saved network float16',
                                image_encoder='qwen_default for every aspect',
                                c='rounded mean Leiden count; Python round'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    with md.open('x') as f:
        f.write(table(records))
    print(f'Completed {len(records)} vector sets in {time.monotonic() - start:.2f} seconds; '
          f'max k20 difference vs sheet={result["max_abs_difference_vs_sheet"]}; {args.out}')


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
