"""Post-verdict Stage 1 diagnostics; CPU only, at most eight threads.

Reads run artifacts from --outputs (main checkout by default), writes only in
this round. CPU fit inference rebuilds 04/05's saved architectures and weights;
it is diagnostic inference, distinct from their training-device held replay.
No labels choose a recipe, seed, eligibility or outcome here.
"""
import argparse
import importlib.util
import json
import time
from functools import lru_cache
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import triu
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             fowlkes_mallows_score, v_measure_score)
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
import ct_score as sc
import ct_held
from pav_prompts import PROMPTS as QWEN_INSTRUCTIONS

SEPT = Path('/project/CoSiR-buddy_prototype_conditioning')
TARGETS = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts')
PROMPTS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
METRICS = dict(ami=adjusted_mutual_info_score, v=v_measure_score,
               ari=adjusted_rand_score, fmi=fowlkes_mallows_score)
COMPARATORS = dict(B1=('core', 'k40', 'P3'), B2=('paper', 'strict', 'P1'),
                   B3=('paper', 'strict', 'P4'), B4=('paper', 'k40', 'P1'),
                   B5=('paper', 'k40', 'P4'))


def read_verdict(path):
    """The first file boundary, before any diagnostic inputs or computations."""
    path = Path(path)
    assert path.is_file(), 'diagnostics require verdict_stage1.json first'
    record = json.loads(path.read_text())
    assert record['stage'] == 1 and record['outcome_id'] in ('S1-GO', 'S1-MM', 'S1-IN', 'S1-NO')
    assert set(record['eligibility']['eligible_seeds']) == set(sc.ARM_ORDER)
    assert record['eligibility']['eligible_seeds']['K1'] == sc.SEEDS
    assert record['eligibility']['eligible_seeds']['B1'] == sc.SEEDS
    return record


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    assert (np.linalg.norm(x, axis=1) > 0).all()
    return sc.l2(x)


def union_knn(x, k=20, block=256):
    """Development probe's exact cosine union graph, including its tie rule."""
    x = normalize(x)
    n = len(x)
    assert 0 < k < n
    edges = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = x[start:stop] @ x.T
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        neighbors = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        edges.extend(zip(np.repeat(np.arange(start, stop), k).tolist(),
                         neighbors.ravel().tolist()))
    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.simplify(multiple=True, loops=True)
    assert graph.is_simple() and not graph.is_directed() and 'weight' not in graph.es.attributes()
    return graph


@lru_cache(maxsize=1)
def september_graph_helpers():
    path = SEPT / 'src/conditional_buddy/buddy_graph.py'
    spec = importlib.util.spec_from_file_location('ct_diag_september_graph', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_graph(x, k=20):
    """September's mutual graph and both repairs; float32 math on CPU."""
    helpers = september_graph_helpers()
    x = normalize(x)
    assert 0 < k < len(x)
    adjacency = helpers.mutual_knn(x, K=k, device='cpu', use_half=False)
    adjacency, degree = helpers.ensure_min_degree(adjacency, x, x, device='cpu', use_half=False)
    adjacency, connectivity = helpers.ensure_connected(adjacency, x, x, alpha=.5,
                                                       device='cpu', use_half=False)
    assert connected_components(adjacency, directed=False, return_labels=False) == 1
    upper = triu(adjacency, k=1).tocoo()
    graph = ig.Graph(n=len(x), edges=list(zip(upper.row.tolist(), upper.col.tolist())), directed=False)
    graph.simplify(multiple=True, loops=True)
    assert graph.is_simple() and 'weight' not in graph.es.attributes()
    return graph, dict(degree=degree, connectivity=connectivity)


def leiden_partitions(graph):
    """Ordinary modularity is resolution 1; no resolution search."""
    result = []
    for seed in range(3):
        partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, seed=seed)
        result.append(partition.membership)
    return np.asarray(result, dtype=np.int64)


def vote_transfer(fit, communities, held, k=20):
    """September vote and nearest-winning-label tie rule; reuse neighbor search."""
    assert communities.ndim == 2 and communities.shape[1] == len(fit)
    assert fit.shape[1] == held.shape[1] and 0 < k <= len(fit)
    neighbors = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=1).fit(fit)
    assigned = np.empty((len(communities), len(held)), dtype=np.int64)
    fallback = np.zeros(len(communities), dtype=np.int64)
    for start in range(0, len(held), 128):
        indices = neighbors.kneighbors(held[start:start + 128], return_distance=False)
        for offset, row in enumerate(indices):
            for seed, partition in enumerate(communities):
                labels = partition[row]
                values, counts = np.unique(labels, return_counts=True)
                winners = values[counts == counts.max()]
                assigned[seed, start + offset] = next(label for label in labels if label in winners)
                fallback[seed] += int(counts.max() == 1 and len(values) > k / 2)
    return assigned, fallback.tolist()


def partition_metrics(ps, labels):
    """All metrics per partition; random permutations preserve its exact counts."""
    assert ps.ndim == 2 and labels.shape == (ps.shape[1],)
    assert np.issubdtype(ps.dtype, np.integer) and np.issubdtype(labels.dtype, np.integer)
    mask = labels >= 0
    assert mask.any()
    rng = np.random.default_rng(0)
    records = []
    for p in ps:
        random = rng.permutation(p)
        records.append(dict(count=int(len(np.unique(p))),
                            scores={key: float(metric(labels[mask], p[mask])) for key, metric in METRICS.items()},
                            random={key: float(metric(labels[mask], random[mask])) for key, metric in METRICS.items()}))
    return records


def diagnose_vectors(x, labels, kind, fixed=None, fit=None, fit_labels=None):
    """One vector set; reuse 03's partitions and its sheet partition function."""
    assert kind in ('k1', 'percept')
    if fixed is None:
        fixed = sc.output_partitions(x, kind)
    readings = {key: partition_metrics(ps, labels) for key, ps in fixed.items()}
    graph = union_knn(x)
    leiden = leiden_partitions(graph)
    counts = [len(np.unique(p)) for p in leiden]
    c = int(round(float(np.mean(counts))))
    # Match the sheet's k-means input, including no re-normalization of K1.
    kx = normalize(x) if kind == 'percept' else np.asarray(x, dtype=np.float32)
    kc = fixed['r20'] if c == 20 else np.asarray(sc.partitions(kx, c))
    readings.update(leiden3a=partition_metrics(leiden, labels),
                    kmeans_at_leiden_count=partition_metrics(kc, labels))
    record = dict(held=readings, leiden_counts=counts, rounded_mean_count=c,
                  graph=dict(vertices=graph.vcount(), edges=graph.ecount()))
    if fit is not None:
        assert len(fit) == len(fit_labels) and fit.shape[1] == x.shape[1]
        in_sample = sc.output_partitions(fit, kind)
        record['fit'] = {key: partition_metrics(ps, fit_labels) for key, ps in in_sample.items()}
        graph, repair = fit_graph(fit)
        fit_communities = leiden_partitions(graph)
        transferred, fallback = vote_transfer(fit, fit_communities, x)
        record['held']['leiden3b'] = partition_metrics(transferred, labels)
        record['fit']['leiden3b'] = partition_metrics(fit_communities, fit_labels)
        record['leiden3b'] = dict(counts=[int(len(np.unique(p))) for p in fit_communities],
                                  fallback_counts=fallback, repair=repair,
                                  graph_vertices=graph.vcount(), graph_edges=graph.ecount())
    return record, leiden, fixed


def leiden_margin(partitions, labels, eligible, n_resamples=2000, workers=8):
    """Reuse 03's paired bootstrap with one three-partition reading per arm."""
    fixed = {arm: {seed: dict(r20=ps) for seed, ps in runs.items()}
             for arm, runs in partitions.items()}
    return sc.paired_bootstrap(fixed, labels, eligible, n_resamples=n_resamples,
                               workers=workers, comparator_readings=('r20',))


def saved_partitions(path, source, held_sha256):
    """Read 03's frozen partitions without writing to the main checkout."""
    path = Path(path)
    assert sc._sha_file(path) == source['partitions_sha256'], path
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['partitions_sha256'] == source['partitions_sha256']
    assert meta['identity']['held_sha256'] == held_sha256
    assert meta['identity']['source_sha256'] == source['output_sha256']
    with np.load(path, allow_pickle=False) as z:
        result = {key: z[key] for key in z.files}
    assert set(result) in ({'r20'}, {'r20', 'own'})
    for key, ps in result.items():
        assert ps.shape == (3 if key == 'r20' else 1, sc.HELD_ROWS)
        assert np.issubdtype(ps.dtype, np.integer)
    return result


def target_images(aspect, names, held_sha256, root=TARGETS):
    path = Path(root) / f'{PROMPTS[aspect]}_held'
    meta = json.loads((path / 'meta.json').read_text())
    assert meta['held_sha256'] == held_sha256
    assert meta['instruction'] == QWEN_INSTRUCTIONS[PROMPTS[aspect]]
    assert meta['shape'] == [sc.HELD_ROWS, 2048] and meta['dtype'] == 'float16'
    with np.load(path / 'ids.npz', allow_pickle=False) as ids:
        assert ids['painting_names'].tolist() == names
    raw = np.load(path / 'img.npy', mmap_mode='r')
    assert raw.shape == (sc.HELD_ROWS, 2048) and raw.dtype == np.float16
    return normalize(raw)


class FitInference:
    """Cache the three fit input readers; apply 04/05's builders on CPU."""

    def __init__(self, output_root, held_names):
        import ct_d2
        import ct_percept as pc
        import torch
        self.d2, self.pc, self.torch = ct_d2, pc, torch
        torch.set_num_threads(8)
        self.root = Path(output_root)
        s = pc.load_frozen()
        self.sample_sha256 = pc.pc_data.fingerprint(s)
        ids = np.flatnonzero(pc.pc_data.partitions(s) == 'fit')
        self.names = sorted(s['all_painting_ids'][ids].tolist())
        assert len(self.names) == len(set(self.names)) == 13400
        assert not set(self.names) & set(held_names)
        self.labels = ct_held.painting_labels(self.names)
        self.inputs = {}
        self.sources = {}

    def features(self, kind):
        if kind not in self.inputs:
            reader = dict(qwen_default=ct_held.qwen_default_images,
                          clip_image=ct_held.clip_global_images, patch=ct_held.clip_patches)[kind]
            x = reader(self.names)
            # Core image_inputs normalizes default Qwen a second time.
            self.inputs[kind] = x
        return self.inputs[kind]

    def predict(self, row, aspect, seed):
        torch, pc, d2 = self.torch, self.pc, self.d2
        if row == 'K1':
            meta = d2._weights_info(self.root, aspect, seed)
            assert meta['config']['sample_sha256'] == self.sample_sha256
            path = d2._weight_path(self.root, aspect, seed)
            _, views = d2.ms_train.build_model(d2.ROWS['D2'], list(d2.ASPECTS), torch.device('cpu'))
            view = views[aspect]
            view.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model = view['img']
            x = self.features('qwen_default')
            width, dtype = 2048, np.float16
        else:
            settings, cut, arm = COMPARATORS[row]
            if row == 'B1' and aspect == 'style':
                arm = 'P2'
            path = self.root / 'percept' / settings / 'stage2' / cut / f'{arm}__s{seed}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            c = meta['config']
            assert (c['settings'], c['cut'], c['arm'], c['seed']) == (settings, cut, arm, seed)
            assert not c['smoke'] and meta['sample_sha256'] == self.sample_sha256
            assert sc._sha_file(path.with_suffix('.pt')) == meta['weights_sha256']
            model = pc.build_mapper(c, meta['K'])
            state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
            model.load_state_dict(state['mapper'])
            kind = pc.ARMS[arm][1]
            x = self.features(kind)
            if kind == 'qwen_default':
                x = x / np.linalg.norm(x, axis=1, keepdims=True)
            width, dtype = meta['K'], np.float32
        model.eval()
        prob = np.empty((len(self.names), width), dtype=dtype)
        with torch.no_grad():
            for start in range(0, len(x), 256):
                value = model(torch.from_numpy(np.ascontiguousarray(x[start:start + 256])))
                if row != 'K1':
                    value = value.sigmoid()
                prob[start:start + len(value)] = value.numpy().astype(dtype)
        assert np.isfinite(prob).all()
        if row == 'K1':
            assert np.all(np.abs(np.linalg.norm(prob.astype(np.float32), axis=1) - 1) < 1e-3)
        else:
            assert ((prob >= 0) & (prob <= 1)).all()
        self.sources[f'{row}/{aspect}/{seed}'] = dict(path=str(path.with_suffix('.pt')),
                                                     weights_sha256=meta['weights_sha256'],
                                                     training_device=meta['config']['device'],
                                                     inference_device='cpu')
        return prob.astype(np.float32)


def paper_runs(verdict, output_root):
    """All R-COL runs, including collapsed runs without a Stage 2 mapper."""
    record = verdict['sources']['rcol']['record']
    rows = []
    for run in record['runs']:
        arm = 'P1' if run['version'] == 'clip_roberta' else 'P4'
        path = Path(output_root) / 'percept/paper/stage2' / run['cut'] / f"{arm}__s{run['seed']}.json"
        row = {key: run[key] for key in ('version', 'seed', 'cut', 'kept_count', 'share', 'collapsed')}
        row['stage2_dev_auc'] = None
        row['stage2_status'] = 'not_trained_kept_count_under_2'
        entry = next(key for key, (_, cut, mapper) in COMPARATORS.items()
                     if key != 'B1' and (cut, mapper) == (run['cut'], arm))
        row['held_status'] = verdict['sources']['held_outputs'][entry][str(run['seed'])]['status']
        row['eligible'] = run['seed'] in verdict['eligibility']['eligible_seeds'][entry]
        if run['kept_count'] >= 2:
            meta = json.loads(path.read_text())
            assert meta['K'] == run['kept_count']
            c = meta['config']
            assert (c['settings'], c['cut'], c['arm'], c['seed']) == ('paper', run['cut'], arm, run['seed'])
            row.update(stage2_dev_auc=meta['dev_auc'], stage2_status='trained',
                       stage2_sidecar_sha256=sc._sha_file(path))
        rows.append(row)
    return rows


def markdown(report):
    lines = ['Diagnostics only; all computations followed the saved verdict.', '',
             '| Row | Aspect | Seeds | k20 AMI | Leiden 3a AMI | Leiden count | k-means at c AMI | Leiden 3b AMI | Fit k20 AMI |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    groups = {}
    for record in report['records']:
        groups.setdefault((record['row'], record['aspect']), []).append(record)
    for (row, aspect), records in groups.items():
        cells = []
        for population, reading in (('held', 'r20'), ('held', 'leiden3a'), (None, None),
                                    ('held', 'kmeans_at_leiden_count'), ('held', 'leiden3b'), ('fit', 'r20')):
            values = []
            for r in records:
                if population is None:
                    values.append(np.mean(r['leiden_counts']))
                elif reading in r.get(population, {}):
                    values.append(np.mean([p['scores']['ami'] for p in r[population][reading]]))
            cells.append(f'{np.mean(values):.4f}' if values else '—')
        lines.append(f'| {row} | {aspect} | {len(records)} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| Margin | d | Paired 95% interval |', '|---|---:|---:|']
    for name, margin in report['margins'].items():
        lo, hi = margin['interval']
        lines.append(f"| {name} | {margin['d']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ['', 'All finite saved runs appear above, including collapsed paper runs; only the verdict’s eligible seeds enter margins.',
              'JSON contains V, ARI, FMI and random baselines per partition, plus every paper run’s topic count, share, collapse and dev AUC.', '']
    return '\n'.join(lines)


@threadpool_limits.wrap(limits=8)
def run(verdict_path, output_root=sc.OUTPUTS, destination=HERE / 'diag_stage1.json', *, workers=8, n_resamples=2000):
    """Compute diagnostics after the sealed verdict, without modifying it."""
    verdict = read_verdict(verdict_path)
    verdict_hash = sc._sha_file(verdict_path)
    destination, output_root = Path(destination).resolve(), Path(output_root)
    assert destination.is_relative_to(HERE), 'diagnostics must be written in this round'
    assert destination.suffix == '.json'
    assert destination.name != 'verdict_stage1.json'
    assert Path(verdict_path).resolve() not in (destination, destination.with_suffix('.md'))
    started = time.monotonic()
    names, held_hash = ct_held.read_held_list(output_root / 'held/held_paintings.json')
    assert held_hash == verdict['held_sha256'] and names == ct_held.held_names()
    labels = ct_held.painting_labels(names)
    eligible = verdict['eligibility']['eligible_seeds']
    inference = FitInference(output_root, names)
    fixed, leiden, records = {}, {}, []
    held_sources = verdict['sources']['held_outputs']
    for aspect in PROMPTS:
        fixed[aspect], leiden[aspect] = {}, {}
        arms = sc.ARM_ORDER if aspect == 'emotion' else ('K1', 'B1')
        for arm in arms:
            fixed[aspect][arm], leiden[aspect][arm] = {}, {}
            for seed in sc.SEEDS:
                source = held_sources[arm][str(seed)] if aspect == 'emotion' else None
                if source is not None and source['status'] != 'finite':
                    continue
                stem = sc.HELD_STEMS[arm]
                if aspect != 'emotion':
                    stem = f'd2/{aspect}' if arm == 'K1' else f'percept/core__k40/{"P2" if aspect == "style" else "P3"}'
                path = output_root / 'held' / f'{stem}__s{seed}.npz'
                if source is not None:
                    assert sc._sha_file(path) == source['output_sha256']
                    assert sc._sha_file(path.with_suffix('.json')) == source['sidecar_sha256']
                width = None
                if arm in ('B2', 'B3'):
                    settings, cut, mapper = COMPARATORS[arm]
                    width = next(r['kept_count'] for r in verdict['sources']['rcol']['record']['runs']
                                 if (r['version'], r['cut'], r['seed']) ==
                                 ('clip_roberta' if mapper == 'P1' else 'qwen_qwen', cut, seed))
                x = sc.load_held_output(path, arm, names, held_hash, expected_width=width)
                assert x is not None
                prior = saved_partitions(source['partitions'], source, held_hash) if source is not None else None
                fit = inference.predict(arm, aspect, seed)
                held_meta = json.loads(path.with_suffix('.json').read_text())
                assert held_meta['weights_sha256'] == inference.sources[f'{arm}/{aspect}/{seed}']['weights_sha256'], \
                    'fit and held inference weights differ'
                record, ps, r20 = diagnose_vectors(x, labels[aspect], 'k1' if arm == 'K1' else 'percept',
                                                  prior, fit, inference.labels[aspect])
                record.update(row=arm, aspect=aspect, model_seed=seed,
                              eligible=seed in eligible[arm], source=str(path), source_sha256=sc._sha_file(path),
                              sidecar_sha256=sc._sha_file(path.with_suffix('.json')))
                records.append(record)
                fixed[aspect][arm][seed], leiden[aspect][arm][seed] = r20, ps
                print(f'Diagnostics {arm}/{aspect}/s{seed}: {time.monotonic() - started:.1f}s', flush=True)
    encoder = ct_held.qwen_default_images(names)
    for aspect in PROMPTS:
        for row, x in (('target', target_images(aspect, names, held_hash)), ('encoder', encoder)):
            record, _, _ = diagnose_vectors(x, labels[aspect], 'k1')
            record.update(row=row, aspect=aspect, model_seed=None)
            records.append(record)
    margins = dict(leiden_emotion=leiden_margin(leiden['emotion'], labels['emotion'], eligible,
                                               n_resamples=n_resamples, workers=workers))
    for aspect in ('style', 'genre'):
        margins[aspect] = sc.paired_bootstrap(fixed[aspect], labels[aspect],
                                              {arm: sc.SEEDS for arm in ('K1', 'B1')},
                                              n_resamples=n_resamples, workers=workers)
    report = dict(schema=1, diagnostic_only=True, verdict_sha256=verdict_hash,
                  held_sha256=held_hash, eligibility=eligible, records=records, margins=margins,
                  held_statuses={arm: {seed: source['status'] for seed, source in runs.items()}
                                 for arm, runs in held_sources.items()},
                  fit_names_sha256=ct_held.names_sha256(inference.names), fit_rows=len(inference.names),
                  fit_sources=inference.sources, paper_runs=paper_runs(verdict, output_root),
                  code_sha256={str(path): sc._sha_file(path) for path in
                               (Path(__file__), Path(sc.__file__), SEPT / 'src/conditional_buddy/buddy_graph.py')},
                  seconds=time.monotonic() - started,
                  protocol=dict(threads=8, device='cpu', leiden_seeds=[0, 1, 2], knn=20,
                                leiden3a='union cosine undirected unweighted simple; ModularityVertexPartition resolution 1',
                                leiden3b='fit mutual20 + degree/MST repair alpha=.5; held20 vote, nearest winning label breaks ties',
                                fit_graph_dtype='float32 on CPU (September GPU path uses half)',
                                c='Python round(mean Leiden count) per vector set',
                                random='default_rng(0), independent permutations per partition preserving exact sizes',
                                bootstrap_resamples=n_resamples, bootstrap='ct_score.paired_bootstrap; saved eligibility',
                                fit_inference='04/05 builders and saved weights on CPU; D2 cast through float16'))
    assert sc._sha_file(verdict_path) == verdict_hash, 'verdict changed during diagnostics'
    sc._atomic_json(destination, report)
    destination.with_suffix('.md').write_text(markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verdict', type=Path, default=sc.OUTPUTS.parent / 'verdict_stage1.json')
    parser.add_argument('--outputs', type=Path, default=sc.OUTPUTS)
    parser.add_argument('--out', type=Path, default=HERE / 'diag_stage1.json')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    import ct_stage0
    ct_stage0.require_pass()  # V1: no diagnostics without a Stage 0 pass
    run(args.verdict, args.outputs, args.out, workers=args.workers)


if __name__ == '__main__':
    main()
