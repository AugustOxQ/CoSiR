#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent SC7 attribution; replay reads frozen r14/r28 only.

Held is controller-only, after committed picks. No other round scorer imports.
AMI boot numerics adapted from permitted ns_c13 and our r29rd_c1_core;
shared seed draw follows the explicit SC7 task, rows/queries before seeds.
"""
import os
import sys
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from scipy.special import gammaln
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROWS = ('instructed_graphs', 'content_correction', 'buddy', 'relational')
COUNTS = dict(emotion=9, style=27, genre=10)
R14 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/held/k11=40')
R28 = Path('/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1/held')
R29 = Path('/data/SSD2/cosir_r29/20261112_siglip_only')
TARGET = Path('/project/CoSiR/src/test/20261112_siglip_main/records/ablation_siglip1.json')
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
FEATURES = Path('/data/SSD2/pre_extract/artelingo/features')
GENRE = Path('/data/SSD/wikiart_genre')
SPEC = Path('/project/CoSiR/docs/superpowers/specs')
PROTOCOL = [SPEC/'2026-11-12-siglip-only-adapter-design.md', SPEC/'2026-11-03-new-scores-design.md',
            Path('/project/CoSiR/src/test/20261103_new_scores/ns_c13.py'),
            Path('/project/CoSiR/src/test/20261112_siglip_main/r28_ablation.py'),
            Path('/project/CoSiR/src/test/20261029_retrieval/rt_metrics.py'),
            Path('/project/CoSiR/src/test/20261109_r20_figures/r20_ret.py')]
CACHE = HERE/'r29rd_s2abl_cache'
INPUTS = {}
NOTES = [
    'SC7 attribution only: no read-out pick, LEVEL unit, WIN/LEVEL/BELOW outcome or ladder. K* is fixed siglip1_prefix; five seeds 0..4; each removal is paired to the same training seed.',
    'Point protocol: r28_ablation.py module docstring (only its docstring was inspected), r29 spec 3.4 Ablation and D18; AMI at 9/27/10 clusters, stored float16 -> float32 without renormalisation, MiniBatchKMeans n_init=3 batch_size=4096 states=0,1,2, eight threads, all 12281 paintings fit before masking labels, states averaged within seed then seeds equally.',
    'mAP uses float32 unit rows, cosine products computed in float64 from those rows (rt_metrics module docstring), no query expansion, labelled gallery and queries, excludes the query itself, drops queries with no relevant gallery item. AP and means use float64; exact cosine ties retain ascending canonical painting-row order, following our independent C1 choice.',
    'Intervals: fresh numpy default_rng(0) for each cell, 2000 AMI / 10000 mAP draws, numpy linear percentiles at 2.5/97.5. AMI samples ALL painting rows, then five training seeds, following r17 spec 3.4 and ns_c13.draws; labels masked after row resampling; saved partitions are resampled, never refit.',
    'Both arms use the SAME row/query draw and SAME training-seed draw (removal s pairs to K* s), exactly as requested. mAP is r17 shared-seed paired retrieval (apart=False / told-versus-untold form). Generic r17 two-seeded AMI and apart=True retrieval instead draw comparator seeds independently after candidate seeds; that alternative would change the intervals, but contradicts the explicit task pairing, so is not used.',
    'd is removal seed mean minus K* seed mean; per-seed differences are also recorded. AMI n is all sampled rows (12281), n_labelled separately; mAP n is retained labelled queries. Gallery rows with unknown genre are excluded before retrieval; there are 9949 labelled genre paintings.',
    'Sidecars may identify canonical rows by their newline-joined name hash rather than listing names. Rebuild the split independently from caption extraction sample IDs and default_rng(42), group-caption 70/10/20 allocation as in our C1; assert canonical held hash. Emotion plurality ties use alphabetical order; conflicting/missing genre is -1.',
    'r28 has no recorded intervals. Replay compares only independently recomputed per-seed values, seed means, per-seed changes and changes to frozen r28 within 1e-9. For interval rows it also checks d is inside the independently calculated interval; this is a sanity check, not an interval reproduction claim.',
    'C1 JSON stores AMI means only. kstar_check compares those means directly and per-seed values against saved independent C1 partition caches when available, explicitly distinguishing the cache targets. Missing per-seed targets prevent a complete consistency check.',
    'Held requires the controller assertion --picks-committed; this program reads no r29 records or forbidden scorer outputs. --rows must name present removals explicitly; all requested seed/aspect export pairs are preflighted before any r29 held model file is read. Replay and fixtures never open r29 model exports.',
    'CUDA (controller only) accelerates cosine matrix products/sorting; TF32 is disabled. AMI fitting/intervals and retrieval intervals remain on CPU. CPU and CUDA floating-point tie order can differ; certify each backend with the frozen replay before held. All files actually read as data/cache/source provenance are sha256 listed; installed library source files are represented by package versions.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def track(path):
    path = Path(path).resolve()
    INPUTS[str(path)] = sha(path)
    return path


def read_json(path):
    return json.loads(track(path).read_text())


def writable(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE) or not (path.name.startswith('r29rd_s2abl') or path.name == 's2abl_held.json'):
        raise ValueError('Output must be an s2abl-prefixed file in rederive/ (or s2abl_held.json)')
    return path


def write_json(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def names_hash(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()


def load_frame():
    """Copied/adapted from our independent C1 held split and labels, not a scorer."""
    records = read_json(ANNOTATIONS)
    meta = read_json(FEATURES/'metadata.json')
    ids = np.concatenate([np.load(track(FEATURES/'shards'/f'shard_{s:05d}'/'sample_ids.npy'), allow_pickle=False)
                          for s in range(meta['num_shards'])])
    groups = {}
    group = np.array([groups.setdefault(records[int(i)]['painting'], len(groups)) for i in ids])
    counts = np.bincount(group)
    order = np.random.default_rng(42).permutation(len(counts))
    shares = counts[order] / counts.sum()
    starts = np.cumsum(shares) - shares
    parts = np.empty(len(counts), dtype=np.int64)
    parts[order] = np.searchsorted(np.cumsum([.7, .1, .2])[:2], starts, side='right')
    wanted = np.sort(ids[parts[group] == 2])
    names = sorted({records[int(i)]['painting'] for i in wanted})
    assert (len(names), len(wanted)) == (12281, 61744)
    assert names_hash(names) == 'a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876'
    lookup = {v: i for i, v in enumerate(names)}
    codes = {v: i for i, v in enumerate(sorted({r['emotion'] for r in records}))}
    votes = np.zeros((len(names), len(codes)), dtype=np.int64)
    for i in wanted:
        r = records[int(i)]
        votes[lookup[r['painting']], codes[r['emotion']]] += 1
    styles = {v: i for i, v in enumerate(sorted({r['art_style'] for r in records}))}
    painting_style = {}
    for r in records:
        assert painting_style.setdefault(r['painting'], r['art_style']) == r['art_style']
    genres = {}
    for p in (GENRE/'genre_train.csv', GENRE/'genre_val.csv'):
        with track(p).open() as f:
            for relative, code in csv.reader(f):
                name, code = Path(relative).stem, int(code)
                if genres.setdefault(name, code) != code:
                    genres[name] = -1
    labels = dict(emotion=votes.argmax(1), style=np.array([styles[painting_style[n]] for n in names]),
                  genre=np.array([genres.get(n, -1) for n in names]))
    assert (labels['genre'] >= 0).sum() == 9949
    return names, labels


def load_export(path, names, aspect, seed):
    meta = read_json(path.with_suffix('.json'))
    for k, expected in (('aspect', aspect), ('seed', seed)):
        if k in meta:
            assert meta[k] == expected, (path, k)
    hashes = []
    for doc in (meta, meta.get('config', {}), meta.get('config', {}).get('graph_identity', {})):
        for key in ('list_sha256', 'held_sha256', 'painting_names_sha256', 'names_sha256'):
            value = doc.get(key)
            if isinstance(value, dict):
                value = value.get('held')
            if isinstance(value, str):
                hashes.append(value)
    explicit = next((meta[k] for k in ('painting_names', 'names', 'keys') if isinstance(meta.get(k), list)), None)
    assert hashes or explicit is not None, (path, 'missing row identity')
    assert all(v == names_hash(names) for v in hashes), (path, 'row hash mismatch')
    if explicit is not None:
        assert explicit == names, (path, 'row order mismatch')
    with np.load(track(path), allow_pickle=False) as z:
        key = 'img' if 'img' in z else 'emb'
        raw = z[key]
    assert raw.dtype == np.float16 and raw.ndim == 2 and raw.shape == (len(names), 2048)
    assert np.isfinite(raw).all()
    checksum = meta.get('arrays', {}).get(key, {}).get('sha256')
    if checksum:
        assert checksum == hashlib.sha256(raw.tobytes()).hexdigest()
    return raw.astype(np.float32)


def partitions(x, k):
    with threadpool_limits(8):
        return np.stack([MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=s)
                         .fit_predict(np.asarray(x, dtype=np.float32)) for s in range(3)]).astype(np.int32)


def ami_points(parts, y):
    valid = y >= 0
    return np.array([np.mean([adjusted_mutual_info_score(y[valid], p[valid]) for p in states]) for states in parts])


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (n > 0).all()
    return x/n


def map_ap(x, y, device='cpu', batch_size=128):
    import torch
    torch.set_num_threads(8)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Controller requested CUDA but unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    valid = np.flatnonzero(y >= 0)
    z, labels = unit(x)[valid], y[valid]
    t = torch.as_tensor(z, dtype=torch.float64, device=device)
    yl = torch.as_tensor(labels, device=device)
    aps = []
    with torch.inference_mode():
        for start in range(0, len(z), batch_size):
            stop = min(start+batch_size, len(z))
            own = torch.arange(start, stop, device=device)
            scores = t[start:stop] @ t.T
            scores[torch.arange(stop-start, device=device), own] = -torch.inf
            ranking = torch.argsort(scores, dim=1, descending=True, stable=True)
            ranking = ranking[ranking != own[:, None]].reshape(stop-start, len(z)-1)
            rel = yl[ranking] == yl[start:stop, None]
            count = rel.sum(1)
            ranks = torch.arange(1, len(z), device=device, dtype=torch.float64)
            ap = ((rel.cumsum(1).double()/ranks)*rel).sum(1) / count.clamp_min(1)
            ap[count == 0] = float('nan')
            aps.append(ap.cpu().numpy())
    ap = np.concatenate(aps)
    return ap, valid


@lru_cache(maxsize=32)
def log_factorials(n):
    return gammaln(np.arange(n+1, dtype=np.float64)+1)


def fast_ami(C):
    """Independent exact hypergeometric EMI, copied from our C1 numerics."""
    C = C[C.sum(1) > 0][:, C.sum(0) > 0]
    n = int(C.sum())
    if not n or C.shape == (1, 1):
        return 1.
    a, b = C.sum(1), C.sum(0)
    i, j = np.nonzero(C)
    values = C[i, j].astype(np.float64)
    mi = max(0., float(np.sum(values/n*(np.log(values)+np.log(n)-np.log(a[i])-np.log(b[j])))))
    ha, hb = -np.sum(a/n*np.log(a/n)), -np.sum(b/n*np.log(b/n))
    au, ac = np.unique(a, return_counts=True)
    bu, bc = np.unique(b, return_counts=True)
    g, emi = log_factorials(n), 0.
    for ai, multiplicity in zip(au, ac):
        ks = np.arange(1, min(int(ai), int(bu.max()))+1, dtype=np.int64)[None, :]
        bj = bu[:, None]
        valid = (ks <= bj) & (ks >= np.maximum(1, ai+bj-n))
        lp = (g[ai]+g[n-ai]+g[bj]+g[n-bj]-g[n]-g[ks]-g[ai-ks]
              -g[np.maximum(bj-ks, 0)]-g[np.maximum(n-ai-bj+ks, 0)])
        term = np.exp(np.where(valid, lp, -np.inf))*(ks/n)*(np.log(n)+np.log(ks)-np.log(ai)-np.log(bj))
        emi += multiplicity*float(np.sum(term*bc[:, None]))
    den = (ha+hb)/2-emi
    eps = np.finfo(np.float64).eps
    den = min(den, -eps) if den < 0 else max(den, eps)
    return float((mi-emi)/den)


def ami_weighted(parts, y, weights, seeds):
    valid = y >= 0
    yy, ww = y[valid], weights[valid]
    ny = int(yy.max(initial=-1))+1
    values = {}
    for s in np.unique(seeds):
        scores = []
        for p in parts[s]:
            pp = p[valid]
            nk = int(pp.max(initial=-1))+1
            C = np.bincount(yy*nk+pp, weights=ww, minlength=ny*nk).reshape(ny, nk).astype(np.int64)
            scores.append(fast_ami(C))
        values[s] = float(np.mean(scores))
    return float(np.mean([values[s] for s in seeds]))


_BOOT = None

def boot_init(removal, kstar, y):
    global _BOOT
    threadpool_limits(1)
    _BOOT = removal, kstar, y


def ami_task(task):
    removal, kstar, y = _BOOT
    rows, seeds = task
    w = np.bincount(rows, minlength=len(y))
    return ami_weighted(removal, y, w, seeds)-ami_weighted(kstar, y, w, seeds)


def ami_interval(removal, kstar, y, draws=2000, workers=8):
    assert removal.shape == kstar.shape and removal.ndim == 3
    rng = np.random.default_rng(0)
    tasks = [(rng.integers(0, len(y), size=len(y)), rng.choice(len(removal), size=len(removal), replace=True))
             for _ in range(draws)]
    if workers == 1:
        boot_init(removal, kstar, y)
        values = [ami_task(task) for task in tasks]
    else:
        # spawn is safe even when retrieval has already initialised Torch/CUDA.
        with mp.get_context('spawn').Pool(workers, initializer=boot_init, initargs=(removal, kstar, y)) as pool:
            values = pool.map(ami_task, tasks, chunksize=4)
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def map_interval(removal, kstar, draws=10000):
    assert removal.shape == kstar.shape and removal.ndim == 2
    finite = np.isfinite(removal)
    assert np.array_equal(finite, np.isfinite(kstar)) and np.all(finite == finite[:1])
    diff = (removal-kstar)[:, finite[0]]
    ns, n = diff.shape
    assert n > 0
    rng, boot = np.random.default_rng(0), np.empty(draws)
    for start in range(0, draws, 128):
        size = min(128, draws-start)
        weights = np.empty((size, n), dtype=np.float64)
        selected = np.empty((size, ns), dtype=np.int64)
        for j in range(size):
            weights[j] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            selected[j] = rng.choice(ns, size=ns, replace=True)
        means = weights @ diff.T/n
        boot[start:start+size] = np.take_along_axis(means, selected, axis=1).mean(1)
    return tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))


def score_seed(path, names, aspect, seed, args):
    # Bind raw file hashes, labels/order protocol, code, backend and batch size.
    meta = track(path.with_suffix('.json'))
    track(path)
    identity = dict(npz=INPUTS[str(path.resolve())], sidecar=INPUTS[str(meta)],
                    names=names_hash(names), labels=hashlib.sha256(args.labels[aspect].tobytes()).hexdigest(),
                    code=sha(__file__), device=args.device, batch=args.batch_size)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    p = CACHE/f'r29rd_s2abl_{key}.npz'
    side = p.with_suffix('.json')
    if p.exists() and side.exists():
        saved = read_json(side)
        assert saved['identity'] == identity
        assert saved['sha256'] == sha(p)
        for source, digest in saved['inputs'].items():
            assert INPUTS.get(source, digest) == digest
            INPUTS[source] = digest
        with np.load(track(p), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    before = time.monotonic()
    x = load_export(path, names, aspect, seed)
    part = partitions(x, COUNTS[aspect])
    ap, qids = map_ap(x, args.labels[aspect], args.device, args.batch_size)
    result = dict(parts=part, ap=ap, qids=qids)
    CACHE.mkdir(exist_ok=True)
    with p.open('wb') as f:
        np.savez(f, **result)
    write_json(side, dict(identity=identity, sha256=sha(p), inputs=dict(INPUTS)))
    print(f'{aspect} s{seed} {path.parent.name}: {time.monotonic()-before:.1f}s', flush=True)
    return result


def compare(target, key, value, recorded, measure):
    diff = float(np.max(np.abs(np.asarray(value)-np.asarray(recorded))))
    return dict(target=str(target), key=key, value=value, recorded_value=recorded,
                abs_diff=diff, pass_=diff <= 1e-9, measure=measure)


def replay_comparisons(cells, record, seeds):
    comparisons = []
    for cell in cells:
        _, row, aspect, metric = cell['id'].split('/')
        for arm, field in (('k11=40', 'kstar'), (row, 'removal')):
            expected = record['arms'][arm][aspect]
            base = f'arms/{arm}/{aspect}/{metric}'
            values = cell[field+'_per_seed']
            comparisons.append(compare(TARGET, base+'_per_seed', values,
                                       [expected[metric+'_per_seed'][s] for s in seeds], metric))
            if seeds == list(range(5)):
                comparisons.append(compare(TARGET, base+'_mean', cell[field+'_mean'], expected[metric+'_mean'], metric))
        delta = record['deltas_vs_cstar'][row][aspect][metric]
        comparisons.append(compare(TARGET, f'deltas_vs_cstar/{row}/{aspect}/{metric}/per_seed', cell['d_per_seed'],
                                   [delta['per_seed'][s] for s in seeds], metric))
        if seeds == list(range(5)):
            comparisons.append(compare(TARGET, f'deltas_vs_cstar/{row}/{aspect}/{metric}/delta', cell['d'], delta['delta'], metric))
    return comparisons


def c1_check(cells, arms, labels, aspects):
    path = HERE/'c1_rederived.json'
    record = read_json(path)
    checks = []
    missing = []
    per_seed = {}
    for aspect in aspects:
        values = ami_points(np.stack([v['parts'] for v in arms['kstar'][aspect]]), labels[aspect]).tolist()
        per_seed[aspect] = values
        targets = [c for c in record['cells'] if c['id'] == f'ami/{aspect}/p1']
        assert len(targets) == 1
        checks.append(compare(path, f'cells/id=ami/{aspect}/p1/value', float(np.mean(values)), targets[0]['value'], 'ami'))
        for seed, value in enumerate(values):
            files = sorted((HERE/'r29rd_c1_cache').glob(f'*/candidate_held_ami_{aspect}_s{seed}.npz'))
            if not files:
                missing.append(f'{aspect}/s{seed}: C1 JSON has no per-seed field and no saved partition cache found')
            for p in files:
                with np.load(track(p), allow_pickle=False) as z:
                    expected = float(ami_points(z['parts'][None], labels[aspect])[0])
                checks.append(compare(p, f'parts/AMI/mean_states/seed={seed}', value, expected, 'ami'))
    return dict(complete=not missing and all(c['pass_'] for c in checks), kstar_per_seed=per_seed,
                mean_checks_pass=all(c['pass_'] for c in checks), checks=checks, missing=missing,
                notes=['Mean target is C1 JSON; per-seed targets are independently saved C1 partitions because the JSON omits per-seed values.'])


def run(args):
    started = time.monotonic()
    for p in [Path(__file__), *PROTOCOL]:
        track(p)  # Hash provenance only; no scorer code is imported or executed.
    seeds = args.seeds if args.mode == 'replay' else list(range(5))
    reference = R14 if args.mode == 'replay' else R29/'cstar/siglip1/held/siglip1_prefix'
    root = R28 if args.mode == 'replay' else R29/'ablation/siglip1/siglip1_prefix/held'
    if args.mode == 'held':
        if not args.picks_committed:
            raise ValueError('Held is controller-only after committed picks: supply --picks-committed')
        if args.points_only:
            raise ValueError('Held always includes full intervals')
        replay = read_json(args.replay_record)
        if not (replay.get('replay_certified') and replay.get('device') == args.device
                and all(c['pass_'] for c in replay['replays'])):
            raise ValueError('Held requires a complete passing frozen point replay on the requested device')
    paths = [folder/f'{a}__s{s}.npz' for folder in [reference, *[root/r for r in args.rows]]
             for a in args.aspects for s in seeds]
    missing = [str(p) for p in paths for p in (p, p.with_suffix('.json')) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Requested rows are not fully present: '+', '.join(missing[:4]))
    names, labels = load_frame()
    args.labels = labels
    record = read_json(TARGET) if args.mode == 'replay' else None
    arms = {r: {a: [] for a in args.aspects} for r in ['kstar', *args.rows]}
    cells, intervals = [], []
    result = dict(mode=args.mode, complete=False, rows=args.rows, aspects=args.aspects, seeds=seeds,
                  cells=cells, kstar_check=None, replays=[], notes=NOTES.copy(), inputs=INPUTS, device=args.device)
    import scipy, sklearn
    result['versions'] = dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    if seeds != list(range(5)) or args.aspects != list(COUNTS) or args.points_only:
        result['notes'].append('Partial replay diagnostic: not a complete 24-cell certification; requested subset/points-only is recorded explicitly.')
    for row in ['kstar', *args.rows]:
        folder = reference if row == 'kstar' else root/row
        for aspect in args.aspects:
            for seed in seeds:
                if args.time_budget and time.monotonic()-started >= args.time_budget:
                    result['stopped_at_budget'] = True
                    result['elapsed_seconds'] = time.monotonic()-started
                    write_json(args.output, result)
                    print('Time budget reached; completed seeds cached. Rerun to resume.', flush=True)
                    return
                arms[row][aspect].append(score_seed(folder/f'{aspect}__s{seed}.npz', names, aspect, seed, args))
            if row == 'kstar':
                continue
            k, r, y = arms['kstar'][aspect], arms[row][aspect], labels[aspect]
            kp, rp = np.stack([v['parts'] for v in k]), np.stack([v['parts'] for v in r])
            ka, ra = np.stack([v['ap'] for v in k]), np.stack([v['ap'] for v in r])
            assert all(np.array_equal(v['qids'], k[0]['qids']) for v in k+r)
            for metric, kv, rv in [('ami', ami_points(kp, y), ami_points(rp, y)),
                                   ('map', np.nanmean(ka, axis=1), np.nanmean(ra, axis=1))]:
                do_interval = not args.points_only and (args.mode == 'held' or row in args.interval_rows)
                count = (2000 if metric == 'ami' else 10000) if do_interval else 0
                lo = hi = None
                interval_started = time.monotonic()
                if do_interval:
                    print(f'interval {row}/{aspect}/{metric}: {count} draws', flush=True)
                    lo, hi = (ami_interval(rp, kp, y, count, args.workers) if metric == 'ami'
                              else map_interval(ra, ka, count))
                km, rm = float(kv.mean()), float(rv.mean())
                cell = dict(id=f'ablation/{row}/{aspect}/{metric}', kstar_mean=km, removal_mean=rm,
                            kstar_per_seed=kv.tolist(), removal_per_seed=rv.tolist(), d=rm-km,
                            d_per_seed=(rv-kv).tolist(), lo=lo, hi=hi, n_draws=count,
                            interval_seconds=time.monotonic()-interval_started,
                            n=len(y) if metric == 'ami' else int(np.isfinite(ka[0]).sum()), n_labelled=int((y >= 0).sum()))
                cells.append(cell)
                if do_interval and args.mode == 'replay':
                    intervals.append(dict(id=cell['id'], d=cell['d'], lo=lo, hi=hi, d_inside=lo <= cell['d'] <= hi,
                                          note='No interval target exists in r28.'))
            if args.mode == 'replay':
                result['replays'] = replay_comparisons(cells, record, seeds)
            result['interval_sanity'] = intervals
            result['elapsed_seconds'] = time.monotonic()-started
            write_json(args.output, result)
    if args.mode == 'held':
        result['kstar_check'] = c1_check(cells, arms, labels, args.aspects)
        replay = read_json(args.replay_record)
        result['replays'] = replay['replays']
        result['replay_record_sha256'] = INPUTS[str(Path(args.replay_record).resolve())]
        result['replay_certified'] = bool(replay.get('replay_certified')) and all(c['pass_'] for c in replay['replays'])
        if not result['replay_certified']:
            result['notes'].append('Replay is incomplete or failed; the controller must not treat these SC7 values as certified.')
    else:
        result['replay_certified'] = (args.rows == list(ROWS) and args.aspects == list(COUNTS) and seeds == list(range(5))
                                      and len(cells) == 24 and all(c['pass_'] for c in result['replays']))
    result['complete'] = len(cells) == len(args.rows)*len(args.aspects)*2
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(args.output, result)
    maxima = {m: max((r['abs_diff'] for r in result['replays'] if r['measure'] == m), default=None) for m in ('ami', 'map')}
    print(json.dumps(dict(output=str(args.output), cells=len(cells), max_abs_diff=maxima,
                          replay_certified=result.get('replay_certified'), seconds=result['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('held', 'replay'):
        p = sub.add_parser(mode)
        p.add_argument('--rows', nargs='+', choices=ROWS, required=mode == 'held', default=list(ROWS))
        p.add_argument('--aspects', nargs='+', choices=list(COUNTS), default=list(COUNTS))
        p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        p.add_argument('--batch-size', type=int, default=128)
        p.add_argument('--workers', type=int, choices=range(1, 9), default=8)
        p.add_argument('--output', type=Path)
        p.add_argument('--points-only', action='store_true')
        p.add_argument('--time-budget', type=float, default=0)
        if mode == 'replay':
            p.add_argument('--seeds', nargs='+', type=int, choices=range(5), default=list(range(5)))
            p.add_argument('--interval-rows', nargs='+', choices=ROWS, default=['instructed_graphs'])
        else:
            p.add_argument('--picks-committed', action='store_true')
            p.add_argument('--replay-record', type=Path, default=HERE/'r29rd_s2abl_replay.json')
    args = parser.parse_args()
    args.output = writable(args.output or HERE/('s2abl_held.json' if args.mode == 'held' else 'r29rd_s2abl_replay.json'))
    if len(set(args.rows)) != len(args.rows) or len(set(args.aspects)) != len(args.aspects):
        parser.error('Duplicate rows/aspects are not allowed')
    if args.mode == 'replay' and len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate seeds are not allowed')
    run(args)


if __name__ == '__main__':
    main()
