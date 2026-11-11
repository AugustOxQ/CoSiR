"""Independent D16 re-derivation. No r21 imports or stage 2 score inputs.

`derive` reads held/<arm>/<aspect>__s<seed>.npz under --arm-root (the
held directory, not its parent). `regress` uses r10's no-buddy arm.
Only the two specified D16 output trees may be written. CPU only.

Clustering preserves saved unit rows, casting float16 to float32 without
renormalising, as r17 does. Retrieval normalises once, ranks float64 cosine
with stable gallery-order ties, and drops singleton queries (r20 protocol).
Bootstrap RNG/order are supplied by the explicitly permitted frozen kernels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
R17 = HERE.parent / '20261103_new_scores'
R20 = HERE.parent / '20261109_r20_figures'
ROOT = Path('/data/SSD2/cosir_r21/20261111_dec_heads')
FULL_OUT = ROOT / 'd16'
SMOKE_OUT = ROOT / 'd16_smoke'
ARM_ROOT = ROOT / 'full/held'
PICKS = ROOT / 'scores/stage1_picks.json'
R10 = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
CSTAR_ROOT = R10 / 'full/held'
CSTAR = 'k11=40'
SEEDS = tuple(range(5))
COUNTS = dict(emotion=9, style=27, genre=10)
LABELLED = dict(emotion=12281, style=12281, genre=9949)
DRAW_AMI, DRAW_MAP = 2000, 10000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def names_sha(names):
    names = np.asarray(names)
    require(names.ndim == 1 and names.dtype.kind in 'US', 'need actual painting IDs')
    require(len(set(names.tolist())) == len(names), 'duplicate painting IDs')
    return hashlib.sha256('\n'.join(names.tolist()).encode()).hexdigest()


def write_json(path, doc):
    path = Path(path).resolve()
    require(any(path.is_relative_to(root.resolve()) for root in (FULL_OUT, SMOKE_OUT)),
            'outputs must be under d16/ or d16_smoke/')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')


def older_paths():
    if str(R17) not in sys.path:
        sys.path.insert(0, str(R17))
    import ns_paths
    ns_paths.add_rounds()
    if str(R20) not in sys.path:
        sys.path.insert(0, str(R20))


def held_data(aspects):
    older_paths()
    import ns_data
    names = np.asarray(ns_data.keys('artelingo', 'held'))
    require(len(names) == 12281 and names.tolist() == sorted(names.tolist()), 'wrong held IDs')
    labels = {a: ns_data.labels('artelingo', 'held', a) for a in aspects}
    for a, y in labels.items():
        require(y.dtype == np.int64 and y.shape == (12281,), f'{a}: wrong labels')
        require(int((y >= 0).sum()) == LABELLED[a], f'{a}: wrong labelled count')
        require(len(np.unique(y[y >= 0])) == COUNTS[a], f'{a}: wrong class count')
    return names, labels


def check_identity(meta, names, arm, aspect, seed, width=2048):
    """C6: bind positional rows to extraction IDs via the ordered list hash."""
    expected = dict(kind='held', arm=arm, aspect=aspect, seed=seed, scale='full',
                    rows=len(names), width=width, dtype='float16', list_sha256=names_sha(names))
    for key, value in expected.items():
        require(meta.get(key) == value, f'export {arm}/{aspect}/s{seed}: {key} mismatch')
    require(not meta.get('vc3_problems'), 'export failed VC3')


def load_export(root, arm, aspect, seed, names, hashes):
    require(Path(arm).name == arm and arm not in ('.', '..'), 'invalid arm name')
    path = Path(root) / arm / f'{aspect}__s{seed}.npz'
    side = path.with_suffix('.json')
    # Hash the exact sidecar bytes parsed; verify the array against that sidecar.
    raw = side.read_bytes()
    hashes[str(side)] = hashlib.sha256(raw).hexdigest()
    meta = json.loads(raw)
    check_identity(meta, names, arm, aspect, seed)
    before = sha256(path)
    with np.load(path, allow_pickle=False) as z:
        require(z.files == ['img'], f'{path}: expected only img')
        x = z['img']
    require(sha256(path) == before, f'{path}: export changed while reading')
    hashes[str(path)] = before
    require(x.shape == (len(names), 2048) and x.dtype == np.float16, f'{path}: shape/dtype')
    digest = array_sha(x)
    require(meta.get('sha256') == digest, f'{path}: array hash mismatch')
    require(meta.get('arrays', {}).get('img') ==
            dict(shape=list(x.shape), dtype='float16', sha256=digest), f'{path}: array descriptor mismatch')
    x = x.astype(np.float32)
    require(np.isfinite(x).all(), f'{path}: nonfinite rows')
    require(np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3), f'{path}: nonunit rows')
    return x


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    require(x.ndim == 2 and np.isfinite(x).all(), 'invalid embedding matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    require((norm > 0).all(), 'zero embedding row')
    return x / np.maximum(norm, 1e-12)


def partitions(x, k, threads=8):
    """Own r17-compatible call: all rows, including unlabelled paintings."""
    x = np.asarray(x).astype(np.float32)
    require(x.ndim == 2 and len(x) >= k >= 2 and np.isfinite(x).all(), 'invalid k-means input')
    require(np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3), 'k-means rows must be unit')
    require(1 <= threads <= 8, 'threads must be 1..8')
    with threadpool_limits(limits=threads):
        return np.stack([MiniBatchKMeans(n_clusters=k, random_state=s, n_init=3,
                                        batch_size=4096).fit(x).labels_
                         for s in (0, 1, 2)]).astype(np.int32)


def ami_value(parts, labels):
    y, parts = np.asarray(labels), np.asarray(parts)
    require(parts.shape == (3, len(y)) and y.ndim == 1, 'partition/label shape mismatch')
    keep = y >= 0
    require(keep.any(), 'no labelled rows')
    return float(np.mean([adjusted_mutual_info_score(y[keep], p[keep]) for p in parts]))


def label_ap(x, labels, names, workers=8):
    """Own ranking/AP, with actual painting IDs retained for paired resampling.

    Input is already normalised float32, like r20's score_rows input.
    Singleton queries are dropped; singleton paintings remain in the gallery.
    """
    x, y, names = np.asarray(x), np.asarray(labels), np.asarray(names)
    require(x.dtype == np.float32 and x.ndim == 2 and y.shape == (len(x),), 'invalid AP input')
    require(len(names) == len(x) and np.isfinite(x).all(), 'invalid AP rows')
    names_sha(names)
    require(1 <= workers <= 8, 'workers must be 1..8')
    keep = np.flatnonzero(y >= 0)
    gallery, lab = x[keep].astype(np.float64), y[keep]
    _, inverse, counts = np.unique(lab, return_inverse=True, return_counts=True)
    queries = np.flatnonzero(counts[inverse] > 1)
    require(len(queries) > 0, 'no queries with relevant gallery items')

    def block(q):
        sim = gallery[q] @ gallery.T
        sim[np.arange(len(q)), q] = -np.inf
        ranking = np.argsort(-sim, axis=1, kind='stable')[:, :-1]
        relevant = lab[ranking] == lab[q, None]
        precision = np.cumsum(relevant, axis=1) / np.arange(1, len(gallery))
        return (precision * relevant).sum(axis=1) / relevant.sum(axis=1)

    blocks = [queries[i:i + 256] for i in range(0, len(queries), 256)]
    # At most eight simultaneous blocks; BLAS is one thread per worker.
    with threadpool_limits(limits=1):
        if workers == 1:
            ap = [block(q) for q in blocks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                ap = list(pool.map(block, blocks))
    return dict(ap=np.concatenate(ap).astype(np.float64), query_ids=names[keep[queries]],
                dropped=int(len(keep) - len(queries)))


def paired_values(arm, cstar):
    require(set(arm) == set(cstar) == set(SEEDS), 'need matching seeds 0..4')
    a = np.asarray([arm[s] for s in SEEDS], dtype=np.float64)
    c = np.asarray([cstar[s] for s in SEEDS], dtype=np.float64)
    require(a.shape == c.shape == (5,) and np.isfinite([a, c]).all(), 'invalid seed values')
    return dict(d=float(np.mean(a - c)), arm_mean=float(a.mean()), cstar_mean=float(c.mean()),
                arm_seeds=a.tolist(), cstar_seeds=c.tolist())


def word(lo, hi):
    require(np.isfinite([lo, hi]).all() and lo <= hi, 'invalid interval')
    return 'raises' if lo > 0 else 'lowers' if hi < 0 else 'no clear change'


def chosen_arms(picks):
    selected = picks.get('stage1b')
    require(isinstance(selected, dict) and set(selected) == {'plain', 'start', 'teacher'},
            'stage1b picks are not ready: require plain, start, teacher')
    arms = ['dec-off']
    for form in ('plain', 'start', 'teacher'):
        value = selected[form]
        if isinstance(value, dict):
            value = value.get('pick', value.get('arm'))
        require(isinstance(value, str) and value and Path(value).name == value,
                f'{form}: require a picked arm name')
        arms.append(value)
    require(len(set(arms)) == 4, 'need four distinct arms')
    return arms


def ami_interval(arm, cstar, y, draws, workers=8):
    older_paths()
    import rs_ablation
    return rs_ablation.held_effect({s: {'r20': arm[s]} for s in SEEDS},
                                   {s: {'r20': cstar[s]} for s in SEEDS}, y, draws, workers)


def stacked_ap(results):
    require(set(results) == set(SEEDS), 'need AP for seeds 0..4')
    first = results[0]
    for r in results.values():
        require(np.array_equal(r['query_ids'], first['query_ids']) and r['dropped'] == first['dropped'],
                'query painting order differs across seeds')
    return dict(first, ap=np.stack([results[s]['ap'] for s in SEEDS]))


def derive(arms=None, arm_root=ARM_ROOT, out=FULL_OUT, aspects=tuple(COUNTS),
           draws_ami=DRAW_AMI, draws_map=DRAW_MAP):
    start = time.monotonic()
    require(draws_ami > 0 and draws_map > 0, 'draw counts must be positive')
    require(aspects and len(set(aspects)) == len(aspects) and set(aspects) <= set(COUNTS), 'invalid aspects')
    raw_picks = PICKS.read_bytes()
    picks_sha = hashlib.sha256(raw_picks).hexdigest()
    arms = chosen_arms(json.loads(raw_picks)) if arms is None else list(arms)
    require(arms and len(set(arms)) == len(arms), 'need distinct arms')
    names, labels = held_data(aspects)
    hashes, cells = {}, []
    doc = dict(cells=cells, inputs=dict(picks_path=str(PICKS), picks_sha256=picks_sha,
               export_sha256=hashes, draws_ami=draws_ami, draws_map=draws_map,
               code_sha256=sha256(__file__), seeds=list(SEEDS), list_sha256=names_sha(names),
               arm_root=str(Path(arm_root).resolve()), cstar_root=str(CSTAR_ROOT),
               label_sha256={a: array_sha(y) for a, y in labels.items()}, bootstrap_rng_seed=0,
               clustering='stored unit float16 cast to float32; no renormalisation',
               retrieval='unit float32, float64 cosine, stable gallery-order ties; singleton queries dropped'))
    older_paths()
    import r20_ret
    for aspect in aspects:
        y = labels[aspect]
        base_parts, base_ap = {}, {}
        for name, root in [(CSTAR, CSTAR_ROOT)] + [(a, arm_root) for a in arms]:
            part = {k: {} for k in (COUNTS[aspect], 20)}
            aps = {}
            for seed in SEEDS:
                x = load_export(root, name, aspect, seed, names, hashes)
                for k in part:
                    part[k][seed] = partitions(x, k)
                aps[seed] = label_ap(unit(x), y, names)
                print(f'{aspect} {name} seed={seed} scored', file=sys.stderr, flush=True)
            ap = stacked_ap(aps)
            if not base_parts:
                base_parts, base_ap = part, ap
                continue
            for metric, k in (('ami_class', COUNTS[aspect]), ('ami20', 20)):
                point = paired_values({s: ami_value(part[k][s], y) for s in SEEDS},
                                      {s: ami_value(base_parts[k][s], y) for s in SEEDS})
                interval = ami_interval(part[k], base_parts[k], y, draws_ami)
                cells.append(dict(arm=name, aspect=aspect, metric=metric, **point,
                                  lo=interval['lo'], hi=interval['hi'],
                                  word=word(interval['lo'], interval['hi'])))
            point = paired_values({s: float(ap['ap'][s].mean()) for s in SEEDS},
                                  {s: float(base_ap['ap'][s].mean()) for s in SEEDS})
            interval = r20_ret.paired(ap, base_ap, draws=draws_map)
            cells.append(dict(arm=name, aspect=aspect, metric='map', **point,
                              lo=interval['lo'], hi=interval['hi'],
                              word=word(interval['lo'], interval['hi'])))
    doc['seconds'] = time.monotonic() - start
    write_json(Path(out) / 'd16_table.json', doc)
    return doc


def regress():
    start = time.monotonic()
    arm = 'k11=40-no_buddy'
    doc = derive([arm], CSTAR_ROOT, SMOKE_OUT, ('emotion',))
    checks, ref_hashes = [], {}

    def reference(path):
        ref_hashes[str(path)] = sha256(path)
        return json.loads(path.read_text())

    def check(name, actual, expected):
        diff = float(abs(actual - expected))
        checks.append(dict(name=name, actual=actual, expected=expected, abs_diff=diff, passed=diff <= 1e-9))

    ami = reference(R10 / 'ablation/held.json')['results']['buddy']['aspects']['emotion']
    ret = reference(R20 / 'ret_full/ablation_ret.json')['aspects']['emotion']['arms']
    for c in doc['cells']:
        if c['metric'] == 'ami20':
            ref = dict(d=ami['effect'], lo=ami['lo'], hi=ami['hi'],
                       arm_mean=ami['arm_ami'], cstar_mean=ami['final_ami'])
        elif c['metric'] == 'map':
            ref = dict(ret[arm]['comparison'], arm_mean=ret[arm]['map'], cstar_mean=ret[CSTAR]['map'])
        else:
            continue
        for key, value in ref.items():
            check(f'{c["metric"]}/{key}', c[key], value)
        for s in SEEDS:
            a = ami['arm_seed_ami'][str(s)] if c['metric'] == 'ami20' else ret[arm]['per_seed'][s]
            b = ami['final_seed_ami'][str(s)] if c['metric'] == 'ami20' else ret[CSTAR]['per_seed'][s]
            check(f'{c["metric"]}/arm/s{s}', c['arm_seeds'][s], a)
            check(f'{c["metric"]}/cstar/s{s}', c['cstar_seeds'][s], b)
    # r17 committed records retain only test means (ranges are validation).
    # Check that mean, and independently replay its frozen scorer per seed.
    rec = reference(R17 / 'records/outcomes.json')
    c = next(c for c in doc['cells'] if c['metric'] == 'ami_class')
    mean = next(r['cand'] for r in rec['cells'] if r['id'] == 'SC1/artelingo/qwen/emotion/ami/9/plain')
    check('ami_class/r17_record_mean', c['cstar_mean'], mean)
    import ns_metrics
    names, labels = held_data(('emotion',))
    for s in SEEDS:
        x = load_export(CSTAR_ROOT, CSTAR, 'emotion', s, names, doc['inputs']['export_sha256'])
        frozen = ns_metrics.score_partitions('ami', labels['emotion'], ns_metrics.partitions(x, 9, 'artelingo'))
        check(f'ami_class/r17_frozen_scorer/s{s}', c['cstar_seeds'][s], frozen)
    maximum = max(r['abs_diff'] for r in checks)
    passed = all(r['passed'] for r in checks)
    result = dict(passed=passed, cells=len(doc['cells']), max_abs=maximum, checks=checks,
                  seconds=time.monotonic() - start, reference_sha256=ref_hashes,
                  inputs=doc['inputs'], class_count_reference_note=
                  'Held per-seed values are absent from records; mean checked against outcomes.json, '
                  'each seed checked against frozen ns_metrics replay.')
    write_json(SMOKE_OUT / 'regress.json', result)
    print(f'D16 REGRESS {"PASS" if passed else "FAIL"} cells={len(doc["cells"])} max_abs={maximum:.12g}', flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('derive')
    d.add_argument('--arms', nargs='+')
    d.add_argument('--arm-root', type=Path, default=ARM_ROOT)
    d.add_argument('--out', type=Path, default=FULL_OUT)
    d.add_argument('--aspects', nargs='+', choices=tuple(COUNTS), default=tuple(COUNTS))
    d.add_argument('--draws-ami', type=int, default=DRAW_AMI)
    d.add_argument('--draws-map', type=int, default=DRAW_MAP)
    sub.add_parser('regress')
    args = vars(parser.parse_args())
    command = args.pop('command')
    with threadpool_limits(limits=8):
        if command == 'regress':
            return regress()
        doc = derive(**args)
    print(f'D16 DERIVE cells={len(doc["cells"])} seconds={doc["seconds"]:.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
