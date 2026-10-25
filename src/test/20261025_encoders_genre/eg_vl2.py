"""VL2: independent stage-ii recomputation from the EG/RB/CT rules.

No scorer is imported. Partitions live in memory; scorer caches and lift.json
are read only after the independent calculation has finished. Unspecified
MiniBatchKMeans arguments use sklearn defaults, as rb_ami.py does.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess

import numpy as np
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
import threadpoolctl
from threadpoolctl import threadpool_limits

import eg_paths
from eg_encoders import ENCODERS

N_ROWS = 29232
N_DRAWS = 2000
SEEDS = np.arange(5)
TOL = 1e-6
_BOOTSTRAP = None


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def names_sha256(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


class Inputs:
    """Hash every consumed input, including sidecars and raw store arrays."""
    def __init__(self):
        self.hashes = {}
        self.absent = set()

    def track(self, path):
        path = Path(path).resolve()
        if str(path) not in self.hashes:
            self.hashes[str(path)] = file_sha256(path)
        return path

    def exists(self, path):
        exists = Path(path).is_file()
        if not exists:
            self.absent.add(str(Path(path).resolve()))
        return exists

    def json(self, path, keys):
        path = self.track(path)
        value = json.loads(path.read_text())
        assert isinstance(value, dict) and set(keys) <= value.keys(), f'{path}: keys {keys}'
        return value


def rederive_encoder(encoder, outputs, names, labels, missing, inputs,
                     store_root=None, store_count=61402, workers=8, n_draws=N_DRAWS):
    """One frozen encoder entry: missing inputs suppress its entire outcome."""
    e = encoder
    record = dict(encoder=e.store, student=e.student, own=e.own, frozen=e.frozen,
                  status='missing', outcome=None)
    for name in (e.student, e.own, e.frozen, e.store):
        if name in missing:
            record['reason'] = f'{name}: {missing[name]}'
            return record, {}
    root = Path(store_root) if store_root is not None else eg_paths.PRE / 'artelingo_full'
    exports = [eg_paths.p_student(outputs, e.student, a, s)
               for a in eg_paths.ASPECTS for s in SEEDS]
    exports += [eg_paths.p_own(outputs, e.own, s) for s in SEEDS]
    required = exports + [p.with_suffix('.json') for p in exports]
    required += [root / e.store / 'img.npy']
    # The original Qwen store may carry ids.npz rather than a generic index.
    index = root / e.store / 'index.json'
    if e.store == 'qwen3vl_emb_2b' and not inputs.exists(index):
        required.append(root / e.store / 'ids.npz')
    else:
        required.append(index)
    absent = [str(p) for p in required if not inputs.exists(p)]
    if absent:
        record['reason'] = 'missing files: ' + ', '.join(absent)
        return record, {}
    return _score_encoder(e, outputs, names, labels, inputs, root, store_count, workers, n_draws)


def l2_float32(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norm > 0).all()
    return x / norm


def partitions(vectors):
    """Our own k-means calls, all P rows, before masking, at eight threads."""
    x = np.asarray(vectors, dtype=np.float32)
    assert x.ndim == 2 and len(x) >= 20 and np.isfinite(x).all()
    assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3)
    with threadpool_limits(limits=8):
        return np.stack([MiniBatchKMeans(
            n_clusters=20, random_state=s, n_init=3, batch_size=4096
        ).fit_predict(x) for s in range(3)]).astype(np.int64)


def read_export(path, inputs, names, arm, seed, aspect=None):
    key, width, dtype = ('img', 2048, np.float16) if aspect else ('prob', 40, np.float32)
    meta = inputs.json(path.with_suffix('.json'),
                       ['p_sha256', 'arm', 'seed', 'rows', 'width'])
    assert meta['p_sha256'] == names_sha256(names), f'{path}: P hash'
    assert (meta['arm'], meta['seed'], meta['rows'], meta['width']) == (
        arm, int(seed), len(names), width), f'{path}: export identity'
    if aspect:
        assert meta['aspect'] == aspect, f'{path}: aspect'
    with np.load(inputs.track(path), allow_pickle=False) as archive:
        assert key in archive.files, f'{path}: missing {key}'
        x = archive[key]
    assert x.dtype == dtype and x.shape == (len(names), width), f'{path}: shape/dtype'
    assert np.isfinite(x).all(), f'{path}: nonfinite rows'
    if aspect:
        assert np.all(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1) < 1e-3)
    else:
        assert ((x >= 0) & (x <= 1)).all(), f'{path}: not sigmoid probabilities'
    return x


def read_frozen(root, encoder, names, inputs, count):
    path = root / encoder.store
    index_path = path / 'index.json'
    if inputs.exists(index_path):
        index = inputs.json(index_path, ['images', 'n_images', 'dim', 'dtype', 'smoke',
                                        'backbone', 'dataset'])
        assert index['backbone'] == encoder.store and index['dataset'] == root.name
        assert index['dim'] == encoder.width and index['dtype'] == 'float16' and not index['smoke']
        stored = [r['painting'] for r in index['images']]
        assert len(stored) == index['n_images'] == count
    else:
        assert encoder.store == 'qwen3vl_emb_2b'
        with np.load(inputs.track(path / 'ids.npz'), allow_pickle=False) as ids:
            key = 'image_painting_ids' if 'image_painting_ids' in ids.files else 'painting_names'
            assert key in ids.files
            ordered = ids[key]
            assert ordered.ndim == 1 and len(ordered) == count
            stored = ordered.tolist()
    assert all(isinstance(n, str) for n in stored) and len(set(stored)) == count
    raw = np.load(inputs.track(path / 'img.npy'), mmap_mode='r', allow_pickle=False)
    assert raw.dtype == np.float16 and raw.shape == (count, encoder.width)
    lookup = {n: i for i, n in enumerate(stored)}
    assert len(set(names)) == len(names) and set(names) <= lookup.keys()
    rows = np.array([lookup[n] for n in names], dtype=np.int64)
    assert [stored[i] for i in rows] == names, 'frozen name order'
    return l2_float32(raw[rows])


def outcome(emotion):
    assert np.isfinite([emotion['lift'], *emotion['interval']]).all(), 'lift and bounds must be finite'
    if emotion['lift'] <= 0:
        return 'L-FAILS'
    return 'L-HOLDS' if emotion['interval'][0] > 0 else 'L-QUAL'


def _score_encoder(e, outputs, names, labels, inputs, root, count, workers, n_draws):
    p = {}
    frozen = partitions(read_frozen(root, e, names, inputs, count))
    p[f'{e.frozen}/default'] = frozen
    own_r20, own_own = [], []
    for s in SEEDS:
        prob = read_export(eg_paths.p_own(outputs, e.own, s), inputs, names, e.own, s)
        r20 = partitions(l2_float32(prob))
        own = np.argmax(prob, axis=1).astype(np.int64)[None, :]
        p[f'{e.own}/P3__s{s}__r20'] = r20
        p[f'{e.own}/P3__s{s}__own'] = own
        own_r20.append(r20)
        own_own.append(own)
    aspects = {}
    for a in eg_paths.ASPECTS:
        students = []
        for s in SEEDS:
            x = read_export(eg_paths.p_student(outputs, e.student, a, s), inputs,
                            names, e.student, s, a)
            part = partitions(x)
            p[f'{e.student}/{a}__s{s}'] = part
            students.append(part)
        aspects[a] = score_aspect(labels[a], np.stack(students), np.stack(own_r20),
                                 np.stack(own_own), frozen, n_draws, workers)
    return dict(encoder=e.store, student=e.student, own=e.own, frozen=e.frozen,
                status='scored', outcome=outcome(aspects['emotion']), aspects=aspects), p


def compare_partitions(outputs, ours, p_sha, n_rows, inputs):
    """Consume scorer caches only with independently computed arrays in hand."""
    found = []
    for name, array in ours.items():
        path = Path(outputs) / 'stage_ii/partitions' / (name + '.npz')
        where = f'partitions.{name}'
        if not inputs.exists(path) or not inputs.exists(path.with_suffix('.json')):
            found.append(dict(path=where, ours='present', theirs='<missing>'))
            continue
        meta = inputs.json(path.with_suffix('.json'),
                           ['identity', 'sha256', 'shape', 'k', 'states', 'threads'])
        identity = meta['identity']
        assert isinstance(identity, dict) and {
            'arm', 'name', 'kind', 'p_sha256', 'source_sha256',
            'sklearn', 'numpy', 'rb_ami_sha256'} <= identity.keys(), f'{path}: identity keys'
        assert isinstance(identity['source_sha256'], str) and len(identity['source_sha256']) == 64
        arm, partition_name = name.split('/')
        expected = dict(p_sha256=p_sha, arm=arm, name=partition_name,
                        kind='own' if partition_name.endswith('__own') else 'r20',
                        sklearn=sklearn.__version__, numpy=np.__version__,
                        rb_ami_sha256=file_sha256(inputs.track(
                            eg_paths.TEST / '20261025_recipe_backbones/rb_ami.py')))
        for key in expected:
            found += differences(expected[key], identity[key], f'{where}.identity.{key}')
        with np.load(inputs.track(path), allow_pickle=False) as cache:
            assert 'labels' in cache.files, f'{path}: labels key'
            theirs = cache['labels']
        assert theirs.dtype == np.int64 and theirs.ndim == 2, f'{path}: dtype/rank'
        assert theirs.shape[1] == n_rows and (theirs >= 0).all(), f'{path}: rows/labels'
        array_sha = hashlib.sha256(np.ascontiguousarray(theirs).tobytes()).hexdigest()
        found += differences(array_sha, meta['sha256'], f'{where}.sha256')
        found += differences(list(theirs.shape), meta['shape'], f'{where}.shape')
        for key, value in dict(k=20, states=[0, 1, 2], threads=8).items():
            found += differences(value, meta[key], f'{where}.{key}')
        found += differences(array, theirs, where)
    return found


def mean_score(labels, parts, indices, seeds=None):
    """A: mean of per-seed means of partition AMIs, with repeats in I/S."""
    if parts.ndim == 2:  # seedless frozen arm
        parts = parts[None, ...]
    if seeds is None:
        seeds = np.arange(len(parts))
    scores = {int(s): float(np.mean([
        adjusted_mutual_info_score(labels[indices], p[indices])
        for p in parts[s]])) for s in np.unique(seeds)}
    return float(np.mean([scores[int(s)] for s in seeds]))


def _init_bootstrap(labels, student, r20, own, labelled):
    global _BOOTSTRAP
    _BOOTSTRAP = labels, student, r20, own, labelled


def _bootstrap_chunk(bounds):
    labels, student, r20, own, labelled = _BOOTSTRAP
    start, stop = bounds
    rng = np.random.default_rng(0)
    values = []
    with threadpool_limits(limits=1):
        for b in range(stop):
            indices = labelled[rng.integers(0, len(labelled), size=len(labelled))]
            sk = rng.choice(SEEDS, size=5, replace=True)
            so = rng.choice(SEEDS, size=5, replace=True)
            if b < start:
                continue
            a_student = mean_score(labels, student, indices, sk)
            a_own = max(mean_score(labels, r20, indices, so),
                        mean_score(labels, own, indices, so))
            values.append(a_student - a_own)
    return values


def score_aspect(labels, student, r20, own, frozen, n_draws=N_DRAWS, workers=8):
    """Recompute one encoder/aspect; tiny arrays/draw counts support rule tests."""
    assert 1 <= workers <= 8 and n_draws > 0
    assert labels.dtype == np.int64 and labels.ndim == 1
    n = len(labels)
    for p, shape in ((student, (5, 3, n)), (r20, (5, 3, n)),
                     (own, (5, 1, n)), (frozen, (3, n))):
        assert p.shape == shape and p.dtype == np.int64 and (p >= 0).all()
    labelled = np.flatnonzero(labels >= 0)
    assert len(labelled) > 0
    scores = {name: mean_score(labels, p, labelled) for name, p in
              [('student', student), ('own_r20', r20), ('own_own', own), ('frozen', frozen)]}
    best = 'own_r20' if scores['own_r20'] >= scores['own_own'] else 'own_own'
    edges = np.linspace(0, n_draws, min(workers, n_draws) + 1, dtype=int)
    tasks = list(zip(edges[:-1], edges[1:]))
    args = (labels, student, r20, own, labelled)
    if workers == 1:
        _init_bootstrap(*args)
        values = [_bootstrap_chunk(task) for task in tasks]
    else:
        with multiprocessing.get_context('fork').Pool(
                len(tasks), initializer=_init_bootstrap, initargs=args) as pool:
            values = pool.map(_bootstrap_chunk, tasks)
    interval = np.percentile(np.concatenate(values), [2.5, 97.5]).tolist()
    lift = scores['student'] - scores[best]
    assert np.isfinite([lift, *interval]).all(), 'lift and bounds must be finite'
    return dict(n_labelled=len(labelled), n_draws=n_draws, A=scores, own_best=best,
                lift=lift, interval=interval,
                student_minus_frozen=scores['student'] - scores['frozen'])


def differences(ours, theirs, path=''):
    """All semantic differences; partitions require exact cell equality."""
    def serializable(value):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [serializable(v) for v in value]
        return value

    def mismatch(a, b, where=path):
        return [dict(path=where, ours=serializable(a), theirs=serializable(b))]

    if isinstance(ours, np.ndarray):
        if not isinstance(theirs, np.ndarray):
            return mismatch(str(ours.dtype), str(type(theirs)))
        if ours.shape != theirs.shape:
            return mismatch(list(ours.shape), list(theirs.shape), path + '.shape')
        found = []
        for cell in np.argwhere(ours != theirs):
            index = tuple(cell)
            found += mismatch(ours[index].item(), theirs[index].item(),
                              path + ''.join(f'[{i}]' for i in cell))
        return found
    if isinstance(ours, dict) and isinstance(theirs, dict):
        found = []
        for key in sorted(ours.keys() | theirs.keys()):
            where = f'{path}.{key}' if path else key
            if key not in ours or key not in theirs:
                found += mismatch(ours.get(key, '<missing>'), theirs.get(key, '<missing>'), where)
            else:
                found += differences(ours[key], theirs[key], where)
        return found
    if isinstance(ours, (list, tuple)) and isinstance(theirs, (list, tuple)):
        if len(ours) != len(theirs):
            return mismatch(ours, theirs)
        return [d for i, (a, b) in enumerate(zip(ours, theirs))
                for d in differences(a, b, f'{path}[{i}]')]
    if isinstance(ours, (int, float)) and not isinstance(ours, bool):
        if not isinstance(theirs, (int, float)) or isinstance(theirs, bool):
            return mismatch(ours, theirs)
        if not np.isfinite(ours) or not np.isfinite(theirs) or abs(ours - theirs) > TOL:
            return mismatch(ours, theirs)
        return []
    return [] if type(ours) is type(theirs) and ours == theirs else mismatch(ours, theirs)


def read_labels(names, inputs, annotations_path=None, genre_paths=None):
    """RB §3.1: global sorted codes, plurality emotion, first-caption style."""
    annotations_path = Path(annotations_path or os.environ.get(
        'COSIR_ARTELINGO_ANNOTATIONS', '/data/PDD/artelingo/artelingo_train.json'))
    annotations = json.loads(inputs.track(annotations_path).read_text())
    assert isinstance(annotations, list) and annotations
    assert all(isinstance(r, dict) and all(isinstance(r.get(k), str) and r[k]
               for k in ('painting', 'emotion', 'art_style')) for r in annotations)
    emotions = {v: i for i, v in enumerate(sorted({r['emotion'] for r in annotations}))}
    styles = {v: i for i, v in enumerate(sorted({r['art_style'] for r in annotations}))}
    votes, first_style = {}, {}
    wanted = set(names)
    for row in annotations:
        name = row['painting']
        if name in wanted:
            if name not in votes:
                votes[name] = np.zeros(len(emotions), dtype=np.int64)
                first_style[name] = styles[row['art_style']]
            votes[name][emotions[row['emotion']]] += 1
    assert votes.keys() == wanted, 'P names without captions'
    if genre_paths is None:
        root = Path(os.environ.get('COSIR_WIKIART_GENRE_DIR') or '/data/SSD/wikiart_genre')
        genre_paths = [root / 'genre_train.csv', root / 'genre_val.csv']
    assert genre_paths
    genre_ids = {}
    for path in genre_paths:
        with inputs.track(path).open() as handle:
            for row in csv.reader(handle):
                assert len(row) == 2, f'{path}: genre columns'
                name, label = Path(row[0]).stem, int(row[1])
                assert name and 0 <= label < 10
                genre_ids[name] = label if name not in genre_ids else (
                    label if genre_ids[name] == label else -1)
    return dict(emotion=np.array([np.argmax(votes[n]) for n in names], dtype=np.int64),
                style=np.array([first_style[n] for n in names], dtype=np.int64),
                genre=np.array([genre_ids.get(n, -1) for n in names], dtype=np.int64))


def versions():
    return dict(sklearn=sklearn.__version__, numpy=np.__version__,
                scipy=scipy.__version__, threadpoolctl=threadpoolctl.__version__)


def _semantic_record(record):
    # Commit/source provenance describes each implementation; reproducibility
    # settings and all statistical fields must agree.
    keys = ('p_sha256', 'n_rows', 'n_draws', 'label_sha256', 'label_counts',
            'versions', 'threads', 'rb_ami_sha256', 'encoders')
    result = {k: record.get(k, '<missing>') for k in keys}
    if isinstance(result['versions'], dict):
        result['versions'] = {k: result['versions'].get(k, '<missing>')
                              for k in ('sklearn', 'numpy')}
    if isinstance(result['encoders'], dict):
        result['encoders'] = {k: {field: value for field, value in v.items() if field != 'reason'}
                              for k, v in result['encoders'].items()}
    return result


def _cached_result(path, config):
    cached = json.loads(path.read_text())
    assert {'config', 'input_hashes', 'absent_inputs', 'agree', 'differences',
            'code_commit', 'recomputed'} <= cached.keys(), f'{path}: report keys'
    valid = cached['config'] == config and all(
        Path(p).is_file() and file_sha256(p) == sha for p, sha in cached['input_hashes'].items())
    valid = valid and all(not Path(p).is_file() for p in cached['absent_inputs'])
    assert valid, f'{path}: stale result; preserve/archive it before rerunning'
    return cached


def run(outputs, workers=8, smoke=False, *, p_path=None, pick_path=None,
        annotations_path=None, genre_paths=None, n_rows=N_ROWS, n_draws=None,
        p_sha256=eg_paths.P_SHA256, store_root=None, store_count=61402):
    """Write a VL2 report. Optional raw-input paths/sizes permit tiny tests.

    CLI runs use the frozen full P and 2,000 draws, or 20 for --smoke.
    Smoke outputs are restricted to /tmp/claude-0/.
    """
    n_draws = (20 if smoke else N_DRAWS) if n_draws is None else n_draws
    assert 1 <= workers <= 8 and n_draws > 0
    outputs = Path(outputs)
    assert not smoke or eg_paths.smoke_ok(outputs), '--smoke writes only under /tmp/claude-0/'
    p_path = Path(p_path) if p_path is not None else eg_paths.RB_PART1 / 'P.json'
    pick_path = Path(pick_path) if pick_path is not None else eg_paths.RB_PART1 / 'pick.json'
    config = dict(n_rows=n_rows, n_draws=n_draws, p_sha256=p_sha256, smoke=smoke,
                  p_path=str(p_path.resolve()), pick_path=str(pick_path.resolve()),
                  annotations_path=str(annotations_path) if annotations_path else None,
                  genre_paths=[str(p) for p in genre_paths] if genre_paths is not None else None,
                  store_root=str(store_root) if store_root else None, store_count=store_count,
                  annotations_env=os.environ.get('COSIR_ARTELINGO_ANNOTATIONS'),
                  genre_env=os.environ.get('COSIR_WIKIART_GENRE_DIR'))
    output = outputs / 'stage_ii/rederive_vl2.json'
    if output.exists():
        return _cached_result(output, config)
    inputs = Inputs()
    for path in (Path(__file__), Path(eg_paths.__file__),
                 eg_paths.ROUND / 'eg_encoders.py', eg_paths.RULE,
                 eg_paths.TEST / '20261025_recipe_backbones/DECISION_RULE.md',
                 eg_paths.TEST / '20261020_claim_test/DECISION_RULE.md'):
        inputs.track(path)
    rb_ami_path = eg_paths.TEST / '20261025_recipe_backbones/rb_ami.py'
    inputs.track(rb_ami_path)  # provenance only; never imported
    p = inputs.json(p_path, ['names', 'sha256'])
    names = p['names']
    assert isinstance(names, list) and all(isinstance(n, str) and n for n in names)
    assert len(names) == len(set(names)) == n_rows and names == sorted(names), 'P names/order'
    assert names_sha256(names) == p['sha256'] == p_sha256, 'P hash'
    if 'counts' in p:
        assert sum(p['counts'].values()) == n_rows
    labels = read_labels(names, inputs, annotations_path, genre_paths)
    label_hashes = {a: hashlib.sha256(np.ascontiguousarray(y, dtype=np.int64).tobytes()).hexdigest()
                    for a, y in labels.items()}
    counts = {a: int(np.count_nonzero(y >= 0)) for a, y in labels.items()}
    pick = inputs.json(pick_path, ['p_sha256', 'n_rows', 'label_sha256', 'label_counts'])
    assert pick['p_sha256'] == p_sha256 and pick['n_rows'] == n_rows, 'pick P identity'
    assert pick['label_sha256'] == label_hashes, 'RB pick label hashes'
    assert pick['label_counts'] == counts, 'RB pick label counts'
    index = inputs.json(outputs / 'stage_ii/p/index.json', ['missing'])
    assert isinstance(index['missing'], dict) and all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in index['missing'].items())
    if 'p_sha256' in index:
        assert index['p_sha256'] == p_sha256
    derived, independent_partitions = {}, {}
    for e in ENCODERS:
        print(f'VL2: deriving {e.student} ({e.store})', flush=True)
        derived[e.student], parts = rederive_encoder(
            e, outputs, names, labels, index['missing'], inputs, store_root=store_root,
            store_count=store_count, workers=workers, n_draws=n_draws)
        independent_partitions.update(parts)
    ours = dict(p_sha256=p_sha256, n_rows=n_rows, n_draws=n_draws,
                label_sha256=label_hashes, label_counts=counts, encoders=derived,
                versions=versions(), threads=8,
                rb_ami_sha256=inputs.hashes[str(rb_ami_path.resolve())])
    # Only now may the scorer's numbers/partitions be consumed.
    theirs = inputs.json(outputs / 'stage_ii/lift.json',
                         ['p_sha256', 'n_rows', 'n_draws', 'label_sha256', 'label_counts',
                          'versions', 'threads', 'rb_ami_sha256', 'code_commit', 'sources', 'encoders'])
    assert isinstance(theirs['encoders'], dict)
    assert all(isinstance(v, dict) and {'status', 'outcome'} <= v.keys()
               and v['status'] in ('scored', 'missing')
               and (v['status'] != 'scored' or isinstance(v.get('aspects'), dict))
               for v in theirs['encoders'].values()), 'lift encoder schema'
    found = differences(_semantic_record(ours), _semantic_record(theirs))
    found += compare_partitions(outputs, independent_partitions, p_sha256, n_rows, inputs)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=eg_paths.ROUND, text=True).strip()
    report = dict(agree=not found, differences=found, code_commit=commit,
                  input_hashes=inputs.hashes, absent_inputs=sorted(inputs.absent),
                  p_sha256=p_sha256, label_sha256=label_hashes, recomputed=ours,
                  config=config, versions=versions(), threads=8,
                  rb_ami_sha256=inputs.hashes[str(rb_ami_path.resolve())],
                  kmeans=MiniBatchKMeans(n_clusters=20, n_init=3, batch_size=4096).get_params())
    report['kmeans']['random_state'] = [0, 1, 2]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    eg_paths.check_imports()
    result = run(args.outputs, args.workers, args.smoke)
    print(f'VL2 agree={result["agree"]}; differences={len(result["differences"])}')
    return 0 if result['agree'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
