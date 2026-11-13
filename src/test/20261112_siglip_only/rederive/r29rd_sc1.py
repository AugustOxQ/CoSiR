#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent r29 SC1 derivation. Imports no project/round modules.

Sources: r29 spec §§3.2/3.4/4/6, r23 §§3.3/3.4, r22 §3.4;
permitted r29_model/r29_captions and r23 tcl_c13 for architecture/ID conventions.
Default invocation requires all six checkpoints and scores every validation query.
--query-limit samples queries only: the galleries always remain complete.
"""
import os
import sys
import time

START = time.monotonic()
sys.dont_write_bytecode = True
for _key, _value in dict(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                         OPENBLAS_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1',
                         HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='').items():
    os.environ[_key] = _value

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
FEATURES = Path('/data/SSD2/pre_extract/artelingo/features')
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
STORE_ROOT = Path('/data/SSD2/pre_extract/artelingo_full')
STORES = dict(siglip1=STORE_ROOT / 'siglip_so400m',
              siglip1_prefix=STORE_ROOT / 'siglip_so400m_capprefix')
WEIGHTS = Path('/data/SSD2/cosir_r29/20261112_siglip_only/cstar/siglip1/weights')
GENRE_DIR = Path('/data/SSD/wikiart_genre')
ASPECTS = ('emotion', 'style', 'genre')
DIRECTIONS = ('t2i', 'i2t', 't2t')
ARMS = ('siglip1', 'siglip1_prefix', 'p1_siglip1', 'p1_siglip1_prefix')
VAL_SHA256 = '1430d4b56e4a76dbe3e1c4d699e1db5c92967c5f2048b74b6cbcdf1589809e2e'
NOTES = [
    'Gallery order for exact cosine ties: painting names ascending for images; sample IDs ascending for captions (r22/r23 convention).',
    'Painting-emotion plurality ties use the alphabetically earliest emotion, as the frozen label convention; all nine emotions, including something else, are retained.',
    'Float16 stored rows are promoted to float32 and L2-normalised once before towers; towers, exact GELU, output L2 and cosine use float32 CPU. AP arithmetic, cell means and nine-cell group means use float64; no intermediate float32 AP storage.',
    'A query with no relevant labelled gallery row after own-row removal has nonfinite AP and is excluded; counts report finite APs only.',
    'Own-row exclusion applies to the identical caption sample ID in t2t only; paired paintings/captions remain in cross-modal galleries.',
    'The requested prefixed store records prefix "represent the caption: " (lowercase) in index.json, while the specification writes "Represent the caption: ". The specified stored vectors are used as supplied, without re-encoding.',
    'p1_form reports the independently stronger P1 form even if K1 wins; its use as the comparator form is conditional on K2 winning, per D20.',
]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f'Output outside rederive/: {path}')
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and np.isfinite(norm).all() and (norm > 0).all()
    return x / norm


def join_rows(stored, requested):
    """Only explicit identities establish row positions; never a dataset image_id."""
    lookup = {identity: row for row, identity in enumerate(stored)}
    assert len(lookup) == len(stored), 'duplicate store identities'
    assert len(set(requested)) == len(requested), 'duplicate requested identities'
    return np.asarray([lookup[identity] for identity in requested], dtype=np.int64)


def validation_frame():
    annotations = read_json(ANNOTATIONS)
    assert len(annotations) == 308723
    metadata = read_json(FEATURES / 'metadata.json')
    assert metadata['backbone_model'] == 'openai/clip-vit-base-patch32'
    # Reproduce the frozen painting-grouped split from its original sample-ID order.
    # These are real extraction IDs, not range(n) used as an assumed store join.
    ids = np.concatenate([np.load(FEATURES / 'shards' / f'shard_{shard:05d}' / 'sample_ids.npy',
                                  allow_pickle=False)
                          for shard in range(metadata['num_shards'])])
    assert ids.dtype == np.int64 and len(ids) == len(np.unique(ids)) == len(annotations)
    assert ids.min() >= 0 and ids.max() < len(annotations)
    painting_to_group = {}
    group_ids = []
    for sample_id in ids:
        painting = annotations[int(sample_id)]['painting']
        group_ids.append(painting_to_group.setdefault(painting, len(painting_to_group)))
    group_ids = np.asarray(group_ids, dtype=np.int64)
    counts = np.bincount(group_ids)
    permutation = np.random.default_rng(42).permutation(len(counts))
    shares = counts[permutation] / counts.sum()
    starts = np.cumsum(shares) - shares
    part = np.empty(len(counts), dtype=np.int64)
    part[permutation] = np.searchsorted(np.cumsum([0.7, 0.1, 0.2])[:2], starts, side='right')
    caption_ids = np.sort(ids[part[group_ids] == 1])
    names = sorted({annotations[int(i)]['painting'] for i in caption_ids})
    assert (len(names), len(caption_ids)) == (6152, 30872)
    assert hashlib.sha256('\n'.join(names).encode()).hexdigest() == VAL_SHA256
    lookup = {name: row for row, name in enumerate(names)}
    tp = np.asarray([lookup[annotations[int(i)]['painting']] for i in caption_ids], dtype=np.int64)
    # Every annotation of each selected painting must be present.
    expected_ids = {i for i, row in enumerate(annotations) if row['painting'] in lookup}
    assert set(caption_ids) == expected_ids
    emotions = sorted({r['emotion'] for r in annotations})
    styles = sorted({r['art_style'] for r in annotations})
    assert len(emotions) == 9 and len(styles) == 27
    emotion_code = {name: i for i, name in enumerate(emotions)}
    style_code = {name: i for i, name in enumerate(styles)}
    caption_emotion = np.asarray([emotion_code[annotations[int(i)]['emotion']] for i in caption_ids])
    emotion_votes = np.zeros((len(names), len(emotions)), dtype=np.int64)
    np.add.at(emotion_votes, (tp, caption_emotion), 1)
    image_emotion = emotion_votes.argmax(axis=1)
    style_by_name = {}
    for row in annotations:
        assert style_by_name.setdefault(row['painting'], row['art_style']) == row['art_style']
    image_style = np.asarray([style_code[style_by_name[name]] for name in names], dtype=np.int64)
    genre_by_name = {}
    for path in (GENRE_DIR / 'genre_train.csv', GENRE_DIR / 'genre_val.csv'):
        with path.open() as stream:
            for relative_path, raw_label in csv.reader(stream):
                name, label = Path(relative_path).stem, int(raw_label)
                assert 0 <= label < 10
                previous = genre_by_name.setdefault(name, label)
                if previous != label:
                    genre_by_name[name] = -1
    image_genre = np.asarray([genre_by_name.get(name, -1) for name in names], dtype=np.int64)
    assert np.count_nonzero(image_genre >= 0) == 4960
    labels = dict(emotion=dict(t=caption_emotion, i=image_emotion),
                  style=dict(t=image_style[tp], i=image_style),
                  genre=dict(t=image_genre[tp], i=image_genre))
    return dict(names=names, ids=caption_ids, tp=tp, labels=labels,
                plurality_ties=int(((emotion_votes == emotion_votes.max(1, keepdims=True)).sum(1) > 1).sum()))


def load_store(source, frame, images=False):
    folder = STORES[source]
    index_bytes = (folder / 'index.json').read_bytes()
    index = json.loads(index_bytes)
    progress = read_json(folder / 'progress.json')
    digest = hashlib.sha256(index_bytes).hexdigest()
    assert progress['index_sha256'] == digest and progress['gate']['ok'] is True
    assert progress['txt_done'] == index['n_texts'] == 308723
    assert index['smoke'] is False and index['backbone'] == 'siglip_so400m'
    assert index['dim'] == 1152 and index['dtype'] == 'float16'
    records = index['texts']
    stored_ids = [row['sample_id'] for row in records]
    positions = join_rows(stored_ids, frame['ids'].tolist())
    for position, painting_row in zip(positions, frame['tp']):
        assert records[position]['painting'] == frame['names'][painting_row], 'caption painting mismatch'
    selected_names = set(frame['names'])
    assert sum(row['painting'] in selected_names for row in records) == len(frame['ids'])
    raw = np.load(folder / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == (308723, 1152) and raw.dtype == np.float16
    vectors = dict(t=unit(raw[positions]))
    if images:
        assert progress['img_done'] == index['n_images'] == 61402
        image_names = [row['painting'] for row in index['images']]
        positions = join_rows(image_names, frame['names'])
        raw = np.load(folder / 'img.npy', mmap_mode='r', allow_pickle=False)
        assert raw.shape == (61402, 1152) and raw.dtype == np.float16
        vectors['i'] = unit(raw[positions])
    return vectors, digest


def load_state(source, aspect, store_digest):
    path = WEIGHTS / source / f'{aspect}__s0.pt'
    info = read_json(path.with_suffix('.json'))
    config, rb = info['config'], info['config']['rb']
    assert config['aspect'] == aspect and config['seed'] == 0
    assert rb['width'] == rb['cap_width'] == 1152 and rb['cap_source'] == source
    assert rb['cap_store_sha256'] == store_digest
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = {f'{side}.{name}' for side in ('cap', 'img')
                for name in ('trunk.0.weight', 'trunk.0.bias', 'output.weight', 'output.bias')}
    assert set(state) == expected, 'unexpected architecture/state keys'
    for side in ('cap', 'img'):
        for name, shape in (('trunk.0.weight', (1024, 1152)), ('trunk.0.bias', (1024,)),
                            ('output.weight', (2048, 1024)), ('output.bias', (2048,))):
            value = state[f'{side}.{name}']
            assert value.shape == shape and value.dtype == torch.float32 and torch.isfinite(value).all()
    return state


def tower(x, state, side, batch_size=1024):
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            z = torch.from_numpy(x[start:start + batch_size])
            z = F.linear(z, state[f'{side}.trunk.0.weight'], state[f'{side}.trunk.0.bias'])
            z = F.gelu(z, approximate='none')
            z = F.linear(z, state[f'{side}.output.weight'], state[f'{side}.output.bias'])
            assert torch.isfinite(z).all() and (z.norm(dim=1) > 0).all()
            result.append(F.normalize(z, dim=1).numpy())
    return np.concatenate(result)


def ranked_ap(order, query_labels, gallery_labels):
    """Gallery order already excludes own rows. Mask unlabelled rows only here."""
    ranked_labels = gallery_labels[order]
    relevant = (ranked_labels == query_labels[:, None]) & (query_labels[:, None] >= 0)
    n_relevant = relevant.sum(axis=1)
    labelled_rank = np.cumsum(ranked_labels >= 0, axis=1)
    precision = np.cumsum(relevant, axis=1, dtype=np.float64)
    precision /= np.maximum(labelled_rank, 1)
    precision *= relevant
    ap = precision.sum(axis=1, dtype=np.float64) / np.maximum(n_relevant, 1)
    ap[(query_labels < 0) | (n_relevant == 0)] = np.nan
    return ap


def selected_queries(n, limit):
    if not limit or limit >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(0).choice(n, size=limit, replace=False))


def score_direction(vectors, labels, direction, query_limit=0, batch_size=128):
    qs, gs = dict(t2i=('t', 'i'), i2t=('i', 't'), t2t=('t', 't'))[direction]
    queries = selected_queries(len(vectors[qs]), query_limit)
    gallery = torch.from_numpy(vectors[gs])
    aps = {aspect: [] for aspect in labels}
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            chosen = queries[start:start + batch_size]
            scores = (torch.from_numpy(vectors[qs][chosen]) @ gallery.T).numpy()
            if direction == 't2t':
                scores[np.arange(len(chosen)), chosen] = -np.inf
            # Stable descending sort: equal scores keep original gallery order.
            order = np.argsort(-scores, axis=1, kind='stable')
            if direction == 't2t':
                order = order[order != chosen[:, None]].reshape(len(chosen), len(gallery) - 1)
            for aspect, lab in labels.items():
                aps[aspect].append(ranked_ap(order, lab[qs][chosen], lab[gs]))
    return {aspect: np.concatenate(chunks) for aspect, chunks in aps.items()}


def finite_query_counts(frame):
    """Exact scoring eligibility derived from labels, without any cosine/ranking."""
    result = {}
    for aspect, labels in frame['labels'].items():
        for direction in DIRECTIONS:
            qs, gs = dict(t2i=('t', 'i'), i2t=('i', 't'), t2t=('t', 't'))[direction]
            query_labels, gallery_labels = labels[qs], labels[gs]
            valid = gallery_labels[gallery_labels >= 0]
            histogram = np.bincount(valid)
            number = np.zeros(len(query_labels), dtype=np.int64)
            labelled = (query_labels >= 0) & (query_labels < len(histogram))
            number[labelled] = histogram[query_labels[labelled]]
            if direction == 't2t':
                number[labelled] -= 1
            result[f'{aspect}/{direction}'] = int(((query_labels >= 0) & (number > 0)).sum())
    return result


def decision(arms):
    kstar = None
    if all(arm in arms for arm in ARMS[:2]):
        kstar = 'siglip1_prefix' if arms['siglip1_prefix']['group'] > arms['siglip1']['group'] else 'siglip1'
    form = None
    if all(arm in arms for arm in ARMS[2:]):
        form = 'prefixed' if arms['p1_siglip1_prefix']['group'] > arms['p1_siglip1']['group'] else 'plain'
    return kstar, ('prefixed' if kstar == 'siglip1_prefix' else 'plain') if kstar else None, form


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-limit', type=int, default=0, help='0 = all; otherwise deterministic subset, full galleries')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--output', type=Path, default=HERE / 'sc1_rederived.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--inspect', action='store_true', help='derive counts and check checkpoint availability without scoring')
    args = parser.parse_args()
    assert args.query_limit >= 0 and args.batch_size > 0
    full = not args.query_limit and set(args.arms) == set(ARMS)
    if not full and not args.inspect and args.output.resolve() == HERE / 'sc1_rederived.json':
        parser.error('Subset/partial runs require a different --output; never write partial SC1 as final')
    checkpoint_paths = [WEIGHTS / source / f'{aspect}__s0.pt' for source in ARMS[:2] for aspect in ASPECTS]
    missing = [str(path) for path in checkpoint_paths if not path.is_file() or not path.with_suffix('.json').is_file()]
    if not args.inspect:
        required = [path for path in checkpoint_paths if path.parent.name in args.arms]
        absent = [str(path) for path in required if not path.is_file() or not path.with_suffix('.json').is_file()]
        if absent:
            parser.error('Required checkpoint/sidecar missing: ' + ', '.join(absent))
    frame = validation_frame()
    eligibility = finite_query_counts(frame)
    if args.inspect:
        result = dict(paintings=len(frame['names']), captions=len(frame['ids']),
                      expected_counts_per_arm=eligibility, plurality_ties=frame['plurality_ties'],
                      missing_checkpoints=missing, elapsed_seconds=time.monotonic() - START)
        write_json(HERE / 'r29rd_inspection.json', result)
        print(json.dumps(result), flush=True)
        return
    vectors, digests = {}, {}
    for source in STORES:
        if any(arm.removeprefix('p1_') == source for arm in args.arms):
            vectors[source], digests[source] = load_store(source, frame, images=source == 'siglip1')
    if 'siglip1' not in vectors:
        plain, _ = load_store('siglip1', frame, images=True)
        image_vectors = plain['i']
    else:
        image_vectors = vectors['siglip1']['i']
    for source in vectors:
        vectors[source]['i'] = image_vectors
    arms, counts, timings = {}, {}, {}
    for arm in args.arms:
        started = time.monotonic()
        source = arm.removeprefix('p1_')
        cells, arm_counts = {}, {}
        if arm.startswith('p1_'):
            jobs = [(frame['labels'], vectors[source])]
        else:
            jobs = []
            # Lazy aspect processing keeps only one aspect's embeddings in memory.
        for aspect in ((None,) if arm.startswith('p1_') else ASPECTS):
            if aspect is None:
                labs, vec = jobs[0]
            else:
                state = load_state(source, aspect, digests[source])
                vec = dict(t=tower(vectors[source]['t'], state, 'cap'),
                           i=tower(vectors[source]['i'], state, 'img'))
                labs = {aspect: frame['labels'][aspect]}
            for direction in DIRECTIONS:
                values = score_direction(vec, labs, direction, args.query_limit, args.batch_size)
                for head, ap in values.items():
                    key = f'{head}/{direction}'
                    valid = np.isfinite(ap)
                    assert valid.any(), f'no finite AP: {arm}/{key}'
                    cells[key] = float(np.mean(ap[valid], dtype=np.float64))
                    arm_counts[key] = int(valid.sum())
                    if not args.query_limit:
                        assert arm_counts[key] == eligibility[key]
                print(f'{arm} {aspect or "all"}/{direction}: done', flush=True)
        cells = {f'{a}/{d}': cells[f'{a}/{d}'] for a in ASPECTS for d in DIRECTIONS}
        arms[arm] = dict(cells=cells, group=float(np.mean(list(cells.values()), dtype=np.float64)))
        counts[arm] = {key: arm_counts[key] for key in cells}
        timings[arm] = time.monotonic() - started
        print(f'{arm}: group={arms[arm]["group"]!r}, seconds={timings[arm]:.3f}', flush=True)
    kstar, kstar_form, p1_form = decision(arms) if full else (None, None, None)
    notes = list(NOTES)
    if not full:
        notes.append(f'VALIDATION ONLY: query_limit={args.query_limit}; requested arms={args.arms}; galleries complete; SC1 and P1 decisions suppressed.')
    write_json(args.output, dict(arms=arms, kstar=kstar, kstar_form=kstar_form,
                                p1_form=p1_form, counts=counts, notes=notes))
    runtime = time.monotonic() - START
    provenance = dict(complete=full, query_limit=args.query_limit, painting_count=len(frame['names']),
                      caption_count=len(frame['ids']), expected_counts_per_arm=eligibility,
                      validation_names_sha256=VAL_SHA256, plurality_ties=frame['plurality_ties'],
                      elapsed_seconds=runtime, arm_seconds=timings,
                      script_sha256=sha256(__file__), annotation_sha256=sha256(ANNOTATIONS),
                      store_index_sha256=digests, numpy=np.__version__, torch=torch.__version__,
                      device='cpu', threads=torch.get_num_threads(), batch_size=args.batch_size)
    write_json(args.output.with_suffix('.run.json'), provenance)
    print(json.dumps(dict(output=str(args.output), elapsed_seconds=runtime, kstar=kstar,
                          kstar_form=kstar_form, p1_form=p1_form)), flush=True)


if __name__ == '__main__':
    main()
