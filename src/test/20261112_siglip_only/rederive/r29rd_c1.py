#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent r29 stage 1c. Modes: validate, replay, held.

Only the explicit held command with --picks-committed can read/score K* held
exports. Development and smoke runs use validation or old K0 held inputs only.
All outputs/caches are confined to rederive/. Older C13 sources are adapted,
never imported. Public libraries and our own SC1/SC2/core are the only imports.
"""
import os
import sys
import time
START = time.monotonic()
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import r29rd_c1_core as C
import r29rd_sc1 as rd
import r29rd_sc2 as rd2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ARM = 'siglip1_prefix'
SEEDS = tuple(range(5))
ASPECTS = rd.ASPECTS
R11 = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R14 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
R23 = Path('/data/SSD2/cosir_r23/20261106_text_claim')
TEST_ROOT = Path('/project/CoSiR/src/test')
R17_RECORDS = TEST_ROOT / '20261103_new_scores/records'
R14_RECORDS = TEST_ROOT / '20261102_backbones_sweep/records'
R23_RECORDS = TEST_ROOT / '20261106_text_claim/records'
K0_TEXT_WEIGHTS = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40')
HELD_SHA = 'a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876'
INST = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
CACHE = HERE / 'r29rd_c1_cache'
ALLOW_CANDIDATE_HELD = False
NOTES = [
    'P1 image mAP uses the r17 SC7 untold candidate: unit-normalise emotion/style/genre heads, concatenate in that order, divide by float32 sqrt(3). Q1 image mAP uses the r14 told aspect head. They have separate arm-level cross-fit read-out picks; image_map[aspect].pick is told/Q1 and .p1_pick is untold/P1.',
    'Both image-map pick paths use each told aspect head\'s five-seed R0 validation mAP range as the LEVEL/selection unit. The common arm-level pick is repeated per aspect in the requested output structure.',
    'AMI clusters every painting before masking negative labels at scoring, including in row resampling. Stored candidate float16 rows are promoted to float32 without another L2, as r28/r29 specify. Plain/teacher store inputs are unit-normalised as in the r17 comparator protocol.',
    'Image mAP uses labelled queries/gallery only; fits use all fit paintings without labels. Validation parity halves are defined before label masking; average the two half means equally. Fit the final picked map on all validation rows.',
    'Image menu: none, center, PCA-whiten64/256/1024 (capped at width), each crossed with k=0/1/3/5/10. Covariance is float64 with n-1 divisor, eigenvalues descending, epsilon=1e-6*largest eigenvalue; maps and ranking rows are float32. Equal-variance eigenvector sign choices cancel in cosine.',
    'Image selection uses mean gain in told R0 range units, guard against losses greater than that range, the r11 0.001 tie band and R0/no-expansion/none-center-whiten/smaller-k/larger-dimension order. A 1e-12 comparison epsilon follows the permitted r17 independent derivation; zero ranges invalidate the selection rather than divide by zero.',
    'Text means are fit on all validation text/image output rows in float64 then cast to float32. Centering and expansion rank all gallery rows; unlabelled rows leave only AP. The own caption sample ID leaves t2t gallery, paired paintings remain cross-modal. Text APs are retained as float32 to match r23 per-unit files, with float64 means and intervals; SC1 kept double AP before its mean.',
    'Exact cosine ties keep ascending gallery row order: image painting names, caption extraction sample IDs. Painting plurality ties take the alphabetically first of all nine emotions; unknown/conflicting genre is -1.',
    'Each comparison has a fresh default_rng(0). AMI samples every painting row then five candidate seeds, with three k-means states averaged per seed, 2000 draws. Image mAP samples labelled queries then shared seeds, 10000 draws. Text samples sorted painting units with scored queries then shared seeds; repeats weight query sums and counts, 10000 draws. Percentiles use numpy linear interpolation.',
    'Forward diagnostics compare our image tower with stored exports after float16 rounding; image cells use the stored rows. CUDA is controller-only, TF32 disabled; AMI/k-means/bootstraps stay on CPU with at most eight worker threads/processes.',
    'Read-out picks and ranges must be committed by the controller before the held command. --picks-committed is an explicit assertion of that precondition; the held report pins our validation-picks SHA256. This program makes no commits and reads no forbidden pick record.',
    'A missing or uncertified replay marks its affected comparison INCONCLUSIVE; it contributes neither WIN nor LEVEL. SC3 reports the level supported by decided cells and whether inconclusive cells could raise it, per D22. Replay reports distinguish full interval checks from sampled K0 forward/ranking checks.',
    'The specified prefixed store metadata uses lowercase represent the caption:; use its supplied rows unchanged. P1 caption form is plain as fixed in the task; its own r23 validation pick is used per cell.',
]


def names_hash(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()


def source_hash():
    return hashlib.sha256(''.join(rd.sha256(p) for p in [__file__, C.__file__, rd.__file__, rd2.__file__]).encode()).hexdigest()


def cached(name, identity, compute):
    digest = hashlib.sha256((source_hash() + json.dumps(identity, sort_keys=True, default=str) + C.DEVICE).encode()).hexdigest()
    folder = CACHE / digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}
    result = compute()
    with path.open('wb') as stream:
        np.savez(stream, **result)
    return result


def frame(split):
    if split == 'val':
        return rd.validation_frame()
    assert split == 'held'
    annotations = rd.read_json(rd.ANNOTATIONS)
    meta = rd.read_json(rd.FEATURES / 'metadata.json')
    ids = np.concatenate([np.load(rd.FEATURES / 'shards' / f'shard_{s:05d}' / 'sample_ids.npy', allow_pickle=False)
                          for s in range(meta['num_shards'])])
    groups, group = {}, []
    for sample_id in ids:
        name = annotations[int(sample_id)]['painting']
        group.append(groups.setdefault(name, len(groups)))
    group = np.asarray(group)
    counts = np.bincount(group)
    order = np.random.default_rng(42).permutation(len(counts))
    shares = counts[order] / counts.sum()
    start = np.cumsum(shares) - shares
    parts = np.empty(len(counts), dtype=np.int64)
    parts[order] = np.searchsorted(np.cumsum([.7, .1, .2])[:2], start, side='right')
    wanted = np.sort(ids[parts[group] == 2])
    names = sorted({annotations[int(i)]['painting'] for i in wanted})
    assert (len(names), len(wanted)) == (12281, 61744) and names_hash(names) == HELD_SHA
    lookup = {name: row for row, name in enumerate(names)}
    tp = np.asarray([lookup[annotations[int(i)]['painting']] for i in wanted], dtype=np.int64)
    codes = {v: k for k, v in enumerate(sorted({r['emotion'] for r in annotations}))}
    text_emotion = np.asarray([codes[annotations[int(i)]['emotion']] for i in wanted])
    votes = np.zeros((len(names), len(codes)), dtype=np.int64)
    np.add.at(votes, (tp, text_emotion), 1)
    styles = {v: k for k, v in enumerate(sorted({r['art_style'] for r in annotations}))}
    painting_style = {}
    for r in annotations:
        assert painting_style.setdefault(r['painting'], r['art_style']) == r['art_style']
    style = np.asarray([styles[painting_style[n]] for n in names])
    genres = {}
    for p in (rd.GENRE_DIR / 'genre_train.csv', rd.GENRE_DIR / 'genre_val.csv'):
        with p.open() as stream:
            for relative, code in csv.reader(stream):
                name, code = Path(relative).stem, int(code)
                if genres.setdefault(name, code) != code:
                    genres[name] = -1
    genre = np.asarray([genres.get(n, -1) for n in names])
    assert (genre >= 0).sum() == 9949
    return dict(names=names, ids=wanted, tp=tp, labels=dict(emotion=dict(t=text_emotion, i=votes.argmax(1)),
                style=dict(t=style[tp], i=style), genre=dict(t=genre[tp], i=genre)))


def export(path, fr):
    meta = rd.read_json(path.with_suffix('.json'))
    hashes = []
    for doc in (meta, meta.get('config', {}), meta.get('config', {}).get('graph_identity', {})):
        for key in ('list_sha256', 'held_sha256', 'painting_names_sha256', 'names_sha256'):
            value = doc.get(key)
            if isinstance(value, dict):
                value = value.get('dev') if len(fr['names']) == 6152 else value.get('held')
            if isinstance(value, str):
                hashes.append(value)
    assert hashes and all(h == names_hash(fr['names']) for h in hashes), (path, 'painting-name hash mismatch', hashes)
    with np.load(path, allow_pickle=False) as archive:
        raw = archive['img']
    assert raw.dtype == np.float16 and raw.shape == (len(fr['names']), 2048) and np.isfinite(raw).all()
    if meta.get('arrays', {}).get('img', {}).get('sha256'):
        assert meta['arrays']['img']['sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
    return raw


def candidate_images(split, aspect, seed, fr):
    if split == 'held' and not ALLOW_CANDIDATE_HELD:
        raise RuntimeError('K* held access is controller-only after committed picks')
    p = rd2.CANDIDATE_EXPORTS.parent / split / ARM / f'{aspect}__s{seed}.npz'
    return export(p, fr), p


def k0_images(split, aspect, seed, fr):
    if split == 'val':
        p = rd2.K0_EXPORTS / f'{aspect}__{rd2.GRAPHS[aspect]}__s{seed}.npz'
    else:
        p = R14 / f'cstar/siglip1/held/k11=40/{aspect}__s{seed}.npz'
    return export(p, fr), p


def stored(folder, side, fr):
    """Join explicit sample IDs/painting names, never an assumed dataset image ID."""
    folder = Path(folder)
    requested = fr['ids'].tolist() if side == 't' else fr['names']
    if (folder / 'index.json').is_file():
        index = rd.read_json(folder / 'index.json')
        records = index['texts' if side == 't' else 'images']
        names = [r['sample_id' if side == 't' else 'painting'] for r in records]
        positions = rd.join_rows(names, requested)
        if side == 't':
            assert all(records[p]['painting'] == fr['names'][pid] for p, pid in zip(positions, fr['tp']))
    else:
        with np.load(folder / 'ids.npz', allow_pickle=False) as archive:
            field = 'sample_ids' if side == 't' else 'painting_names'
            positions = rd.join_rows(archive[field].tolist(), requested)
            if side == 't':
                painting_field = next((k for k in ('caption_painting_ids', 'painting_ids', 'painting_names')
                                       if k in archive and archive[k].shape == archive[field].shape), None)
                if painting_field:
                    assert np.array_equal(archive[painting_field][positions], np.asarray(fr['names'])[fr['tp']])
    raw = np.load(folder / ('txt.npy' if side == 't' else 'img.npy'), mmap_mode='r', allow_pickle=False)
    return C.unit(raw[positions])


def teacher_image(split, aspect, fr):
    suffix = 's2img' if split == 'val' else 'held'
    return stored(rd.STORE_ROOT / f'qwen3vl_emb_2b_aspect_prompts/{INST[aspect]}_{suffix}', 'i', fr)


def plain_image(fr):
    return stored(rd.STORES['siglip1'], 'i', fr)


def load_candidate_vectors(fr, seed, aspect, digest, raw):
    state = rd2.load_state(ARM, aspect, seed, digest)
    return dict(t=C.tower(raw['t'], state, 'cap'), i=C.tower(raw['i'], state, 'img'))


def untold(heads):
    return (np.concatenate([C.unit(heads[a]) for a in ASPECTS], axis=1) / np.float32(np.sqrt(3))).astype(np.float32)


def plain_text_file(row, aspect, seed, direction, readout, split='held'):
    head = 'all' if row in ('plain', 'siglip_so400m') else aspect
    return R23 / f'scores/art_{split}/caption/{row}/{head}__s{seed}__{direction}__{readout}.npz'


def read_ap(path, aspect, expected_pid):
    with np.load(path, allow_pickle=False) as archive:
        ap = archive['ap__' + aspect].astype(np.float64)
        assert np.array_equal(archive['q_pid'], expected_pid), (path, 'query painting IDs differ')
    return ap


def comparator_text(aspect, direction, comparator, fr):
    stage = rd.read_json(R23_RECORDS / 'stage1_art.json')
    row = 'siglip_so400m' if comparator == 'p1' else 'teacher'
    pick = stage['picks']['caption'][row][aspect][direction]
    pid = fr['tp'] if direction.startswith('t') else np.arange(len(fr['names']))
    ap = read_ap(plain_text_file(row, aspect, 'x', direction, pick), aspect, pid)
    return ap, pick


def raw_range(ap_values):
    return float(np.ptp(ap_values))


def validation(args):
    fr = frame('val')
    raw, digest = rd2.inputs(ARM, fr)
    image_values = {a: [] for a in ASPECTS}
    ami_values = {a: [] for a in ASPECTS}
    tables = {kind: {a: [] for a in ASPECTS} for kind in ('told', 'untold')}
    text = {f'{a}/{d}': {r: [] for r in C.TEXT_MENU} for a in ASPECTS for d in rd.DIRECTIONS}
    checks = {}
    for seed in args.seeds:
        heads, identities = {}, {}
        for aspect in ASPECTS:
            weight = rd.WEIGHTS / ARM / f'{aspect}__s{seed}.pt'
            projected = None
            if args.forward_only_validation:
                def diagnostic_project(seed=seed, aspect=aspect):
                    value = load_candidate_vectors(fr, seed, aspect, digest, raw)
                    return dict(text_output=value['t'], image_output=value['i'])
                projected = cached(f'validation_diagnostic_forward_{aspect}_s{seed}',
                    dict(weights=rd.sha256(weight), store=digest, names=names_hash(fr['names'])), diagnostic_project)
                image, path = projected['image_output'].astype(np.float16), None
            else:
                image, path = candidate_images('val', aspect, seed, fr)
            heads[aspect] = image
            identity = dict(export=rd.sha256(path) if path else 'diagnostic_forward_only', weights=rd.sha256(weight), store=digest,
                            query_limit=args.query_limit, batch=args.batch_size)
            identities[aspect] = identity
            def compute(image=image, seed=seed, aspect=aspect):
                vectors = dict(t=projected['text_output'], i=projected['image_output']) if projected is not None else load_candidate_vectors(fr, seed, aspect, digest, raw)
                result = dict(image_output=vectors['i'], text_output=vectors['t'],
                              parts=C.partitions(image, rd2.CLASS_COUNTS[aspect]))
                return result
            vectors = cached(f'val_vectors_{aspect}_s{seed}', identity, compute)
            if not args.forward_only_validation:
                checks[f'{aspect}/s{seed}'] = rd2.forward_comparison(vectors['image_output'], image)
            y = fr['labels'][aspect]['i']
            ami_values[aspect].append(C.part_value(vectors['parts'], y))
            ap = cached(f'val_r0_{aspect}_s{seed}', identity,
                        lambda image=image, y=y: dict(ap=C.image_ap(image, image, y, 'none+k0', args.query_limit, args.batch_size)[0]))['ap']
            image_values[aspect].append(float(np.nanmean(ap)))
            for direction in rd.DIRECTIONS:
                def score(direction=direction, aspect=aspect, vectors=vectors):
                    values, _ = C.text_ap(vectors['text_output'], vectors['image_output'], fr['labels'][aspect],
                                          direction, limit=args.query_limit, batch_size=args.batch_size)
                    return values
                values = cached(f'val_text_{aspect}_{direction}_s{seed}', identity, score)
                for readout, ap in values.items():
                    text[f'{aspect}/{direction}'][readout].append(float(np.nanmean(ap.astype(np.float64))))
            readouts = ['none+k0', 'center+k0', 'center+k10'] if args.quick else C.menu(2048)
            table = cached(f'val_pick_told_{aspect}_s{seed}', {**identity, 'menu': readouts},
                           lambda image=image, y=y: {k: np.asarray(v) for k, v in C.crossfit(image, y, readouts, args.query_limit, args.batch_size).items()})
            tables['told'][aspect].append({k: float(v) for k, v in table.items()})
            print(f'validation {aspect}/s{seed}: AMI, R0, text/readouts, told crossfit complete', flush=True)
        joined = untold(heads)
        readouts = ['none+k0', 'center+k0', 'center+k10'] if args.quick else C.menu(joined.shape[1])
        def joined_table():
            result = C.crossfit_multi(joined, {a: fr['labels'][a]['i'] for a in ASPECTS}, readouts,
                                      args.query_limit, args.batch_size)
            return {a + '/' + k: np.asarray(v) for a, row in result.items() for k, v in row.items()}
        table = cached(f'val_pick_untold_s{seed}', {'heads': identities, 'menu': readouts}, joined_table)
        for aspect in ASPECTS:
            tables['untold'][aspect].append({k.split('/', 1)[1]: float(v) for k, v in table.items() if k.startswith(aspect + '/')})
        print(f'validation s{seed}: untold crossfit complete', flush=True)
    complete = args.seeds == list(SEEDS) and not args.query_limit and not args.quick and not args.forward_only_validation
    ranges = {a: raw_range(image_values[a]) for a in ASPECTS}
    picks = dict(text={}, image_map={}, ami_range={a: raw_range(ami_values[a]) for a in ASPECTS})
    for cell, rows in text.items():
        means = {r: float(np.mean(values)) for r, values in rows.items()}
        pick = max(C.TEXT_MENU, key=means.get) if complete else None
        picks['text'][cell] = dict(pick=pick, means=means, range=raw_range(rows[pick]) if pick else None,
                                   per_seed=rows)
    image_tables, image_picks = {}, {}
    for kind in ('told', 'untold'):
        names = list(tables[kind]['emotion'][0])
        rows = {r: {a: float(np.mean([t[r] for t in tables[kind][a]])) for a in ASPECTS} for r in names}
        if complete:
            chosen, scores, kept = C.select_image(rows, ranges)
        else:
            chosen, scores, kept = None, {}, {}
        image_picks[kind] = chosen
        image_tables[kind] = dict(table=rows, scores=scores, kept=kept, pick=chosen)
    for a in ASPECTS:
        picks['image_map'][a] = dict(pick=image_picks['told'], p1_pick=image_picks['untold'], range=ranges[a])
    result = dict(complete=complete, kstar=ARM, p1_form='plain', picks=picks, image_tables=image_tables,
                  ami_per_seed=ami_values, image_r0_per_seed=image_values, forward_check=rd2.forward_summary(checks) if checks else {'pending': 'stored validation exports unavailable; diagnostic uses float16-quantised own forwards only'},
                  source_sha256=source_hash(), elapsed_seconds=time.monotonic()-START,
                  device=C.DEVICE, query_limit=args.query_limit, seeds=args.seeds, notes=NOTES)
    if not complete and args.output.resolve() == HERE / 'r29rd_c1_picks.json':
        raise ValueError('Subset validation must use a separate --output filename')
    rd.write_json(args.output, result)
    print(f'validation complete={complete}; seconds={result["elapsed_seconds"]:.3f}; {args.output}', flush=True)


def point_target(family, aspect, comp):
    if family == 'ami' or comp == 'p1':
        rows = rd.read_json(R17_RECORDS / 'outcomes.json')['cells']
        sc = 'SC1' if family == 'ami' and comp == 'p1' else 'SC3' if family == 'ami' else 'SC7'
        metric = 'ami' if family == 'ami' else 'map'
        return next(c for c in rows if c['sc'] == sc and c['set'] == 'artelingo' and c['encoder'] == 'siglip1'
                    and c['aspect'] == aspect and c['metric'] == metric and c.get('kind') != 'fixed20')
    rows = rd.read_json(R14_RECORDS / 'verdicts.json')['cells']
    return next(c for c in rows if c['encoder'] == 'siglip1' and c['sc'] == 'SC6' and c['task'] == 'ret' and c['aspect'] == aspect)


def target_differences(stats, target, intervals=True):
    expected = dict(value=target.get('cand', target.get('candidate_value')), comparator_value=target.get('comp', target.get('comparator_value')),
                    d=target['d'], lo=target['lo'], hi=target['hi'])
    return {k: abs(stats[k] - v) for k, v in expected.items() if k in stats and (intervals or k not in ('lo', 'hi'))}


def replay_ami(args):
    fr = frame('held')
    rows = []
    for aspect in args.aspects:
        y = fr['labels'][aspect]['i']
        parts = []
        for seed in SEEDS:
            image, path = k0_images('held', aspect, seed, fr)
            result = cached(f'k0_ami_{aspect}_s{seed}', {'export': rd.sha256(path)},
                            lambda image=image: dict(parts=C.partitions(image, rd2.CLASS_COUNTS[aspect])))
            parts.append(result['parts'])
        candidate = np.stack(parts)
        for comp in ('p1', 'q1'):
            features = plain_image(fr) if comp == 'p1' else teacher_image('held', aspect, fr)
            identity = hashlib.sha256(features.tobytes()).hexdigest()
            other = cached(f'k0_ami_{aspect}_{comp}', {'features': identity},
                           lambda: dict(parts=C.partitions(features, rd2.CLASS_COUNTS[aspect])))['parts']
            if args.draws:
                stats = C.ami_interval(candidate, other, y, args.draws, args.workers)
            else:
                cv, bv = C.part_value(candidate, y), C.part_value(other, y)
                stats = dict(value=cv, comparator_value=bv, d=cv-bv)
            differences = target_differences(stats, point_target('ami', aspect, comp), args.draws == 2000)
            rows.append(dict(id=f'ami/{aspect}/{comp}', stats=stats, abs_diff=differences,
                             max_abs_diff=max(differences.values()), full_interval=args.draws == 2000))
            print(f'K0 AMI {aspect}/{comp}: max diff={rows[-1]["max_abs_diff"]:.3g}', flush=True)
    return rows


def saved_image_ap(aspect, fr):
    path = R11 / f'stage5/perquery/instructed__{aspect}__h{aspect}__s-.npz'
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive['rows'], np.flatnonzero(fr['labels'][aspect]['i'] >= 0))
        return archive['ap'].astype(np.float64)


def replay_image(args):
    val, held = frame('val'), frame('held')
    stage = rd.read_json(R14_RECORDS / 'stage1_siglip1.json')['artret_picks']
    untold_pick = rd.read_json(R17_RECORDS / 'stage1_picks.json')['untold/artelingo/siglip1']['pick']
    rows = []
    heads_v = {s: {a: k0_images('val', a, s, val)[0] for a in ASPECTS} for s in SEEDS}
    heads_h = {s: {a: k0_images('held', a, s, held)[0] for a in ASPECTS} for s in SEEDS}
    for aspect in args.aspects:
        y = held['labels'][aspect]['i']
        for comp in ('p1', 'q1'):
            pick = untold_pick if comp == 'p1' else stage['cstar']['pick']
            candidates = []
            for seed in SEEDS:
                v = untold(heads_v[seed]) if comp == 'p1' else heads_v[seed][aspect]
                h = untold(heads_h[seed]) if comp == 'p1' else heads_h[seed][aspect]
                identity = dict(val=hashlib.sha256(v.tobytes()).hexdigest(), held=hashlib.sha256(h.tobytes()).hexdigest(),
                                pick=pick, limit=args.query_limit, batch=args.batch_size)
                candidates.append(cached(f'k0_image_ap_{aspect}_{comp}_s{seed}', identity,
                    lambda v=v, h=h: dict(ap=C.image_ap(v, h, y, pick, args.query_limit, args.batch_size)[0]))['ap'])
            if comp == 'p1':
                v, h = plain_image(val), plain_image(held)
                other = C.image_ap(v, h, y, stage['frozen']['pick'], args.query_limit, args.batch_size)[0]
            else:
                other = saved_image_ap(aspect, held)
                if args.query_limit:
                    other = other[rd.selected_queries(len(other), args.query_limit)]
            if args.query_limit:
                cv, bv = float(np.mean(candidates)), float(other.mean())
                stats = dict(value=cv, comparator_value=bv, d=cv-bv)
                differences = {}
            else:
                stats = C.retrieval_interval(np.stack(candidates), other, draws=args.ret_draws)
                differences = target_differences(stats, point_target('image_map', aspect, comp), args.ret_draws == 10000)
            rows.append(dict(id=f'image_map/{aspect}/{comp}', stats=stats, abs_diff=differences,
                             max_abs_diff=max(differences.values()) if differences else None,
                             full_interval=not args.query_limit and args.ret_draws == 10000))
            print(f'K0 image mAP {aspect}/{comp}: max diff={rows[-1]["max_abs_diff"]}', flush=True)
    return rows


def k0_text_vectors(fr, aspect, seed):
    raw = dict(t=stored(rd.STORE_ROOT / 'qwen3vl_emb_2b', 't', fr), i=stored(rd.STORE_ROOT / 'qwen3vl_emb_2b', 'i', fr))
    state = torch.load(K0_TEXT_WEIGHTS / f'{aspect}__s{seed}.pt', map_location='cpu', weights_only=True)
    # The permitted r23 independent path normalises both already-unit arrays once more.
    return {side: C.tower(C.unit(raw[side]), state, tower_side, batch_size=4096) for side, tower_side in [('t', 'cap'), ('i', 'img')]}


def replay_text(args):
    val, held = frame('val'), frame('held')
    stage = rd.read_json(R23_RECORDS / 'stage1_art.json')
    targets = rd.read_json(R23 / 'stage2/cells.json')
    full_scoring = not args.query_limit and args.seeds == list(SEEDS)
    rows = []
    for aspect in args.aspects:
        projected = {}
        for seed in args.seeds:
            def project(fr, split, seed=seed):
                return cached(f'k0_text_vectors_{aspect}_{split}_s{seed}',
                    dict(weights=rd.sha256(K0_TEXT_WEIGHTS / f'{aspect}__s{seed}.pt'),
                         names=names_hash(fr['names']), forward_batch=4096),
                    lambda: k0_text_vectors(fr, aspect, seed))
            projected[seed] = project(val, 'val'), project(held, 'held')
        for direction in rd.DIRECTIONS:
            pick = stage['picks']['caption']['cstar'][aspect][direction]
            pid = held['tp'] if direction.startswith('t') else np.arange(len(held['names']))
            frozen = np.stack([read_ap(plain_text_file('cstar', aspect, s, direction, pick), aspect, pid) for s in SEEDS])
            computed = {}
            ap_differences, mean_differences = [], []
            for seed, (v, h) in projected.items():
                fitted = {side: C.means(x) for side, x in v.items()}
                def score(v=v, h=h, fitted=fitted):
                    ap, queries = C.text_ap(h['t'], h['i'], held['labels'][aspect], direction, [pick], fitted,
                                             args.query_limit, args.batch_size)
                    return dict(ap=ap[pick], queries=queries)
                scored = cached(f'k0_text_ap_{aspect}_{direction}_s{seed}',
                    dict(weight=rd.sha256(K0_TEXT_WEIGHTS / f'{aspect}__s{seed}.pt'),
                         pick=pick, limit=args.query_limit, batch=args.batch_size, forward_batch=4096), score)
                actual, queries = scored['ap'].astype(np.float64), scored['queries']
                expected = frozen[seed, queries]
                assert np.array_equal(np.isfinite(actual), np.isfinite(expected))
                ap_differences.append(float(np.nanmax(np.abs(actual-expected))))
                mean_differences.append(abs(float(np.nanmean(actual))-float(np.nanmean(expected))))
                computed[seed] = actual
            candidate = np.stack([computed[s] for s in SEEDS]) if full_scoring else frozen
            for comp in ('plain', 'teacher'):
                rp = stage['picks']['caption'][comp][aspect][direction]
                other = read_ap(plain_text_file(comp, aspect, 'x', direction, rp), aspect, pid)
                stats = C.retrieval_interval(candidate, other, pid, args.ret_draws)
                target = next(c for c in targets if c.get('sc') == 'SC1' and c.get('cell') == 'label' and c['aspect'] == aspect
                              and c['direction'] == direction and c['comparator'] == comp)
                differences = dict(d=abs(stats['d']-target['point']))
                if args.ret_draws == 10000:
                    differences.update(lo=abs(stats['lo']-target['lower']), hi=abs(stats['hi']-target['upper']))
                rows.append(dict(id=f'text/{aspect}/{direction}/{comp}', stats=stats, abs_diff=differences,
                    max_abs_diff=max(differences.values()), scoring_max_abs_diff=max(ap_differences),
                    scoring_mean_abs_diff=max(mean_differences), scoring_queries=len(queries), scoring_seeds=args.seeds,
                    full_scoring=full_scoring, interval_inputs='own_forward_AP' if full_scoring else 'frozen_per_unit_AP',
                    full_interval=args.ret_draws == 10000))
            other, _ = comparator_text(aspect, direction, 'p1', held)
            side_rows = rd.read_json(R23 / 'stage3/part5_side_rows.json')['sections']['label']
            side = next(x for x in side_rows if x['row'] == 'siglip_so400m' and x['aspect'] == aspect
                        and x['direction'] == direction and x['set'] == 'art_held')
            rows[-1]['p1_side_point_abs_diff'] = abs(float(np.nanmean(other))-side['map'])
            print(f'K0 text {aspect}/{direction}: AP max diff={max(ap_differences):.3g}; '
                  f'scored mean diff={max(mean_differences):.3g}; interval diff={rows[-1]["max_abs_diff"]:.3g}; '
                  f'full_scoring={full_scoring}', flush=True)
    return rows


def replays(args):
    reports = {}
    for family in (('ami', 'image_map', 'text') if args.family == 'all' else (args.family,)):
        rows = {'ami': replay_ami, 'image_map': replay_image, 'text': replay_text}[family](args)
        full_intervals = all(r['full_interval'] for r in rows)
        full_scoring = all(r.get('full_scoring', r['max_abs_diff'] is not None) for r in rows)
        differences = [r['max_abs_diff'] for r in rows if r['max_abs_diff'] is not None]
        max_diff = max(differences) if differences else None
        scoring_diff = max((r.get('scoring_max_abs_diff', 0.) for r in rows), default=0.)
        reports[family] = dict(rows=rows, max_abs_diff=max_diff, scoring_max_abs_diff=scoring_diff,
                               full_intervals=full_intervals, full_scoring=full_scoring, aspects=args.aspects,
                               interval_gate_1e9=full_intervals and max_diff is not None and max_diff <= 1e-9,
                               scoring_gate_1e9=full_scoring and max_diff is not None and max_diff <= 1e-9)
    result = dict(replays=reports, source_sha256=source_hash(), device=C.DEVICE,
                  elapsed_seconds=time.monotonic()-START, query_limit=args.query_limit, notes=NOTES)
    rd.write_json(args.output, result)
    print(f'replays seconds={result["elapsed_seconds"]:.3f}; {args.output}', flush=True)


def replay_gate(reports, family, aspect):
    matches = []
    for report in reports:
        item = report.get('replays', {}).get(family)
        if item and aspect in item['aspects']:
            matches.append(item)
    return any(r['interval_gate_1e9'] and r['scoring_gate_1e9'] for r in matches)


def held_cells(args):
    global ALLOW_CANDIDATE_HELD
    if not args.picks_committed:
        raise ValueError('Held scoring requires controller assertion --picks-committed; no K* held input has been read')
    picks_doc = rd.read_json(args.picks)
    assert picks_doc['complete'] and picks_doc['source_sha256'] == source_hash(), 'Incomplete/stale validation picks'
    ALLOW_CANDIDATE_HELD = True
    val, held = frame('val'), frame('held')
    picks = picks_doc['picks']
    reports = [rd.read_json(p) for p in args.replay_reports]
    assert all(r['source_sha256'] == source_hash() for r in reports), 'Stale replay source'
    val_raw, digest = rd2.inputs(ARM, val)
    held_raw, held_digest = rd2.inputs(ARM, held)
    assert digest == held_digest
    candidate = {a: dict(ami=[], p1_map=[], q1_map=[], text={d: [] for d in rd.DIRECTIONS}) for a in ASPECTS}
    checks = {}
    r14_stage = rd.read_json(R14_RECORDS / 'stage1_siglip1.json')['artret_picks']
    for seed in SEEDS:
        heads_v, heads_h, identities = {}, {}, {}
        for aspect in ASPECTS:
            v, vp = candidate_images('val', aspect, seed, val)
            h, hp = candidate_images('held', aspect, seed, held)
            heads_v[aspect], heads_h[aspect] = v, h
            identity = dict(val=rd.sha256(vp), held=rd.sha256(hp), weight=rd.sha256(rd.WEIGHTS / ARM / f'{aspect}__s{seed}.pt'),
                            picks=rd.sha256(args.picks))
            identities[aspect] = identity
            y = held['labels'][aspect]['i']
            candidate[aspect]['ami'].append(cached(f'candidate_held_ami_{aspect}_s{seed}', identity,
                           lambda h=h: dict(parts=C.partitions(h, rd2.CLASS_COUNTS[aspect])))['parts'])
            pick = picks['image_map'][aspect]['pick']
            candidate[aspect]['q1_map'].append(cached(f'candidate_held_told_{aspect}_s{seed}', {**identity, 'pick': pick},
                           lambda v=v, h=h, y=y, pick=pick: dict(ap=C.image_ap(v, h, y, pick, batch_size=args.batch_size)[0]))['ap'])
            def project(aspect=aspect, seed=seed):
                vvec = load_candidate_vectors(val, seed, aspect, digest, val_raw)
                hvec = load_candidate_vectors(held, seed, aspect, digest, held_raw)
                return dict(val_t=vvec['t'], val_i=vvec['i'], held_t=hvec['t'], held_i=hvec['i'])
            vectors = cached(f'candidate_held_forward_{aspect}_s{seed}', identity, project)
            checks[f'held/{aspect}/s{seed}'] = rd2.forward_comparison(vectors['held_i'], h)
            checks[f'val/{aspect}/s{seed}'] = rd2.forward_comparison(vectors['val_i'], v)
            fitted = dict(t=C.means(vectors['val_t']), i=C.means(vectors['val_i']))
            for direction in rd.DIRECTIONS:
                r = picks['text'][f'{aspect}/{direction}']['pick']
                def score(direction=direction, r=r, vectors=vectors, aspect=aspect):
                    ap, _ = C.text_ap(vectors['held_t'], vectors['held_i'], held['labels'][aspect], direction, [r], fitted,
                                      batch_size=args.batch_size)
                    return dict(ap=ap[r])
                candidate[aspect]['text'][direction].append(cached(f'candidate_held_text_{aspect}_{direction}_s{seed}',
                                                                  {**identity, 'readout': r}, score)['ap'])
            print(f'held {aspect}/s{seed}: AMI, told image, text and forward complete', flush=True)
        vjoin, hjoin = untold(heads_v), untold(heads_h)
        for aspect in ASPECTS:
            pick = picks['image_map'][aspect]['p1_pick']
            y = held['labels'][aspect]['i']
            candidate[aspect]['p1_map'].append(cached(f'candidate_held_untold_{aspect}_s{seed}', {'heads': identities, 'pick': pick},
                lambda y=y, pick=pick: dict(ap=C.image_ap(vjoin, hjoin, y, pick, batch_size=args.batch_size)[0]))['ap'])
    cells = []
    for aspect in ASPECTS:
        y = held['labels'][aspect]['i']
        parts = np.stack(candidate[aspect]['ami'])
        for comp in ('p1', 'q1'):
            features = plain_image(held) if comp == 'p1' else teacher_image('held', aspect, held)
            other = C.partitions(features, rd2.CLASS_COUNTS[aspect])
            stats = C.ami_interval(parts, other, y, 2000, args.workers)
            r = picks['ami_range'][aspect]
            gate = replay_gate(reports, 'ami', aspect)
            cells.append(dict(id=f'ami/{aspect}/{comp}', **stats, r=r, n_rows=len(y), n_labelled=int((y>=0).sum()),
                              outcome=C.outcome(stats['lo'], r) if gate else 'INCONCLUSIVE'))
            if comp == 'p1':
                other_ap = C.image_ap(plain_image(val), plain_image(held), y, r14_stage['frozen']['pick'], batch_size=args.batch_size)[0]
            else:
                other_ap = saved_image_ap(aspect, held)
            stats = C.retrieval_interval(np.stack(candidate[aspect][comp+'_map']), other_ap)
            r = picks['image_map'][aspect]['range']
            gate = replay_gate(reports, 'image_map', aspect)
            cells.append(dict(id=f'image_map/{aspect}/{comp}', **stats, r=r,
                              candidate_embedding='untold' if comp == 'p1' else 'told',
                              outcome=C.outcome(stats['lo'], r) if gate else 'INCONCLUSIVE'))
            for direction in rd.DIRECTIONS:
                other_ap, _ = comparator_text(aspect, direction, comp, held)
                pid = held['tp'] if direction.startswith('t') else np.arange(len(held['names']))
                stats = C.retrieval_interval(np.stack(candidate[aspect]['text'][direction]), other_ap, pid)
                r = picks['text'][f'{aspect}/{direction}']['range']
                gate = replay_gate(reports, 'text', aspect)
                cells.append(dict(id=f'text/{aspect}/{direction}/{comp}', **stats, r=r,
                                  outcome=C.outcome(stats['lo'], r) if gate else 'INCONCLUSIVE'))
    assert len(cells) == 30 and len({c['id'] for c in cells}) == 30
    result = dict(picks=picks, cells=cells, sc3=C.ladder(cells), forward_check=rd2.forward_summary(checks),
                  replays=reports, notes=NOTES, picks_sha256=rd.sha256(args.picks),
                  elapsed_seconds=time.monotonic()-START)
    rd.write_json(args.output, result)
    print(f'held SC3={result["sc3"]}; seconds={result["elapsed_seconds"]:.3f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('validate', 'replay', 'held'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--aspects', nargs='+', choices=ASPECTS, default=list(ASPECTS))
    parser.add_argument('--forward-only-validation', action='store_true', help='subset diagnostic only when stored validation exports are absent')
    parser.add_argument('--quick', action='store_true', help='validation smoke: reduced image menu, no final picks')
    parser.add_argument('--family', choices=('all', 'ami', 'image_map', 'text'), default='all')
    parser.add_argument('--draws', type=int, default=2000, help='AMI bootstrap draws, 0 for point replay only')
    parser.add_argument('--ret-draws', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=8, choices=range(1,9))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--picks', type=Path, default=HERE/'r29rd_c1_picks.json')
    parser.add_argument('--picks-committed', action='store_true')
    parser.add_argument('--replay-reports', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    assert args.query_limit >= 0 and args.draws >= 0 and args.ret_draws > 0 and args.batch_size > 0
    args.output = args.output or HERE / {'validate':'r29rd_c1_picks.json', 'replay':f'r29rd_c1_replay_{args.family}.json',
                                       'held':'c1_rederived.json'}[args.mode]
    if not args.output.resolve().is_relative_to(HERE):
        parser.error('All outputs must stay inside rederive/')
    if args.mode != 'held' and args.output.resolve() == HERE/'c1_rederived.json':
        parser.error('Only the complete held command may write c1_rederived.json')
    if args.forward_only_validation and (args.mode != 'validate' or not args.query_limit or not args.quick):
        parser.error('--forward-only-validation is limited to subset --quick validation diagnostics')
    if args.mode == 'held' and (args.query_limit or args.quick or args.seeds != list(SEEDS)):
        parser.error('Held output requires all five seeds, all queries and the full read-out menu')
    if args.mode == 'held' and not args.picks_committed:
        parser.error('Controller must commit picks before --picks-committed; no K* held inputs read')
    C.set_device(args.device)
    {'validate':validation, 'replay':replays, 'held':held_cells}[args.mode](args)


if __name__ == '__main__':
    main()
