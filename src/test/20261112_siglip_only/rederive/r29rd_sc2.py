#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent r29 SC2 pass rule; imports only our SC1 module and public libraries.

Modes: check-k0 (image regression), check-text (K* seed 0/subset), inspect, run.
The full run requires all 15 candidate weight/export pairs and writes the final
SC2 JSON only after computing every cell. No round scoring code is imported.
"""
import os
import sys
import time

START = time.monotonic()
sys.dont_write_bytecode = True
for _key, _value in dict(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                         PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='').items():
    os.environ[_key] = _value

import argparse
from pathlib import Path

# Import SC1 before other numerical libraries: it enforces the CPU/thread policy.
import r29rd_sc1 as rd
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits
import torch

HERE = Path(__file__).resolve().parent
BASELINE_RECORDS = Path('/project/CoSiR/src/test/20261112_siglip_main/records')
K0_EXPORTS = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/train/k11=40/R6-5')
CANDIDATE_EXPORTS = Path('/data/SSD2/cosir_r29/20261112_siglip_only/cstar/siglip1/val')
GRAPHS = dict(emotion='Gres', style='Fres05', genre='R0o')
CLASS_COUNTS = dict(emotion=9, style=27, genre=10)
SEEDS = tuple(range(5))
AMI_STATES = (0, 1, 2)
AMI_ROWS = ('Cluster all 6,152 validation paintings, including unlabelled genre rows; '
            'mask negative labels only when scoring AMI (genre: 4,960 scored rows). '
            'r17, 2026-11-03-new-scores-design.md §3.3 explicitly prescribes this population. '
            'The r28 valpick docstring specifies float16-to-float32 rows without further normalization.')
NOTES = [
    'For AMI the explicit SC2/r28 input convention (stored float16 to float32, no L2) controls the generic r17 unit-row wording; r17 controls clustering all rows before masking labels.',
    'MiniBatchKMeans uses exactly the specified n_clusters, n_init=3, batch_size=4096 and random_state=0/1/2, with other parameters at installed sklearn defaults; eight numerical-library threads. AMI uses sklearn arithmetic normalization.',
    'Image mAP casts stored float16 to float32 then L2-normalises once, removes unlabelled paintings from queries and gallery, excludes the query itself, and breaks exact cosine ties by sorted painting order. AP and means are float64.',
    'Text cells reuse our SC1 individual-caption input, forward, labels and AP logic; no centering/expansion, no float16 output quantisation for text retrieval.',
    'Painting plurality ties use alphabetical emotion order and all nine emotion classes including something else; genre CSV conflicts/missing entries are unlabelled.',
    'Image cells always use stored validation img rows. The forward diagnostic quantises our independently computed image outputs to float16 and compares elements in float32, reporting exact-equality share and largest difference; no unstated equality-share acceptance threshold is invented.',
    'K0 means/ranges/best are frozen r28 record values: image m is the recorded score, text m is the mean of nine recorded r0 means; ranges/best use per-seed group means. No rounded spec threshold is used in a decision.',
    'CPU forward batches are 1024 and ranking batches 128. Subset checks sample query positions with default_rng(0), keeping full galleries; the full run samples nothing.',
    'The selected prefix store metadata spells represent the caption: in lowercase while the prose capitalizes it; use the specified stored rows unchanged, as in SC1.',
]


def mean(values):
    return float(np.mean(values, dtype=np.float64))


def selection():
    doc = rd.read_json(HERE / 'sc1_rederived.json')
    assert set(doc['arms']) == set(rd.ARMS), 'SC1 must contain all four full arms'
    arm, form, _ = rd.decision(doc['arms'])
    assert doc['kstar'] == arm and doc['kstar_form'] == form
    expected = {'emotion/t2i': 30872, 'emotion/i2t': 6152, 'emotion/t2t': 30872,
                'style/t2i': 30872, 'style/i2t': 6152, 'style/t2t': 30872,
                'genre/t2i': 24886, 'genre/i2t': 4960, 'genre/t2t': 24886}
    assert doc['counts'][arm] == expected, 'SC1 is incomplete or a query subset'
    return arm, doc


def frozen_baseline():
    image_doc = rd.read_json(BASELINE_RECORDS / 'valpick.json')
    assert image_doc['complete'] is True and image_doc['seeds'] == list(SEEDS)
    image = image_doc['encoders']['siglip1']
    assert image['names_sha256'] == rd.VAL_SHA256 and image['n_val'] == 6152
    image_cells = {f'{metric}/{aspect}': image['cells'][f'{metric}/{aspect}']['per_seed']
                   for aspect in rd.ASPECTS for metric in ('ami', 'map')}
    assert all(len(values) == 5 and np.isfinite(values).all() for values in image_cells.values())
    image_per_seed = np.mean(list(image_cells.values()), axis=0, dtype=np.float64)
    image_m = float(image['score'])
    assert abs(mean(image_per_seed) - image_m) < 1e-12
    picks = rd.read_json(BASELINE_RECORDS / 'text.json')['ours_siglip1']['picks']
    text_cells = {f'{aspect}/{direction}': picks[aspect][direction]['per_seed']['r0']
                  for aspect in rd.ASPECTS for direction in rd.DIRECTIONS}
    text_means = [picks[a][d]['means']['r0'] for a in rd.ASPECTS for d in rd.DIRECTIONS]
    assert all(len(values) == 5 and np.isfinite(values).all() for values in text_cells.values())
    for values, recorded_mean in zip(text_cells.values(), text_means):
        assert abs(mean(values) - recorded_mean) < 1e-12
    text_per_seed = np.mean(list(text_cells.values()), axis=0, dtype=np.float64)
    text_m = mean(text_means)
    baseline = {}
    for group, m, values in [('image', image_m, image_per_seed), ('text', text_m, text_per_seed)]:
        spread = float(np.max(values) - np.min(values))
        baseline[group] = dict(m=m, r=spread, best=float(np.max(values)), lower=m - spread)
    for group, key, expected in [('image', 'm', '0.3862917081'), ('image', 'r', '0.0029788602'),
                                 ('text', 'm', '0.2593512884'), ('text', 'r', '0.0020107136')]:
        assert f'{baseline[group][key]:.10f}' == expected, (group, key, baseline[group][key])
    return baseline, dict(image=image_cells, text=text_cells), dict(image=image_per_seed.tolist(), text=text_per_seed.tolist())


def level_for(image, text, baseline):
    ib, tb = baseline['image'], baseline['text']
    if image > ib['best'] and text > tb['best']:
        return 3
    passes = image >= ib['lower'] and text >= tb['lower']
    if passes and text >= tb['m']:
        return 2
    return 1 if passes else 0


def export_rows(path, frame, aspect, seed):
    """Stored image rows in sorted validation order, certified by the sidecar hash."""
    info = rd.read_json(path.with_suffix('.json'))
    config = info.get('config', {})
    digests = []
    for doc in (info, config, config.get('graph_identity', {})):
        for key in ('names_sha256', 'painting_names_sha256', 'list_sha256'):
            if key in doc:
                value = doc[key]
                if isinstance(value, dict):
                    value = value.get('dev', value.get('val'))
                if isinstance(value, str):
                    digests.append(value)
    own_digest = rd.hashlib.sha256('\n'.join(frame['names']).encode()).hexdigest()
    assert digests and all(value == own_digest for value in digests), (path, 'painting-order hash mismatch', digests)
    for doc in (config, info):
        if 'aspect' in doc:
            assert doc['aspect'] == aspect
        if 'seed' in doc:
            assert doc['seed'] == seed
        if 'output_rows' in doc:
            assert doc['output_rows'] == len(frame['names'])
    if 'output_sha256' in info:
        assert info['output_sha256'] == rd.sha256(path), (path, 'export file hash differs')
    with np.load(path, allow_pickle=False) as archive:
        raw = archive['img']
    assert raw.shape == (6152, 2048) and raw.dtype == np.float16 and np.isfinite(raw).all()
    assert (np.linalg.norm(raw.astype(np.float32), axis=1) > 0).all()
    return raw


def ami_cell(raw, labels, n_clusters):
    x = np.asarray(raw, dtype=np.float32, order='C')
    assert x.shape[0] == len(labels) and np.isfinite(x).all()
    labelled = labels >= 0
    assert len(np.unique(labels[labelled])) == n_clusters
    values = []
    with threadpool_limits(limits=8):
        for state in AMI_STATES:
            partition = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, batch_size=4096,
                                        random_state=state).fit_predict(x)
            values.append(float(adjusted_mutual_info_score(labels[labelled], partition[labelled],
                                                          average_method='arithmetic')))
    return mean(values), values


def image_map(raw, labels, query_limit=0, batch_size=128):
    labelled = labels >= 0
    x, labels = rd.unit(raw[labelled]), labels[labelled]
    queries = rd.selected_queries(len(x), query_limit)
    gallery = torch.from_numpy(x)
    all_ap = []
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            chosen = queries[start:start + batch_size]
            scores = (torch.from_numpy(x[chosen]) @ gallery.T).numpy()
            scores[np.arange(len(chosen)), chosen] = -np.inf
            order = np.argsort(-scores, axis=1, kind='stable')
            order = order[order != chosen[:, None]].reshape(len(chosen), len(x) - 1)
            all_ap.append(rd.ranked_ap(order, labels[chosen], labels))
    ap = np.concatenate(all_ap)
    return mean(ap[np.isfinite(ap)]), int(np.isfinite(ap).sum())


def load_state(arm, aspect, seed, store_digest):
    """Generalise SC1's seed-0 state validation without importing round factories."""
    path = rd.WEIGHTS / arm / f'{aspect}__s{seed}.pt'
    info = rd.read_json(path.with_suffix('.json'))
    config, rb = info['config'], info['config']['rb']
    assert config['aspect'] == aspect and config['seed'] == seed
    assert rb['width'] == rb['cap_width'] == 1152 and rb['cap_source'] == arm
    assert rb['cap_store_sha256'] == store_digest
    assert config['names_sha256']['dev'] == rd.VAL_SHA256
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = {f'{side}.{name}' for side in ('cap', 'img')
                for name in ('trunk.0.weight', 'trunk.0.bias', 'output.weight', 'output.bias')}
    assert set(state) == expected
    for side in ('cap', 'img'):
        for name, shape in [('trunk.0.weight', (1024, 1152)), ('trunk.0.bias', (1024,)),
                            ('output.weight', (2048, 1024)), ('output.bias', (2048,))]:
            value = state[f'{side}.{name}']
            assert value.shape == shape and value.dtype == torch.float32 and torch.isfinite(value).all()
    return state


def inputs(arm, frame):
    text, digest = rd.load_store(arm, frame, images=arm == 'siglip1')
    if arm == 'siglip1':
        image = text['i']
    else:
        image = rd.load_store('siglip1', frame, images=True)[0]['i']
    return dict(t=text['t'], i=image), digest


def forward_comparison(projected, stored):
    quantised = projected.astype(np.float16)
    difference = np.abs(quantised.astype(np.float32) - stored.astype(np.float32))
    equal = quantised == stored
    return dict(elements=int(equal.size), equal_elements=int(equal.sum()),
                share_equal=float(equal.mean(dtype=np.float64)), max_abs_diff=float(difference.max()),
                rows=int(len(stored)), equal_rows=int(equal.all(axis=1).sum()),
                max_abs_diff_before_quantisation=float(np.max(np.abs(projected - stored.astype(np.float32)))))


def forward_summary(cells):
    total = sum(v['elements'] for v in cells.values())
    equal = sum(v['equal_elements'] for v in cells.values())
    return dict(cells=cells, elements=total, equal_elements=equal, share_equal=equal / total,
                max_abs_diff=max(v['max_abs_diff'] for v in cells.values()),
                max_abs_diff_before_quantisation=max(v['max_abs_diff_before_quantisation'] for v in cells.values()))


def text_cells_for(vectors, labels, query_limit, batch_size):
    result, counts = {}, {}
    for direction in rd.DIRECTIONS:
        values = rd.score_direction(vectors, labels, direction, query_limit, batch_size)
        for aspect, ap in values.items():
            finite = np.isfinite(ap)
            assert finite.any()
            key = f'{aspect}/{direction}'
            result[key] = mean(ap[finite])
            counts[key] = int(finite.sum())
    return result, counts


def k0_check(args):
    baseline, expected, _ = frozen_baseline()
    frame = rd.validation_frame()
    cells, differences, states, counts = {}, {}, {}, {}
    for aspect in args.aspects:
        for seed in args.seeds:
            started = time.monotonic()
            path = K0_EXPORTS / f'{aspect}__{GRAPHS[aspect]}__s{seed}.npz'
            raw = export_rows(path, frame, aspect, seed)
            labels = frame['labels'][aspect]['i']
            ami, state_values = ami_cell(raw, labels, CLASS_COUNTS[aspect])
            map_value, n_queries = image_map(raw, labels, args.query_limit, args.batch_size)
            for kind, value in [('ami', ami), ('map', map_value)]:
                cell = f'{kind}/{aspect}'
                key = f'{cell}/s{seed}'
                cells[key] = value
                differences[key] = abs(value - expected['image'][cell][seed]) if kind == 'ami' or not args.query_limit else None
            states[f'{aspect}/s{seed}'] = state_values
            counts[f'{aspect}/s{seed}'] = n_queries
            print(f'K0 {aspect}/s{seed}: AMI diff={differences[f"ami/{aspect}/s{seed}"]:.3g}; '
                  f'mAP diff={differences[f"map/{aspect}/s{seed}"]}; seconds={time.monotonic() - started:.3f}', flush=True)
    max_differences = {kind: max(v for key, v in differences.items() if key.startswith(kind + '/') and v is not None)
                       if any(key.startswith(kind + '/') and v is not None for key, v in differences.items()) else None
                       for kind in ('ami', 'map')}
    passed = all(value is not None and value <= 1e-6 for value in max_differences.values())
    complete = set(args.aspects) == set(rd.ASPECTS) and set(args.seeds) == set(SEEDS) and not args.query_limit
    report = dict(kind='independent_K0_image_regression', complete=complete, pass_check=passed,
                  seeds=args.seeds, aspects=args.aspects, query_limit=args.query_limit,
                  cells=cells, abs_diff=differences, max_abs_diff=max_differences,
                  ami_per_state=states, scored_queries=counts, k0=baseline, ami_rows=AMI_ROWS,
                  elapsed_seconds=time.monotonic() - START, provenance=provenance(args))
    rd.write_json(args.output, report)
    print(f'K0 regression complete={complete}, pass={passed}, max_abs_diff={max_differences}', flush=True)
    if not passed:
        raise SystemExit(1)


def text_check(args):
    arm, sc1 = selection()
    frame = rd.validation_frame()
    raw, digest = inputs(arm, frame)
    cells, counts = {}, {}
    forward = {}
    for aspect in rd.ASPECTS:
        state = load_state(arm, aspect, args.seed, digest)
        vectors = dict(t=rd.tower(raw['t'], state, 'cap'), i=rd.tower(raw['i'], state, 'img'))
        values, n_queries = text_cells_for(vectors, {aspect: frame['labels'][aspect]}, args.query_limit, args.batch_size)
        cells.update(values)
        counts.update(n_queries)
        path = CANDIDATE_EXPORTS / arm / f'{aspect}__s{args.seed}.npz'
        if path.is_file() and path.with_suffix('.json').is_file():
            stored = export_rows(path, frame, aspect, args.seed)
            forward[aspect] = forward_comparison(vectors['i'], stored)
        print(f'{arm} {aspect}/s{args.seed}: text done', flush=True)
    comparable = args.seed == 0 and not args.query_limit
    difference = {key: abs(value - sc1['arms'][arm]['cells'][key]) for key, value in cells.items()} if comparable else None
    report = dict(kind='independent_SC2_text_validation', kstar=arm, seed=args.seed, query_limit=args.query_limit,
                  full_galleries=True, cells=cells, group=mean(list(cells.values())), counts=counts,
                  sc1_abs_diff=difference, forward_check=forward or {'pending': 'candidate validation exports unavailable'},
                  elapsed_seconds=time.monotonic() - START, provenance=provenance(args))
    rd.write_json(args.output, report)
    print(f'Text check group={report["group"]!r}, seconds={report["elapsed_seconds"]:.3f}', flush=True)
    if difference and max(difference.values()) > 1e-6:
        raise SystemExit('SC1 seed-0 text replay differs by more than 1e-6')


def missing_inputs(arm):
    missing = []
    for seed in SEEDS:
        for aspect in rd.ASPECTS:
            for base, suffix in [(rd.WEIGHTS, '.pt'), (CANDIDATE_EXPORTS, '.npz')]:
                path = base / arm / f'{aspect}__s{seed}{suffix}'
                for candidate in (path, path.with_suffix('.json')):
                    if not candidate.is_file():
                        missing.append(str(candidate))
    return missing


def provenance(args):
    return dict(script_sha256=rd.sha256(__file__), sc1_script_sha256=rd.sha256(rd.__file__),
                sc1_result_sha256=rd.sha256(HERE / 'sc1_rederived.json'),
                k0_valpick_sha256=rd.sha256(BASELINE_RECORDS / 'valpick.json'),
                k0_text_sha256=rd.sha256(BASELINE_RECORDS / 'text.json'),
                validation_names_sha256=rd.VAL_SHA256, numpy=np.__version__, torch=torch.__version__,
                sklearn=sklearn.__version__, device='cpu', threads=torch.get_num_threads(), batch_size=args.batch_size,
                kmeans_parameters=MiniBatchKMeans(n_clusters=9, n_init=3, batch_size=4096, random_state=0).get_params())


def full_run(args):
    arm, sc1 = selection()
    missing = missing_inputs(arm)
    if missing:
        raise SystemExit('Required weights/exports/sidecars missing:\n' + '\n'.join(missing))
    # The independent K0 image regression must have completed before SC2.
    check = rd.read_json(HERE / 'r29rd_sc2_k0_check.json')
    assert check['complete'] and check['pass_check'], 'Run the complete K0 self-check first'
    assert check['provenance']['script_sha256'] == rd.sha256(__file__), 'K0 check predates this implementation'
    assert check['provenance']['k0_valpick_sha256'] == rd.sha256(BASELINE_RECORDS / 'valpick.json')
    baseline, _, _ = frozen_baseline()
    frame = rd.validation_frame()
    raw, digest = inputs(arm, frame)
    expected_counts = rd.finite_query_counts(frame)
    cells = dict(image={f'{m}/{a}': [] for a in rd.ASPECTS for m in ('ami', 'map')},
                 text={f'{a}/{d}': [] for a in rd.ASPECTS for d in rd.DIRECTIONS})
    forward, ami_states, counts, timings, exports = {}, {}, {}, {}, {}
    for seed in SEEDS:
        started = time.monotonic()
        for aspect in rd.ASPECTS:
            path = CANDIDATE_EXPORTS / arm / f'{aspect}__s{seed}.npz'
            stored = export_rows(path, frame, aspect, seed)
            labels = frame['labels'][aspect]['i']
            ami, state_values = ami_cell(stored, labels, CLASS_COUNTS[aspect])
            map_value, n_queries = image_map(stored, labels, batch_size=args.batch_size)
            assert n_queries == int((labels >= 0).sum())
            cells['image'][f'ami/{aspect}'].append(ami)
            cells['image'][f'map/{aspect}'].append(map_value)
            state = load_state(arm, aspect, seed, digest)
            vectors = dict(t=rd.tower(raw['t'], state, 'cap'), i=rd.tower(raw['i'], state, 'img'))
            key = f'{aspect}/s{seed}'
            forward[key] = forward_comparison(vectors['i'], stored)
            ami_states[key] = state_values
            exports[key] = rd.sha256(path)
            values, query_counts = text_cells_for(vectors, {aspect: frame['labels'][aspect]}, 0, args.batch_size)
            for cell, value in values.items():
                assert query_counts[cell] == expected_counts[cell]
                cells['text'][cell].append(value)
                counts[f'{cell}/s{seed}'] = query_counts[cell]
                if seed == 0:
                    assert abs(value - sc1['arms'][arm]['cells'][cell]) <= 1e-6, 'SC1 text mismatch: ' + cell
            print(f'{arm} {key}: image/text/forward done', flush=True)
        timings[str(seed)] = time.monotonic() - started
    per_seed = {group: np.mean(list(values.values()), axis=0, dtype=np.float64).tolist() for group, values in cells.items()}
    means = {group: mean(values) for group, values in per_seed.items()}
    level = level_for(means['image'], means['text'], baseline)
    report = dict(kstar=arm, per_seed=per_seed, mean=means, cells=cells, k0=baseline, level=level,
                  status='pass' if level >= 1 else 'fail', forward_check=forward_summary(forward), ami_rows=AMI_ROWS,
                  notes=NOTES)
    rd.write_json(args.output, report)
    rd.write_json(args.output.with_suffix('.run.json'), dict(elapsed_seconds=time.monotonic() - START,
                  per_seed_seconds=timings, ami_per_state=ami_states, text_counts=counts,
                  export_sha256=exports, caption_store_sha256=digest, provenance=provenance(args)))
    print(f'SC2 means={means}, level={level}, status={report["status"]}, seconds={time.monotonic() - START:.3f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'check-k0', 'check-text', 'run'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--aspects', nargs='+', choices=rd.ASPECTS, default=list(rd.ASPECTS))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=0)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    assert args.query_limit >= 0 and args.batch_size > 0
    defaults = {'inspect': 'r29rd_sc2_inspection.json', 'check-k0': 'r29rd_sc2_k0_check.json',
                'check-text': 'r29rd_sc2_text_check.json', 'run': 'sc2_rederived.json'}
    args.output = args.output or HERE / defaults[args.mode]
    if args.mode != 'run' and args.output.resolve() == HERE / 'sc2_rederived.json':
        parser.error('Only the complete run may write sc2_rederived.json')
    if args.mode == 'run' and (args.query_limit or args.seeds != list(SEEDS) or args.aspects != list(rd.ASPECTS)):
        parser.error('run must compute every cell for all five seeds, without subset arguments')
    if args.mode == 'check-k0':
        k0_check(args)
    elif args.mode == 'check-text':
        text_check(args)
    elif args.mode == 'run':
        full_run(args)
    else:
        arm, _ = selection()
        baseline, cells, values = frozen_baseline()
        report = dict(kstar=arm, k0=baseline, k0_per_seed=values, k0_cells=cells,
                      missing_inputs=missing_inputs(arm), ami_rows=AMI_ROWS,
                      elapsed_seconds=time.monotonic() - START)
        rd.write_json(args.output, report)
        print(f'K*={arm}; K0={baseline}; missing inputs={len(report["missing_inputs"])}', flush=True)


if __name__ == '__main__':
    main()
