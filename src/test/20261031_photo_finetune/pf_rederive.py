"""C13: derive from the rule, frozen protocol and saved vectors; read r13 records only in compare.

No r13 implementation is imported except pf_paths. Clustering, labels, retrieval,
cross-fitting, read-out selection and resampling belong to the frozen r10–r12 code.
The saved float16 vectors feed clustering unchanged; retrieval gets unit float32.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import pf_paths as paths  # installs the frozen rounds on sys.path
import ac_cluster
import ac_data
import ac_select
import ac_stats
import rs_vh
import rt_readout
import rt_stats
from ac_stats import outcome, verdict  # frozen rule, unchanged in r13


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')


def changed(settings):
    return sum(settings[c] != paths.BASE[c] for c in paths.COLUMNS)


def pick_sweep(rows):
    """D9: inclusive best - 0.5, fewest changed columns, then lowest row."""
    candidates = {int(n): e for n, e in rows.items() if int(n) != 0 and e['status'] == 'ok'}
    assert candidates, 'no successful sweep candidate'
    best = max(e['score'] for e in candidates.values())
    tied = sorted(n for n, e in candidates.items() if e['score'] >= best - paths.TIE_MARGIN)
    n = min(tied, key=lambda n: (changed(candidates[n]['settings']), n))
    return {'row': n, 'settings': dict(candidates[n]['settings']), 'best_score': best, 'tied_rows': tied}


def row9(rows):
    """§3.5 combination; failed/missing smoke rows cannot contribute settings."""
    assert '1' in rows and rows['1']['status'] == 'ok', 'row 1 must succeed; stop'
    settings = dict(paths.BASE)
    for column in paths.COLUMNS:
        candidates = [n for n in paths.ROW9_SOURCES if str(n) in rows
                      and rows[str(n)]['status'] == 'ok'
                      and paths.GRID[n][column] != paths.BASE[column]
                      and rows[str(n)]['score'] > rows['1']['score']]
        if candidates:
            n = min(candidates, key=lambda n: (-rows[str(n)]['score'], n))
            settings[column] = paths.GRID[n][column]
    equals = next((n for n, s in paths.GRID.items() if settings == s), None)
    return {'settings': settings, 'skipped': equals is not None, 'equals': equals}


def saved_export(run_id, split):
    """Check export identity, bytes and agreement with its run record.

    Weight hashes must agree with run.json. This audit does not open checkpoints;
    checkpoint validation is the exporter's separate responsibility.
    """
    path = paths.OUTPUTS / 'exports' / run_id / split / 'emotion.npz'
    side = read_json(path.with_suffix('.json'))
    keys = ac_data.keys(split)
    assert side['keys_sha256'] == rs_vh.names_sha(keys), f'{path}: keys hash'
    assert side['run_id'] == run_id and side['split'] == split, f'{path}: export identity'
    assert side['smoke'] is bool(paths.SMOKE), f'{path}: smoke identity'
    weights = side['weights_sha256']
    assert isinstance(weights, str) and len(weights) == 64 and set(weights) <= set('0123456789abcdef')
    check_export_run(run_id, side)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img'}, f'{path}: array fields'
        x = ac_data._array(z['img'], len(keys), 2048, np.float16, unit=True)
    assert rs_vh.array_sha(x) == side['array_sha256'], f'{path}: array hash'
    return x


def frozen_export(seed, split):
    """r12 export checks without ac_data.stored's additional r10 sidecar read."""
    path = paths.R12_OUT / 'exports/cstar' / split / f'emotion__s{seed}.npz'
    side = read_json(path.with_suffix('.json'))
    keys = ac_data.keys(split)
    assert side['keys_sha256'] == rs_vh.names_sha(keys), f'{path}: keys hash'
    assert side['split'] == split and side['seed'] == seed and side['aspect'] == 'emotion'
    assert side['smoke'] is False
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img'}
        x = ac_data._array(z['img'], len(keys), 2048, np.float16, unit=True)
    assert rs_vh.array_sha(x) == side['array_sha256'], f'{path}: array hash'
    return x


def nearcopy_arrays(train_keys, train, val_keys, val, test_keys, test, block=1024):
    """D12 in float32. Exact nearest ties use validation, then sorted key order.

    The train-only count uses symmetric blocks and masks self matches. No train
    photo is removed for matching another train photo. Histogram clips the tiny
    float32 overshoot above one into the final bin.
    """
    assert block > 0
    for keys, x in ((train_keys, train), (val_keys, val), (test_keys, test)):
        ac_data._key_list(keys)
        assert keys == sorted(keys) and len(keys) == len(x)
    assert not (set(train_keys) & (set(val_keys) | set(test_keys)))
    assert not set(val_keys) & set(test_keys)
    train, val, test = map(ac_data._unit, (train, val, test))
    held = np.concatenate((val, test))
    nearest, split = np.empty(len(train), np.float32), np.empty(len(train), np.int64)
    copies = np.zeros(len(train), dtype=bool)
    with threadpool_limits(8):
        for lo in range(0, len(train), block):
            hi = min(lo + block, len(train))
            sims = train[lo:hi] @ held.T
            idx = sims.argmax(axis=1)
            nearest[lo:hi] = sims[np.arange(hi - lo), idx]
            split[lo:hi] = idx >= len(val)
            for other in range(lo, len(train), block):
                end = min(other + block, len(train))
                sims = train[lo:hi] @ train[other:end].T
                if other == lo:
                    np.fill_diagonal(sims, -np.inf)
                match = sims >= paths.NEAR_COPY_CUT
                copies[lo:hi] |= match.any(axis=1)
                copies[other:end] |= match.any(axis=0)
    removed = nearest >= paths.NEAR_COPY_CUT
    sources = sorted({k.split('/', 1)[0] for k in train_keys})
    by_source = {s: {'val': 0, 'test': 0} for s in sources}
    for i in np.flatnonzero(removed):
        by_source[train_keys[i].split('/', 1)[0]][('val', 'test')[split[i]]] += 1
    edges = np.linspace(.80, 1., 21)
    counts, _ = np.histogram(np.minimum(nearest, 1.), bins=edges)
    return {'cut': paths.NEAR_COPY_CUT, 'n_train': len(train), 'n_removed': int(removed.sum()),
            'removed_keys': [k for k, keep in zip(train_keys, removed) if keep],
            'by_source': by_source, 'hist': {'edges': edges.tolist(), 'counts': counts.tolist()},
            'within_train_copies': int(copies.sum())}


def store_split_keys():
    """Read only image rows of the index; never materialize its text/label rows.

    Held keys come from r12's frozen key loader; train split membership is image
    metadata, with ignored membership discarded. PF_SMOKE limits train only.
    """
    import ijson
    with (paths.STORE / 'index.json').open('rb') as handle:
        images = list(ijson.items(handle, 'images.item'))
    stored = [f"{r['source']}/{r['image_name']}" for r in images]
    ac_data._key_list(stored)
    split_keys = {s: [] for s in paths.SPLITS}
    for key, row in zip(stored, images):
        splits = set(row['splits']) - {'ignored'}
        assert len(splits) == 1 and splits <= set(paths.SPLITS), (key, splits)
        split_keys[next(iter(splits))].append(key)
    for s in paths.SPLITS:
        split_keys[s].sort()
        assert len(split_keys[s]) == paths.N[s], (s, len(split_keys[s]))
    for s in ('val', 'test'):
        assert split_keys[s] == ac_data.keys(s), f'{s}: store and frozen key loader differ'
    if paths.SMOKE:
        split_keys['train'] = split_keys['train'][:paths.SMOKE]
    return stored, split_keys


def derive_nearcopy():
    stored, split_keys = store_split_keys()
    x = np.load(paths.STORE / 'img.npy', mmap_mode='r', allow_pickle=False)
    assert x.shape == (len(stored), 2048) and x.dtype == np.float16
    arrays = [x[ac_data._gather(stored, split_keys[s])] for s in paths.SPLITS]
    return nearcopy_arrays(split_keys['train'], arrays[0], split_keys['val'], arrays[1],
                           split_keys['test'], arrays[2])


def sweep_score(stored, settings, status='ok', workers=8):
    labels = ac_data.labels('val')
    ami = ac_cluster.ami(ac_cluster.partitions(stored), labels)
    # The frozen cross-fit accepts one seed as None (not the incomplete set {0}).
    table, _ = ac_select.arm_table(
        {('emotion', None): ac_data._unit(stored)}, {'emotion': labels},
        {'emotion': paths.R12_R_MAP}, fixed=ac_select.readout_from_name(paths.SWEEP_READOUT),
        workers=workers, keys=ac_data.keys('val'))
    mp = table[paths.SWEEP_READOUT]['map']['emotion']
    g_clu = (ami - paths.TEACHER_AMI) / paths.R12_R_CLU
    g_map = (mp - paths.TEACHER_MAP) / paths.R12_R_MAP
    return {'settings': None if settings is None else dict(settings),
            'changed': None if settings is None else changed(settings), 'status': status,
            'ami': ami, 'map': mp, 'g_clu': g_clu, 'g_map': g_map, 'score': min(g_clu, g_map)}


def run_status(run_id):
    """Training status is taken from run.json, independently of result records."""
    status = read_json(paths.OUTPUTS / 'runs' / run_id / 'run.json')['status']
    assert status in ('ok', 'failed'), (run_id, status)
    return status


def check_export_run(run_id, side):
    run = read_json(paths.OUTPUTS / 'runs' / run_id / 'run.json')
    assert run['status'] == 'ok', f'{run_id}: export of an unsuccessful run'
    assert side['weights_sha256'] == run['weights_sha256'], f'{run_id}: export weights_sha256 differs from run'


def validate_runs(sheet, nearcopy):
    """Audit D10 and the fit set against independently derived decisions.

    Only run.json, export sidecars and label-free store membership are read.
    Failed sweep runs must attest to the same fit keys as successful runs.
    """
    _, split_keys = store_split_keys()
    train = split_keys['train']
    removed = set(nearcopy['removed_keys'])
    assert removed <= set(train), 'removed keys are not all train keys'
    fit = [k for k in train if k not in removed]
    fit_sha = rs_vh.names_sha(fit)
    root = paths.OUTPUTS / 'runs'
    for path in sorted(root.rglob('run.json')):
        run = read_json(path)
        assert run.get('fit_keys_sha256') == fit_sha, f'{path}: fit_keys_sha256 differs from derived fit set'
    for s in paths.SEEDS:
        path = root / f'final/s{s}/run.json'
        run = read_json(path)
        assert run['status'] == 'ok', f'final/s{s} failed; cannot give a verdict'
        assert run['settings'] == sheet['pick']['settings'], f'{path}: settings differ from derived pick'
        assert type(run['train_seed']) is int and run['train_seed'] == paths.train_seed(s), f'{path}: train_seed'
        assert type(run['cstar_seed']) is int and run['cstar_seed'] == s, f'{path}: cstar_seed'
    # Include sidecars for any extra saved split, even if scoring does not use it.
    exports = paths.OUTPUTS / 'exports'
    for path in sorted(exports.rglob('emotion.json')):
        run_id = path.parent.parent.relative_to(exports).as_posix()
        check_export_run(run_id, read_json(path))
    return fit


def derive_sweep(workers=8):
    rows = {'0': sweep_score(frozen_export(0, 'val'), None, 'reference', workers)}

    def add(n, settings):
        run_id = f'sweep/row{n}'
        status = run_status(run_id)
        if status == 'failed':
            rows[str(n)] = {'settings': dict(settings), 'changed': changed(settings), 'status': status,
                            **dict.fromkeys(('ami', 'map', 'g_clu', 'g_map', 'score'))}
        else:
            rows[str(n)] = sweep_score(saved_export(run_id, 'val'), settings, workers=workers)

    for n, settings in paths.GRID.items():
        if (paths.OUTPUTS / 'runs' / f'sweep/row{n}/run.json').exists():
            add(n, settings)
            assert n != 1 or rows['1']['status'] == 'ok', 'row 1 failed; stop'
    ninth = row9(rows)
    if (paths.OUTPUTS / 'runs/sweep/row9/run.json').exists():
        assert not ninth['skipped'], 'row 9 ran despite equalling an earlier row'
        add(9, ninth['settings'])
    elif ninth['skipped']:
        rows['9'] = {'settings': dict(ninth['settings']), 'changed': changed(ninth['settings']),
                     'status': 'skipped', **dict.fromkeys(('ami', 'map', 'g_clu', 'g_map', 'score'))}
    return {'rows': rows, 'row9': ninth, 'pick': pick_sweep(rows)}


def seed_ranges(get_stored, workers=8):
    """All-gallery R0, not the cross-fit scores used for picking read-outs."""
    return ac_stats.measure_seed_ranges(
        lambda s: ac_data._unit(get_stored(s, 'val')), lambda s: get_stored(s, 'val'),
        ac_data.labels('val'), ac_data.emotion_dist('val'), workers=workers)


def readout_menu(get_stored, ranges, workers=8):
    scores, pick = ac_select.arm_table(
        {('emotion', s): ac_data._unit(get_stored(s, 'val')) for s in paths.SEEDS},
        {'emotion': ac_data.labels('val')}, {'emotion': ranges['map']['r']},
        workers=workers, keys=ac_data.keys('val'))
    return {'pick': rt_readout.name(pick), 'scores': scores}


def comparator_ap(arm, stage='stage2'):
    path = paths.R12_OUT / stage / 'perquery' / f'affection__{arm}__emotion__s-.npz'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'ap', 'rows'}, path
        ap, rows = z['ap'], z['rows']
    n = len(ac_data.keys('test'))
    assert ap.dtype == np.float64 and ap.shape == (n,) and np.isfinite(ap).all(), path
    assert rows.dtype == np.int64 and np.array_equal(rows, np.arange(n)), f'{path}: query identity'
    assert ((ap >= 0) & (ap <= 1)).all(), path
    return {'ap': ap, 'rows': rows, 'dropped': 0, 'point': float(ap.mean()), 'per_seed': None}


def comparator_partitions(arm):
    path = paths.R12_OUT / 'partitions' / f'affection__test__{arm}__s-.npz'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'labels'}, path
        parts = z['labels']
    assert parts.dtype == np.int32 and parts.shape == (3, len(ac_data.keys('test'))), path
    assert ((parts >= 0) & (parts < 20)).all(), path
    return parts


def check_draws(draws_clu, draws_ret):
    assert type(draws_clu) is int and 0 < draws_clu <= 2000, 'clustering draws must be in 1..2000'
    assert type(draws_ret) is int and 0 < draws_ret <= 10000, 'retrieval draws must be in 1..10000'
    assert paths.SMOKE or (draws_clu, draws_ret) == (2000, 10000), 'reduced draws require PF_SMOKE > 0'


def score_criteria(get_stored, ranges, readout, candidate='photo', only=None, workers=8,
                   draws_clu=2000, draws_ret=10000):
    """r13 SC5 is r12 SC8. Frozen wrappers supply all numerical statistics.

    Rebind only the candidate's name/read-out and the SC numbering; comparator
    picks, AP rows and partitions are frozen r12 outputs. No r13 result is read.
    """
    check_draws(draws_clu, draws_ret)
    picks = read_json(paths.R12_OUT / 'stage1/picks.json')
    picks['affection']['readout']['cstar'] = readout
    labels = ac_data.labels('test')
    ids = ('SC1', 'SC2', 'SC3', 'SC4', 'SC5') if only is None else only
    cand_parts, cand_ap = None, None
    result = {}
    for sid in ids:
        old = 'SC8' if sid == 'SC5' else sid
        comp = ac_stats.comparator_arm(old, picks)
        if sid in ('SC1', 'SC3'):
            if cand_parts is None:
                cand_parts = np.stack([ac_cluster.partitions(get_stored(s, 'test')) for s in paths.SEEDS])
            paired = ac_cluster.paired(cand_parts, comparator_partitions(comp), labels, draws=draws_clu, workers=workers)
            entry = ac_stats.cluster_sc(old, paired, picks, ranges)
        else:
            if cand_ap is None:
                cand_ap = ac_stats.arm_retrieval(
                    lambda s: ac_data._unit(get_stored(s, 'val')),
                    lambda s: ac_data._unit(get_stored(s, 'test')),
                    readout, labels, seeds=paths.SEEDS, workers=workers)
            entry = ac_stats.retrieval_sc(old, cand_ap, comparator_ap(comp, 'stage3' if sid == 'SC5' else 'stage2'),
                                          picks, ranges, {}, draws=draws_ret)
        entry['candidate'] = candidate
        result[sid] = entry
    return result


def derive(workers=8, draws_clu=2000, draws_ret=10000):
    """The allowed-data-only pass; no r13 records, gates or partitions are read."""
    start = time.monotonic()
    check_draws(draws_clu, draws_ret)
    sheet = derive_sweep(workers)
    for s in paths.SEEDS:
        assert run_status(f'final/s{s}') == 'ok', f'final/s{s} failed; cannot give a verdict'
    cache = {}

    def stored(s, split):
        if (s, split) not in cache:
            cache[s, split] = saved_export(f'final/s{s}', split)
        return cache[s, split]

    nearcopy = derive_nearcopy()
    validate_runs(sheet, nearcopy)
    ranges = seed_ranges(stored, workers)
    readout = readout_menu(stored, ranges, workers)
    sc = score_criteria(stored, ranges, readout['pick'], workers=workers,
                        draws_clu=draws_clu, draws_ret=draws_ret)
    result = {'nearcopy': nearcopy, 'sheet': sheet,
              'stage2': {'ranges': ranges, 'r12_ranges': read_json(paths.R12_OUT / 'stage0b/ranges.json'),
                         'readout': readout},
              'stage3': {'sc': sc, 'verdict': verdict({sid: sc[sid]['outcome'] for sid in ('SC1', 'SC2', 'SC3', 'SC4')}),
                         'claims': {'SC5': sc['SC5']['pass']}, 'draws_clu': draws_clu, 'draws_ret': draws_ret},
              'smoke': paths.SMOKE, 'seconds': time.monotonic() - start}
    write_json(paths.OUTPUTS / 'rederive/theirs.json', result)
    return result


def compare_documents(ours, theirs, smoke=False):
    """One item per leaf, with removed keys compared as a set.

    Numeric measurements use absolute 1e-6 tolerance. Settings, integer counts,
    picks, booleans and outcomes are exact. Missing evidence always fails. Smoke
    requires all derived sweep rows and ignores extra recorded rows; ranges,
    menu, SCs and choices still all compare.
    """
    missing = object()
    items = {}

    def visit(a, b, trail):
        name = '.'.join(trail)
        if trail == ('sheet', 'rows') and smoke and isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(b):
                visit(a.get(key, missing), b[key], (*trail, key))
            return
        if isinstance(a, dict) or isinstance(b, dict):
            ad, bd = a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}
            for key in sorted(set(ad) | set(bd)):
                visit(ad.get(key, missing), bd.get(key, missing), (*trail, str(key)))
            return
        exact_list = name == 'nearcopy.removed_keys' or name.endswith('.tied_rows')
        if not exact_list and (isinstance(a, list) or isinstance(b, list)):
            al, bl = a if isinstance(a, list) else [], b if isinstance(b, list) else []
            for i in range(max(len(al), len(bl))):
                visit(al[i] if i < len(al) else missing, bl[i] if i < len(bl) else missing, (*trail, str(i)))
            return
        ok = False
        if a is not missing and b is not missing:
            if name == 'nearcopy.removed_keys':
                ok = (isinstance(a, list) and isinstance(b, list) and len(a) == len(set(a))
                      and len(b) == len(set(b)) and set(a) == set(b))
            elif isinstance(a, float) and isinstance(b, float) and 'settings' not in trail:
                ok = bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= paths.TOL['c13'])
            else:
                ok = type(a) is type(b) and a == b
        items[name] = {'ours': '<missing>' if a is missing else a,
                       'theirs': '<missing>' if b is missing else b, 'ok': ok}

    visit(ours, theirs, ())
    return {'items': items, 'ok': bool(items) and all(e['ok'] for e in items.values())}


def compare():
    """The only step that opens r13 result records. Never called by derive."""
    theirs_full = read_json(paths.OUTPUTS / 'rederive/theirs.json')
    theirs = {k: theirs_full[k] for k in ('nearcopy', 'sheet', 'stage2', 'stage3')}
    ours = {}
    for key, path in (('nearcopy', paths.NEARCOPY_FILE), ('sheet', paths.SHEET_FILE),
                      ('stage2', paths.STAGE2_FILE), ('stage3', paths.OUTPUTS / 'stage3/verdict.json')):
        ours[key] = read_json(path) if path.is_file() else {}
    # Stage 3 also contains descriptive rows and run metadata outside C13.
    ours['stage3'] = {k: v for k, v in ours['stage3'].items()
                      if k in ('sc', 'verdict', 'claims', 'draws_clu', 'draws_ret')}
    result = compare_documents(ours, theirs, smoke=paths.SMOKE > 0)
    if theirs_full.get('smoke') != paths.SMOKE:
        result['items']['smoke'] = {'ours': paths.SMOKE, 'theirs': theirs_full.get('smoke'), 'ok': False}
        result['ok'] = False
    write_json(paths.OUTPUTS / 'rederive/rederive.json', result)
    failed = sum(not e['ok'] for e in result['items'].values())
    print(f"C13 {'PASS' if result['ok'] else 'FAIL'}: {len(result['items'])} items, {failed} mismatches")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    derive_parser = sub.add_parser('derive', help='recompute from saved vectors without r13 result records')
    derive_parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    derive_parser.add_argument('--draws-clu', type=int, default=None, help='smoke only: 1..2000 draws')
    derive_parser.add_argument('--draws-ret', type=int, default=None, help='smoke only: 1..10000 draws')
    sub.add_parser('compare', help='compare theirs.json with r13 result records')
    args = parser.parse_args(argv)
    if args.command == 'compare':
        return 0 if compare()['ok'] else 1
    if not paths.SMOKE and (args.draws_clu is not None or args.draws_ret is not None):
        parser.error('draw overrides require PF_SMOKE > 0')
    draws_clu = 2000 if args.draws_clu is None else args.draws_clu
    draws_ret = 10000 if args.draws_ret is None else args.draws_ret
    try:
        check_draws(draws_clu, draws_ret)
    except AssertionError as exc:
        parser.error(str(exc))
    result = derive(args.workers, draws_clu=draws_clu, draws_ret=draws_ret)
    print(f"C13 derived: {result['stage3']['verdict']}, {result['seconds']:.1f}s; rederive/theirs.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
